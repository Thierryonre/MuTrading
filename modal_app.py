"""Modal jobs (used sparingly - limited credits).

    modal run modal_app.py::score_corpus --model finbert
    modal run modal_app.py::score_corpus --model nosible
"""
import io

import modal

app = modal.App("mutrading")

score_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.5.1", "transformers==4.51.3", "pandas", "pyarrow", "numpy<2.3")
)

MODELS = {
    "finbert": "ProsusAI/finbert",
    "nosible": "NOSIBLE/financial-sentiment-v1.1-base",
}


@app.function(image=score_image, gpu="T4", timeout=3 * 3600, max_containers=4)
def score(model_key: str, texts: list[str], batch_size: int = 128) -> bytes:
    """Return parquet bytes with p_pos, p_neu, p_neg for each input text (same order)."""
    import numpy as np
    import pandas as pd
    import torch
    from transformers import (AutoModelForCausalLM, AutoModelForSequenceClassification,
                              AutoTokenizer)

    name = MODELS[model_key]
    dev = "cuda"
    out = np.zeros((len(texts), 3), dtype=np.float32)  # pos, neu, neg
    # sort by length for efficient batching
    order = np.argsort([len(t) for t in texts])
    if model_key == "finbert":
        tok = AutoTokenizer.from_pretrained(name)
        model = AutoModelForSequenceClassification.from_pretrained(name).half().to(dev).eval()
        lab = {v.lower(): k for k, v in model.config.id2label.items()}
        idx = [lab["positive"], lab["neutral"], lab["negative"]]
        for s in range(0, len(texts), batch_size):
            b = order[s:s + batch_size]
            enc = tok([texts[i] for i in b], padding=True, truncation=True, max_length=128,
                      return_tensors="pt").to(dev)
            with torch.no_grad():
                p = model(**enc).logits.float().softmax(-1)[:, idx].cpu().numpy()
            out[b] = p
    else:
        tok = AutoTokenizer.from_pretrained(name)
        tok.padding_side = "left"
        model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.float16).to(dev).eval()
        label_ids = [tok.encode(w, add_special_tokens=False)[0] for w in ["positive", "neutral", "negative"]]
        sys_prompt = "Classify the financial sentiment as positive, neutral, or negative."

        def fmt(t):
            return tok.apply_chat_template([{"role": "system", "content": sys_prompt},
                                            {"role": "user", "content": t}],
                                           tokenize=False, add_generation_prompt=True, enable_thinking=False)
        for s in range(0, len(texts), batch_size):
            b = order[s:s + batch_size]
            enc = tok([fmt(texts[i]) for i in b], padding=True, truncation=True, max_length=256,
                      return_tensors="pt").to(dev)
            with torch.no_grad():
                logits = model(**enc, logits_to_keep=1).logits[:, -1, :].float()
            out[b] = logits[:, label_ids].softmax(-1).cpu().numpy()
        if len(texts) <= 10:
            print([tok.decode(i) for i in label_ids])
    buf = io.BytesIO()
    pd.DataFrame(out, columns=["p_pos", "p_neu", "p_neg"]).to_parquet(buf)
    return buf.getvalue()


@app.local_entrypoint()
def score_corpus(model: str = "finbert", chunk: int = 60000, limit: int = 0):
    import os
    from pathlib import Path

    import pandas as pd

    data_dir = Path(os.environ.get("RLT_DATA_DIR", "data")) / "cache"
    corpus = pd.read_parquet(data_dir / "corpus.parquet", columns=["id", "text"])
    if limit:
        corpus = corpus.iloc[:limit]
    texts = corpus.text.tolist()
    chunks = [texts[i:i + chunk] for i in range(0, len(texts), chunk)]
    results = list(score.map([model] * len(chunks), chunks))
    probs = pd.concat([pd.read_parquet(io.BytesIO(r)) for r in results], ignore_index=True)
    probs.insert(0, "id", corpus.id.values)
    dest = data_dir / f"scores_{model}{'_sample' if limit else ''}.parquet"
    probs.to_parquet(dest)
    print("saved", dest, len(probs))
    print(probs.describe())


# ----------------------------------------------------------------------------- RL training on CPU
import os as _os

_LZ_DIR = _os.path.expanduser("~/rlt/LightZero")
# only the small pre-built feature arrays are shipped (copied here by scripts/stage_modal_data.sh)
_CACHE = _os.path.expanduser("~/rlt/modal_data")

train_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("build-essential", "git")
    .pip_install("numpy==1.26.4")  # match local env (NumPy 2 breaks some LightZero ctree calls)
    .pip_install("torch", index_url="https://download.pytorch.org/whl/cpu")
    .pip_install("DI-engine @ git+https://github.com/opendilab/DI-engine.git@ce54f6baa8d39d5b98adaf9a7e82560a444a2b1a")
    .add_local_dir(_LZ_DIR, "/opt/LightZero", copy=True,
                   ignore=[".git", "**/__pycache__", "**/*.so", "build", "*.egg-info"])
    .run_commands("cd /opt/LightZero && pip install -e . && pip install pandas pyarrow tabulate")
    .env({"RLT_DATA_DIR": "/data", "PYTHONUNBUFFERED": "1"})
    .add_local_dir(_CACHE, "/data/cache")
    .add_local_python_source("rltrading")
)
exp_vol = modal.Volume.from_name("rltrading-exp", create_if_missing=True)


@app.function(image=train_image, cpu=4.0, memory=6144, timeout=6 * 3600, volumes={"/vol": exp_vol},
              max_containers=6)
def train_eval(argv: list[str]) -> dict:
    """Train one agent (argv for rltrading.train_lz), select checkpoint on val, backtest on test.
    Returns the small result files {relative_path: bytes}."""
    import glob
    import os

    os.makedirs("/vol/rlt", exist_ok=True)
    os.chdir("/vol/rlt")
    from rltrading import backtest, train_lz
    from rltrading import config as C
    exp_dir = train_lz.main(argv + ["--threads", "4"])
    exp_vol.commit()
    return _evaluate(exp_dir)


def _evaluate(exp_dir):
    import glob
    import os

    from rltrading import backtest
    from rltrading import config as C
    summary = backtest.evaluate_exp(exp_dir)
    out_dir = C.RESULTS_DIR / "runs" / summary["name"]
    files = {os.path.relpath(p, C.RESULTS_DIR): open(p, "rb").read()
             for p in glob.glob(str(out_dir / "*"))}
    for rel, b in files.items():  # also keep a copy on the volume in case the local client disconnects
        dest = os.path.join("/vol/results", rel)
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        open(dest, "wb").write(b)
    exp_vol.commit()
    return files


@app.function(image=train_image, cpu=2.0, memory=4096, timeout=2 * 3600, volumes={"/vol": exp_vol},
              max_containers=6)
def eval_only(exp_dir: str, run_args: dict | None = None) -> dict:
    """Backtest an already-trained experiment on the volume (exp_dir relative to /vol/rlt)."""
    import json
    import os
    os.chdir("/vol/rlt")
    if run_args is not None:
        json.dump(run_args, open(os.path.join(exp_dir, "run_args.json"), "w"))
    return _evaluate(exp_dir)


@app.local_entrypoint()
def sweep(spec: str = "", seeds: str = "0", common: str = ""):
    """spec: comma list of algo:sentiment, e.g. "muzero:none,stochastic_muzero:finbert"."""
    from pathlib import Path
    jobs = []
    for item in spec.split(","):
        algo, sent = item.split(":")
        for s in seeds.split(","):
            jobs.append(["--algo", algo, "--sentiment", sent, "--seed", s] + common.split())
    res_dir = Path(__file__).parent / "results"
    for files in train_eval.map(jobs, return_exceptions=True):
        if isinstance(files, Exception):
            print("FAILED:", repr(files)[:2000])
            continue
        for rel, b in files.items():
            p = res_dir / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(b)
            print("saved", p)


@app.local_entrypoint()
def recover(only: str):
    """Backtest trained runs on the volume that have checkpoints but no results yet
    (e.g. after a crash in the evaluation step). `only`: comma list of exact exp dir names
    (with LightZero timestamp suffix) - never pass runs that are still training."""
    import json
    import re
    from pathlib import Path
    names = [e.path.split("/")[-1] for e in exp_vol.listdir("rlt/exp")]
    done = {e.path.split("/")[-1] for e in exp_vol.listdir("results/runs")} if _vol_has("results/runs") else set()
    jobs = []
    for n in names:
        base = re.sub(r"_\d{6}_\d{6}$", "", n)
        if n not in only.split(",") or base in done:
            continue
        try:
            args = json.loads(b"".join(exp_vol.read_file(f"rlt/exp/{base}/run_args.json")))
        except Exception:
            print("no run_args for", n)
            continue
        jobs.append((f"exp/{n}", args))
    print("recovering", [j[0] for j in jobs])
    res_dir = Path(__file__).parent / "results"
    for files in eval_only.starmap(jobs, return_exceptions=True):
        if isinstance(files, Exception):
            print("FAILED:", repr(files)[:2000])
            continue
        for rel, b in files.items():
            p = res_dir / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(b)
            print("saved", p)


def _vol_has(path):
    try:
        exp_vol.listdir(path)
        return True
    except Exception:
        return False
