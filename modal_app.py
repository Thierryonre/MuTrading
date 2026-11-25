"""Modal jobs (used sparingly - limited credits).

    modal run modal_app.py::score_corpus --model finbert
    modal run modal_app.py::score_corpus --model nosible
"""
import io

import modal

app = modal.App("rltrading")

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
