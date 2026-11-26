# RLTrading — MuZero-family agents + free news/forum sentiment for daily trading

Can the MuZero family of model-based RL algorithms (via [LightZero](https://github.com/opendilab/LightZero))
learn to trade daily US equities/ETFs, and does adding free news / forum sentiment help?

**Short answer: no.** Across 7 MuZero-family algorithms, 2 sentiment models, long/short and long-only action spaces
and multiple seeds, no configuration *reliably* beat buy-and-hold on the 2019-07 → 2022-12 test period.
One run (Stochastic MuZero + NOSIBLE LLM sentiment, long-only, seed 0) did — Sharpe 1.11 vs 0.65 — but the
other two seeds of the same configuration scored 0.29 and −0.47 (E5). See [Conclusions](#conclusions) and the
[Experiment log](#experiment-log) for every test and result.

## Setup in one paragraph

* **Universe:** 10 ETFs (SPY QQQ IWM DIA XLF XLE XLK XLV GLD TLT) + 15 large caps
  (AAPL MSFT AMZN GOOGL JPM XOM JNJ WMT NVDA BAC INTC KO PG CSCO PFE). Daily bars from Yahoo Finance (free).
* **Splits (chronological):** train 2010-01 → 2017-12, validation 2018-01 → 2019-06 (checkpoint selection),
  **test 2019-07 → 2022-12** (touched once per run; includes the COVID crash, 2021 rally and 2022 bear market).
* **Environment:** one asset per episode; at each close the agent chooses a target position
  (discrete {-1, 0, +1} or continuous [-1, 1] for Sampled MuZero). Reward = position × next-day log return − 5 bps × |Δposition|.
  Training episodes are random 128-day windows of random assets; evaluation runs every asset over the full split.
* **Observation (24 price features + optional 6 sentiment features + current position):** vol-normalised returns (1/5/20d and last 10 days),
  realised vol, SMA distances, RSI, volume z-score, range, SPY return, VIX z-score/change; sentiment features below.
  Normalised with training-period statistics only.
* **Algorithms (LightZero):** MuZero, EfficientZero, Gumbel MuZero, Stochastic MuZero, Sampled MuZero,
  Sampled EfficientZero, UniZero.
* **Metrics:** equal-weight portfolio of the 25 per-asset strategies: Sharpe, CAGR, max drawdown, turnover;
  compared against buy-and-hold, flat, SMA-200 trend, 20-day momentum and simple sentiment rules.

## Sentiment pipeline (no paid APIs)

* **Text:** the free Hugging Face corpus
  [`financial-news-multisource`](https://huggingface.co/datasets/Brianferrell787/financial-news-multisource)
  (ungated mirror `idleengine/financial-news-multisource`; research / non-commercial use). Subsets used:
  CNBC headlines (2007–2022), NYT business desk (2010–2021), finance-keyword-filtered headlines from 9 other outlets
  (2007–2022), S&P 500 / FinSen / CNBC small sets, and **Reddit** finance forums (r/wallstreetbets, r/stocks, r/investing, …; 2008–2025, dense only from 2020).
  ≈474k texts after cleaning/dedup.
* **Models:** [ProsusAI/FinBERT](https://huggingface.co/ProsusAI/finbert) (classic), and
  [NOSIBLE financial-sentiment v1.1](https://huggingface.co/NOSIBLE/financial-sentiment-v1.1-base)
  (2025, Qwen3-0.6B fine-tuned LLM classifier — the "newer" option). Scored on a Modal T4.
* **No look-ahead:** timestamped items published after 15:30 ET count for the next session; date-only items are
  shifted one trading day. Market sentiment is z-scored per source with trailing 252-day stats, then averaged.
* **Features:** `mkt_sent1/5/20` (market news sentiment, EMA 1/5/20d), `mkt_newsvol`,
  `asset_sent5` / `asset_newscnt` (keyword-tagged news per asset), `forum_sent5` / `forum_vol` (Reddit, upvote-weighted).

## Reproduce

```bash
# Linux / WSL, python 3.10
git clone https://github.com/opendilab/LightZero && pip install -e LightZero   # + torch CPU
pip install -r requirements.txt
export RLT_DATA_DIR=~/rlt/data PYTHONPATH=$PWD
python -m rltrading.data.prices                    # prices
python -m rltrading.data.news                      # news/forum parquet (~3.6 GB)
python -m rltrading.sentiment.corpus               # build text corpus
modal run modal_app.py::score_corpus --model finbert   # (or nosible) GPU scoring
python -m rltrading.sentiment.index finbert        # daily features -> data/processed/
python scripts/sentiment_ic.py finbert nosible     # predictive power check
python -m rltrading.backtest --baselines
python -m rltrading.train_lz --algo stochastic_muzero --sentiment finbert --seed 0
python -m rltrading.backtest --exp exp/stochastic_muzero_finbert_seed0
```

## Experiment log

Results are appended chronologically. All returns are net of 5 bps per unit of position change.

### E0 — Baselines (equal-weight portfolio of 25 assets)

See `results/baselines_finbert.csv`.

| Strategy | Val Sharpe | Val CAGR | Test Sharpe | Test CAGR | Test MaxDD |
|---|---|---|---|---|---|
| Buy & hold | 0.69 | 10.1% | 0.65 | 12.7% | -29.4% |
| Flat (cash) | 0 | 0 | 0 | 0 | 0 |
| Long above SMA-200 | 0.19 | 1.5% | 0.91 | 9.4% | -16.4% |
| 20-day momentum long/short | -0.08 | -1.6% | 0.21 | 2.3% | -29.4% |
| Long if 5d market sentiment < 0 (contrarian) | 0.57 | 7.3% | 0.10 | 0.0% | -29.6% |
| Long if 5d market sentiment > 0 (follow) | -0.22 | -1.7% | 0.81 | 7.9% | -12.4% |

The two sentiment rules flip between validation and test — an early warning that the sentiment signal is weak/unstable.

### E1 — Does sentiment predict returns? (rank IC, FinBERT vs NOSIBLE LLM)

Full tables in `results/sentiment_ic.md`. Market features evaluated on SPY; asset features as daily cross-sectional IC.

| Feature | FinBERT train / val / test (5-day IC) | NOSIBLE train / val / test (5-day IC) |
|---|---|---|
| `mkt_sent5` (5d market news sentiment) | −0.063 / −0.076 / −0.004 | −0.065 / −0.072 / +0.037 |
| `mkt_sent20` | −0.063 / −0.090 / +0.033 | −0.058 / −0.105 / +0.043 |
| `mkt_sent1` (1-day IC) | −0.011 / +0.002 / +0.044 | 0.000 / −0.058 / +0.075 |
| `asset_sent5` (cross-sectional) | +0.007 / +0.040 / +0.010 | +0.004 / +0.022 / −0.004 |
| `forum_sent5` (Reddit) | +0.022 / −0.056 / −0.025 | +0.040 / −0.128 / +0.015 |
| `mkt_newsvol` (news volume) | −0.004 / −0.028 / +0.101 | same |

Findings:
* Signals are weak (|IC| ≲ 0.1) and **unstable**: market news sentiment was mildly *contrarian* in train and validation
  (high sentiment → lower next-week returns) but turned *momentum-like* in the test period. Anything learned on 2010–2019
  about sentiment is likely to hurt in 2019–2022.
* The two models agree closely (daily correlation 0.78–0.86 across features). The newer NOSIBLE LLM gives slightly larger
  magnitudes, but in the same unstable directions.
* Reddit forum data is too sparse before 2017 (<1k posts/yr) to train on, and its sign flips too. **News beats forums
  here** because it covers the whole period; forum features are tested only as an add-on ablation.

### E2 — Model-free reference: PPO (stable-baselines3), 1M steps, seed 0

| Run | Val Sharpe (selected ckpt) | Test Sharpe | Test CAGR | Test MaxDD | Avg position |
|---|---|---|---|---|---|
| PPO, no sentiment | see `results/runs/ppo_none_seed0` | 0.16 | 1.2% | −25.0% | +0.45 |
| PPO + FinBERT sentiment | see `results/runs/ppo_finbert_seed0` | 0.43 | 5.6% | −17.1% | +0.40 |

Both are below buy-and-hold (0.65).

### E3 — MuZero family sweep, seed 0 (LightZero, 120k env steps ≈ 30k gradient steps; MuZero pilots 150k)

Common settings: MLP world model (latent 128), 25 MCTS simulations, γ = 0.95, td_steps 5, replay ratio 0.25, batch 256,
SSL head shrunk to 256 wide (see *Engineering notes*). Discrete actions {short, flat, long}; Sampled MuZero uses a continuous
position in [−1, 1] with K = 10 sampled actions. Checkpoint chosen by validation Sharpe among ≤ 8 evenly spaced checkpoints.
Full table: `results/summary.md`; equity curves: `results/equity_curves_test.png`.

| Algorithm | Sentiment | Val Sharpe | **Test Sharpe** | Test CAGR | Test MaxDD | Turnover (×/yr) | Avg position |
|---|---|---|---|---|---|---|---|
| *Buy & hold* | – | 0.69 | **0.65** | 12.7% | −29.4% | 0.3 | 1.00 |
| MuZero | none | 0.68 | −0.25 | −3.3% | −26.8% | 40 | 0.44 |
| MuZero | FinBERT | 0.19 | −0.07 | −0.7% | −13.4% | 31 | 0.20 |
| MuZero | NOSIBLE | -0.32 | -0.03 | -0.5% | -9.4% | 34 | 0.23 |
| EfficientZero | none | 0.47 | 0.34 | 3.9% | −23.0% | 37 | 0.24 |
| EfficientZero | FinBERT | 0.60 | −0.09 | −1.9% | −18.6% | 28 | 0.23 |
| Gumbel MuZero | none | −0.36 | −0.25 | −7.4% | −49.2% | 5 | −0.99 |
| Gumbel MuZero | FinBERT | 0.19 | −1.04 | −5.1% | −17.0% | 41 | 0.12 |
| Stochastic MuZero | none | −0.47 | −0.37 | −3.9% | −19.0% | 36 | 0.33 |
| Stochastic MuZero | FinBERT | 0.59 | 0.02 | −0.6% | −22.1% | 28 | 0.18 |
| Stochastic MuZero | NOSIBLE | 0.60 | 0.29 | 3.2% | −19.1% | 29 | 0.38 |
| Sampled MuZero | none | −0.49 | −0.72 | −6.0% | −20.5% | 41 | 0.14 |
| Sampled MuZero | FinBERT | −0.46 | 0.07 | 0.2% | −16.3% | 36 | 0.14 |
| UniZero (transformer world model) | none | 1.14 | 0.09 | 0.0% | −27.3% | 28 | 0.66 |
| UniZero | FinBERT | 0.85 | 0.41 | 6.3% | −28.3% | 26 | 0.72 |
| PPO (E2) | none / FinBERT | 0.21 / 1.79 | 0.16 / 0.43 | 1.2% / 5.6% | −25.0% / −17.1% | 68 / 76 | 0.45 / 0.40 |

Observations:
* **No MuZero variant beats buy-and-hold** on the test period; the best (EfficientZero without sentiment 0.34,
  Stochastic MuZero + NOSIBLE 0.29) roughly halve the benchmark's Sharpe with lower drawdowns, mostly because they hold
  a partial long position (avg ≈ 0.25–0.4) while trading ~30–40 times a year.
* **Validation Sharpe does not predict test Sharpe** (e.g. MuZero none: 0.68 → −0.25). The validation window
  (18 months) is short and checkpoint selection on it is largely noise-fitting.
* Sentiment helped 4 of 5 algorithm pairs on test (MuZero, Stochastic, Sampled, PPO) but hurt EfficientZero; with one
  seed each and Sharpe standard errors of ≈ ±0.5 over 3.5 years, none of these differences is statistically meaningful.
* **Gumbel MuZero failed to learn** a useful policy here: every trained checkpoint was worse on validation than the random
  initialisation (val Sharpe −0.4 to −2.1), so the untrained net (≈ always short) was selected. With only 3 actions the
  Gumbel top-k machinery gives little advantage and its policy targets appear to collapse.
* **UniZero** (run locally, finished last) had the *highest validation Sharpe of any run* (1.14, above buy-and-hold's 0.69) and then scored 0.09 on test — the clearest example of validation over-fitting in this study. With FinBERT it reached 0.41, mostly by staying ~70% long.
* **Sampled MuZero** (continuous sizing) was the weakest and the most expensive (≈2× compute of discrete MuZero).
* The planning advantage of MuZero relies on a learnable model of the environment. Here the agent's action never
  affects the next observation (except its own position), so the "dynamics" to learn are just the market's — essentially
  unpredictable noise — and the search mostly averages over noise.

### E4 — Seed variance and long-only action space (local CPU, after Modal cancelled these jobs)

| Run | Val Sharpe | **Test Sharpe** | Test CAGR | Test MaxDD | Turnover | Avg position |
|---|---|---|---|---|---|---|
| Stochastic MuZero, none, seed 0 / **seed 1** | −0.47 / 0.19 | −0.37 / **−0.36** | −3.9% / −3.7% | −19.0% / −20.2% | 36 / 38 | 0.33 / 0.27 |
| Stochastic MuZero, NOSIBLE, seed 0 / **seed 1** | 0.60 / 0.84 | 0.29 / **0.13** | 3.2% / 0.8% | −19.1% / −12.3% | 29 / 28 | 0.38 / 0.34 |
| EfficientZero, none, **long-only** {flat, long} | 0.55 | 0.13 | 0.8% | −15.8% | 28 | 0.24 |
| **Stochastic MuZero, NOSIBLE, long-only** | 0.32 | **1.11** | **13.2%** | **−12.2%** | 25 | 0.60 |

* Second seeds are consistent in sign with the first: no-sentiment Stochastic MuZero loses money on both seeds,
  NOSIBLE-sentiment Stochastic MuZero is weakly positive on both (0.29, 0.13) — a small but repeatable sentiment effect,
  still well below buy-and-hold.
* **Stochastic MuZero + NOSIBLE + long-only is the first agent to beat buy-and-hold** on the test period
  (Sharpe 1.11 vs 0.65, CAGR 13.2% vs 12.7%, max drawdown −12% vs −29%). Year by year it trailed in the up-years
  (2019 +4% vs +13%, 2021 +22% vs +34%) but cut exposure in the COVID crash (average position 0.74 in Jan 2020 →
  0.26 in Apr 2020) and in 2022 (+1% vs −15%). Its validation Sharpe was only 0.32, i.e. it was **not** the
  configuration one would have picked from validation alone — treat this as a single lucky-or-not seed until the
  robustness runs in E5 (more seeds + a no-sentiment control) are in.
* Long-only did not help EfficientZero (0.34 → 0.13): it stayed mostly flat (avg position 0.24).

### E5 — Is the long-only winner real? More seeds + controls (local CPU)

| Run | Val Sharpe | **Test Sharpe** | Test CAGR | Test MaxDD | Turnover | Avg position |
|---|---|---|---|---|---|---|
| Stochastic MuZero, NOSIBLE, long-only, seed 0 | 0.32 | **1.11** | 13.2% | −12.2% | 25 | 0.60 |
| Stochastic MuZero, NOSIBLE, long-only, seed 1 | 0.42 | 0.29 | 2.7% | −15.6% | 29 | 0.57 |
| Stochastic MuZero, NOSIBLE, long-only, seed 2 | 0.62 | −0.47 | −1.5% | −8.4% | 15 | 0.07 |
| Stochastic MuZero, no sentiment, long-only, seed 0 | 0.45 | 0.29 | 2.2% | −11.5% | 36 | 0.39 |
| Stochastic MuZero, no sentiment, long-only, seed 1 | 0.52 | 0.35 | 4.0% | −20.2% | 31 | 0.53 |
| Stochastic MuZero, FinBERT, long-only, seed 0 | 0.33 | 0.21 | 2.0% | −21.2% | 25 | 0.50 |

**Seed ensembles** (average of the per-seed daily portfolio returns — no selection on test):

| Ensemble | Test Sharpe | CAGR | MaxDD |
|---|---|---|---|
| NOSIBLE, long-only (3 seeds) | 0.62 | 4.8% | −10.1% |
| No sentiment, long-only (2 seeds) | 0.35 | 3.2% | −13.0% |
| NOSIBLE, long/short (2 seeds) | 0.25 | 2.1% | −13.7% |
| No sentiment, long/short (2 seeds) | −0.41 | −3.7% | −17.6% |
| *Buy & hold* | *0.65* | *12.7%* | *−29.4%* |

* The 1.11 Sharpe of seed 0 **did not replicate** (seeds 1 and 2: 0.29, −0.47). Seed 2 learned to sit almost
  entirely in cash. The validation Sharpe ranks the three seeds in *exactly the wrong order*, so there was no way to pick
  seed 0 in advance.
* Averaged over seeds, the ordering is consistent: **long-only > long/short, and NOSIBLE sentiment > no sentiment**
  (by ≈0.3–0.6 Sharpe). The 3-seed NOSIBLE long-only ensemble roughly matches buy-and-hold's Sharpe (0.62 vs 0.65) with a
  third of its drawdown, but with far lower returns (4.8% vs 12.7% a year) because it is only ~40% invested on average.
  A risk-matched comparison (buy-and-hold scaled to the same volatility) would be about equal — i.e. no real edge.
* The improvement from sentiment is mostly in 2020 and 2022, where cutting exposure helped; it is consistent with the
  E1 finding that the news signal is weak and regime-dependent.

## Conclusions

1. **The MuZero family does not trade daily equities better than buy-and-hold here.** 7 algorithms × sentiment
   options × action spaces, 25 assets, 3.5-year out-of-sample test including a crash, a rally and a bear market.
   The best *robust* result (multi-seed average) is ≈ buy-and-hold's Sharpe at much lower return.
2. **Why model-based planning does not help:** MuZero's strength is planning through a learnable model of how actions
   change the world. In single-asset trading the agent's action only changes its own position; the rest of the "world
   model" is next-day market returns, which are close to unpredictable. MCTS then mostly averages noise, and Gumbel /
   Sampled variants add machinery without new information. Stochastic MuZero, which explicitly models chance, was the
   most competitive MuZero variant, which fits this picture.
3. **Sentiment:** free news sentiment (≈474k headlines + Reddit posts, no paid APIs) has small, unstable predictive power
   (|IC| ≲ 0.1, sign flips between periods). The 2025 LLM classifier (NOSIBLE, Qwen3-0.6B) was slightly better than
   FinBERT both in IC and as an RL input. Forum (Reddit) data is too sparse before 2017 to be useful for training.
4. **Validation-based selection is unreliable** with an 18-month validation window: validation Sharpe was uncorrelated
   (or anti-correlated) with test Sharpe across runs. Any single-run "winner" should be treated as noise until
   replicated across seeds — this repo's own best run is an example.
5. **Constraints help more than algorithms:** removing shorting and adding sentiment mattered more than the choice of
   MuZero variant; a simple long-only SMA-200 trend rule (test Sharpe 0.91) beat every RL agent.

Ideas not tried (would need more compute): many more assets for training (hundreds of stocks), cross-sectional
(portfolio) action spaces, risk-adjusted reward (e.g. differential Sharpe), walk-forward retraining, and much longer
validation periods / multi-seed checkpoint selection.

## Engineering notes

* **LightZero patches** (`patches/lightzero.patch`, applied to commit `003772de`):
  three UniZero bugs for vector observations (duplicate kwarg, wrong kwarg names), a NumPy-2 scalar-conversion bug in
  Gumbel MuZero, and a 2× faster python loop in the MuZero replay-buffer target computation.
  The default 1024-wide self-supervised projection head was shrunk to 256 in configs — together ≈2.5× faster CPU
  training.
* **Look-ahead safeguards:** features use data up to the decision close only; news after 15:30 ET (or date-only items)
  is shifted to the next session; normalisation statistics come from the training period; the test split was only
  evaluated once per run with a checkpoint chosen on validation. Execution at the close that generated the signal
  and log-return approximation for shorts are simplifications.
* **Compute:** local 8-core CPU (WSL2) for most training; Modal for GPU sentiment scoring (2 × ~475k texts on T4s,
  ≈$0.8) and parallel CPU training (≈$6 total). Several Modal jobs were cancelled by the platform and re-run locally.
* **Data licences:** price data from Yahoo Finance; the news corpus is research/non-commercial only.
