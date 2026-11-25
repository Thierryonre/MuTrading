# RLTrading — MuZero-family agents + free news/forum sentiment for daily trading

Can the MuZero family of model-based RL algorithms (via [LightZero](https://github.com/opendilab/LightZero))
learn to trade daily US equities/ETFs, and does adding free news / forum sentiment help?

**Status:** work in progress — see [Experiment log](#experiment-log) for every test and result.

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
