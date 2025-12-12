# Development Guide - Martingale Hacks SPY NLP Trading System

This document outlines the step-by-step development plan to build a production-ready Kaggle competition entry.

## Quick Start

```bash
git clone https://github.com/PauloTuppy/martingale-hacks-spy-nlp.git
cd martingale-hacks-spy-nlp
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Development Phases

### Phase 1: Data Loading & Baseline (Weeks 1-2)

**Goal:** Establish a reproducible data pipeline and baseline model to beat.

#### 1.1: Data Exploration & Cleaning
- Load SPY price data (OHLCV) from Kaggle competition
- Load daily aggregated financial text (headlines, tweets, earnings calls)
- Join data by date; ensure no forward-looking bias
- Check for missing values, outliers, date misalignment
- **Output:** Cleaned CSV files in `data/processed/`

#### 1.2: Feature Engineering v1 (Price Only)
- Daily returns, log returns
- Lagged returns (1d, 3d, 5d, 10d)
- Rolling volatility (10d, 20d SMA)
- Simple trend flag: SMA(10) > SMA(20)
- Volume changes, rolling volume
- **Output:** Features dataframe with all price-based features

#### 1.3: Target Definition
- Define target: next-day return (simple or log), or binary/ternary class (Up/Neutral/Down)
- Align targets: day t features predict day t+1 returns
- Check target distribution
- **Output:** Features + target DataFrame, ready for modeling

#### 1.4: Time-Series CV Setup
- Implement expanding-window CV (no data leakage)
- Fold 1: Train on first 4 years, validate on next 1 year
- Fold 2: Train on first 5 years, validate on next 1 year
- ... up to 4-5 folds
- **Notebook:** `01_eda_and_features.ipynb`

#### 1.5: Baseline Models
- LightGBM: simple hyperparams, just to establish a floor
- XGBoost: baseline comparison
- Track CV score: RMSE on target + Sharpe ratio of resulting strategy
- **Notebook:** `02_baseline_models.ipynb`

### Phase 2: Text Features & Model Diversity (Weeks 2-3)

**Goal:** Introduce NLP signals and expand model portfolio.

#### 2.1: Text Feature Extraction
- TF-IDF or bag-of-words on daily aggregated text
- Sentiment scores (positive/negative counts, or pre-trained sentiment model)
- Optionally: simple topic clustering or keyword extraction
- **Output:** Daily text feature vectors

#### 2.2: BERT/FinBERT Embeddings
- Use `transformers` library to load a pretrained model (BERT-base or FinBERT)
- Mean-pool embeddings per day (aggregate all daily headlines into one 768-d vector)
- Dimensionality reduction (PCA, or just use raw embeddings)
- **Output:** Daily embedding vectors

#### 2.3: Multi-Model Training
- **Model A (Price-only):** LightGBM on price features only
- **Model B (Text-only):** LightGBM on TF-IDF + sentiment only
- **Model C (Combined):** LightGBM on price + TF-IDF + sentiment + embeddings
- **Model D (Neural, optional):** Simple 1-2 layer neural network on embeddings + price
- Evaluate each model on CV Sharpe; report train/val RMSE and Sharpe
- **Notebook:** `03_nlp_features_and_embeddings.ipynb`

### Phase 3: Trading Logic & Risk Overlay (Weeks 2-3)

**Goal:** Convert model predictions to tradeable signals with risk management.

#### 3.1: Trading Rules
- Threshold-based: If predicted_return > T_up, Buy; if < T_down, Sell; else Hold
- Tune T_up and T_down using CV Sharpe (grid search or Bayesian optimization)
- Hysteresis: Only switch from Buy to Sell if predicted_return < T_down - margin (to reduce churn)

#### 3.2: Risk Overlay
- **Volatility scaling:** Reduce position size if rolling 20d volatility > 75th percentile
  - Normal position: +1 (full long), or -1 (full short)
  - High vol: position *= 0.5 or 0.7
- **Position limits:** Cap position at ±1 (no leveraging)
- **Max drawdown stop (optional):** If cumulative drawdown > 15%, stop trading for N days

#### 3.3: Backtest Logic
- Daily PnL = position[t] * return[t+1]
- Cumulative return = cumsum(daily_pnl)
- Sharpe ratio = annual_return / annual_volatility
- Metrics: Sharpe, max drawdown, win rate, avg trade duration
- **Notebook:** `04_ensemble_and_trading.ipynb`

### Phase 4: Ensemble & Hyperparameter Tuning (Weeks 3-4)

**Goal:** Combine models to maximize CV Sharpe while controlling overfitting.

#### 4.1: Ensemble Methods
- **Simple average:** mean(pred_A, pred_B, pred_C)
- **Weighted average:** w_A*pred_A + w_B*pred_B + w_C*pred_C (tune weights on CV Sharpe)
- **Voting:** Convert each model to discrete signal, take majority vote
- **Stacking:** Train a meta-model (e.g., logistic regression) on top of individual model predictions

#### 4.2: Hyperparameter Tuning
- Grid/random search over:
  - LightGBM: learning_rate, num_leaves, max_depth
  - Trading thresholds: T_up, T_down
  - Ensemble weights: w_A, w_B, w_C
- Use CV Sharpe as objective; early stop if val Sharpe stagnates
- Track all runs in a results table

#### 4.3: Validation vs. Leaderboard
- Compare CV Sharpe against public leaderboard Sharpe
- If they diverge significantly, investigate potential leakage or overfitting
- Avoid "LB chasing": optimize for CV, not public LB

### Phase 5: Final Training & Submission (Week 4-5)

**Goal:** Generate test predictions and package for Devpost.

#### 5.1: Final Model Training
- Retrain best ensemble on full training data (respecting temporal order)
- Use final hyperparameters determined from CV
- No validation split; use all data to maximize signal

#### 5.2: Test Predictions
- Generate predictions on test set
- Apply trading logic + risk overlay
- Output: CSV with columns [date, signal] or [date, position] in Kaggle-specified format

#### 5.3: Kaggle Submission
- Create `05_final_submission.ipynb` as the "golden notebook"
- Ensure reproducibility: set random seeds, pin dependencies
- Kaggle will run this notebook and evaluate on hidden test set

#### 5.4: Strategy Memo
- Write 1-2 page PDF:
  - Problem statement
  - Data sources and features
  - Model architecture (text encoder, gradient boosted ensemble, risk overlay)
  - Key results: CV Sharpe, public/private LB Sharpe, max drawdown
  - Failure cases: what scenarios hurt performance?
  - Future improvements
- **Output:** `reports/strategy_memo.pdf`

#### 5.5: Devpost Submission
- Link Kaggle notebook + strategy memo PDF
- Optional: 2-min video walkthrough
- Brief description emphasizing novelty (not copycats)

## Key Metrics to Track

Create a `results.csv` file that logs:

```
model_name, cv_fold, train_rmse, val_rmse, cv_sharpe, public_lb_sharpe, date
```

Keep this updated after each experiment to avoid re-running models.

## Avoiding Common Pitfalls

1. **Data Leakage:** Never use future information (t+1 and beyond) in features for day t.
2. **LB Chasing:** Optimize CV, not public leaderboard. The private test is what counts (70% of score).
3. **Overfitting:** If CV Sharpe >> public Sharpe, investigate overfitting.
4. **Turnover:** High-frequency trades reduce Sharpe due to transaction costs (even if not explicitly modeled).
5. **Regime Shifts:** Financial markets change; ensure model doesn't break on new regimes.

## Useful Commands

```bash
# Update dependencies if needed
pip install --upgrade -r requirements.txt

# Run a single Jupyter notebook
jupyter notebook notebooks/01_eda_and_features.ipynb

# List all commits
git log --oneline

# Create a new branch for experimentation
git checkout -b experiment/new-nlp-features

# Commit changes
git add .
git commit -m "Add BERT embeddings to text features"
git push origin experiment/new-nlp-features
```

## Resources

- [LightGBM Docs](https://lightgbm.readthedocs.io/)
- [Transformers (HuggingFace)](https://huggingface.co/docs/transformers/)
- [FinBERT Pretrained Model](https://huggingface.co/ProsusAI/finbert)
- [Kaggle API](https://github.com/Kaggle/kaggle-api)

## Timeline

- **Week 1:** Phases 1.1-1.4 (data loading, baseline, CV setup)
- **Week 2:** Phases 2.1-3.3 (text features, multi-model, trading logic)
- **Week 3:** Phase 4 (ensemble, tuning)
- **Week 4-5:** Phase 5 (final training, memo, Devpost)

Good luck!
