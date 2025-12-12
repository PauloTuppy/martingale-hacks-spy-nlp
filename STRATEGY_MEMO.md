# MARTINGALE HACKS: STRATEGY MEMO
## NLP-Driven S&P 500 Trading Signals with Risk Overlay

**Submission by:** PauloTuppy  
**Competition:** Martingale Hacks Winter 2025  
**Kaggle Notebook:** https://www.kaggle.com/code/paulotuppy/martingale-spy-text-signals  

---

### PROBLEM STATEMENT

Transform unstructured financial text (news, tweets, earnings calls) into daily Buy/Sell/Hold trading signals for SPY, optimized for Sharpe ratio on hidden test set. The challenge requires balancing predictive accuracy (RMSE) with risk-adjusted returns while avoiding overfitting to public leaderboard.

---

### ARCHITECTURE & METHODOLOGY

#### 1. Data Sources & Features
- **Price Data:** 10 years SPY OHLCV history
- **Text Data:** Daily aggregated financial news, earnings calls, market commentary
- **Feature Engineering:**
  - Price: Returns (simple, log, lagged), volatility (10d, 20d), technicals (SMA 10/20/50), trend, momentum
  - Text: TF-IDF vectors, word count, sentiment (positive/negative keyword counts), presence indicator
  - Total: 25+ engineered features per day

#### 2. Machine Learning Pipeline
- **Model:** LightGBM gradient boosted regressor (100 estimators, max_depth=6)
- **Time-Series CV:** 3-fold expanding window (no forward-looking bias)
  - Fold 1: Train 4y → Validate 1y
  - Fold 2: Train 5y → Validate 1y  
  - Fold 3: Train 6y → Validate 1y
- **Ensemble:** Multi-model averaging
  - Model A: Price-only features
  - Model B: Text-only features
  - Model C: Combined features
  - Final: Weighted ensemble optimized for CV Sharpe

#### 3. Trading Logic
- **Signal Generation:** Threshold-based
  - Buy if predicted_return > 1% (tuned)
  - Sell if predicted_return < -1% (tuned)
  - Hold otherwise
- **Position Size:** Full unit (±1) before risk overlay
- **Hysteresis:** Prevents excessive switching between signals

---

### RISK MANAGEMENT

#### Volatility Scaling
- **Implementation:** Adaptive position sizing based on 20-day rolling volatility
- **Logic:** Reduce exposure to 0.7x when volatility > 75th percentile
- **Effect:** Smoother equity curve, lower max drawdown

#### Position Limits
- Cap maximum position at ±1 (no leverage)
- Prevents catastrophic losses from extreme positions
- Aligns with realistic trading constraints

#### Drawdown Control  
- Optional: Stop trading for N days after drawdown exceeds 15%
- Allows recovery period before resuming trading

#### Risk Metrics Tracked
- **Sharpe Ratio:** (mean_PnL / std_PnL) × √252
- **Max Drawdown:** Largest cumulative loss from peak
- **Win Rate:** Percentage of profitable trades
- **Turnover:** Frequency of position changes

---

### PERFORMANCE RESULTS

#### Cross-Validation Metrics (Hidden from Public LB)
- **Average CV Sharpe:** 0.45-0.65 (varies by fold, market regime)
- **Average RMSE:** 0.008-0.012 (daily return RMSE)
- **Max Drawdown:** -8% to -12% depending on regime

#### Key Insights
- Sharpe ratio benefits significantly from risk overlay (0.3-0.4 improvement)
- Text features alone underperform price (slight signal, high noise)
- Combined features show synergy (0.15+ Sharpe improvement vs. individual)
- Volatility scaling reduces tail risk without sacrificing return potential

---

### BIGGEST FAILURES & LESSONS LEARNED

#### 1. **LB Chasing Trap**
- **Failure:** Optimized hyperparameters on public leaderboard (30% of score)
- **Lesson:** CV Sharpe diverged from LB Sharpe; final performance suffered
- **Fix:** Locked hyperparameters after initial CV tuning; prioritized CV robustness

#### 2. **Text Feature Overfitting**
- **Failure:** Complex BERT embeddings overfit to training period sentiment
- **Lesson:** Text signal decays in new regimes; simple TF-IDF more robust
- **Fix:** Reverted to TF-IDF; added text feature decay weighting

#### 3. **Turnover Penalty**
- **Failure:** High-frequency trading signals hurt Sharpe (implied transaction costs)
- **Lesson:** Threshold tuning critical; small changes → large turnover swings
- **Fix:** Added hysteresis and turnover penalty to objective; reduced switching by 40%

#### 4. **Regime Shifts**
- **Failure:** 2020 COVID crash: model predicted "hold" during 30% drop
- **Lesson:** Crisis volatility unprecedented; historical CV doesn't cover tail events
- **Fix:** Trained regime-aware sub-models; blend dynamically based on VIX

---

### TECHNICAL IMPLEMENTATION

**Language:** Python  
**Libraries:** pandas, numpy, scikit-learn, LightGBM, transformers  
**Compute:** Kaggle kernel (CPU, 2-3 min execution)  
**Validation:** Time-series aware CV; no leakage checks  
**Reproducibility:** Fixed random seeds; pinned all dependencies  

---

### FUTURE IMPROVEMENTS

1. **Ensemble diversity:** Add LSTM, XGBoost, CatBoost models
2. **Sentiment upgrade:** Fine-tune FinBERT on financial corpora
3. **Adaptive risk:** Dynamic risk limits based on regime detector
4. **Transaction costs:** Explicit modeling of bid-ask and slippage
5. **Real-time updates:** Streaming data ingestion for production deployment

---

### CONCLUSION

This submission demonstrates an end-to-end NLP-to-trading pipeline with disciplined risk management. The key innovation is the **risk overlay framework** that dynamically scales exposure based on volatility, balancing return potential with drawdown control. By combining multiple feature types and respecting temporal ordering through time-series CV, the model achieves consistent Sharpe ratios while avoiding the pitfalls of leaderboard optimization.

**Core Strength:** Robust handling of regime shifts through multi-model ensemble and volatility-aware position sizing.  
**Core Challenge:** Capturing non-stationary text signal decay while maintaining predictive power.
