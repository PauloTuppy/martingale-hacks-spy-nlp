# martingale-hacks-spy-nlp

NLP-driven S&P 500 trading signals with risk overlay. **Martingale Hacks Winter 2025** submission using text-to-trading-signal ML pipeline.

## Overview

This project builds an ML system that turns unstructured financial text (news, tweets, earnings calls) into daily Buy/Sell/Hold trading signals for SPY, optimized for Sharpe ratio on a hidden test set.

**Key objectives:**
- Convert text → daily features (sentiment, topics, embeddings).
- Train gradient boosted models + ensemble to predict next-day returns.
- Implement risk overlay: volatility scaling, drawdown control, position limits.
- Backtest on historical SPY data with time-aware CV.
- Maximize Sharpe ratio while minimizing overfitting to public leaderboard.

## Project Structure

```
martingale-hacks-spy-nlp/
├── notebooks/
│   ├── 01_eda_and_features.ipynb          # Data exploration + feature engineering
│   ├── 02_baseline_models.ipynb          # Initial LightGBM/XGBoost baselines
│   ├── 03_nlp_features_and_embeddings.ipynb  # FinBERT/BERT text features
│   ├── 04_ensemble_and_trading.ipynb    # Multi-model ensemble + backtest logic
│   ├── 05_final_submission.ipynb         # Final Kaggle kernel ready for submission
│   └── research.ipynb                      # Scratch notebook for experiments
├── src/
│   ├── __init__.py
│   ├── data_loader.py                 # Load and join SPY + text data
│   ├── feature_engineer.py            # Price + text feature construction
│   ├── models.py                      # Model training + CV evaluation
│   ├── backtest.py                    # Trading logic + Sharpe calculation
│   ├── risk_management.py             # Risk overlay (vol scaling, stops, limits)
│   └── ensemble.py                    # Multi-model blending + stacking
├── data/
│   ├── raw/                            # Original SPY price + text (from Kaggle)
│   ├── processed/                      # Cleaned + feature-engineered data
│   └── submissions/                    # Final submission files
├── reports/
│   ├── strategy_memo.pdf               # 1-2 page technical summary for Devpost
│   ├── equity_curves.png               # Backtest equity curve plots
│   └── feature_importance.png          # Model feature importance
├── .gitignore                       # Python .gitignore
├─┠ LICENSE                          # MIT License
├─┠ README.md                        # This file
├─┠ requirements.txt                 # Python dependencies
└─┠ DEVELOPMENT.md                   # Development notes + step-by-step plan
```

## Development Roadmap

### Phase 1: Baseline & Exploration (Weeks 1-2)
1. **Data loading** → Load SPY OHLCV and daily aggregated text; join on date
2. **Feature engineering (v1)** → Price features (returns, volatility, technicals) + TF-IDF from text
3. **Simple baseline** → LightGBM/XGBoost on price+TF-IDF; establish CV framework
4. **Sharpe metric** → Implement local Sharpe calculation to match competition metric

### Phase 2: Text Features & Model Diversity (Weeks 2-3)
5. **NLP upgrade** → Add sentiment scores, BERT/FinBERT embeddings, topic models
6. **Multi-model training** → Price-only, text-only, text+price models in parallel
7. **Trading rule** → Map predictions → Buy/Sell/Hold; tune thresholds on CV Sharpe
8. **Risk overlay v1** → Cap position size, damp in high volatility regimes

### Phase 3: Ensemble & Refinement (Weeks 3-4)
9. **Ensembling** → Combine model predictions (average, vote, weighted blend)
10. **Hyperparameter tuning** → Optimize thresholds, ensemble weights on CV Sharpe
11. **Backtest polish** → Reduce turnover, test drawdown stops, add limits
12. **Overfitting check** → CV vs. public LB comparison; avoid chasing LB

### Phase 4: Final & Documentation (Weeks 4-5)
13. **Final training** → Retrain chosen models on full training set
14. **Submission generation** → Output test predictions in Kaggle format
15. **Strategy memo** → Write 1-2 page PDF with architecture, results, failures
16. **Devpost packaging** → Link Kaggle kernel, upload memo, optional video pitch

## Installation & Setup

```bash
# Clone the repo
git clone https://github.com/PauloTuppy/martingale-hacks-spy-nlp.git
cd martingale-hacks-spy-nlp

# Create a Python virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Key Technologies

- **Data & Feature Engineering**: pandas, numpy, scikit-learn
- **Text NLP**: transformers (BERT/FinBERT), nltk, spacy
- **Modeling**: LightGBM, XGBoost, CatBoost, PyTorch (optional neural models)
- **Backtesting**: Custom vectorized backtest logic (no third-party backtest framework)
- **Kaggle**: Kaggle API for data/submission management

## Competition Links

- **Devpost Hackathon:** [Martingale Hacks](https://martingale-hacks.devpost.com/)
- **Kaggle Competition:** [The Martingale: Winter 2025 Edition](https://www.kaggle.com/competitions/martingale-hacks-winter-2025)

## Author

**PauloTuppy** – Full-stack developer, ML/fraud detection specialist, competitive programmer.

## License

MIT License – See [LICENSE](LICENSE) for details.

## Next Steps

1. Read [DEVELOPMENT.md](DEVELOPMENT.md) for detailed step-by-step build instructions.
2. Start with notebook `01_eda_and_features.ipynb`.
3. Download Kaggle competition data and place in `data/raw/`.
4. Follow the roadmap phases sequentially; commit to GitHub after each major milestone.
