# GitHub Open Source Intelligence (OSINT) – Data Collection, Analysis and Machine Learning

An end-to-end, college-level Data Science project that collects **public** GitHub repository data, cleans and
explores it, engineers leakage-free features, trains machine-learning models to estimate a repository's star
count, and presents everything in an interactive **Streamlit** dashboard.

**Workflow:** Data Collection → Cleaning → EDA → Feature Engineering → Machine Learning → Evaluation → Prediction → Visualization

---

## 1. Project Overview
*GitHub OSINT* means gathering and analysing information that is openly published on GitHub (repository
metadata, activity timestamps, public owner statistics) to learn something useful – here: what makes
repositories popular, which technologies are trending and how active projects are.

## 2. Problem Statement
Millions of public repositories exist, but popularity is highly uneven and hard to judge by eye. Analysing
open-source data helps developers choose libraries, helps maintainers benchmark their projects, and helps
researchers study technology trends. This project asks: **can we estimate a repository's stars from its
observable characteristics (forks, watchers, issues, age, activity, language, license ...)?**

## 3. Objectives
- Collect public repository + owner data via the GitHub REST API (pagination, rate limits, error handling).
- Clean and validate the data with a documented pipeline.
- Explore popularity, language, activity and relationship patterns visually.
- Build leakage-free features and train **Linear Regression** and **Random Forest** models.
- Evaluate with MAE, MSE, RMSE and R², select the best model, and serve predictions in a dashboard.

## 4. Architecture
```
GitHub API → Data Collection → Raw Dataset (data/raw) → Data Cleaning → Processed Dataset (data/processed)
          → EDA (outputs/figures) → Feature Engineering → Machine Learning (models/) → Prediction → Streamlit Dashboard
```
```
github-osint-datascience/
├── app.py                      Streamlit dashboard
├── requirements.txt  .env.example  .gitignore
├── data/raw | data/processed   CSV datasets (+ cleaning_report.json)
├── models/                     github_star_predictor.pkl, metrics.json, test_predictions.csv
├── outputs/figures/            13 PNG charts from EDA
├── notebooks/                  01..04 walkthrough notebooks
├── src/
│   ├── config.py               paths, constants, logging
│   ├── pipeline.py             one-command pipeline
│   ├── data_collection/        github_api.py, sample_data.py (offline demo data)
│   ├── data_processing/        clean_data.py
│   ├── analysis/               eda.py
│   └── ml/                     features.py, train_model.py, predict.py
└── tests/                      test_data_cleaning.py, test_prediction.py
```

## 5. Dataset
| Group | Fields |
|---|---|
| Repository | `repository_name, full_name, owner, description, language, stars, forks, watchers, open_issues, size, created_at, updated_at, pushed_at, license, topics, default_branch, archived, is_fork, visibility, repository_url` |
| Owner | `owner_login, owner_type, followers, public_repositories, account_created_at` |
| Meta | `collected_at` (used as the reference date for ages) |

> **Important API detail:** the GitHub *Search* API returns `watchers_count` **equal to the star count**. The
> true watcher number is `subscribers_count`, only available from the repository detail endpoint. The collector
> uses that endpoint (needs a token), and the cleaner automatically removes `watchers` if it is a copy of `stars`.

## 6. Data Cleaning (`src/data_processing/clean_data.py`)
1. Numeric conversion; **negative counts → invalid → NaN**.
2. Dates parsed to timezone-aware datetimes; `updated_at < created_at` repaired.
3. Rows missing `full_name`, `stars` or `created_at` dropped.
4. Exact duplicates and duplicate repositories (`full_name`) removed.
5. Language names normalised (`python`, `js`, `cpp` → `Python`, `JavaScript`, `C++`); missing → `Unknown`.
6. Descriptions cleaned (HTML, URLs, emoji codes, whitespace); topics lower-cased, de-duplicated.
7. Missing license → `No License`; other missing numerics → median.
8. **Outliers:** `size`, `open_issues`, `followers`, `public_repositories` are winsorised at the 99.5th percentile.
   `stars`/`forks` are *not* capped – genuine popularity is the signal, not noise.
9. Constant/duplicate columns (`visibility`, `default_branch`, `owner`) dropped.

Before/after statistics (rows, columns, missing values, duplicates, dtypes) are shown on the dashboard's
**Data Cleaning** page and stored in `data/processed/cleaning_report.json`.

## 7. Exploratory Data Analysis
13 figures are saved to `outputs/figures/` (bar charts, histograms, scatter plots, box plots, correlation heatmap)
and the same views are interactive in the dashboard. Typical patterns (real GitHub data behaves similarly; the
bundled demo data is synthetic, so confirm on your own collection):
- Stars are **extremely right-skewed** – a few repositories hold most stars → log scales are used.
- Stars correlate strongly with forks and watchers; weakly with open issues; age helps older projects accumulate stars.
- Python/JavaScript dominate repository counts; median stars differ by language.
- Many repositories are pushed recently, a long tail is inactive.

## 8. Machine Learning
**Target:** `stars`. **Features:** `language` (one-hot), `forks`, `watchers`, `open_issues`, `repository_age_days`,
`days_since_last_update`, `has_license`, `has_description`, `has_topics`, `is_archived`.

- **Preprocessing** (`ColumnTransformer`): median imputation → `log1p` → standard scaling for counts; scaling for day
  counts; one-hot encoding for language (rare/unseen languages handled); binary flags passed through.
- **Split:** 80 % train / 20 % test, `random_state=42`.
- **Models:** Linear Regression (baseline, explainable) and Random Forest Regressor (non-linear, 300 trees).
- **Metrics:** MAE, MSE, RMSE, R². Also RMSE/R² on the log scale – used to select the best model because squared error
  in raw stars is dominated by a handful of mega-popular repositories.

### Why `log1p(stars)`?
Star counts span several orders of magnitude. Training on `log1p(stars)` makes the target roughly bell-shaped, stops
big repositories from dominating the loss, and turns multiplicative effects into additive ones that linear models
handle well. Predictions are converted back with `expm1()` (implemented with `TransformedTargetRegressor`).

### Avoiding data leakage
- `stars` and **any star-derived feature** (`fork_star_ratio`, `issue_star_ratio`, `watcher_star_ratio`) are **never**
  model inputs – they contain the answer. They exist only in `add_ratio_features_for_eda()` for exploration.
- `assert_no_leakage()` blocks any feature whose name contains the target; a unit test checks it.
- The Search-API `watchers_count` (a copy of stars) is detected and removed.
- Only information observable *before* the outcome is used (no future data); the scaler/imputer are fitted on the
  training split only because they live inside the pipeline.

## 9. Results
Demo run on the **synthetic sample dataset** (500 repositories → 400 train / 100 test):

| Model | MAE | RMSE | R² | RMSE (log) | R² (log) |
|---|--:|--:|--:|--:|--:|
| Linear Regression | 482.4 | 974.7 | 0.607 | 0.520 | 0.794 |
| Random Forest | 500.3 | 1045.9 | 0.547 | 0.546 | 0.772 |

**Selected model: Linear Regression** – the synthetic data was generated with log-linear relationships, so the simple
model wins. On real GitHub data, relationships are messier and Random Forest often matches or beats it; the training
script picks the winner automatically from your data. Feature importance (Random Forest) is shown on the dashboard.
Run on real data and replace this table with your own numbers.

## 10. Installation
```bash
git clone <repository-url>
cd github-osint-datascience

python -m venv venv
# Windows
venv\Scripts\activate
# Linux/macOS
source venv/bin/activate

pip install -r requirements.txt
```
Configure your token (optional but recommended) – copy `.env.example` to `.env`:
```text
GITHUB_TOKEN=your_token
```

### Run
```bash
python -m src.data_collection.github_api --keyword "machine learning" --language Python --min-stars 100 --limit 500
python -m src.data_processing.clean_data
python -m src.analysis.eda
python -m src.ml.train_model
streamlit run app.py
```
No API access? Use the offline demo data (synthetic) – everything else works identically:
```bash
python -m src.pipeline --sample        # collect(sample) → clean → EDA → train
streamlit run app.py
```
Or one command for real data: `python -m src.pipeline --keyword "machine learning" --language Python --limit 500`.

Tests: `python -m pytest -q`

### Notes on collection
- The Search API returns at most 1,000 results per query; larger `--limit` values automatically move the star window down.
- Enrichment (true watchers, owner stats) needs ~1 request per repo/owner. With a token (5,000/hour) 500 repos is easy;
  without one it is disabled automatically (60/hour) and the model trains without `watchers`.
- Rate limits, retries, timeouts, missing fields and duplicate repositories are all handled.

## 11. Prediction
`python -m src.ml.predict` or the **Star Prediction** page returns the predicted stars, a typical error range
(10th–90th percentile of test errors) and a Low / Moderate / High popularity label (tertiles of training stars).
**It is an estimate, not a guarantee** – stars also depend on marketing, timing, community and luck.

## 12. Limitations & Future Work
Snapshot data (no star history); forks/watchers are correlated with the target and partly *consequences* of
popularity, so the model describes association, not causation; README quality, contributors and release data are not
used. Ideas: gradient boosting, time-series of stars, text features from descriptions, SQLite storage.

## 13. OSINT Ethics
This project uses **only publicly available GitHub information** through the official API for **educational
public-data analysis**. It does **not** and must not be extended to: bypass authentication, access private
repositories, scrape private information, collect passwords or tokens, exploit GitHub, or perform unauthorized
reconnaissance. Tokens live in `.env` (git-ignored, never hard-coded). Respect GitHub's API rate limits, Terms of
Service and Acceptable Use Policies. Do not use personal data to profile or harass individuals.
