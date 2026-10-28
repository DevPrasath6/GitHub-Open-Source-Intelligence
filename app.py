"""Streamlit dashboard for the GitHub OSINT data science project.

Run:  streamlit run app.py
"""
from __future__ import annotations

import json

import pandas as pd
import plotly.express as px
import streamlit as st

from src.analysis import eda
from src.config import CLEANING_REPORT, DATA_PROCESSED, DATA_RAW, METRICS_PATH, MODEL_PATH, PREDICTIONS_PATH
from src.data_processing.clean_data import load_cleaned_data, profile_dataframe
from src.ml.predict import DISCLAIMER, predict_stars
from src.ml.train_model import comparison_table, train_models
from src.pipeline import run_full_pipeline
from src.data_collection.github_api import GitHubClient

# Works on old and new Streamlit: new versions use **STRETCH, old ones use_container_width=True
_ST_VERSION = tuple(int(x) for x in st.__version__.split(".")[:2] if x.isdigit())
STRETCH = {"width": "stretch"} if _ST_VERSION >= (1, 50) else {"use_container_width": True}

st.set_page_config(page_title="GitHub OSINT Analytics", page_icon="🔎", layout="wide")


def _mtime(path) -> float:
    return path.stat().st_mtime if path.exists() else 0.0


@st.cache_data(show_spinner=False)
def get_clean(_mt: float) -> pd.DataFrame:
    return load_cleaned_data(DATA_PROCESSED)


@st.cache_data(show_spinner=False)
def get_raw(_mt: float) -> pd.DataFrame:
    return pd.read_csv(DATA_RAW)


@st.cache_data(show_spinner=False)
def get_metrics(_mt: float) -> dict:
    return json.loads(METRICS_PATH.read_text())


@st.cache_resource(show_spinner=False)
def get_model(_mt: float):
    from src.ml.predict import load_model
    return load_model(MODEL_PATH)


# ------------------------------------------------------------------ sidebar
st.sidebar.title("🔎 GitHub OSINT")
page = st.sidebar.radio("Navigate", ["Dashboard", "Data Explorer", "Data Cleaning", "EDA",
                                     "Machine Learning", "Star Prediction", "Live GitHub Collection", "Ethics & About"])
with st.sidebar.expander("Pipeline controls"):
    st.caption("Generates a synthetic demo dataset, cleans it and retrains the model.")
    if st.button("Run sample pipeline"):
        with st.spinner("Running pipeline..."):
            run_full_pipeline(sample=True)
        st.cache_data.clear()
        st.cache_resource.clear()
        st.rerun()

if not DATA_PROCESSED.exists():
    st.title("GitHub Open Source Intelligence (OSINT)")
    st.warning("No cleaned dataset found yet. Collect real data with "
               "`python -m src.pipeline --keyword \"machine learning\" --language Python` "
               "or generate a demo dataset below.")
    if st.button("Generate sample data & run full pipeline", type="primary"):
        with st.spinner("Running pipeline..."):
            run_full_pipeline(sample=True)
        st.cache_data.clear()
        st.rerun()
    st.stop()

df = get_clean(_mtime(DATA_PROCESSED))
_src_file = DATA_RAW.with_name("data_source.txt")
data_source = _src_file.read_text().strip() if _src_file.exists() else "sample"
if data_source == "sample" and page != "Live GitHub Collection":
    st.warning("You are viewing SYNTHETIC sample data (links are fake). Open **Live GitHub Collection** "
               "in the sidebar to fetch real repositories from GitHub.")
elif page != "Live GitHub Collection":
    st.success("Data source: live GitHub API")
metrics = get_metrics(_mtime(METRICS_PATH)) if METRICS_PATH.exists() else None

# ------------------------------------------------------------------ pages
if page == "Dashboard":
    st.title("GitHub OSINT – Repository Intelligence Dashboard")
    top = df.loc[df["stars"].idxmax()]
    c = st.columns(5)
    c[0].metric("Total repositories", f"{len(df):,}")
    c[1].metric("Average stars", f"{df['stars'].mean():,.0f}")
    c[2].metric("Average forks", f"{df['forks'].mean():,.0f}")
    c[3].metric("Top language", df["language"].mode().iat[0])
    c[4].metric("Most starred", top["repository_name"], f"{top['stars']:,} ★")
    l, r = st.columns(2)
    l.plotly_chart(eda.plotly_top_repos(df, 10), **STRETCH)
    lang = df["language"].value_counts().head(10).reset_index()
    r.plotly_chart(px.bar(lang, x="count", y="language", orientation="h", title="Repositories by Language")
                   .update_yaxes(autorange="reversed"), **STRETCH)

elif page == "Data Explorer":
    st.title("Data Explorer")
    c1, c2, c3 = st.columns([2, 1, 2])
    langs = c1.multiselect("Language", sorted(df["language"].unique()))
    min_stars = c2.number_input("Minimum stars", 0, int(df["stars"].max()), 0, step=100)
    query = c3.text_input("Search repository name")
    view = df
    if langs:
        view = view[view["language"].isin(langs)]
    view = view[view["stars"] >= min_stars]
    if query:
        view = view[view["full_name"].str.contains(query, case=False, na=False)]
    st.caption(f"{len(view):,} of {len(df):,} repositories")
    cols = ["full_name", "language", "stars", "forks", "watchers", "open_issues", "license",
            "topics", "created_at", "pushed_at", "repository_url"]
    st.dataframe(view[[c for c in cols if c in view]], **STRETCH, hide_index=True,
                 column_config={"repository_url": st.column_config.LinkColumn("link")})
    st.download_button("Download filtered CSV", view.to_csv(index=False), "filtered_repositories.csv")

elif page == "Data Cleaning":
    st.title("Data Cleaning")
    if not DATA_RAW.exists():
        st.info("Raw dataset not found.")
    else:
        before = profile_dataframe(get_raw(_mtime(DATA_RAW)))
        after = profile_dataframe(df)
        a, b = st.columns(2)
        for col, title, p in [(a, "Before cleaning", before), (b, "After cleaning", after)]:
            with col:
                st.subheader(title)
                m = st.columns(3)
                m[0].metric("Rows", f"{p['rows']:,}")
                m[1].metric("Columns", p["columns"])
                m[2].metric("Duplicates", p["duplicate_rows"] + p["duplicate_repositories"]
                            if title.startswith("Before") else p["duplicate_repositories"])
                st.write(f"**Missing values:** {p['total_missing']:,}")
                if p["missing_values"]:
                    st.dataframe(pd.Series(p["missing_values"], name="missing").to_frame())
                if title.startswith("After"):
                    st.write("**Data types**")
                    st.dataframe(pd.Series(p["dtypes"], name="dtype").to_frame())
        if CLEANING_REPORT.exists():
            with st.expander("Cleaning steps applied"):
                st.json(json.loads(CLEANING_REPORT.read_text())["steps"])

elif page == "EDA":
    st.title("Exploratory Data Analysis")
    tabs = st.tabs(["Stars by language", "Top repositories", "Stars vs forks", "Distribution",
                    "Activity", "Correlation"])
    figs = [eda.plotly_stars_by_language(df), eda.plotly_top_repos(df), eda.plotly_stars_vs_forks(df),
            eda.plotly_stars_distribution(df), eda.plotly_activity(df), eda.plotly_correlation(df)]
    for tab, fig in zip(tabs, figs):
        tab.plotly_chart(fig, **STRETCH)

elif page == "Machine Learning":
    st.title("Machine Learning – Star Prediction")
    if st.button("Retrain models"):
        with st.spinner("Training..."):
            train_models(df)
        st.cache_data.clear()
        st.cache_resource.clear()
        st.rerun()
    if not metrics:
        st.info("No trained model yet - click 'Retrain models'.")
    else:
        c = st.columns(3)
        c[0].metric("Training rows", f"{metrics['n_train']:,}")
        c[1].metric("Testing rows", f"{metrics['n_test']:,}")
        c[2].metric("Best model", metrics["best_model"])
        st.subheader("Model comparison (test set)")
        st.dataframe(comparison_table(metrics).round(3), **STRETCH)
        st.caption("MAE/RMSE/R² are in stars; *_log metrics are on log1p(stars), the scale the models "
                   "were trained on and the one used to pick the best model.")
        l, r = st.columns(2)
        fi = pd.Series(metrics["feature_importance"]).head(10).reset_index()
        fi.columns = ["feature", "importance"]
        l.plotly_chart(px.bar(fi, x="importance", y="feature", orientation="h",
                              title="Random Forest feature importance").update_yaxes(autorange="reversed"),
                       **STRETCH)
        if PREDICTIONS_PATH.exists():
            pr = pd.read_csv(PREDICTIONS_PATH)
            best_col = metrics["best_model"].lower().replace(" ", "_")
            fig = px.scatter(pr, x="actual", y=best_col, log_x=True, log_y=True, hover_name="full_name",
                             title=f"Actual vs predicted stars ({metrics['best_model']})", opacity=0.6)
            lim = [pr["actual"].min(), pr["actual"].max()]
            fig.add_scatter(x=lim, y=lim, mode="lines", name="perfect", line=dict(dash="dash"))
            r.plotly_chart(fig, **STRETCH)
        st.info("**No data leakage:** `stars` and every star-derived ratio are excluded from the inputs. "
                "Training on log1p(stars) tames the heavy right-skew; predictions are converted back with expm1.")

elif page == "Star Prediction":
    st.title("Star Prediction")
    if not (MODEL_PATH.exists() and metrics):
        st.info("Train a model first (Machine Learning page).")
        st.stop()
    with st.form("predict"):
        c1, c2, c3 = st.columns(3)
        language = c1.selectbox("Programming Language", metrics.get("languages") or sorted(df["language"].unique()))
        forks = c2.number_input("Forks", 0, 1_000_000, 100)
        watchers = c3.number_input("Watchers", 0, 1_000_000, 50)
        open_issues = c1.number_input("Open Issues", 0, 100_000, 10)
        age = c2.number_input("Repository Age (days)", 0, 6000, 730)
        since = c3.number_input("Days Since Last Update", 0, 6000, 14)
        d1, d2, d3 = st.columns(3)
        has_license = d1.checkbox("Has License", True)
        has_description = d2.checkbox("Has Description", True)
        has_topics = d3.checkbox("Has Topics", True)
        go = st.form_submit_button("Predict Stars", type="primary")
    if go:
        res = predict_stars(language, forks, watchers, open_issues, age, since, has_license,
                            has_description, has_topics, model=get_model(_mtime(MODEL_PATH)), metrics=metrics)
        st.success(f"Predicted Stars: {res['predicted_stars']:,}")
        st.write(f"**{res['popularity']}** · typical range ≈ {res['range_low']:,} – {res['range_high']:,} "
                 "(10th–90th percentile of test-set errors)")
        st.warning(DISCLAIMER)

elif page == "Live GitHub Collection":
    st.title("Live GitHub Collection")
    st.caption("Fetches real, public repositories from the GitHub REST API, then cleans, analyses and retrains.")
    client_has_token = GitHubClient().has_token
    with st.form("live"):
        c1, c2 = st.columns(2)
        keyword = c1.text_input("Search keyword", "machine learning")
        language = c2.text_input("Programming language (blank = any)", "Python")
        c3, c4 = st.columns(2)
        min_stars = c3.number_input("Minimum stars", 0, 1_000_000, 100, step=50)
        limit = c4.number_input("Number of repositories", 10, 5000, 100, step=50)
        token = st.text_input("GitHub token (optional, only used for this run, not saved)", type="password",
                              help="Without a token GitHub allows ~60 requests/hour; enrichment is skipped.")
        enrich = st.checkbox("Also fetch true watchers + owner stats (needs a token; slower)", value=True)
        go = st.form_submit_button("Fetch live data", type="primary")
    if not (client_has_token or token):
        st.info("No token detected: repositories are fetched (stars, forks, issues, dates) but watchers and "
                "owner stats are skipped.")
    if go:
        try:
            with st.spinner("Calling the GitHub API - this can take a few minutes..."):
                run_full_pipeline(sample=False, keyword=keyword, language=language or None,
                                  min_stars=int(min_stars), limit=int(limit), token=token or None, enrich=enrich)
            st.cache_data.clear()
            st.cache_resource.clear()
            st.success("Live data collected, cleaned and model retrained. Open Data Explorer to browse it.")
        except Exception as exc:  # show a friendly error instead of a stack trace
            st.error(f"Collection failed: {exc}")

else:
    st.title("Ethics & About")
    st.markdown("""
**Scope.** This project performs *educational OSINT on public data only* – repository metadata and public
profile statistics returned by the official GitHub REST API.

**It does not:** bypass authentication, access private repositories, scrape private information, collect
passwords or tokens, exploit GitHub, or perform unauthorized reconnaissance.

**Good practice.** The GitHub token is read from `.env` (never committed), API rate limits are respected, and
GitHub's Terms of Service and Acceptable Use Policies apply. Predictions are estimates, not guarantees.
""")
