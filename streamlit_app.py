"""
Streamlit App — 50 Startups CRISP-DM Interactive Dashboard
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from src.eda import DataLoader, EDA
from src.modeling import MLModeling
from src.discussion import ExpertDiscussion
from src.report import CRISPDMReport

OUTPUT_DIR = "output"
FIGURES_DIR = os.path.join(OUTPUT_DIR, "figures")
MODELS_DIR = os.path.join(OUTPUT_DIR, "models")
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
DATA_PATH = os.path.join(DATA_DIR, "50_Startups.csv")

st.set_page_config(page_title="50 Startups CRISP-DM", layout="wide", page_icon="📊")


@st.cache_resource(show_spinner=False)
def run_pipeline():
    os.makedirs(FIGURES_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)

    loader = DataLoader(local_path=DATA_PATH)
    df = loader.load()
    df_encoded = loader.encode_state()

    eda = EDA(df, FIGURES_DIR)
    eda_results = {}
    eda_results["correlation_with_profit"] = eda.correlation_analysis()
    eda_results["correlation_fig"] = eda._plot_correlation_heatmap(
        df.select_dtypes(include=[__import__('numpy').number]).corr()
    )
    eda_results["distribution_fig"] = eda.distribution_analysis()
    eda_results["boxplot_fig"] = eda.boxplot_by_state()
    eda_results["scatter_fig"] = eda.scatter_plots_vs_profit()

    modeler = MLModeling(df_encoded, target="Profit", output_dir=MODELS_DIR)
    modeler.prepare_data()
    modeler.train_linear_regression()
    modeler.train_random_forest()
    modeler.train_xgboost()

    eval_summary = modeler.get_evaluation_summary()
    ranking = modeler.get_final_ranking()
    aggregated_importance = modeler.get_aggregated_feature_importance()
    modeler.save_models()

    feature_order = [r["Feature"] for _, r in ranking.iterrows()]
    fs_df, fs_fig = eda.forward_selection_analysis(
        modeler.X_train, modeler.X_test,
        modeler.y_train, modeler.y_test,
        feature_order,
    )

    comp_df, rank_table, multi_fig = modeler.multi_algorithm_feature_selection(figures_dir=FIGURES_DIR)

    residual_figs = {}
    for name in ["LinearRegression", "RandomForest", "XGBoost"]:
        result = modeler.models.get(name)
        if result:
            residual_figs[name] = eda.residual_plots(result["y_pred_test"], modeler.y_test.values, name)

    importance_figs = {}
    for name, importance in modeler.feature_importances.items():
        importance_figs[name] = eda.feature_importance_plot(importance, name)

    discussion = ExpertDiscussion(eda_results, modeler.models)
    rounds = discussion.run_all_rounds(eval_summary, ranking, aggregated_importance)

    report = CRISPDMReport(OUTPUT_DIR)
    business_conclusion = discussion.consensus["business_conclusion"] if discussion.consensus else {}
    budget_allocation = discussion.consensus.get("budget_allocation", []) if discussion.consensus else []
    report.generate(df, eda_results["correlation_with_profit"], eval_summary, ranking,
                    budget_allocation, business_conclusion, forward_selection_df=fs_df)

    return {
        "df": df,
        "df_encoded": df_encoded,
        "eda_results": eda_results,
        "modeler": modeler,
        "eval_summary": eval_summary,
        "ranking": ranking,
        "aggregated_importance": aggregated_importance,
        "fs_df": fs_df,
        "fs_fig": fs_fig,
        "comp_df": comp_df,
        "rank_table": rank_table,
        "multi_fig": multi_fig,
        "residual_figs": residual_figs,
        "importance_figs": importance_figs,
        "discussion": discussion,
        "rounds": rounds,
        "report": report,
        "business_conclusion": business_conclusion,
        "budget_allocation": budget_allocation,
    }


def render_overview(data):
    st.header("Project Overview")
    st.markdown("""
This project applies the **CRISP-DM** methodology to the Kaggle **50 Startups** dataset,
simulating a **5-expert, 5-round discussion** to identify factors driving **Profit**.
""")

    col1, col2, col3 = st.columns(3)
    col1.metric("Records", len(data["df"]))
    col2.metric("Features", len(data["df_encoded"].columns) - 1)
    col3.metric("Models", len(data["modeler"].models))

    st.subheader("Workflow")
    workflow_path = os.path.join(OUTPUT_DIR, "workflow.png")
    if os.path.exists(workflow_path):
        st.image(workflow_path, width='stretch')


def render_eda(data):
    st.header("Phase 1 & 2: Exploratory Data Analysis")

    st.subheader("Dataset Preview")
    st.dataframe(data["df"], width='stretch')

    st.subheader("Descriptive Statistics")
    st.dataframe(data["df"].describe().T, width='stretch')

    st.subheader("Correlation with Profit")
    corr = data["eda_results"]["correlation_with_profit"]
    corr_df = pd.DataFrame({"Feature": corr.index, "Correlation": corr.values})
    st.dataframe(corr_df, width='stretch', hide_index=True)

    st.subheader("Correlation Heatmap")
    st.pyplot(data["eda_results"]["correlation_fig"])

    st.subheader("Distribution Analysis")
    st.pyplot(data["eda_results"]["distribution_fig"])

    st.subheader("Boxplot by State")
    st.pyplot(data["eda_results"]["boxplot_fig"])

    st.subheader("Scatter Plots vs Profit")
    st.pyplot(data["eda_results"]["scatter_fig"])


def render_modeling(data):
    st.header("Phase 3 & 4: Modeling")

    st.subheader("Model Evaluation Summary")
    st.dataframe(data["eval_summary"], width='stretch', hide_index=True)

    st.subheader("Feature Importance Ranking")
    st.dataframe(data["ranking"], width='stretch', hide_index=True)

    st.subheader("Aggregated Feature Importance")
    agg = data["aggregated_importance"]
    agg_df = pd.DataFrame({"Feature": list(agg.keys()), "Importance": list(agg.values())})
    st.bar_chart(agg_df.set_index("Feature"))

    for name, fig in data["importance_figs"].items():
        st.subheader(f"Feature Importance — {name}")
        st.pyplot(fig)

    for name, fig in data["residual_figs"].items():
        st.subheader(f"Residual Analysis — {name}")
        st.pyplot(fig)


def render_feature_selection(data):
    st.header("Feature Selection Analysis")

    st.subheader("Forward Feature Selection (Linear Regression)")
    st.dataframe(data["fs_df"], width='stretch', hide_index=True)
    st.pyplot(data["fs_fig"])

    st.subheader("Multi-Algorithm Feature Selection Comparison")
    st.markdown("Comparing 5 methods: SFS (Forward), RFE, SelectKBest, Lasso (L1), Random Forest")
    st.pyplot(data["multi_fig"])

    st.subheader("Comparison Data")
    st.dataframe(data["comp_df"], width='stretch', hide_index=True)

    st.subheader("Feature Rankings by Algorithm")
    st.dataframe(data["rank_table"], width='stretch')


def render_discussion(data):
    st.header("Multi-Round Expert Discussion")

    expert_colors = {
        "RD": "#3498db",
        "Marketing": "#e74c3c",
        "Administration": "#2ecc71",
        "Governor": "#f39c12",
        "ML_Expert": "#9b59b6",
    }

    for r in data["rounds"]:
        with st.expander(f"Round {r['round']}: {r['title']}", expanded=(r["round"] == 5)):
            if "claims" in r:
                for c in r["claims"]:
                    speaker = c.get("speaker_id", "?")
                    color = expert_colors.get(speaker, "#333")
                    st.markdown(f"**<span style='color:{color}'>{c['expert']}</span>**", unsafe_allow_html=True)
                    st.markdown(c["claim"])
                    if "limitations" in c:
                        for lim in c["limitations"]:
                            st.markdown(f"- {lim}")
                    st.divider()

            elif "findings" in r:
                for f in r["findings"]:
                    speaker = f.get("speaker_id", "?")
                    color = expert_colors.get(speaker, "#333")
                    st.markdown(f"**<span style='color:{color}'>{f['expert']}</span>**", unsafe_allow_html=True)
                    st.markdown(f["claim"])
                    st.divider()

            elif "challenges" in r:
                for ch in r["challenges"]:
                    speaker = ch.get("speaker_id", "?")
                    color = expert_colors.get(speaker, "#333")
                    st.markdown(f"**<span style='color:{color}'>{ch['expert']}</span>**", unsafe_allow_html=True)
                    st.markdown(ch["claim"])
                    if "limitations" in ch:
                        for lim in ch["limitations"]:
                            st.markdown(f"-  {lim}")
                    st.divider()

            elif "consensus" in r:
                c = r["consensus"]
                st.subheader("Votes")
                for v in c["votes"]:
                    st.markdown(f"- **{v['expert']}**: {v['vote']}")

                st.subheader("Final Ranking")
                rank_df = pd.DataFrame(c["final_ranking"])
                st.dataframe(rank_df, width='stretch', hide_index=True)

                st.subheader("Budget Allocation")
                budget_df = pd.DataFrame(c["budget_allocation"])
                st.dataframe(budget_df, width='stretch', hide_index=True)

                st.subheader("Business Conclusion")
                for s in c["business_conclusion"]["summary"]:
                    st.markdown(f"- {s}")


def render_report(data):
    st.header("Phase 5 & 6: Evaluation & Report")

    report_path = os.path.join(OUTPUT_DIR, "CRISP_DM_Report.md")
    if os.path.exists(report_path):
        with open(report_path, "r", encoding="utf-8") as f:
            report_content = f.read()
        st.markdown(report_content)
    else:
        st.warning("Report not generated yet.")


def _load_fig(filename):
    path = os.path.join(FIGURES_DIR, filename)
    if os.path.exists(path):
        fig, ax = plt.subplots(figsize=(10, 8))
        img = plt.imread(path)
        ax.imshow(img)
        ax.axis("off")
        fig.tight_layout()
        return fig
    return None


def main():
    st.sidebar.title("50 Startups CRISP-DM")
    st.sidebar.markdown("---")

    with st.sidebar:
        page = st.radio(
            "Navigation",
            ["Overview", "EDA", "Modeling", "Feature Selection", "Expert Discussion", "Report"],
        )

    with st.spinner("Running CRISP-DM pipeline..."):
        data = run_pipeline()

    if page == "Overview":
        render_overview(data)
    elif page == "EDA":
        render_eda(data)
    elif page == "Modeling":
        render_modeling(data)
    elif page == "Feature Selection":
        render_feature_selection(data)
    elif page == "Expert Discussion":
        render_discussion(data)
    elif page == "Report":
        render_report(data)


if __name__ == "__main__":
    main()
