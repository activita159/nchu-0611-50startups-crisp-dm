"""
Kaggle 50 Startups CRISP-DM Expert Discussion Framework
========================================================
Five-domain-expert multi-round discussion analyzing factors
influencing Profit in the 50 Startups dataset.
"""

import sys
import os
import io

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from src.eda import DataLoader, EDA
from src.modeling import MLModeling
from src.discussion import ExpertDiscussion
from src.report import CRISPDMReport

OUTPUT_DIR = "output"
FIGURES_DIR = os.path.join(OUTPUT_DIR, "figures")
MODELS_DIR = os.path.join(OUTPUT_DIR, "models")
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
DATA_PATH = os.path.join(DATA_DIR, "50_Startups.csv")
LOG_PATH = os.path.join(OUTPUT_DIR, "discussion_transcript.txt")


def print_section(title: str):
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")


def log_chinese(content: str, f: io.TextIOWrapper):
    f.write(content + "\n")


def main():
    os.makedirs(FIGURES_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)

    with open(LOG_PATH, "w", encoding="utf-8") as log:
        log.write("=" * 60 + "\n")
        log.write("  50 Startups CRISP-DM — 專家討論逐字稿\n")
        log.write("=" * 60 + "\n\n")

        # ═══════════════════════════════════════════════════
        # Phase 1 & 2: Data Loading + EDA
        # ═══════════════════════════════════════════════════
        print_section("Phase 1 & 2: Data Loading & EDA")

        loader = DataLoader(local_path=DATA_PATH)
        df = loader.load()
        print(f"[OK] Loaded {len(df)} records")
        print(f"     Columns: {list(df.columns)}")
        print(f"     Missing values: {loader.check_missing()['Missing Count'].sum()}")

        df_encoded = loader.encode_state()

        eda = EDA(df, FIGURES_DIR)
        eda_results = eda.run_all()
        correlation = eda_results["correlation_with_profit"]

        print("\n[EDA] Correlation with Profit:")
        for name, value in correlation.items():
            print(f"      {name}: {value:.4f}")

        log.write("=== EDA Results ===\n")
        log.write("Profit 與各變數之相關係數：\n")
        for name, value in correlation.items():
            log.write(f"  {name}: {value:.4f}\n")
        log.write("\n")

        # ═══════════════════════════════════════════════════
        # Phase 3 & 4: Data Preparation + Modeling
        # ═══════════════════════════════════════════════════
        print_section("Phase 3 & 4: Data Preparation & Modeling")

        modeler = MLModeling(df_encoded, target="Profit", output_dir=MODELS_DIR)
        modeler.prepare_data()
        modeler.train_linear_regression()
        modeler.train_random_forest()
        modeler.train_xgboost()

        eval_summary = modeler.get_evaluation_summary()
        print("\n[Modeling] Evaluation Summary:")
        print(eval_summary.to_string(index=False))

        ranking = modeler.get_final_ranking()
        print("\n[Modeling] Feature Importance Ranking:")
        print(ranking.to_string(index=False))

        aggregated_importance = modeler.get_aggregated_feature_importance()
        modeler.save_models()
        print(f"\n[OK] Models saved to {MODELS_DIR}")

        # ── Forward Feature Selection ──
        print_section("Forward Feature Selection (Linear Regression)")
        feature_order = [r["Feature"] for _, r in ranking.iterrows()]
        fs_df = eda.forward_selection_analysis(
            modeler.X_train, modeler.X_test,
            modeler.y_train, modeler.y_test,
            feature_order,
        )
        print("\n[Forward Selection] LR metrics by number of features:")
        print(fs_df.to_string(index=False))

        log.write("=== 模型評估摘要 ===\n")
        log.write(eval_summary.to_string(index=False) + "\n\n")
        log.write("=== 特徵重要性綜合排名 ===\n")
        log.write(ranking.to_string(index=False) + "\n\n")

        # Residual & Importance plots
        for name in ["LinearRegression", "RandomForest", "XGBoost"]:
            result = modeler.models.get(name)
            if result:
                eda.residual_plots(result["y_pred_test"], modeler.y_test.values, name)

        for name, importance in modeler.feature_importances.items():
            eda.feature_importance_plot(importance, name)

        # ═══════════════════════════════════════════════════
        # Expert Discussion (5 Rounds)
        # ═══════════════════════════════════════════════════
        print_section("Multi-Round Expert Discussion")

        discussion = ExpertDiscussion(eda_results, modeler.models)
        rounds = discussion.run_all_rounds(eval_summary, ranking, aggregated_importance)

        for r in rounds:
            print(f"\n--- Round {r['round']}: {r['title']} ---")
            log.write(f"\n{'─' * 50}\n")
            log.write(f"Round {r['round']}: {r['title']}\n")
            log.write(f"{'─' * 50}\n")

            if "claims" in r:
                for c in r["claims"]:
                    expert_id = c.get("speaker_id", "?")
                    snippet = c["claim"][:100].replace("\n", " ")
                    print(f"  [{expert_id}]: {snippet}...")
                    log.write(f"\n[{c['expert']}]\n{c['claim']}\n")

            elif "findings" in r:
                for f in r["findings"]:
                    expert_id = f.get("speaker_id", "?")
                    snippet = f["claim"][:100].replace("\n", " ")
                    print(f"  [{expert_id}]: {snippet}...")
                    log.write(f"\n[{f['expert']}]\n{f['claim']}\n")

            elif "challenges" in r:
                for ch in r["challenges"]:
                    expert_id = ch.get("speaker_id", "?")
                    snippet = ch["claim"][:100].replace("\n", " ")
                    print(f"  [{expert_id}]: {snippet}...")
                    log.write(f"\n[{ch['expert']}]\n{ch['claim']}\n")
                    if "limitations" in ch:
                        log.write("限制因素：\n")
                        for lim in ch["limitations"]:
                            log.write(f"  - {lim}\n")

            elif "consensus" in r:
                c = r["consensus"]
                print(f"  Votes: {len(c['votes'])} experts voted")
                print(f"  Ranking: {[(x['factor'], x['confidence']) for x in c['final_ranking']]}")
                print(f"  Budget: {[(x['category'], f'{x['percentage']}%') for x in c['budget_allocation']]}")
                log.write(f"\n=== 投票結果 ===\n")
                for v in c["votes"]:
                    log.write(f"  {v['expert']}: {v['vote']}\n")
                log.write(f"\n=== 最終排名 ===\n")
                for item in c["final_ranking"]:
                    log.write(f"  Rank {item['rank']}: {item['factor']} (importance={item['importance']:.4f}, confidence={item['confidence']})\n")
                log.write(f"\n=== 預算配置 ===\n")
                for item in c["budget_allocation"]:
                    log.write(f"  {item['category']}: {item['percentage']}% — {item['reason']}\n")
                log.write(f"\n=== 商業結論 ===\n")
                for s in c["business_conclusion"]["summary"]:
                    log.write(f"  - {s}\n")

        # ═══════════════════════════════════════════════════
        # Report Generation
        # ═══════════════════════════════════════════════════
        print_section("Phase 5 & 6: Evaluation & Report")

        report = CRISPDMReport(OUTPUT_DIR)
        business_conclusion = discussion.consensus["business_conclusion"] if discussion.consensus else {}
        budget_allocation = discussion.consensus.get("budget_allocation", []) if discussion.consensus else []

        report.generate(df, correlation, eval_summary, ranking,
                        budget_allocation, business_conclusion,
                        forward_selection_df=fs_df)
        print(f"[OK] CRISP-DM Report: {os.path.join(OUTPUT_DIR, 'CRISP_DM_Report.md')}")
        print(f"[OK] Discussion Transcript: {LOG_PATH}")

        # ═══════════════════════════════════════════════════
        # Final Summary
        # ═══════════════════════════════════════════════════
        print_section("Final Summary")
        best_r2 = eval_summary.loc[eval_summary['Test R2'].idxmax(), 'Test R2']
        print(f"""
    Records       : {len(df)}
    Predictors    : {list(df_encoded.columns)}
    Best Model R2 : {best_r2}

    Core Conclusion: R&D Spend is the dominant factor driving Profit.
    Budget Proposal: R&D 60% | Marketing 25% | Administration 10% | Reserve 5%
    """)

        print("Done. All outputs saved to 'output/' directory.")
        print(f"Full Chinese discussion transcript: {LOG_PATH}")
        return 0


if __name__ == "__main__":
    sys.exit(main())
