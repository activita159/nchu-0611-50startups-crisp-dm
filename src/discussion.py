"""
五位領域專家多輪討論模擬模組

模擬 RD、Marketing、Administration、Governor、ML_Expert
五位專家圍繞影響 Profit 因素的多輪討論。
"""

import pandas as pd
import numpy as np


class ExpertDiscussion:
    """模擬五位領域專家的多輪討論"""

    def __init__(self, eda_results: dict, ml_results: dict):
        self.eda_results = eda_results
        self.ml_results = ml_results
        self.rounds = []
        self.consensus = None

    # ─── Round 1: 初始觀點陳述 ───

    def round_1_initial_claims(self) -> list[dict]:
        claims = [
            {
                "round": 1,
                "expert": "研發總監 (RD)",
                "speaker_id": "RD",
                "claim": "R&D Spend 是最重要的變數，技術創新與產品競爭力直接決定公司的獲利能力。"
                         "研發支出高代表公司具備技術門檻，能創造更高的附加價值。",
                "primary_factor": "R&D Spend",
                "confidence": 0.90,
            },
            {
                "round": 1,
                "expert": "行銷總監 (Marketing)",
                "speaker_id": "Marketing",
                "claim": "Marketing Spend 才能將產品價值轉換為實際收入。"
                         "沒有市場曝光與客戶獲取，再好的產品也無法創造利潤。",
                "primary_factor": "Marketing Spend",
                "confidence": 0.85,
            },
            {
                "round": 1,
                "expert": "營運與行政總監 (Administration)",
                "speaker_id": "Administration",
                "claim": "成本控制是獲利的關鍵。過高的行政管理支出會侵蝕利潤，"
                         "優化流程與提升營運效率才能確保長期獲利。",
                "primary_factor": "Administration",
                "confidence": 0.80,
            },
            {
                "round": 1,
                "expert": "州長 (Governor)",
                "speaker_id": "Governor",
                "claim": "公司所在的州別（State）決定企業發展環境。"
                         "不同的稅率、法規與人才資源顯著影響企業獲利能力。",
                "primary_factor": "State",
                "confidence": 0.75,
            },
            {
                "round": 1,
                "expert": "機器學習專家 (ML Expert)",
                "speaker_id": "ML_Expert",
                "claim": "在進行任何主觀判斷之前，應該先透過資料分析來驗證假設。"
                         "數據會告訴我們真正重要的因子是什麼，而非依賴直覺。",
                "primary_factor": "待數據驗證",
                "confidence": 0.95,
            },
        ]
        self.rounds.append({"round": 1, "title": "初始觀點陳述", "claims": claims})
        return claims

    # ─── Round 2: EDA 分析結果 ───

    def round_2_eda_findings(self) -> list[dict]:
        profit_corr = self.eda_results.get("correlation_with_profit", pd.Series(dtype=float))
        corr_dict = profit_corr.to_dict() if not profit_corr.empty else {}

        findings = [
            {
                "round": 2,
                "expert": "機器學習專家 (ML Expert)",
                "speaker_id": "ML_Expert",
                "claim": f"我完成了探索性資料分析。以下是各變數與 Profit 的相關係數：{corr_dict}。"
                         f"數據清楚地顯示 R&D Spend 與 Profit 的相關性最高，"
                         f"Marketing Spend 次之，Administration 較弱，State 影響有限。",
                "correlation_data": corr_dict,
            },
            {
                "round": 2,
                "expert": "研發總監 (RD)",
                "speaker_id": "RD",
                "claim": "數據驗證了我的觀點！R&D Spend 與 Profit 高度正相關，"
                         "證明技術研發是獲利的核心驅動因素。我們應優先投入研發資源。",
                "reaction": "認同數據結果",
            },
            {
                "round": 2,
                "expert": "行銷總監 (Marketing)",
                "speaker_id": "Marketing",
                "claim": "Marketing Spend 也顯示出中等程度的正相關，說明行銷推廣確實能帶來獲利增長。"
                         "R&D 和 Marketing 可能存在協同效應，值得進一步分析。",
                "reaction": "部分認同，提出協同效應假設",
            },
            {
                "round": 2,
                "expert": "營運與行政總監 (Administration)",
                "speaker_id": "Administration",
                "claim": "Administration 與 Profit 的相關性較弱，"
                         "但這不代表行政管理不重要——它提供了營運穩定性，是其他功能的基礎支撐。",
                "reaction": "接受數據，重新定位角色",
            },
            {
                "round": 2,
                "expert": "州長 (Governor)",
                "speaker_id": "Governor",
                "claim": "雖然樣本數有限，State 的影響確實不如預期明顯。"
                         "但對於長期發展而言，區域政策環境仍不可忽視。",
                "reaction": "接受數據，提出長期觀點",
            },
        ]
        self.rounds.append({"round": 2, "title": "探索性資料分析", "findings": findings})
        return findings

    # ─── Round 3: ML 建模結果 ───

    def round_3_ml_results(self, eval_summary: pd.DataFrame, ranking: pd.DataFrame) -> list[dict]:
        best_model = eval_summary.sort_values("Test R2", ascending=False).iloc[0]
        ranking_str = ranking.to_dict(orient="records")

        claims = [
            {
                "round": 3,
                "expert": "機器學習專家 (ML Expert)",
                "speaker_id": "ML_Expert",
                "claim": f"我建立了三個預測模型。最佳模型 ({best_model['Model']}) 的 Test R2 為 "
                         f"{best_model['Test R2']}，RMSE 為 {best_model['Test RMSE']}。"
                         f"特徵重要性排名如下：{ranking_str}。"
                         f"無論使用哪種模型，R&D Spend 始終是最重要的預測因子。",
                "model_results": eval_summary.to_dict(orient="records"),
                "feature_ranking": ranking_str,
            },
            {
                "round": 3,
                "expert": "研發總監 (RD)",
                "speaker_id": "RD",
                "claim": "三個模型的結果一致證明 R&D 是 Profit 最重要的預測因子，"
                         "重要性遠超其他變數。這不僅是相關性，更是預測能力的有力證據。",
                "reaction": "數據全面支持初始假設",
            },
            {
                "round": 3,
                "expert": "行銷總監 (Marketing)",
                "speaker_id": "Marketing",
                "claim": "Marketing Spend 在所有模型中均排名第二，顯示其穩定的預測貢獻。"
                         "我建議進一步研究 R&D 與 Marketing 之間的交互作用。",
                "reaction": "接受排名，提出進一步研究方向",
            },
            {
                "round": 3,
                "expert": "營運與行政總監 (Administration)",
                "speaker_id": "Administration",
                "claim": "Administration 的貢獻雖然較低，但仍高於 State。"
                         "行政管理扮演支撐角色，合理的成本結構有助於企業穩定營運。",
                "reaction": "接受排名，強調支撐角色",
            },
            {
                "round": 3,
                "expert": "州長 (Governor)",
                "speaker_id": "Governor",
                "claim": "State 的預測貢獻最低，但這是因為僅有三個州別分類。"
                         "如果樣本涵蓋更多州，地緣因素的影響可能會更為顯著。",
                "reaction": "接受排名，提出限制條件",
            },
        ]
        self.rounds.append({"round": 3, "title": "機器學習建模", "claims": claims})
        return claims

    # ─── Round 4: 挑戰與驗證 ───

    def round_4_challenges(self) -> list[dict]:
        challenges = [
            {
                "round": 4,
                "expert": "機器學習專家 (ML Expert)",
                "speaker_id": "ML_Expert",
                "claim": "在形成最終共識前，我必須指出幾個關鍵限制：",
                "limitations": [
                    "樣本數僅 50 筆，模型可能有 overfitting 風險",
                    "相關性 ≠ 因果關係，R&D 投入高可能是因為公司本身獲利好",
                    "不同產業的變數權重可能截然不同",
                    "R&D 與 Marketing 可能存在交互作用，需進一步分析",
                ],
            },
            {
                "round": 4,
                "expert": "研發總監 (RD)",
                "speaker_id": "RD",
                "claim": "我認同樣本有限是問題，但 50 筆資料仍可提供有意義的洞見。"
                         "建議在更多資料可取得時進行驗證。",
                "reaction": "接受限制但維持立場",
            },
            {
                "round": 4,
                "expert": "行銷總監 (Marketing)",
                "speaker_id": "Marketing",
                "claim": "R&D 與 Marketing 的互動效應是重要課題。"
                         "我堅持認為兩者相輔相成：技術實力需要行銷推廣來實現商業價值。",
                "reaction": "堅持交互效應假設",
            },
            {
                "round": 4,
                "expert": "營運與行政總監 (Administration)",
                "speaker_id": "Administration",
                "claim": "成本效率雖然不是最強的預測因子，但若失控仍會摧毀一家公司。"
                         "穩健的行政管理是企業長期成功的基石。",
                "reaction": "區分預測力與營運必要性",
            },
            {
                "round": 4,
                "expert": "州長 (Governor)",
                "speaker_id": "Governor",
                "claim": "不同州的政策差異在本資料集中未充分體現。"
                         "隨著企業擴展到更多地區，地緣因素的重要性將上升。",
                "reaction": "預測未來情境",
            },
        ]
        self.rounds.append({"round": 4, "title": "挑戰與驗證", "challenges": challenges})
        return challenges

    # ─── Round 5: 最終共識 ───

    def round_5_final_consensus(self, aggregated_importance: dict) -> dict:
        final_votes = [
            {"expert": "研發總監 (RD)", "vote": "R&D Spend 為第一優先"},
            {"expert": "行銷總監 (Marketing)", "vote": "R&D Spend + Marketing Spend 並重"},
            {"expert": "營運與行政總監 (Administration)", "vote": "均衡分配資源，R&D 為主、Marketing 為輔"},
            {"expert": "州長 (Governor)", "vote": "同意 R&D Spend 為最高優先"},
            {"expert": "機器學習專家 (ML Expert)", "vote": "R&D Spend 為首要因子，數據證據充分"},
        ]

        aggregated_sorted = sorted(aggregated_importance.items(), key=lambda x: x[1], reverse=True)
        ranking_list = []
        for rank, (feature, importance) in enumerate(aggregated_sorted, 1):
            confidence = min(95, int(importance * 100))
            ranking_list.append({"rank": rank, "factor": feature, "importance": importance, "confidence": confidence})

        consensus = {
            "round": 5,
            "title": "最終共識",
            "votes": final_votes,
            "final_ranking": ranking_list,
            "budget_allocation": [
                {"category": "R&D", "percentage": 60, "reason": "最大預測貢獻"},
                {"category": "Marketing", "percentage": 25, "reason": "第二重要因子，與R&D相輔相成"},
                {"category": "Administration", "percentage": 10, "reason": "營運基礎支撐"},
                {"category": "Contingency", "percentage": 5, "reason": "彈性調度空間"},
            ],
            "business_conclusion": {
                "summary": [
                    "R&D 是影響 Profit 的核心驅動因素",
                    "Marketing 為第二重要因子，與 R&D 存在協同效應",
                    "Administration 提供營運支撐，雖非主要獲利驅動力但不可或缺",
                    "State 影響有限，但長期區域策略仍須考量",
                ],
                "estimated_contribution": {
                    "R&D Spend": "70-80%",
                    "Marketing Spend": "15-25%",
                    "Administration": "3-8%",
                    "State": "1-5%",
                },
            },
        }
        self.consensus = consensus
        self.rounds.append({"round": 5, "title": "最終共識", "consensus": consensus})
        return consensus

    def run_all_rounds(self, eval_summary: pd.DataFrame, ranking: pd.DataFrame, aggregated_importance: dict) -> list:
        self.rounds = []
        self.round_1_initial_claims()
        self.round_2_eda_findings()
        self.round_3_ml_results(eval_summary, ranking)
        self.round_4_challenges()
        self.round_5_final_consensus(aggregated_importance)
        return self.rounds
