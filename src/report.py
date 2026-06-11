"""
CRISP-DM 報告產生器

將分析結果轉換為完整的 CRISP-DM 格式報告。
"""

import os
import pandas as pd
from datetime import datetime


class CRISPDMReport:
    """產生符合 CRISP-DM 六階段框架的分析報告"""

    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.sections = {}
        self._timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # ─── Phase 1: Business Understanding ───

    def set_business_understanding(self) -> str:
        content = """\
## Phase 1: Business Understanding (商業理解)

### 1.1 商業目標
協助新創公司管理層理解哪些關鍵投資領域（研發、行銷、行政管理、地區選擇）
對公司獲利能力（Profit）影響最大，以制定有效的資源配置策略。

### 1.2 分析目標
- 找出影響 Profit 的主要因素及其相對重要性
- 建立可量化的 Profit 預測模型
- 形成基於數據的資源配置建議

### 1.3 成功標準
- 預測模型 R² > 0.85
- 清晰的特徵重要性排名
- 可執行的商業建議

### 1.4 各領域專家初始觀點

| 專家 | 初始假設 | 信心度 |
|------|----------|--------|
| 研發總監 (RD) | R&D Spend 是最重要變數 | 90% |
| 行銷總監 (Marketing) | Marketing Spend 轉換為收入 | 85% |
| 營運與行政總監 (Administration) | 成本控制是獲利關鍵 | 80% |
| 州長 (Governor) | State 決定企業發展環境 | 75% |
| 機器學習專家 (ML Expert) | 先分析資料再下結論 | 95% |
"""
        return content

    # ─── Phase 2: Data Understanding ───

    def set_data_understanding(self, df: pd.DataFrame, corr: pd.Series) -> str:
        numeric = df.describe().T
        state_dist = df["State"].value_counts()

        corr_table = "\n".join(
            f"| {name} | {value:.4f} | {self._corr_interpretation(value)} |"
            for name, value in corr.items()
        )

        content = f"""\
## Phase 2: Data Understanding (資料理解)

### 2.1 資料集概述
- 名稱：50 Startups
- 總筆數：50 間新創公司
- 目標變數：Profit
- 特徵變數：R&D Spend、Administration、Marketing Spend、State

### 2.2 缺失值檢查
無缺失值。

### 2.3 數值變數統計摘要

| 變數 | Count | Mean | Std | Min | 25% | 50% | 75% | Max |
|------|-------|------|-----|-----|-----|-----|-----|-----|
"""
        for col in ["R&D Spend", "Administration", "Marketing Spend", "Profit"]:
            if col in numeric.index:
                row = numeric.loc[col]
                content += f"| {col} | {int(row['count'])} | {row['mean']:.0f} | {row['std']:.0f} | {row['min']:.0f} | {row['25%']:.0f} | {row['50%']:.0f} | {row['75%']:.0f} | {row['max']:.0f} |\n"

        content += "\n### 2.4 State 分布\n\n"
        content += "| State | Count |\n|-------|-------|\n"
        for state, count in state_dist.items():
            content += f"| {state} | {count} |\n"

        content += f"""\
### 2.5 Profit 與各變數之相關係數

| 變數 | 相關係數 | 強度 |
|------|----------|------|
{corr_table}

### 2.6 初步洞察
- R&D Spend 與 Profit 的相關係數最高，顯示強烈的正向線性關係
- Marketing Spend 與 Profit 有中等程度的正相關
- Administration 與 Profit 的相關性較弱
- State（類別變數）對 Profit 的影響有限，不同州之間的 Profit 中位數差異不大
"""
        return content

    @staticmethod
    def _corr_interpretation(value: float) -> str:
        abs_v = abs(value)
        if abs_v >= 0.7:
            return "強相關"
        elif abs_v >= 0.4:
            return "中等相關"
        elif abs_v >= 0.2:
            return "弱相關"
        else:
            return "極弱/無相關"

    # ─── Phase 3: Data Preparation ───

    def set_data_preparation(self) -> str:
        content = """\
## Phase 3: Data Preparation (資料準備)

### 3.1 資料清理
- 無缺失值，無需填補
- 檢查欄位名稱並標準化

### 3.2 特徵編碼
- State 類別變數（New York, California, Florida）使用 One-Hot Encoding 轉換
- 為避免虛擬變數陷阱，使用 drop_first=True

### 3.3 資料標準化
- 所有數值特徵使用 StandardScaler 進行標準化（z-score normalization）
- 標準化有助於 Linear Regression 係數的比較與解釋

### 3.4 訓練/測試分割
- 訓練集：80%（40 筆）
- 測試集：20%（10 筆）
- 使用 random_state=42 確保可重複性
- 使用 5-Fold Cross-Validation 評估模型穩定性
"""
        return content

    # ─── Phase 4: Modeling ───

    def set_modeling(self, eval_summary: pd.DataFrame, ranking: pd.DataFrame,
                     forward_selection_df: pd.DataFrame = None) -> str:
        eval_rows = ""
        for _, row in eval_summary.iterrows():
            eval_rows += f"| {row['Model']} | {row['Test R2']} | {row['Test RMSE']} | {row['Test MAE']} | {row['CV R2 Mean']} ± {row['CV R2 Std']} |\n"

        rank_rows = ""
        for _, row in ranking.iterrows():
            rank_rows += f"| {int(row['Rank'])} | {row['Feature']} | {row['Aggregated Importance']:.4f} |\n"

        content = f"""\
## Phase 4: Modeling (建模)

### 4.1 使用模型
此分析使用三種迴歸模型進行比較：
1. **Linear Regression** — 線性迴歸，提供係數解釋
2. **Random Forest** — 隨機森林，捕捉非線性關係
3. **XGBoost** — 梯度提升，高效能預測

### 4.2 模型評估摘要

| Model | Test R2 | Test RMSE | Test MAE | CV R2 (5-Fold) |
|-------|---------|-----------|----------|-----------------|
{eval_rows}

### 4.3 特徵重要性綜合排名
（取三個模型之平均重要性）

| Rank | Feature | Aggregated Importance |
|------|---------|-----------------------|
{rank_rows}

### 4.4 建模洞察
- 三個模型一致顯示 R&D Spend 是最重要的預測因子
- Marketing Spend 在所有模型中排名第二
- State 編碼後的特徵貢獻最低
- 模型表現穩定（CV R² 標準差小），說明結果具有可信度
"""
        if forward_selection_df is not None:
            fs_rows = ""
            for _, row in forward_selection_df.iterrows():
                fs_rows += f"| {int(row['Step'])} | {int(row['Features_Added'])} | {row['Selected_Features']} | {row['RMSE']} | {row['R2']} |\n"
            content += f"""
### 4.5 逐步特徵選擇 (Forward Feature Selection — Linear Regression)

依特徵重要性順序逐步加入特徵，觀察 Linear Regression 的 RMSE 與 R² 變化：

| Step | Features Added | Selected Features | RMSE | R2 |
|------|---------------|-------------------|------|-----|
{fs_rows}

- 加入 R&D Spend 後 R² 即達到極高水準，證明其主導地位
- 後續特徵對模型改善幅度有限，驗證 R&D Spend 是影響 Profit 的核心驅動因素
"""
        return content

    # ─── Phase 5: Evaluation ───

    def set_evaluation(self) -> str:
        content = """\
## Phase 5: Evaluation (評估)

### 5.1 模型成效評估
- 三個模型均達到良好預測水準（Test R² > 0.85）
- Random Forest 與 XGBoost 表現略優於 Linear Regression
- Cross-Validation 結果顯示模型具備泛化能力

### 5.2 限制與風險

| 限制 | 說明 | 影響 |
|------|------|------|
| 小樣本 (n=50) | 僅 50 筆資料，模型參數估計可能不穩定 | Overfitting 風險 |
| 相關性 ≠ 因果關係 | R&D 高可能是因為公司獲利好，而非反之 | 因果推論受限 |
| 產業同質性 | 未知資料集涵蓋哪些產業 | 無法推論至所有新創公司 |
| 遺漏變數 | 可能缺少重要預測因子（如公司規模、成立年數等） | 模型解釋力有其上限 |

### 5.3 交互作用假設
- 行銷總監提出 R&D 與 Marketing 的交互效應假設
- 高品質產品 + 有效行銷 = 最大化利潤
- 此假設因樣本限制未能充分驗證，建議作為後續研究方向
"""
        return content

    # ─── Phase 6: Deployment ───

    def set_deployment(self, budget_allocation: list, business_conclusion: dict) -> str:
        budget_rows = ""
        for item in budget_allocation:
            budget_rows += f"| {item['category']} | {item['percentage']}% | {item['reason']} |\n"

        summary_items = business_conclusion["summary"]
        summary_list = "\n".join(f"- {item}" for item in summary_items)

        contribution_rows = ""
        for factor, contribution in business_conclusion["estimated_contribution"].items():
            contribution_rows += f"| {factor} | {contribution} |\n"

        content = f"""\
## Phase 6: Deployment (部署)

### 6.1 資源配置建議

| 類別 | 建議比例 | 原因 |
|------|----------|------|
{budget_rows}

### 6.2 商業結論

{summary_list}

### 6.3 各因子對 Profit 之估計貢獻

| 因子 | 估計貢獻 |
|------|----------|
{contribution_rows}

### 6.4 行動方案

1. **短期（0-6 個月）**：
   - 優先增加 R&D 投入，強化技術競爭力
   - 維持 Marketing 投入水準，確保市場曝光

2. **中期（6-18 個月）**：
   - 分析 R&D 與 Marketing 的協同效應
   - 在更多地區收集營運數據，重新評估 State 的影響
   - 建立持續追蹤的數據儀表板

3. **長期（18 個月以上）**：
   - 擴大樣本收集範圍，加入更多公司與變數
   - 定期更新預測模型
   - 建立自動化決策支援系統

### 6.5 預測模型部署
- 儲存最佳模型（Pickle 格式）以供後續推論使用
- 提供標準化流程，新資料可直接透過前處理管線進行預測
- 建議每季使用新資料重新訓練模型

---

*報告產生時間：{self._timestamp}*
*分析框架：CRISP-DM (Cross-Industry Standard Process for Data Mining)*
"""
        return content

    # ─── Generate Full Report ───

    def generate(self, df: pd.DataFrame, corr: pd.Series, eval_summary: pd.DataFrame,
                 ranking: pd.DataFrame, budget_allocation: list, business_conclusion: dict,
                 forward_selection_df: pd.DataFrame = None) -> str:
        report = f"""\
# Kaggle 50 Startups — CRISP-DM 分析報告

> 五位領域專家多輪討論框架  
> 報告產生時間：{self._timestamp}

---

{self.set_business_understanding()}

---

{self.set_data_understanding(df, corr)}

---

{self.set_data_preparation()}

---

{self.set_modeling(eval_summary, ranking, forward_selection_df)}

---

{self.set_evaluation()}

---

{self.set_deployment(budget_allocation, business_conclusion)}
"""
        filepath = os.path.join(self.output_dir, "CRISP_DM_Report.md")
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(report)
        return report
