# 50 Startups CRISP-DM 專家討論框架 — 專案總結

> **Version**: 1.0  
> **Date**: 2026-06-11  
> **Objective**: 以五位領域專家多輪討論結合 CRISP-DM 六階段方法論，分析 50 Startups 資料集中影響 Profit 的主要因素，形成可執行的決策建議。

---

## Workflow 流程圖

![Workflow](output/workflow.png)

---

## 1. 專案概述

此專案模擬 **五位領域專家**（研發總監、行銷總監、行政總監、州長、機器學習專家）圍繞同一資料集進行 **五輪討論**，從各自專業角度提出假設，再透過 **資料分析與機器學習建模** 檢驗，最終達成共識。

### 討論框架

| Round | 主題 | 重點 |
|-------|------|------|
| 1 | 初始觀點陳述 | 各專家提出主觀假設 |
| 2 | 探索性資料分析 (EDA) | 以資料驗證假設 |
| 3 | 機器學習建模 | 三個模型量化評估特徵重要性 |
| 4 | 挑戰與驗證 | 討論限制、風險與交互效應 |
| 5 | 最終共識 | 投票表決與資源配置建議 |

---

## 2. 專案結構

```
nchu-0611/
├── main.py                      # 主程式（完整管線）
├── generate_workflow.py         # 流程圖生成腳本
├── config.yaml                  # 專案設定
├── requirements.txt             # Python 相依套件
├── data/
│   └── 50_Startups.csv          # 資料集（50 筆）
├── src/
│   ├── __init__.py
│   ├── eda.py                   # 資料載入、EDA、視覺化
│   ├── modeling.py              # ML 建模 (LR / RF / XGBoost)
│   ├── discussion.py            # 五位專家多輪討論模擬
│   └── report.py                # CRISP-DM 報告產生器
└── output/
    ├── CRISP_DM_Report.md       # 完整六階段分析報告 (236 行)
    ├── discussion_transcript.txt # 專家討論逐字稿 (135 行)
    ├── workflow.png             # 專案流程圖
    ├── figures/                 # 11 張視覺化圖表
    │   ├── correlation_heatmap.png
    │   ├── distribution_analysis.png
    │   ├── boxplot_by_state.png
    │   ├── scatter_vs_profit.png
    │   ├── forward_selection_metrics.png
    │   ├── residual_LinearRegression.png
    │   ├── residual_RandomForest.png
    │   ├── residual_XGBoost.png
    │   ├── feature_importance_LinearRegression.png
    │   ├── feature_importance_RandomForest.png
    │   └── feature_importance_XGBoost.png
    └── models/                  # 3 個訓練好的模型
        ├── linearregression_model.pkl
        ├── randomforest_model.pkl
        └── xgboost_model.pkl
```

---

## 3. CRISP-DM 六階段對應

| 階段 | 模組 | 主要工作 |
|------|------|----------|
| **Phase 1: Business Understanding** | `report.py` | 定義商業目標、各專家初始觀點 |
| **Phase 2: Data Understanding** | `eda.py` | 相關係數分析、分布視覺化、缺失值檢查 |
| **Phase 3: Data Preparation** | `modeling.py` | One-Hot Encoding (State)、StandardScaler、80/20 分割 |
| **Phase 4: Modeling** | `modeling.py` | Linear Regression、Random Forest、XGBoost + 5-Fold CV |
| **Phase 5: Evaluation** | `report.py` | 比較 R²/RMSE/MAE、殘差診斷、限制分析 |
| **Phase 6: Deployment** | `report.py` | 資源配置建議、行動方案、模型部署策略 |

---

## 4. 核心分析結果

### 4.1 相關係數（Pearson）

| 變數 | 與 Profit 之相關係數 | 強度 |
|------|---------------------|------|
| R&D Spend | 0.9729 | 強相關 |
| Marketing Spend | 0.7478 | 強相關 |
| Administration | 0.2007 | 弱相關 |

### 4.2 模型評估

| Model | Test R² | Test RMSE | Test MAE | CV R² (5-Fold) |
|-------|---------|-----------|----------|-----------------|
| Linear Regression | 0.8987 | 9055.96 | 6961.48 | 0.9279 ± 0.0438 |
| Random Forest | **0.9147** | **8310.36** | **6131.91** | 0.9277 ± 0.0419 |
| XGBoost | 0.8828 | 9742.04 | 7506.70 | 0.9153 ± 0.0435 |

### 4.3 特徵重要性綜合排名（三模型平均）

| Rank | Feature | Importance | Confidence |
|------|---------|------------|------------|
| 1 | R&D Spend | 0.9203 | 95 |
| 2 | Marketing Spend | 0.0481 | 80 |
| 3 | Administration | 0.0188 | 40 |
| 4 | State_New York | 0.0087 | 20 |
| 5 | State_Florida | 0.0040 | 20 |

### 4.4 逐步特徵選擇（Forward Feature Selection — LR）

| Step | Selected Features | RMSE | R² |
|------|-------------------|------|-----|
| 1 | R&D Spend only | 7714.33 | **0.9265** |
| 2 | + Marketing Spend | 8206.33 | 0.9168 |
| 3 | + Administration | 8995.91 | 0.9001 |
| 4 | + State_New York | 9015.11 | 0.8996 |
| 5 | + State_Florida | 9055.96 | 0.8987 |

> **關鍵發現**：僅用 R&D Spend 一項特徵，Linear Regression 即可達到 R² = 0.9265。加入其餘特徵後，R² 不升反降，證實 R&D Spend 對 Profit 的壓倒性主導地位。

---

## 5. 專家共識摘要

### 投票結果

| 專家 | 最終立場 |
|------|----------|
| RD（研發總監） | R&D Spend 為第一優先 |
| Marketing（行銷總監） | R&D + Marketing 並重 |
| Administration（行政總監） | 均衡分配，R&D 為主 |
| Governor（州長） | 同意 R&D 最高優先 |
| ML Expert（機器學習專家） | R&D Spend 數據證據充分 |

### 資源配置建議

| 類別 | 比例 | 原因 |
|------|------|------|
| R&D | 60% | 最大預測貢獻，核心獲利驅動力 |
| Marketing | 25% | 第二重要因子，與 R&D 相輔相成 |
| Administration | 10% | 營運基礎支撐 |
| Contingency | 5% | 彈性調度空間 |

### 各因子估計貢獻

| 因子 | 估計貢獻 |
|------|----------|
| R&D Spend | 70-80% |
| Marketing Spend | 15-25% |
| Administration | 3-8% |
| State | 1-5% |

---

## 6. 限制與風險

| 限制 | 說明 | 影響 |
|------|------|------|
| 小樣本 (n=50) | 僅 50 筆資料 | Overfitting 風險 |
| 相關性 ≠ 因果 | R&D 高可能是結果而非原因 | 因果推論受限 |
| 產業未知 | 未標註行業類別 | 無法推論至所有新創 |
| 遺漏變數 | 缺公司規模/年齡等 | 模型解釋力有上限 |
| XGBoost 過擬合 | Train R² = 1.0, Test R² = 0.88 | 建議增加 regularization |

---

## 7. 部署與行動方案

| 時程 | 行動 |
|------|------|
| **短期 (0-6m)** | 優先增加 R&D 投入；維持 Marketing 支出 |
| **中期 (6-18m)** | 分析 R&D × Marketing 交互效應；擴大地區數據收集 |
| **長期 (18m+)** | 擴大樣本、新增變數；建立自動化決策支援系統 |

### 模型部署
- 三個模型以 Pickle 格式儲存於 `output/models/`
- 新資料可經 StandardScaler + One-Hot Encoding 前處理後直接推論
- 建議每季使用新資料重新訓練

---

## 8. 執行方式

```bash
cd nchu-0611

# 安裝相依套件
pip install -r requirements.txt

# 執行完整管線
python main.py

# 產生流程圖（獨立執行）
python generate_workflow.py

# 查看產出
ls output/
#  - CRISP_DM_Report.md        # 完整 CRISP-DM 報告
#  - discussion_transcript.txt # 專家討論逐字稿
#  - workflow.png              # 流程圖
#  - figures/                  # 11 張視覺化圖表
#  - models/                   # 3 個訓練好的模型
```

---

## 9. 技術棧

| 類別 | 技術 |
|------|------|
| 語言 | Python 3.14 |
| 資料處理 | pandas, numpy |
| 視覺化 | matplotlib, seaborn |
| 機器學習 | scikit-learn, xgboost |
| 統計 | scipy (Q-Q plot) |
| 報告 | Markdown (UTF-8) |

---

*Generated by: **50 Startups CRISP-DM Expert Discussion Framework v1.0** · 2026-06-11*
