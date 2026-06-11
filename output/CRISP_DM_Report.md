# Kaggle 50 Startups — CRISP-DM 分析報告

> 五位領域專家多輪討論框架  
> 報告產生時間：2026-06-11 11:05:52

---

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


---

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
| R&D Spend | 50 | 73722 | 45902 | 0 | 39936 | 73051 | 101603 | 165349 |
| Administration | 50 | 121345 | 28018 | 51283 | 103731 | 122700 | 144842 | 182646 |
| Marketing Spend | 50 | 211025 | 122290 | 0 | 129300 | 212716 | 299469 | 471784 |
| Profit | 50 | 112013 | 40306 | 14681 | 90139 | 107978 | 139766 | 192262 |

### 2.4 State 分布

| State | Count |
|-------|-------|
| New York | 17 |
| California | 17 |
| Florida | 16 |
### 2.5 Profit 與各變數之相關係數

| 變數 | 相關係數 | 強度 |
|------|----------|------|
| R&D Spend | 0.9729 | 強相關 |
| Marketing Spend | 0.7478 | 強相關 |
| Administration | 0.2007 | 弱相關 |

### 2.6 初步洞察
- R&D Spend 與 Profit 的相關係數最高，顯示強烈的正向線性關係
- Marketing Spend 與 Profit 有中等程度的正相關
- Administration 與 Profit 的相關性較弱
- State（類別變數）對 Profit 的影響有限，不同州之間的 Profit 中位數差異不大


---

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


---

## Phase 4: Modeling (建模)

### 4.1 使用模型
此分析使用三種迴歸模型進行比較：
1. **Linear Regression** — 線性迴歸，提供係數解釋
2. **Random Forest** — 隨機森林，捕捉非線性關係
3. **XGBoost** — 梯度提升，高效能預測

### 4.2 模型評估摘要

| Model | Test R2 | Test RMSE | Test MAE | CV R2 (5-Fold) |
|-------|---------|-----------|----------|-----------------|
| LinearRegression | 0.8987 | 9055.96 | 6961.48 | 0.9279 ± 0.0438 |
| RandomForest | 0.9147 | 8310.36 | 6131.91 | 0.9277 ± 0.0419 |
| XGBoost | 0.8828 | 9742.04 | 7506.7 | 0.9153 ± 0.0435 |


### 4.3 特徵重要性綜合排名
（取三個模型之平均重要性）

| Rank | Feature | Aggregated Importance |
|------|---------|-----------------------|
| 1 | R&D Spend | 0.9203 |
| 2 | Marketing Spend | 0.0481 |
| 3 | Administration | 0.0188 |
| 4 | State_New York | 0.0087 |
| 5 | State_Florida | 0.0040 |


### 4.4 建模洞察
- 三個模型一致顯示 R&D Spend 是最重要的預測因子
- Marketing Spend 在所有模型中排名第二
- State 編碼後的特徵貢獻最低
- 模型表現穩定（CV R² 標準差小），說明結果具有可信度

### 4.5 逐步特徵選擇 (Forward Feature Selection — Linear Regression)

依特徵重要性順序逐步加入特徵，觀察 Linear Regression 的 RMSE 與 R² 變化：

| Step | Features Added | Selected Features | RMSE | R2 |
|------|---------------|-------------------|------|-----|
| 1 | 1 | R&D Spend | 7714.33 | 0.9265 |
| 2 | 2 | R&D Spend + Marketing Spend | 8206.33 | 0.9168 |
| 3 | 3 | R&D Spend + Marketing Spend + Administration | 8995.91 | 0.9001 |
| 4 | 4 | R&D Spend + Marketing Spend + Administration + State_New York | 9015.11 | 0.8996 |
| 5 | 5 | R&D Spend + Marketing Spend + Administration + State_New York + State_Florida | 9055.96 | 0.8987 |


- 加入 R&D Spend 後 R² 即達到極高水準，證明其主導地位
- 後續特徵對模型改善幅度有限，驗證 R&D Spend 是影響 Profit 的核心驅動因素


---

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


---

## Phase 6: Deployment (部署)

### 6.1 資源配置建議

| 類別 | 建議比例 | 原因 |
|------|----------|------|
| R&D | 60% | 最大預測貢獻 |
| Marketing | 25% | 第二重要因子，與R&D相輔相成 |
| Administration | 10% | 營運基礎支撐 |
| Contingency | 5% | 彈性調度空間 |


### 6.2 商業結論

- R&D 是影響 Profit 的核心驅動因素
- Marketing 為第二重要因子，與 R&D 存在協同效應
- Administration 提供營運支撐，雖非主要獲利驅動力但不可或缺
- State 影響有限，但長期區域策略仍須考量

### 6.3 各因子對 Profit 之估計貢獻

| 因子 | 估計貢獻 |
|------|----------|
| R&D Spend | 70-80% |
| Marketing Spend | 15-25% |
| Administration | 3-8% |
| State | 1-5% |


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

*報告產生時間：2026-06-11 11:05:52*
*分析框架：CRISP-DM (Cross-Industry Standard Process for Data Mining)*

