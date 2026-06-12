"""
機器學習建模與評估模組

使用 Linear Regression, Random Forest, XGBoost 三種模型，
以量化方式評估各變數對 Profit 的影響力。
"""

import os
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SequentialFeatureSelector, RFE, SelectKBest, f_regression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler

try:
    from xgboost import XGBRegressor
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False


class MLModeling:
    """機器學習建模與評估"""

    def __init__(self, df: pd.DataFrame, target: str, output_dir: str):
        self.df = df
        self.target = target
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.models = {}
        self.results = {}
        self.feature_importances = {}
        self.scaler = StandardScaler()

    def prepare_data(self, test_size: float = 0.2, random_state: int = 42):
        self.y = self.df[self.target]
        self.X = self.df.drop(columns=[self.target])
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=test_size, random_state=random_state
        )
        self.X_train = pd.DataFrame(self.scaler.fit_transform(self.X_train), columns=self.X_train.columns)
        self.X_test = pd.DataFrame(self.scaler.transform(self.X_test), columns=self.X_test.columns)
        return self.X_train, self.X_test, self.y_train, self.y_test

    def train_linear_regression(self) -> dict:
        model = LinearRegression()
        model.fit(self.X_train, self.y_train)
        y_pred_train = model.predict(self.X_train)
        y_pred_test = model.predict(self.X_test)

        importances = dict(zip(self.X.columns, np.abs(model.coef_)))
        total = sum(importances.values()) or 1
        importances = {k: v / total for k, v in importances.items()}

        result = {
            "model": model,
            "y_pred_train": y_pred_train,
            "y_pred_test": y_pred_test,
            "coefficients": dict(zip(self.X.columns, model.coef_)),
            "intercept": model.intercept_,
            "feature_importance": importances,
            "r2_train": r2_score(self.y_train, y_pred_train),
            "r2_test": r2_score(self.y_test, y_pred_test),
            "rmse_train": np.sqrt(mean_squared_error(self.y_train, y_pred_train)),
            "rmse_test": np.sqrt(mean_squared_error(self.y_test, y_pred_test)),
            "mae_train": mean_absolute_error(self.y_train, y_pred_train),
            "mae_test": mean_absolute_error(self.y_test, y_pred_test),
        }

        cv_scores = cross_val_score(model, self.X, self.y, cv=KFold(5, shuffle=True, random_state=42), scoring="r2")
        result["cv_r2_mean"] = cv_scores.mean()
        result["cv_r2_std"] = cv_scores.std()

        self.models["LinearRegression"] = result
        self.feature_importances["LinearRegression"] = importances
        return result

    def train_random_forest(self, n_estimators: int = 100, random_state: int = 42) -> dict:
        model = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state, n_jobs=-1)
        model.fit(self.X_train, self.y_train)
        y_pred_train = model.predict(self.X_train)
        y_pred_test = model.predict(self.X_test)

        importances = dict(zip(self.X.columns, model.feature_importances_))

        result = {
            "model": model,
            "y_pred_train": y_pred_train,
            "y_pred_test": y_pred_test,
            "feature_importance": importances,
            "r2_train": r2_score(self.y_train, y_pred_train),
            "r2_test": r2_score(self.y_test, y_pred_test),
            "rmse_train": np.sqrt(mean_squared_error(self.y_train, y_pred_train)),
            "rmse_test": np.sqrt(mean_squared_error(self.y_test, y_pred_test)),
            "mae_train": mean_absolute_error(self.y_train, y_pred_train),
            "mae_test": mean_absolute_error(self.y_test, y_pred_test),
        }

        cv_scores = cross_val_score(model, self.X, self.y, cv=KFold(5, shuffle=True, random_state=42), scoring="r2")
        result["cv_r2_mean"] = cv_scores.mean()
        result["cv_r2_std"] = cv_scores.std()

        self.models["RandomForest"] = result
        self.feature_importances["RandomForest"] = importances
        return result

    def train_xgboost(self, n_estimators: int = 100, random_state: int = 42) -> dict | None:
        if not HAS_XGBOOST:
            print("[WARNING] XGBoost 未安裝，跳過 XGBoost 模型訓練。")
            return None

        model = XGBRegressor(n_estimators=n_estimators, random_state=random_state, verbosity=0)
        model.fit(self.X_train, self.y_train)
        y_pred_train = model.predict(self.X_train)
        y_pred_test = model.predict(self.X_test)

        importances = dict(zip(self.X.columns, model.feature_importances_))

        result = {
            "model": model,
            "y_pred_train": y_pred_train,
            "y_pred_test": y_pred_test,
            "feature_importance": importances,
            "r2_train": r2_score(self.y_train, y_pred_train),
            "r2_test": r2_score(self.y_test, y_pred_test),
            "rmse_train": np.sqrt(mean_squared_error(self.y_train, y_pred_train)),
            "rmse_test": np.sqrt(mean_squared_error(self.y_test, y_pred_test)),
            "mae_train": mean_absolute_error(self.y_train, y_pred_train),
            "mae_test": mean_absolute_error(self.y_test, y_pred_test),
        }

        cv_scores = cross_val_score(model, self.X, self.y, cv=KFold(5, shuffle=True, random_state=42), scoring="r2")
        result["cv_r2_mean"] = cv_scores.mean()
        result["cv_r2_std"] = cv_scores.std()

        self.models["XGBoost"] = result
        self.feature_importances["XGBoost"] = importances
        return result

    def train_all(self) -> dict:
        self.prepare_data()
        self.train_linear_regression()
        self.train_random_forest()
        self.train_xgboost()
        return self.results

    def get_evaluation_summary(self) -> pd.DataFrame:
        rows = []
        for name, result in self.models.items():
            rows.append({
                "Model": name,
                "Train R2": round(result["r2_train"], 4),
                "Test R2": round(result["r2_test"], 4),
                "Train RMSE": round(result["rmse_train"], 2),
                "Test RMSE": round(result["rmse_test"], 2),
                "Train MAE": round(result["mae_train"], 2),
                "Test MAE": round(result["mae_test"], 2),
                "CV R2 Mean": round(result["cv_r2_mean"], 4),
                "CV R2 Std": round(result["cv_r2_std"], 4),
            })
        return pd.DataFrame(rows)

    def get_aggregated_feature_importance(self) -> dict:
        aggregated = {}
        all_features = list(self.X.columns)
        for feature in all_features:
            values = []
            for name, importance in self.feature_importances.items():
                if feature in importance:
                    values.append(importance[feature])
            aggregated[feature] = float(np.mean(values)) if values else 0.0

        total = sum(aggregated.values()) or 1
        return {k: round(v / total, 4) for k, v in sorted(aggregated.items(), key=lambda x: x[1], reverse=True)}

    def get_final_ranking(self) -> pd.DataFrame:
        agg = self.get_aggregated_feature_importance()
        rows = []
        for rank, (feature, importance) in enumerate(agg.items(), 1):
            rows.append({
                "Rank": rank,
                "Feature": feature,
                "Aggregated Importance": importance,
            })
        return pd.DataFrame(rows)

    def save_models(self):
        os.makedirs(self.output_dir, exist_ok=True)
        for name, result in self.models.items():
            filepath = os.path.join(self.output_dir, f"{name.lower()}_model.pkl")
            with open(filepath, "wb") as f:
                pickle.dump(result["model"], f)

    def multi_algorithm_feature_selection(self, figures_dir: str = None):
        """
        Compare 5 feature selection methods across different numbers of features:
        SFS (Forward), RFE, SelectKBest, Lasso (L1), Random Forest

        Returns:
            comparison_df: DataFrame with RMSE and R2 for each method/feature count
            ranking_table: DataFrame showing feature rankings per method
        """
        if figures_dir is None:
            figures_dir = self.output_dir
        os.makedirs(figures_dir, exist_ok=True)

        X = self.X.copy()
        y = self.y.copy()
        all_features = list(X.columns)
        n_features = len(all_features)

        methods = {
            "SFS (Forward)": self._sfs_forward,
            "RFE": self._rfe_selection,
            "SelectKBest": self._selectkbest_selection,
            "Lasso (L1)": self._lasso_selection,
            "Random Forest": self._rf_selection,
        }

        results = {}
        rankings = {}

        for method_name, method_func in methods.items():
            rmse_list = []
            r2_list = []
            feature_order = method_func(X, y, n_features)

            for k in range(1, n_features + 1):
                selected_features = feature_order[:k]
                X_sel = X[selected_features]
                X_train, X_test, y_train, y_test = train_test_split(
                    X_sel, y, test_size=0.2, random_state=42
                )
                scaler = StandardScaler()
                X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=selected_features)
                X_test = pd.DataFrame(scaler.transform(X_test), columns=selected_features)

                lr = LinearRegression()
                lr.fit(X_train, y_train)
                y_pred = lr.predict(X_test)

                rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                r2 = r2_score(y_test, y_pred)
                rmse_list.append(rmse)
                r2_list.append(r2)

            results[method_name] = {"RMSE": rmse_list, "R2": r2_list}
            rankings[method_name] = feature_order

        comparison_df = pd.DataFrame({
            "Method": list(methods.keys()),
            **{f"RMSE_{k+1}": [results[m]["RMSE"][k] for m in methods] for k in range(n_features)},
            **{f"R2_{k+1}": [results[m]["R2"][k] for m in methods] for k in range(n_features)},
        })

        ranking_table = pd.DataFrame(rankings).T
        ranking_table.columns = [f"Rank {i+1}" for i in range(n_features)]
        ranking_table.index.name = "Algorithm"

        self._plot_multi_algorithm_comparison(results, rankings, figures_dir)

        return comparison_df, ranking_table

    def _sfs_forward(self, X, y, k):
        if k >= len(X.columns):
            return list(X.columns)
        lr = LinearRegression()
        sfs = SequentialFeatureSelector(lr, n_features_to_select=k, direction="forward", cv=5)
        sfs.fit(X, y)
        selected = list(X.columns[sfs.get_support()])
        remaining = [f for f in X.columns if f not in selected]
        return selected + remaining

    def _rfe_selection(self, X, y, k):
        rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        rfe = RFE(rf, n_features_to_select=1)
        rfe.fit(X, y)
        ranks = rfe.ranking_
        feature_rank_pairs = sorted(zip(ranks, X.columns))
        return [f for _, f in feature_rank_pairs]

    def _selectkbest_selection(self, X, y, k):
        skb = SelectKBest(score_func=f_regression, k=min(k, len(X.columns)))
        skb.fit(X, y)
        scores = skb.scores_
        feature_score_pairs = sorted(zip(scores, X.columns), reverse=True)
        return [f for _, f in feature_score_pairs]

    def _lasso_selection(self, X, y, k):
        scaler = StandardScaler()
        X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
        lasso = Lasso(alpha=0.01, random_state=42, max_iter=10000)
        lasso.fit(X_scaled, y)
        coef_abs = np.abs(lasso.coef_)
        feature_coef_pairs = sorted(zip(coef_abs, X.columns), reverse=True)
        return [f for _, f in feature_coef_pairs]

    def _rf_selection(self, X, y, k):
        rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        rf.fit(X, y)
        importances = rf.feature_importances_
        feature_imp_pairs = sorted(zip(importances, X.columns), reverse=True)
        return [f for _, f in feature_imp_pairs]

    def _plot_multi_algorithm_comparison(self, results, rankings, figures_dir):
        methods = list(results.keys())
        n_features = len(results[methods[0]]["RMSE"])
        x_vals = list(range(1, n_features + 1))

        colors = {
            "SFS (Forward)": "#1f77b4",
            "RFE": "#ff7f0e",
            "SelectKBest": "#2ca02c",
            "Lasso (L1)": "#d62728",
            "Random Forest": "#9467bd",
        }

        fig = plt.figure(figsize=(16, 10))
        gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 0.6], hspace=0.35, wspace=0.3)

        ax_rmse = fig.add_subplot(gs[0, 0])
        ax_r2 = fig.add_subplot(gs[0, 1])

        for method in methods:
            color = colors.get(method, "#333333")
            ax_rmse.plot(x_vals, results[method]["RMSE"], "o-", color=color,
                         label=method, linewidth=2, markersize=6)
            ax_r2.plot(x_vals, results[method]["R2"], "o-", color=color,
                       label=method, linewidth=2, markersize=6)

        ax_rmse.set_title("RMSE by Number of Features (All Algorithms)", fontsize=13, fontweight="bold")
        ax_rmse.set_xlabel("Number of Features")
        ax_rmse.set_ylabel("RMSE")
        ax_rmse.legend(loc="best", fontsize=9)
        ax_rmse.grid(True, alpha=0.3)

        ax_r2.set_title("R-squared by Number of Features (All Algorithms)", fontsize=13, fontweight="bold")
        ax_r2.set_xlabel("Number of Features")
        ax_r2.set_ylabel("R-squared")
        ax_r2.legend(loc="best", fontsize=9)
        ax_r2.grid(True, alpha=0.3)

        table_ax = fig.add_subplot(gs[1:, :])
        table_ax.axis("off")

        table_data = []
        for method in methods:
            row = [method]
            for i in range(n_features):
                feat = rankings[method][i] if i < len(rankings[method]) else "-"
                row.append(feat if feat else "-")
            table_data.append(row)

        col_labels = ["Algorithm"] + [f"Rank {i+1}" for i in range(n_features)]
        table = table_ax.table(
            cellText=table_data,
            colLabels=col_labels,
            cellLoc="center",
            loc="center",
            colColours=["#f0f0f0"] * len(col_labels),
        )
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.8)

        for j in range(len(col_labels)):
            table[0, j].set_text_props(fontweight="bold")

        method_colors = [colors.get(m, "#ffffff") for m in methods]
        for i, color in enumerate(method_colors):
            table[i + 1, 0].set_text_props(color=color, fontweight="bold")

        table_ax.set_title("Feature Selection Rankings by Algorithm", fontsize=13, fontweight="bold", pad=20)

        fig.savefig(os.path.join(figures_dir, "multi_algorithm_feature_selection.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)
