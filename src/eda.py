"""
資料載入、探索性資料分析與視覺化模組
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")
plt.rcParams["font.size"] = 12
plt.rcParams["axes.titlesize"] = 14
plt.rcParams["axes.labelsize"] = 12


class DataLoader:
    """載入 50 Startups 資料集並進行初步清理"""

    def __init__(self, url: str | None = None, local_path: str | None = None):
        self.url = url
        self.local_path = local_path
        self.df = None
        self.df_encoded = None

    def load(self) -> pd.DataFrame:
        if self.local_path and os.path.exists(self.local_path):
            self.df = pd.read_csv(self.local_path)
        elif self.url:
            self.df = pd.read_csv(self.url)
        else:
            raise ValueError("需要提供 URL 或本機路徑")

        self._clean_columns()
        return self.df

    def _clean_columns(self):
        self.df.columns = self.df.columns.str.strip()
        if "Profit" in self.df.columns:
            self.df = self.df.rename(columns={"Profit": "Profit"})
        if "State" not in self.df.columns:
            for col in self.df.columns:
                if "state" in col.lower():
                    self.df = self.df.rename(columns={col: "State"})
                    break

    def check_missing(self) -> pd.DataFrame:
        missing = self.df.isnull().sum()
        missing_pct = (self.df.isnull().sum() / len(self.df)) * 100
        return pd.DataFrame({"Missing Count": missing, "Missing %": missing_pct})

    def encode_state(self) -> pd.DataFrame:
        self.df_encoded = pd.get_dummies(self.df, columns=["State"], drop_first=True)
        return self.df_encoded

    def describe(self) -> pd.DataFrame:
        return self.df.describe().T

    def get_state_distribution(self) -> pd.Series:
        return self.df["State"].value_counts()


class EDA:
    """執行探索性資料分析並產生視覺化圖表"""

    def __init__(self, df: pd.DataFrame, output_dir: str):
        self.df = df
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def correlation_analysis(self) -> pd.DataFrame:
        numeric_df = self.df.select_dtypes(include=[np.number])
        corr_matrix = numeric_df.corr()
        self._plot_correlation_heatmap(corr_matrix)
        profit_corr = corr_matrix["Profit"].drop("Profit").sort_values(ascending=False)
        return profit_corr

    def _plot_correlation_heatmap(self, corr_matrix: pd.DataFrame):
        fig, ax = plt.subplots(figsize=(10, 8))
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
        sns.heatmap(
            corr_matrix,
            annot=True,
            fmt=".3f",
            cmap="RdBu_r",
            center=0,
            mask=mask,
            square=True,
            linewidths=0.5,
            ax=ax,
        )
        ax.set_title("Correlation Matrix — 50 Startups Dataset")
        fig.tight_layout()
        fig.savefig(os.path.join(self.output_dir, "correlation_heatmap.png"), dpi=150)
        plt.close(fig)
        return fig

    def distribution_analysis(self):
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        n_cols = len(numeric_cols)
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()

        for i, col in enumerate(numeric_cols):
            ax = axes[i]
            sns.histplot(self.df[col], kde=True, bins=15, ax=ax, color="steelblue", edgecolor="white")
            ax.axvline(self.df[col].mean(), color="red", linestyle="--", label=f"Mean: {self.df[col].mean():.0f}")
            ax.axvline(self.df[col].median(), color="green", linestyle="--", label=f"Median: {self.df[col].median():.0f}")
            ax.set_title(f"Distribution of {col}")
            ax.set_xlabel(col)
            ax.set_ylabel("Frequency")
            ax.legend(fontsize=9)

        for j in range(n_cols, 4):
            axes[j].set_visible(False)

        fig.tight_layout()
        fig.savefig(os.path.join(self.output_dir, "distribution_analysis.png"), dpi=150)
        plt.close(fig)
        return fig

    def boxplot_by_state(self):
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        n_cols = len(numeric_cols)
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()

        for i, col in enumerate(numeric_cols):
            sns.boxplot(x="State", y=col, data=self.df, ax=axes[i], hue="State", palette="Set2", legend=False)
            axes[i].set_title(f"{col} by State")
            axes[i].set_xlabel("State")
            axes[i].set_ylabel(col)

        for j in range(n_cols, 4):
            axes[j].set_visible(False)

        fig.tight_layout()
        fig.savefig(os.path.join(self.output_dir, "boxplot_by_state.png"), dpi=150)
        plt.close(fig)
        return fig

    def scatter_plots_vs_profit(self):
        predictors = ["R&D Spend", "Administration", "Marketing Spend"]
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        for i, col in enumerate(predictors):
            ax = axes[i]
            sns.scatterplot(x=col, y="Profit", data=self.df, hue="State", ax=ax, palette="Set2", s=80, edgecolor="black")
            ax.set_title(f"Profit vs {col}")
            ax.set_xlabel(col)
            ax.set_ylabel("Profit")

            corr = self.df[col].corr(self.df["Profit"])
            ax.text(
                0.05, 0.95,
                f"r = {corr:.4f}",
                transform=ax.transAxes,
                fontsize=11,
                verticalalignment="top",
                bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.8},
            )

        fig.tight_layout()
        fig.savefig(os.path.join(self.output_dir, "scatter_vs_profit.png"), dpi=150)
        plt.close(fig)
        return fig

    def residual_plots(self, y_true: np.ndarray, y_pred: np.ndarray, model_name: str):
        residuals = y_true - y_pred
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        axes[0].scatter(y_pred, residuals, alpha=0.7, edgecolors="black", linewidth=0.5)
        axes[0].axhline(y=0, color="red", linestyle="--")
        axes[0].set_xlabel("Predicted Profit")
        axes[0].set_ylabel("Residuals")
        axes[0].set_title(f"Residual Plot — {model_name}")

        from scipy import stats
        stats.probplot(residuals, dist="norm", plot=axes[1])
        axes[1].set_title(f"Q-Q Plot — {model_name}")

        fig.tight_layout()
        fig.savefig(os.path.join(self.output_dir, f"residual_{model_name}.png"), dpi=150)
        plt.close(fig)
        return fig

    def feature_importance_plot(self, importance: dict, model_name: str):
        sorted_items = sorted(importance.items(), key=lambda x: x[1])
        features, values = zip(*sorted_items)

        fig, ax = plt.subplots(figsize=(8, 5))
        colors = ["#2ecc71" if v > 0.1 else "#3498db" if v > 0.01 else "#95a5a6" for v in values]
        bars = ax.barh(features, values, color=colors, edgecolor="black", linewidth=0.5)
        ax.set_xlabel("Importance")
        ax.set_title(f"Feature Importance — {model_name} (50 Startups)")
        for bar, val in zip(bars, values):
            ax.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height() / 2, f"{val:.4f}", va="center", fontsize=10)
        fig.tight_layout()
        fig.savefig(os.path.join(self.output_dir, f"feature_importance_{model_name}.png"), dpi=150)
        plt.close(fig)
        return fig

    def forward_selection_analysis(self, X_train, X_test, y_train, y_test,
                                    feature_order: list) -> pd.DataFrame:
        from sklearn.linear_model import LinearRegression
        from sklearn.metrics import r2_score, mean_squared_error

        steps = []
        for k in range(1, len(feature_order) + 1):
            selected = feature_order[:k]
            X_tr = X_train[selected]
            X_te = X_test[selected]
            lr = LinearRegression()
            lr.fit(X_tr, y_train)
            y_pred = lr.predict(X_te)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            r2 = r2_score(y_test, y_pred)
            steps.append({
                "Step": k,
                "Features_Added": k,
                "Selected_Features": " + ".join(selected),
                "RMSE": round(rmse, 2),
                "R2": round(r2, 4),
            })

        df_steps = pd.DataFrame(steps)

        fig, ax1 = plt.subplots(figsize=(10, 6))
        ax2 = ax1.twinx()

        x_vals = df_steps["Step"].values
        rmse_vals = df_steps["RMSE"].values
        r2_vals = df_steps["R2"].values

        line1, = ax1.plot(x_vals, rmse_vals, "o-", color="#3498db", linewidth=2.5,
                          markersize=10, markerfacecolor="white", markeredgewidth=2)
        ax1.set_xlabel("Number of Features")
        ax1.set_ylabel("RMSE", color="#3498db")
        ax1.tick_params(axis="y", labelcolor="#3498db")

        line2, = ax2.plot(x_vals, r2_vals, "s--", color="#2ecc71", linewidth=2.5,
                          markersize=10, markerfacecolor="white", markeredgewidth=2)
        ax2.set_ylabel("R2", color="#2ecc71")
        ax2.tick_params(axis="y", labelcolor="#2ecc71")

        for i in range(len(x_vals)):
            ax1.annotate(f"{rmse_vals[i]:.0f}", (x_vals[i], rmse_vals[i]),
                         textcoords="offset points", xytext=(0, 14),
                         ha="center", fontsize=9, color="#3498db")
            ax2.annotate(f"{r2_vals[i]:.4f}", (x_vals[i], r2_vals[i]),
                         textcoords="offset points", xytext=(0, -18),
                         ha="center", fontsize=9, color="#2ecc71")

        ax1.set_title("Linear Regression — Forward Feature Selection (50 Startups)")
        ax1.set_xticks(x_vals)
        ax1.legend([line1, line2], ["RMSE", "R2"], loc="center right", fontsize=11)
        fig.tight_layout()
        fig.savefig(os.path.join(self.output_dir, "forward_selection_metrics.png"), dpi=150)
        plt.close(fig)

        return df_steps, fig

    def run_all(self):
        results = {}
        results["correlation_with_profit"] = self.correlation_analysis()
        self.distribution_analysis()
        self.boxplot_by_state()
        self.scatter_plots_vs_profit()
        return results
