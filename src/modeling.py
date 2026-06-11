"""
機器學習建模與評估模組

使用 Linear Regression, Random Forest, XGBoost 三種模型，
以量化方式評估各變數對 Profit 的影響力。
"""

import os
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
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
