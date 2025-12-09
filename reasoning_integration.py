"""reasoning_integration.py
Integrates perception with Assignment 2 reasoning:
- Rule-based expert system (reused from rules_python.py)
- Machine-learning model (RandomForest) trained on stroke dataset

Exposes a StrokeReasoner class with a `decide` method.
"""
from typing import Dict, Any, Tuple
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

from rules_python import apply_rules

DATASET_PATH = "healthcare-dataset-stroke-data.csv"

class StrokeReasoner:
    def __init__(self, csv_path: str = DATASET_PATH, prob_threshold: float = 0.5):
        self.csv_path = csv_path
        self.prob_threshold = prob_threshold
        self.model = None
        self.scaler = None
        self.feature_columns = None
        self.numeric_columns = ['age', 'avg_glucose_level', 'bmi']
        self._train_model()

    def _preprocess(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        df = df.copy()
        df['bmi'] = pd.to_numeric(df['bmi'], errors='coerce')
        df['bmi'].fillna(df['bmi'].median(), inplace=True)
        # One-hot encode simple categorical fields
        cat_cols = ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']
        df_encoded = pd.get_dummies(df, columns=cat_cols, drop_first=True)
        y = df_encoded['stroke']
        X = df_encoded.drop(columns=['stroke', 'id'])
        return X, y

    def _train_model(self):
        df = pd.read_csv(self.csv_path)
        X, y = self._preprocess(df)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        scaler = StandardScaler()
        X_train_scaled = X_train.copy()
        X_test_scaled = X_test.copy()

        num_idx = [X_train.columns.get_loc(c) for c in self.numeric_columns if c in X_train.columns]
        X_train_scaled.iloc[:, num_idx] = scaler.fit_transform(X_train.iloc[:, num_idx])
        X_test_scaled.iloc[:, num_idx] = scaler.transform(X_test.iloc[:, num_idx])

        model = RandomForestClassifier(
            n_estimators=200,
            random_state=42,
            class_weight='balanced'
        )
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        acc = accuracy_score(y_test, y_pred)
        print(f"[Reasoning] ML model validation accuracy: {acc:.3f}")

        self.model = model
        self.scaler = scaler
        self.feature_columns = X_train.columns

    def _row_to_feature_vector(self, row: Dict[str, Any]) -> np.ndarray:
        # Build a one-row DataFrame with same columns as training
        df_row = pd.DataFrame([row])
        # Apply same preprocessing as during training
        df_row['bmi'] = pd.to_numeric(df_row['bmi'], errors='coerce')
        df_row['bmi'].fillna(df_row['bmi'].median(), inplace=True)

        cat_cols = ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']
        df_encoded = pd.get_dummies(df_row, columns=cat_cols, drop_first=True)

        # Make sure all training feature columns exist
        for col in self.feature_columns:
            if col not in df_encoded.columns:
                df_encoded[col] = 0
        df_encoded = df_encoded[self.feature_columns]

        # Scale numerics
        X_row = df_encoded.copy()
        num_idx = [X_row.columns.get_loc(c) for c in self.numeric_columns if c in X_row.columns]
        X_row.iloc[:, num_idx] = self.scaler.transform(X_row.iloc[:, num_idx])

        return X_row.values[0]

    def decide(self, perceived_row: Dict[str, Any]) -> Dict[str, Any]:
        """Combine rule-based and ML reasoning for a single perceived row."""
        t0 = time.time()
        # Rule-based decision
        rule_label = apply_rules(perceived_row)

        # ML decision
        x_vec = self._row_to_feature_vector(perceived_row)
        prob_stroke = float(self.model.predict_proba([x_vec])[0][1])
        ml_label = int(prob_stroke >= self.prob_threshold)

        # Simple fusion: flag stroke if either component is positive
        final_label = 1 if (rule_label == 1 or ml_label == 1) else 0

        latency = time.time() - t0
        explanation = (
            f"Rule-based={rule_label}, ML={ml_label} (p={prob_stroke:.3f}), threshold={self.prob_threshold}"
        )

        return {
            "rule_label": int(rule_label),
            "ml_label": ml_label,
            "prob_stroke": prob_stroke,
            "final_label": final_label,
            "latency": latency,
            "explanation": explanation,
        }

    def update_threshold(self, new_threshold: float):
        """Feedback mechanism to update ML probability threshold."""
        self.prob_threshold = float(new_threshold)
