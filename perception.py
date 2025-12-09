"""perception.py
Perception module for Assignment 3 (Integrated ISD).
Simulates tabular sensor input using the stroke dataset from Assignment 2.
"""
import pandas as pd
import numpy as np
import time
from typing import Dict, Any

DATASET_PATH = "healthcare-dataset-stroke-data.csv"

class StrokePerception:
    """Perception layer that streams simulated patient records."""

    def __init__(self, csv_path: str = DATASET_PATH):
        self.df = pd.read_csv(csv_path)
        # Basic cleaning similar to Assignment 2
        self.df['bmi'] = pd.to_numeric(self.df['bmi'], errors='coerce')
        self.df['bmi'].fillna(self.df['bmi'].median(), inplace=True)
        self.df.reset_index(drop=True, inplace=True)

    def get_sample(self, idx: int = None, add_noise: bool = True) -> Dict[str, Any]:
        """Return one perceived patient example as a dict.

        If idx is None, a random row is sampled.
        Noise is optionally added to numeric attributes to simulate sensors.
        """
        if idx is None:
            row = self.df.sample(1, random_state=None).iloc[0]
        else:
            row = self.df.iloc[int(idx)]

        row_dict = row.to_dict()

        if add_noise:
            # Add mild Gaussian noise to numeric features
            for col in ['age', 'avg_glucose_level', 'bmi']:
                if col in row_dict and pd.notnull(row_dict[col]):
                    noise = np.random.normal(0, 0.05 * float(row_dict[col]))
                    row_dict[col] = float(row_dict[col]) + noise

        # Perception timestamp
        row_dict['_perceived_at'] = time.time()
        return row_dict

    def get_ground_truth(self, idx: int) -> int:
        """Return ground-truth stroke label for evaluation."""
        return int(self.df.iloc[int(idx)]['stroke'])
