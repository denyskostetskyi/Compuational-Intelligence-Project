from pydantic import BaseModel
from typing import Dict, Any, List
import numpy as np

class ModelReport(BaseModel):
    model_name: str
    target_classes_mapping: Dict[Any, Any]
    accuracy: float
    confusion_matrix: List[List[int]]
    precision: List[float]
    recall: List[float]
    f1: List[float]
    roc_auc: float

    @classmethod
    def from_raw(
        cls,
        model_name: str,
        target_classes_mapping: Dict[Any, Any],
        accuracy: float,
        confusion_matrix: np.ndarray,
        precision: np.ndarray,
        recall: np.ndarray,
        f1: np.ndarray,
        roc_auc: float
    ):
        return cls(
            model_name=model_name,
            target_classes_mapping=target_classes_mapping,
            accuracy=accuracy,
            confusion_matrix=confusion_matrix.tolist(),
            precision=precision.tolist(),
            recall=recall.tolist(),
            f1=f1.tolist(),
            roc_auc=roc_auc
        )

    def show(self):
        print(f"=== Report for {self.model_name} ===")
        print(f"Accuracy: {self.accuracy:.4f}\n")
        print("Classification Report:")
        print("\nPer-Class Metrics:")
        for i, label in self.target_classes_mapping:
            print(f"Class: {label}")
            print(f"  Precision: {self.precision[i]:.4f}")
            print(f"  Recall: {self.recall[i]:.4f}")
            print(f"  F1-score: {self.f1[i]:.4f}")
        print(f"\nMacro-Averaged ROC AUC: {self.roc_auc:.4f}")
