from typing import Any

from torchmetrics import (
    Accuracy,
    MeanSquaredError,
    MetricCollection,
)


def get_classification_metrics(**kwargs: Any) -> MetricCollection:
    return MetricCollection(
        {
            # 'f1': F1Score(**kwargs),
            # 'precision': Precision(**kwargs),
            # 'recall': Recall(**kwargs),
            # 'AUC_ROC': AUROC(**kwargs),
            'Accuracy': Accuracy(**kwargs),

        },
    )

def get_regression_metrics(**kwargs: Any) -> MetricCollection:
    return MetricCollection(
        {
            'MSE': MeanSquaredError(**kwargs),
        },
    )
