"""
svm_model.py
------------
Model definition for the binary classifier (cooling system: ON=1 / OFF=0)
using Support Vector Machine with an RBF kernel.

Features (in this fixed order):
    [body_temp, respiratory_rate, ambient_temp]

A StandardScaler is included inside the Pipeline because SVM with an RBF
kernel is distance-based and is sensitive to unscaled features
(temperature in deg C vs respiratory rate in cycles/min have very
different numeric ranges).
"""

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

FEATURE_COLUMNS = ["body_temp", "respiratory_rate", "ambient_temp"]


def build_svm_pipeline(C: float = 1.0, gamma="scale", probability: bool = True) -> Pipeline:
    """
    Builds the SVM (RBF kernel) pipeline.

    Parameters
    ----------
    C : float
        Regularization strength. Smaller C -> wider margin, more tolerance
        for misclassified points. Larger C -> fits training data tighter.
    gamma : 'scale', 'auto', or float
        RBF kernel coefficient. Controls how far the influence of a single
        training example reaches.
    probability : bool
        If True, enables predict_proba() (useful for setting a confidence
        threshold before triggering the cooling relay).
    """
    pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "svm",
                SVC(
                    kernel="rbf",
                    C=C,
                    gamma=gamma,
                    probability=probability,
                    class_weight="balanced",
                    random_state=42,
                ),
            ),
        ]
    )
    return pipeline


if __name__ == "__main__":
    model = build_svm_pipeline()
    print(model)
