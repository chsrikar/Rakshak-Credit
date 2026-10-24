
from pathlib import Path
import json

import numpy as np
import pandas as pd
import torch
from torch import nn
import joblib


# backend/app/neural_scoring.py
APP_DIR = Path(__file__).resolve().parent
BACKEND_DIR = APP_DIR.parent
ARTIFACT_DIR = BACKEND_DIR / "artifacts"

MODEL_PATH = ARTIFACT_DIR / "rakshak_deep_tabular_model.pt"
PREPROCESSOR_PATH = ARTIFACT_DIR / "rakshak_preprocessor.joblib"
METADATA_PATH = ARTIFACT_DIR / "rakshak_model_metadata.json"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ResidualBlock(nn.Module):
    def __init__(self, width, dropout=0.20):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(width, width),
            nn.BatchNorm1d(width),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(width, width),
            nn.BatchNorm1d(width),
        )
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.activation(x + self.dropout(self.net(x)))


class RakshakDeepTabularNet(nn.Module):
    def __init__(self, input_dim, width=512, blocks=4, dropout=0.20):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Linear(input_dim, width),
            nn.BatchNorm1d(width),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.blocks = nn.Sequential(*[
            ResidualBlock(width, dropout)
            for _ in range(blocks)
        ])
        self.head = nn.Sequential(
            nn.Linear(width, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.blocks(x)
        return self.head(x).squeeze(1)


_model = None
_preprocessor = None
_metadata = None
_feature_columns = None


def load_neural_model():
    global _model, _preprocessor, _metadata, _feature_columns

    if _model is not None:
        return

    for path in (MODEL_PATH, PREPROCESSOR_PATH, METADATA_PATH):
        if not path.exists():
            raise FileNotFoundError(
                f"Missing model artifact: {path}. "
                "Place all three artifacts in backend/artifacts/."
            )

    with open(METADATA_PATH, encoding="utf-8") as f:
        metadata = json.load(f)

    # Checkpoint was created by your own training notebook.
    # Only load trusted model files.
    checkpoint = torch.load(
        MODEL_PATH, map_location=DEVICE, weights_only=False
    )

    feature_columns = checkpoint["feature_columns"]

    model = RakshakDeepTabularNet(
        input_dim=checkpoint["input_dim"],
        width=checkpoint["width"],
        blocks=checkpoint["blocks"],
        dropout=checkpoint["dropout"],
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(DEVICE)
    model.eval()

    import sklearn.compose._column_transformer as ct
    if not hasattr(ct, "_RemainderColsList"):
        class _RemainderColsList(list): pass
        ct._RemainderColsList = _RemainderColsList

    from sklearn.impute import SimpleImputer
    if not hasattr(SimpleImputer, "_fill_dtype"):
        SimpleImputer._fill_dtype = property(
            lambda self: getattr(self, "_fit_dtype", None)
        )

    preprocessor = joblib.load(PREPROCESSOR_PATH)

    _model = model
    _preprocessor = preprocessor
    _metadata = metadata
    _feature_columns = feature_columns


def neural_model_score(features: dict) -> dict:
    if not isinstance(features, dict):
        raise ValueError("Features must be an object.")
    load_neural_model()

    unknown = sorted(set(features) - set(_feature_columns))
    if unknown:
        raise ValueError("Unknown neural model features: " + ", ".join(unknown))

    missing = [
        name for name in _feature_columns
        if name not in features or features[name] is None
    ]
    if missing:
        raise ValueError(
            "Missing neural model features: " + ", ".join(missing)
        )

    numeric_columns = set(_metadata.get("numeric_columns", []))
    for name in numeric_columns:
        value = features[name]
        if isinstance(value, bool) or not isinstance(value, (int, float, np.number)):
            raise ValueError(f"Neural model feature '{name}' must be numeric.")
        if not np.isfinite(float(value)):
            raise ValueError(f"Neural model feature '{name}' must be finite.")

    # Preserve the original training column order and data types.
    frame = pd.DataFrame(
        [{name: features[name] for name in _feature_columns}],
        columns=_feature_columns,
    )

    transformed = _preprocessor.transform(frame)
    if hasattr(transformed, "toarray"):
        transformed = transformed.toarray()

    transformed = np.asarray(transformed, dtype=np.float32)

    if transformed.ndim != 2 or transformed.shape[1] != _model.stem[0].in_features:
        raise ValueError(
            "Preprocessor output dimensions do not match the trained model."
        )

    with torch.no_grad():
        tensor = torch.from_numpy(transformed).to(DEVICE)
        logit = _model(tensor)
        probability = torch.sigmoid(logit).item()

    threshold = float(_metadata.get("review_threshold", 0.28))

    return {
        "risk_score": round(float(probability), 4),
        "threshold": threshold,
        "review_required": probability >= threshold,
        "model_name": "Rakshak Deep Tabular Neural Network",
        "model_type": "synthetic_demo_only",
        "synthetic_only": True,
        "decision": "REVIEW_REQUIRED"
        if probability >= threshold else "REVIEW_NOT_TRIGGERED",
        "warning": (
            "Experimental score trained on synthetic labels. "
            "Not a validated probability of real-world default and "
            "not for automatic lending decisions."
        ),
    }
