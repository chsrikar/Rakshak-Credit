# Rakshak Credit — Backend Deployment & Reliability Guide

## 1. Environment & Dependency Pinning (Step 4H)

The Rakshak Credit backend uses a dual-model architecture:
1. **LightGBM Baseline Model**: 26 cash-flow features + SHAP TreeExplainer.
2. **Deep Tabular Neural Network (4-block ResNet)**: 77 features transformed through a 170-dimensional ColumnTransformer.

### Tested & Supported Versions
- **Python**: `3.11.x`, `3.12.x`, `3.13.x`, or `3.14.x`
- **scikit-learn**: `>= 1.6.0, < 2.0` (artifact generated with `1.6.1`)
- **PyTorch**: `>= 2.2.0, < 3.0`
- **FastAPI**: `>= 0.115.0, < 1.0`
- **LightGBM**: `>= 4.0.0, < 5.0`
- **Joblib**: `>= 1.4.0, < 2.0`

---

## 2. Model Compatibility Shims

When unpickling the `rakshak_preprocessor.joblib` pipeline (fitted under scikit-learn 1.6.1) across newer scikit-learn or Python releases, two internal classes and attributes require shimming:

### Shim 1: `_RemainderColsList`
In scikit-learn 1.6.1, `ColumnTransformer` internally saved remainder columns as an instance of `sklearn.compose._column_transformer._RemainderColsList`. In newer versions, this internal class was refactored.
```python
import sklearn.compose._column_transformer as ct
if not hasattr(ct, "_RemainderColsList"):
    class _RemainderColsList(list): pass
    ct._RemainderColsList = _RemainderColsList
```

### Shim 2: `SimpleImputer._fill_dtype`
When unpickling `SimpleImputer` instances from older scikit-learn versions, the property `_fill_dtype` is expected during transform, which maps to `_fit_dtype`:
```python
from sklearn.impute import SimpleImputer
if not hasattr(SimpleImputer, "_fill_dtype"):
    SimpleImputer._fill_dtype = property(
        lambda self: getattr(self, "_fit_dtype", None)
    )
```

These shims are automatically applied in `backend/app/neural_scoring.py` before unpickling the preprocessor, guaranteeing seamless runtime compatibility.

---

## 3. Clean Environment Verification Procedure

To test in a clean environment from scratch:

```powershell
# 1. Create a clean virtual environment
python -m venv .clean_venv
.\.clean_venv\Scripts\Activate.ps1

# 2. Upgrade pip and install pinned dependencies
python -m pip install --upgrade pip
pip install -r backend/requirements.txt

# 3. Verify Edge Cases & Regression Test Suite (Step 4D & 4F)
$env:PYTHONPATH="backend"
python -m unittest discover -s backend/tests

# 4. Run Model Comparison on held-out test data (Step 4G)
python -m app.compare_models

# 5. Start the backend service
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

---

## 4. Test Suite Matrix

| Test Suite | Description | Expected Result |
|---|---|---|
| `test_01_valid_77_feature_inference_success` | Full 77-feature inference payload | HTTP 200, valid JSON response with risk score & review status |
| `test_02_empty_features_returns_422` | Empty `{}` features dictionary | HTTP 422 with descriptive missing feature listing |
| `test_03_missing_single_feature_returns_422` | Omission of 1 of 77 required features | HTTP 422 explicitly naming the missing feature |
| `test_04_none_feature_value_returns_422` | Passing `None` for a feature value | HTTP 422 |
| `test_05_invalid_numeric_type_returns_422` | Non-numeric string in numeric feature | HTTP 422 |
| `test_06_malformed_json_body_returns_422` | Missing `features` key in payload | HTTP 422 |
| `test_07_unknown_categorical_handled_safely` | Unseen categorical values (e.g. unknown occupation/city) | HTTP 200, handled safely by OHE |
| `test_08_threshold_behavior_high_risk` | High-risk cash-flow stress indicators | HTTP 200, risk score >= 0.28, `REVIEW_REQUIRED` |
| `test_09_missing_model_artifact_returns_503` | Missing PyTorch `.pt` artifact file | HTTP 503 "Missing model artifact" |
| `test_10_missing_preprocessor_artifact_returns_503` | Missing preprocessor `.joblib` file | HTTP 503 "Missing model artifact" |
| `test_11_missing_metadata_artifact_returns_503` | Missing metadata `.json` file | HTTP 503 "Missing model artifact" |
| `test_12_features_wrong_type_non_dict_returns_422` | Passing string/list for `features` | HTTP 422 |
| `test_13_raw_invalid_json_returns_422` | Unparseable raw JSON body | HTTP 422 |
| `test_14_inference_runtime_failure_returns_500` | Simulated neural forward exception | HTTP 500 "Neural scoring failed" |
| `test_15_dimension_mismatch_returns_422` | Preprocessor dimension mismatch | HTTP 422 |

---

## 5. Model Comparison Summary (Step 4G)

Evaluated on the held-out synthetic test set (225 samples, 60 positive defaults, 26.67% base rate):

```
Metric                    | LightGBM Baseline  | Deep Tabular Net   | Delta     
---------------------------------------------------------------------------------
Feature Count             | 26                 | 77                 | +51 features
Decision Threshold        | 0.20               | 0.28               | -         
ROC-AUC                   | 0.6411             | 0.5904             | -0.0507
Accuracy                  | 0.4622             | 0.7289             | +0.2667
Precision                 | 0.3057             | 0.3333             | +0.0276
Recall                    | 0.8000             | 0.0167             | -0.7833
F1 Score                  | 0.4424             | 0.0317             | -0.4107
Brier Score (MSE)         | 0.1861             | 0.2031             | +0.0170
Review Flag Rate          | 69.78%             | 1.33%              | -68.44%
```

Artifact saved at: `artifacts/model_comparison.json`.

---

## 6. Regulatory & Compliance Notice

All models in this repository are trained exclusively on **synthetic cash-flow data** for demonstration and educational purposes. Neither the LightGBM model nor the Deep Tabular Neural Network provides validated default probabilities for real-world automated lending decisions without lawful training data and regulatory audits.

## 7. Step 4G Benchmark Audit (2026-10-24)

`backend/app/compare_models.py --output backend/artifacts/model_comparison_verified.json` reruns the held-out comparison without replacing the existing `artifacts/model_comparison.json`.

- Positive class: `synthetic_missed_payment == 1`. The data dictionary says this is generated by a toy risk formula, not observed repayment.
- Split sizes are 1,050 train / 225 validation / 225 test; applicant IDs have zero pairwise overlap. The comparison reads labels from `data/data/test.csv` only.
- LightGBM consumes the 26 columns in `backend/app/features.py`; its packaged validation-derived threshold is `0.20`.
- Neural inference consumes the checkpoint's 77 columns, transformed by its serialized `ColumnTransformer` to 170 values; checkpoint shape is 170 inputs, 512 width, 4 residual blocks. Its review threshold is `0.28` from `rakshak_model_metadata.json`.
- The benchmark constructs those 77 columns from test rows plus `backend/data/neural_sample_payload.json` defaults where test data has no corresponding field. Thus it is not a clean head-to-head or feature-parity evaluation.
- Recomputed test confusion matrices (rows actual 0/1, columns predicted 0/1): LightGBM `[[56,109],[12,48]]`; neural `[[163,2],[59,1]]`. Neural scores have min 0.1182, median 0.1530, p90 0.23548, max 0.3091; only three scores meet 0.28. This score distribution, severe feature-template mismatch, and synthetic labels explain why the reported recall is 1/60 (0.0167), but do not isolate whether calibration, training, or mapping caused it.
- No model was retrained or replaced. The sklearn 1.9.1 runtime emitted warnings that the neural preprocessor was saved by sklearn 1.6.1; the inference shim permits loading but the version mismatch remains a reproducibility risk.

## 8. MVP demo: Luna evidence extraction and reviewer support

For the MVP demo, **Luna performs both evidence extraction (the former Gemma role) and reviewer explanation**. The legacy `include_gemma` request flag, `components.gemma_evidence`, and `gemma_evidence` response field remain for API compatibility. Their component message and `model_name` identify Luna as the actual provider. No Gemma provider call is made.

The backend loads `backend/.env` (ignored by Git) without overriding process environment variables. Configure only the Luna/OpenAI endpoint, exact account model ID, and API key:

```dotenv
LUNA_API_URL=https://api.openai.com/v1/chat/completions
LUNA_MODEL_NAME=gpt-6-luna
LUNA_API_KEY=<OpenAI API key>
LUNA_TIMEOUT_SECONDS=30
```

Keep API keys out of the frontend, command history, Git, logs, and API responses. Discover model IDs enabled for the configured OpenAI key (prints IDs only):

```powershell
cd backend
python scripts/list_openai_models.py
```

Then run separate real extraction and explanation requests followed by the unified endpoint, all with labeled synthetic smoke input:

```powershell
python scripts/smoke_model_integrations.py
python -m unittest discover -s tests -v
```

From `frontend`, run `npm run build`; run the API with `uvicorn app.main:app --reload` from `backend` while demonstrating the React app. The smoke command prints statuses and latency only, not model content or secrets.

**Document intake:** the underwriting form accepts PDF, PNG, and JPEG documents up to 12 MB each, categorized as PAN/identity, bank statement, ITR, GST, business registration, income evidence, platform settlement, or other. Draft applicant fields, document bytes, and saved analysis responses are Fernet-encrypted in the local SQLite database at `backend/data/rakshak_demo.db`. The local key is `DOCUMENT_ENCRYPTION_KEY` in ignored `backend/.env`; keep it backed up and stable because changing it makes saved records unreadable. Stored filenames are replaced with generated identifiers.

When an authorized borrower requests an estimate, the backend extracts selectable PDF text (up to 40 pages and 6,000 characters per file, with a 14,000-character total cap), masks common PAN, Aadhaar-length, account, IFSC, and phone identifiers, then sends the remaining unverified text to the configured Luna endpoint for evidence extraction. The original file bytes are not sent. Extracted text can also be retrieved by the application owner or lender at `GET /api/v1/underwriting/applications/{application_id}/documents/{document_id}/text`; the lender UI has a **Read text** action. Scanned PDFs and images return `ocr_unavailable` because a Tesseract OCR runtime is not installed. Luna observations are not document verification; review originals. This local demo still lacks malware scanning, a formal retention policy, account provisioning, and production deployment controls. Do not use real personal documents on a shared or network-exposed deployment.

Generate a local encryption key without printing it to the terminal, then append it to the ignored backend environment file:

```powershell
$documentKey = ..\.venv\Scripts\python.exe -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
Add-Content -LiteralPath .env -Value "`nDOCUMENT_ENCRYPTION_KEY=$documentKey"
$documentKey = $null
```

Run those commands from `backend`. Keep the key backed up securely and do not commit it. Existing deterministic financial calculations and independent scoring outputs remain authoritative; Luna cannot approve or reject an application. Neural and LightGBM scores are omitted unless complete source-referenced feature vectors are supplied.
