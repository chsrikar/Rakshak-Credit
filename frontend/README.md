# Rakshak Credit — Four Product UI shell

This is a UI-only React module that narrows the main workspace to exactly four products:
A. Cash-Flow Intelligence
B. Affordability & Credit Capacity
C. AI Underwriting Support
D. Small-Business Intelligence

## Files
- `src/components/RakshakFourProducts.jsx`
- `src/styles/rakshak-four-products.css`

## Integration
1. Copy the component into your React frontend at `src/components/RakshakFourProducts.jsx`.
2. Copy the stylesheet into `src/styles/rakshak-four-products.css`.
3. Ensure the relative stylesheet import in the component matches your project structure. If you place the component elsewhere, update the import path.
4. Render `<RakshakFourProducts />` in the route/page you want to use as the new product workspace.
5. Keep the existing app and routes until you've tested the new view. The product view remains a demonstration shell. It does not invent the complete 77-feature neural or 26-feature LightGBM vectors required by the API; the underwriting desk reports skipped components until those source-backed inputs are supplied.

## Important
- The chart bars and service statuses are clearly marked as placeholders.
- Luna handles evidence extraction and reviewer explanations through the OpenAI API; the legacy `gemma_evidence` response field remains for API compatibility.
- The underwriting form submits applicant-entered fields and notes, not statement files. Extraction labels those values as unverified and reports missing evidence.
- Affordability uses local deterministic EMI arithmetic based on editable demo inputs.
- No approval/rejection decision is generated.
- API keys must remain server-side in FastAPI environment variables; never place them in React.
- Before connecting live underwriting, validate input schemas, use actual model outputs, log model/prompt versions, and require human review.
