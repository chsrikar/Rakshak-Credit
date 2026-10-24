import React, { useEffect, useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import "../styles/borrower-underwriting.css";
import {
  fetchUnderwritingAnalysis, fetchUnderwritingDocuments, saveUnderwritingApplication,
  uploadUnderwritingDocument, removeUnderwritingDocument,
} from "../services/neuralScoring";
import { getDemoUser, signOutDemo } from "../services/demoAuth";

const OCCUPATIONS = [
  ["wage", "Salaried / wage income"], ["kirana", "Shop / small business"],
  ["retail", "Retail / local business"], ["delivery", "Gig / platform work"],
  ["service", "Freelance / service income"], ["startup", "Starting a business"],
  ["not_employed", "No current occupation"], ["other", "Other income source"],
];
const LOAN_PURPOSES = [
  ["working_capital", "Working capital"], ["inventory", "Inventory / stock"],
  ["equipment", "Tools or equipment"], ["business_expansion", "Business expansion"],
  ["education", "Education"], ["healthcare", "Healthcare"], ["home_improvement", "Home improvement"],
  ["personal", "Personal expense"], ["other", "Other purpose"],
];
const DOCUMENT_TYPES = [
  ["identity_pan", "PAN / identity"], ["bank_statement", "Bank statement / account history"],
  ["tax_return", "Income tax return (ITR)"], ["other", "Other supporting document"],
  ["gst_record", "GST return / registration"],
  ["business_registration", "Business registration / licence"], ["income_invoice", "Payslip / invoices / contracts"],
  ["platform_statement", "Platform / gig income statement"],
];
const REQUIRED_DOCUMENTS = [
  ["identity_pan", "PAN / identity"],
  ["bank_statement", "Bank statement / account history"],
  ["tax_return", "Income tax return (ITR)"],
];
const money = (value) => Number.isFinite(Number(value))
  ? new Intl.NumberFormat("en-IN", { style: "currency", currency: "INR", maximumFractionDigits: 0 }).format(Number(value))
  : "Not available";

export default function BorrowerUnderwriting() {
  const navigate = useNavigate();
  const user = getDemoUser();
  const [form, setForm] = useState({ borrowerName: "", occupation: "not_employed", businessName: "", city: "", monthlyInflow: "", monthlyExpenses: "", existingDebt: "0", requestedAmount: "", purpose: "", evidence: "self_reported", notes: "" });
  const [applicationId, setApplicationId] = useState(null);
  const [draftSaved, setDraftSaved] = useState(false);
  const [documents, setDocuments] = useState([]);
  const [requiredFiles, setRequiredFiles] = useState({});
  const [otherCategory, setOtherCategory] = useState("other");
  const [otherFile, setOtherFile] = useState(null);
  const [busy, setBusy] = useState(false);
  const [analysis, setAnalysis] = useState(null);
  const [error, setError] = useState("");
  const [notice, setNotice] = useState("");

  const setField = (key, value) => {
    setForm((current) => ({ ...current, [key]: value }));
    setDraftSaved(false);
    setAnalysis(null);
    setNotice("");
  };
  const complete = Boolean(form.borrowerName.trim() && form.occupation && form.city.trim() &&
    form.monthlyInflow !== "" && form.monthlyExpenses !== "" && form.existingDebt !== "" &&
    [form.monthlyInflow, form.monthlyExpenses, form.existingDebt].every((x) => Number.isFinite(Number(x)) && Number(x) >= 0));

  const uploadedCategories = new Set(documents.map((item) => item.category));
  const missingRequiredDocuments = REQUIRED_DOCUMENTS.filter(([categoryName]) => !uploadedCategories.has(categoryName));
  const documentsComplete = missingRequiredDocuments.length === 0;

  const saveDraft = async () => {
    setError("");
    const result = await saveUnderwritingApplication(form, applicationId || undefined);
    setApplicationId(result.application_id);
    setDraftSaved(true);
    setNotice(`Application ${result.application_id} saved.`);
    return result.application_id;
  };

  useEffect(() => {
    if (!applicationId) return;
    fetchUnderwritingDocuments(applicationId).then(setDocuments).catch((cause) => setError(cause.message || "Could not load evidence files."));
  }, [applicationId]);

  const upload = async (uploadCategory, uploadFile) => {
    if (!uploadFile) return;
    setBusy(true); setError("");
    try {
      // Documents can be uploaded before the borrower has completed their profile.
      // Save the current draft first so the upload always attaches to its application.
      const id = await saveDraft();
      await uploadUnderwritingDocument(id, uploadCategory, uploadFile);
      setDocuments(await fetchUnderwritingDocuments(id));
      setAnalysis(null);
      if (REQUIRED_DOCUMENTS.some(([requiredCategory]) => requiredCategory === uploadCategory)) {
        setRequiredFiles((current) => ({ ...current, [uploadCategory]: null }));
        const picker = document.getElementById(`borrower-document-${uploadCategory}`);
        if (picker) picker.value = "";
      } else {
        setOtherFile(null);
        const picker = document.getElementById("borrower-other-document-file");
        if (picker) picker.value = "";
      }
      setNotice("Evidence uploaded and encrypted. It will be text-extracted and sent to Luna only when you request an estimate.");
    } catch (cause) { setError(cause.message || "Upload failed."); }
    finally { setBusy(false); }
  };

  const remove = async (id) => {
    if (!applicationId || !window.confirm("Remove this evidence file?")) return;
    setError("");
    try {
      await removeUnderwritingDocument(applicationId, id);
      setDocuments(await fetchUnderwritingDocuments(applicationId));
      setAnalysis(null);
      setNotice("Evidence file removed.");
    } catch (cause) { setError(cause.message || "Could not remove that file."); }
  };

  const getEstimate = async () => {
    if (!complete || !documentsComplete) return;
    setBusy(true); setError(""); setNotice("");
    try {
      const id = draftSaved && applicationId ? applicationId : await saveDraft();
      const result = await fetchUnderwritingAnalysis({ ...form, applicationId: id });
      setAnalysis(result);
      setDocuments(await fetchUnderwritingDocuments(id));
    } catch (cause) { setError(cause.message || "Could not calculate the estimate."); }
    finally { setBusy(false); }
  };
  const logout = async () => {
    const completed = await signOutDemo();
    if (!completed) { setError("Could not clear the backend session. Check that the API is running, then try signing out again."); return; }
    navigate("/signin", { replace: true });
  };

  const calculations = analysis?.financial_calculations;
  const amountCeiling = calculations?.illustrative_capacity_range_inr?.maximum ?? calculations?.principal_within_illustrative_capacity;
  const score100 = analysis?.lightgbm_scoring?.rakshak_score != null
    ? Math.round(((analysis.lightgbm_scoring.rakshak_score - 300) / 600) * 100)
    : null;

  return <main className="ru-shell">
    <header className="ru-header">
      <Link to="/" className="ru-brand"><span className="ru-brand-mark">R</span><span className="ru-brand-title">rakshak<span>credit</span></span></Link>
      <div className="ru-header-right"><Link to="/">Overview</Link><span className="ru-dot" /> BORROWER · {user?.username || "DEMO"}<button className="ru-signout" type="button" onClick={logout}>Sign out</button></div>
    </header>
    <section className="ru-banner"><span className="ru-banner-icon">i</span><div><strong>Estimate only.</strong> This demo does not approve or reject loans. Enter your information and supporting evidence; a lender makes any real decision.</div></section>
    <section className="ru-intro"><div><div className="ru-eyebrow">BORROWER APPLICATION</div><h1>Your details and evidence.</h1><p>Tell us about your income and expenses. Add the documents that fit your situation. No loan amount or business purpose is required here.</p></div><div className="ru-id-card"><span>APPLICATION</span><strong>{applicationId || "Not saved"}</strong><small>{draftSaved ? "Draft saved securely" : "Saved when you continue"}</small></div></section>

    <div className="ru-layout borrower-simple-layout">
      <section className="ru-panel ru-form-panel">
        <div className="ru-panel-heading"><div><div className="ru-eyebrow">STEP 01</div><h2>About you</h2></div><span className="ru-step">Required details</span></div>
        <div className="ru-grid two">
          <label className="ru-field">Full name<input value={form.borrowerName} onChange={(event) => setField("borrowerName", event.target.value)} autoComplete="name" /></label>
          <label className="ru-field">City / town<input value={form.city} onChange={(event) => setField("city", event.target.value)} /></label>
          <label className="ru-field">Occupation / income source<select value={form.occupation} onChange={(event) => setField("occupation", event.target.value)}>{OCCUPATIONS.map(([value, label]) => <option key={value} value={value}>{label}</option>)}</select></label>
          <label className="ru-field">Business name <span>(optional)</span><input value={form.businessName} onChange={(event) => setField("businessName", event.target.value)} placeholder="Leave blank if not applicable" /></label>
        </div>
        <div className="ru-section-divider" />
        <div className="ru-panel-heading"><div><div className="ru-eyebrow">STEP 02</div><h2>Monthly finances</h2></div><span className="ru-step">INR</span></div>
        <div className="ru-grid three">
          <label className="ru-field">Income / inflow<input type="number" min="0" value={form.monthlyInflow} onChange={(event) => setField("monthlyInflow", event.target.value)} placeholder="0" /></label>
          <label className="ru-field">Expenses<input type="number" min="0" value={form.monthlyExpenses} onChange={(event) => setField("monthlyExpenses", event.target.value)} placeholder="0" /></label>
          <label className="ru-field">Existing monthly debt<input type="number" min="0" value={form.existingDebt} onChange={(event) => setField("existingDebt", event.target.value)} placeholder="0" /></label>
        </div>
        <p className="ru-section-intro">Use your best monthly estimate. If you currently have no occupation or are starting a business, enter your current income as zero and explain relevant context to your lender.</p>
        <div className="ru-section-divider" />
        <div className="ru-section-divider" />
        <div className="ru-panel-heading"><div><div className="ru-eyebrow">STEP 03</div><h2>Loan request</h2></div><span className="ru-step">Amount or purpose</span></div>
        <p className="ru-section-intro">Enter the amount you need, choose what the loan is for, or provide both. You can leave the amount blank if you are still deciding.</p>
        <div className="ru-grid two">
          <label className="ru-field">Requested amount <span>(optional if purpose selected)</span><input type="number" min="1" max="10000000" value={form.requestedAmount} onChange={(event) => setField("requestedAmount", event.target.value)} placeholder="Amount in INR" /></label>
          <label className="ru-field">Loan purpose <span>(optional if amount entered)</span><select value={form.purpose} onChange={(event) => setField("purpose", event.target.value)}><option value="">Choose a purpose</option>{LOAN_PURPOSES.map(([value, label]) => <option key={value} value={value}>{label}</option>)}</select></label>
        </div>
        <div className="ru-section-divider" />
        <div className="ru-panel-heading"><div><div className="ru-eyebrow">STEP 04</div><h2>Required documents</h2></div><span className="ru-step">All 3 required</span></div>
        <p className="ru-section-intro">Upload each required document before requesting an estimate. Business registration is not required for salaried applicants, people between jobs, or new businesses.</p>
        <div className="ru-required-documents" aria-label="Required borrower documents">{REQUIRED_DOCUMENTS.map(([requiredCategory, label]) => <div className="ru-required-document" key={requiredCategory}><span className={uploadedCategories.has(requiredCategory) ? "is-uploaded" : ""}>{uploadedCategories.has(requiredCategory) ? "✓" : "!"}</span><strong>{label}</strong><small>{uploadedCategories.has(requiredCategory) ? "Received" : "Required"}</small></div>)}</div>
        <div className="ru-required-upload-grid" aria-label="Upload each required document">
          {REQUIRED_DOCUMENTS.map(([requiredCategory, label]) => {
            const uploaded = uploadedCategories.has(requiredCategory);
            return <div className={`ru-required-upload-card${uploaded ? " is-uploaded" : ""}`} key={requiredCategory}>
              <div className="ru-required-upload-heading"><strong>{label}</strong><small>{uploaded ? "Received" : "Required"}</small></div>
              <label className="ru-field">Choose PDF, PNG, or JPEG<input id={`borrower-document-${requiredCategory}`} type="file" accept=".pdf,.png,.jpg,.jpeg,application/pdf,image/png,image/jpeg" onChange={(event) => setRequiredFiles((current) => ({ ...current, [requiredCategory]: event.target.files?.[0] || null }))} /></label>
              <button type="button" className="ru-secondary-button" disabled={!requiredFiles[requiredCategory] || busy} onClick={() => upload(requiredCategory, requiredFiles[requiredCategory])}>{busy ? "Working…" : uploaded ? "Upload another" : `Upload ${label}`}</button>
            </div>;
          })}
        </div>
        <div className="ru-document-upload">
          <div className="ru-panel-heading"><div><div className="ru-eyebrow">OPTIONAL EVIDENCE</div><h3>Other supporting documents</h3></div></div>
          <div className="ru-document-fields">
            <label className="ru-field">Evidence type<select value={otherCategory} onChange={(event) => setOtherCategory(event.target.value)}>{DOCUMENT_TYPES.filter(([value]) => !REQUIRED_DOCUMENTS.some(([required]) => required === value)).map(([value, label]) => <option key={value} value={value}>{label}</option>)}</select></label>
            <label className="ru-field">Choose PDF, PNG, or JPEG<input id="borrower-other-document-file" type="file" accept=".pdf,.png,.jpg,.jpeg,application/pdf,image/png,image/jpeg" onChange={(event) => setOtherFile(event.target.files?.[0] || null)} /></label>
          </div>
          <div className="ru-upload-actions"><small>Up to 12 MB per file · encrypted in local demo storage</small><button type="button" className="ru-secondary-button" disabled={!otherFile || busy} onClick={() => upload(otherCategory, otherFile)}>{busy ? "Working…" : "Upload supporting document"}</button></div>
          <div className="ru-demo-samples"><strong>Trying the demo?</strong><span>Download fictional, invalid samples marked “NOT A REAL DOCUMENT”:</span><div><a href="/demo-documents/synthetic-demo-pan-identity.pdf" download>Identity</a><a href="/demo-documents/synthetic-demo-bank-statement.pdf" download>Bank statement</a><a href="/demo-documents/synthetic-demo-tax-summary.pdf" download>Tax summary</a></div></div>
          <div className="ru-storage-note"><strong>Document analysis in this demo.</strong> Files are encrypted. When you request an estimate, selectable PDF text is extracted, common PAN/account/phone identifiers are masked, and the remaining text is sent to configured Luna for evidence review. Image and scanned-PDF OCR is unavailable here. Extraction is not verification; a lender must inspect the original file.</div>
          {!complete && <div className="ru-document-state">You can upload documents now. Add your name, city, and monthly income, expenses, and debt before requesting an estimate.</div>}
          {complete && !documentsComplete && <div className="ru-document-state">Still required: {missingRequiredDocuments.map(([, label]) => label).join(", ")}.</div>}
          {notice && <div className="ru-document-state is-success" role="status">{notice}</div>}
          {error && <div className="ru-document-state is-error" role="alert">{error}</div>}
          {documents.length > 0 && <div className="ru-document-list" aria-label="Uploaded evidence">{documents.map((item) => <div className="ru-document-item" key={item.id}><span className="ru-document-icon">PDF</span><span><strong>{DOCUMENT_TYPES.find(([value]) => value === item.category)?.[1] || "Supporting evidence"}</strong><small>{(item.size_bytes / (1024 * 1024)).toFixed(2)} MB · Received, pending review</small></span><button type="button" className="ru-document-remove" onClick={() => remove(item.id)}>Remove</button></div>)}</div>}
        </div>
        <button type="button" className="ru-save-draft" disabled={!complete || busy} onClick={() => saveDraft().catch((cause) => setError(cause.message || "Could not save your details."))}>{draftSaved ? "Application details saved" : "Save my details"}</button>
      </section>

      <aside className="ru-side">
        <div className="ru-sticky">
          <div className="ru-eyebrow">STEP 05 · ESTIMATE</div><h2>Your estimate</h2>
          <p className="ru-section-intro">Once your profile and all required documents are provided, see an illustrative borrowing-capacity range and any experimental score the backend can calculate from complete source-backed data.</p>
          <button type="button" className="ru-primary-button ru-estimate-button" disabled={!complete || !documentsComplete || busy} onClick={getEstimate}>{busy ? "Calculating…" : "Predict my amount & score"}</button>
          {complete && !documentsComplete && <small className="ru-validation">Upload all 3 required documents before calculating your estimate.</small>}
          {error && !analysis && <div className="ru-document-state is-error" role="alert">{error}</div>}
          {analysis && <div className="ru-borrower-estimate" aria-live="polite">
            <div className="ru-eyebrow">ILLUSTRATIVE AMOUNT RANGE</div>
            <strong>{amountCeiling != null ? `${money(0)} – ${money(amountCeiling)}` : "Not available"}</strong>
            <small>Based on reported cash flow, a 12-month term, 18% annual rate, 35% repayment share and ₹2,000 safety buffer. This is not an offer.</small>
            <div className="ru-section-divider" />
            <div className="ru-eyebrow">EXPERIMENTAL MODEL SCORE · OUT OF 100</div>
            {score100 != null ? <><strong>{score100} / 100</strong><small>Scaled from the backend Rakshak 300–900 model output. Synthetic-data prototype; not a credit score or decision.</small></> : <><strong>Not available yet</strong><small>No complete source-backed model feature set was supplied, so the backend did not invent a score. Your estimate is still available above.</small></>}
            <p className="ru-footnote">{analysis.regulatory_warning}</p>
          </div>}
          {!analysis && <div className="ru-storage-note"><strong>What you will see</strong><br />An affordability-based amount range. A model score appears only when the backend can calculate it from valid, source-backed inputs.</div>}
        </div>
      </aside>
    </div>
    <footer className="ru-footer"><span>RAKSHAK CREDIT · BORROWER WORKSPACE</span><span>Human lender review required · No automatic decision</span></footer>
  </main>;
}
