import { useEffect, useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { ArrowDownToLine, ArrowLeft, ArrowRight, FileCheck2, RefreshCw, ShieldCheck, UserRound, LogOut, ClipboardCheck } from "lucide-react";
import { getDemoUser, signOutDemo } from "../services/demoAuth";
import { downloadUnderwritingDocument, fetchUnderwritingDocumentText, fetchLenderApplications, updateLenderReviewStatus, type ExtractedDocumentText, type LenderApplicationRecord } from "../services/neuralScoring";
import "../styles/lender-workspace.css";

const labels: Record<string, string> = {
  identity_pan: "PAN / identity", bank_statement: "Bank statement", tax_return: "Income tax return",
  gst_record: "GST record", business_registration: "Business registration", income_invoice: "Income evidence",
  platform_statement: "Platform settlement", other: "Other evidence",
};
const labelFor = (value: string) => labels[value] || value.replace(/_/g, " ");
const money = (value: unknown) => typeof value === "number"
  ? new Intl.NumberFormat("en-IN", { style: "currency", currency: "INR", maximumFractionDigits: 0 }).format(value)
  : "Not supplied";
const pretty = (value: unknown) => typeof value === "string" ? value.replace(/_/g, " ") : "Not available";

export default function LenderWorkspace() {
  const navigate = useNavigate();
  const [items, setItems] = useState<LenderApplicationRecord[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [downloading, setDownloading] = useState("");
  const [updatingReview, setUpdatingReview] = useState("");
  const [readingDocument, setReadingDocument] = useState("");
  const [documentText, setDocumentText] = useState<Record<string, ExtractedDocumentText>>({});
  const user = getDemoUser();

  const refresh = async () => {
    setLoading(true); setError("");
    try { setItems(await fetchLenderApplications()); }
    catch (cause) { setError(cause instanceof Error ? cause.message : "Could not load lender applications."); }
    finally { setLoading(false); }
  };
  useEffect(() => { void refresh(); }, []);
  const logout = async () => {
    const completed = await signOutDemo();
    if (!completed) { setError("Could not clear the backend session. Check that the API is running, then try signing out again."); return; }
    navigate("/signin", { replace: true });
  };

  const download = async (record: LenderApplicationRecord, file: LenderApplicationRecord["documents"][number]) => {
    setDownloading(file.id); setError("");
    try { await downloadUnderwritingDocument(record.application_id, file.id, file.stored_filename); }
    catch (cause) { setError(cause instanceof Error ? cause.message : "Could not download the document."); }
    finally { setDownloading(""); }
  };
  const setReview = async (record: LenderApplicationRecord) => {
    const status = record.review_status === "pending" ? "in_review" : "reviewed";
    setUpdatingReview(record.application_id); setError("");
    try {
      const updated = await updateLenderReviewStatus(record.application_id, status);
      setItems((current) => current.map((item) => item.application_id === record.application_id
        ? { ...item, review_status: updated.review_status, reviewed_by: user?.username || null, reviewed_at: status === "reviewed" ? new Date().toISOString() : null }
        : item));
    } catch (cause) { setError(cause instanceof Error ? cause.message : "Could not update human review status."); }
    finally { setUpdatingReview(""); }
  };
  const readDocument = async (record: LenderApplicationRecord, file: LenderApplicationRecord["documents"][number]) => {
    setReadingDocument(file.id); setError("");
    try {
      const extracted = await fetchUnderwritingDocumentText(record.application_id, file.id);
      setDocumentText((current) => ({ ...current, [file.id]: extracted }));
    } catch (cause) { setError(cause instanceof Error ? cause.message : "Could not read document text."); }
    finally { setReadingDocument(""); }
  };

  return <main className="lender-page">
    <header className="lender-header">
      <Link to="/" className="lender-brand"><span><ShieldCheck size={15} /></span>rakshak<span>credit</span></Link>
      <div className="lender-header-right"><span className="lender-role"><FileCheck2 size={13} /> LENDER REVIEW</span><span>{user?.username || "Lender"}</span><button onClick={logout}><LogOut size={13} /> Sign out</button></div>
    </header>
    <section className="lender-intro">
      <div><Link to="/signin" className="lender-back"><ArrowLeft size={12} /> Role sign-in</Link><div className="lp-kicker">LENDER WORKSPACE · ROLE-RESTRICTED</div><h1>Application review.</h1><p>Review submitted details, evidence, financial calculations and model analysis. Scores are advisory; decisions stay with a human.</p></div>
      <button className="lender-refresh" onClick={() => void refresh()} disabled={loading}><RefreshCw size={13} /> Refresh queue</button>
    </section>
    <section className="lender-summary">
      <article><span>APPLICATIONS</span><strong>{loading ? "—" : items.length}</strong><small>Saved submissions</small></article>
      <article><span>DOCUMENTS</span><strong>{loading ? "—" : items.reduce((count, item) => count + item.documents.length, 0)}</strong><small>Received · not automatically verified</small></article>
      <article><span>HUMAN REVIEW</span><strong>{loading ? "—" : items.filter((item) => item.review_status !== "reviewed").length}</strong><small>Awaiting or in progress</small></article>
    </section>
    {error && <div className="lender-error" role="alert">{error}</div>}
    {loading && <div className="lender-empty">Loading the encrypted application queue…</div>}
    {!loading && !error && items.length === 0 && <div className="lender-empty"><UserRound size={18} /><strong>No borrower drafts yet</strong><span>Submit a borrower demo application to see its details and evidence here.</span><Link to="/underwriting">Open borrower workspace <ArrowRight size={13} /></Link></div>}

    <section className="lender-queue" aria-label="Applications">
      {items.map((record) => {
        const form = record.application as Record<string, unknown>;
        const analysis = record.analyses[0]?.analysis as any;
        const finance = analysis?.financial_calculations;
        const evidence = analysis?.gemma_evidence?.extraction;
        const review = analysis?.luna_decision_support?.decision_support;
        const status = record.review_status || "pending";
        return <article className="lender-application" key={record.application_id}>
          <div className="lender-application-head"><div><div className="lp-kicker">APPLICATION · {record.application_id}</div><h2>{String(form.borrower_name || "Applicant")}</h2><p>{pretty(form.occupation)} · {String(form.city || "Location not stated")}{form.business_name ? ` · ${String(form.business_name)}` : " · No business listed"}</p></div><span className={`lender-pending review-${status}`}>{status === "reviewed" ? "Human review complete" : status === "in_review" ? "Human review in progress" : "Human review required"}</span></div>
          <div className="lender-details-grid">
            <div><span>Monthly income / inflow</span><strong>{money(form.monthly_inflow)}</strong></div>
            <div><span>Monthly expenses</span><strong>{money(form.monthly_expenses)}</strong></div>
            <div><span>Existing debt</span><strong>{money(form.existing_debt)}</strong></div>
            <div><span>Business / occupation</span><strong>{String(form.business_name || pretty(form.occupation))}</strong></div>
            <div><span>Evidence files</span><strong>{record.documents.length} attached</strong></div>
            <div><span>Saved analyses</span><strong>{record.analyses.length}</strong></div>
          </div>
          <div className="lender-documents">
            <div className="lender-subhead"><span><FileCheck2 size={14} /> Evidence files</span><small>{record.documents.length} attached</small></div>
            {record.documents.length === 0 ? <p className="lender-no-docs">No files uploaded. Keep evidence gaps visible during review.</p> : record.documents.map((file) => <div className="lender-document-wrap" key={file.id}><div className="lender-document"><div><strong>{labelFor(file.category)}</strong><small>{file.content_type} · {(file.size_bytes / (1024 * 1024)).toFixed(2)} MB · received {new Date(file.created_at).toLocaleDateString()}</small></div><div className="lender-document-actions"><button onClick={() => void readDocument(record, file)} disabled={readingDocument === file.id}><FileCheck2 size={13} /> {readingDocument === file.id ? "Reading…" : documentText[file.id] ? "Refresh text" : "Read text"}</button><button onClick={() => void download(record, file)} disabled={downloading === file.id}><ArrowDownToLine size={13} /> {downloading === file.id ? "Preparing…" : "Download"}</button></div></div>{documentText[file.id] && <div className={`lender-extracted-text ${documentText[file.id].status}`}><strong>{documentText[file.id].status === "success" ? "Extracted text · unverified" : documentText[file.id].status.replace(/_/g, " ")}</strong><p>{documentText[file.id].text || documentText[file.id].message}</p><small>{documentText[file.id].warning}</small></div>}</div>)}
            <p className="lender-doc-warning">Files are encrypted at rest. Readable PDF text is sent to configured Luna during analysis after common identifiers are masked. Text extraction is not verification; inspect the original. Scanned PDFs and images need OCR, which is not configured in this demo.</p>
          </div>

          <div className="lender-analysis lender-analysis-full">
            <div className="lender-subhead"><span>Underwriting analysis</span><small>{analysis?.status || "Estimate not run"}</small></div>
            {!analysis && <p>No saved analysis is available. The borrower can run the estimate after completing their profile.</p>}
            {analysis && <>
              <p className="lender-analysis-warning">Experimental, synthetic-data support only. Not an approval or rejection. Independently verify borrower statements and documents.</p>
              <div className="lender-analysis-grid">
                <section><h3>Backend financial calculations</h3><div className="lender-stat-grid">
                  <div><span>Capacity range ceiling</span><strong>{money(finance?.illustrative_capacity_range_inr?.maximum ?? finance?.principal_within_illustrative_capacity)}</strong></div>
                  <div><span>Monthly surplus</span><strong>{money(finance?.estimated_monthly_surplus_before_new_loan)}</strong></div>
                  <div><span>Estimated EMI</span><strong>{money(finance?.estimated_monthly_emi)}</strong></div>
                  <div><span>Debt service coverage</span><strong>{typeof finance?.debt_service_coverage_ratio === "number" ? `${finance.debt_service_coverage_ratio.toFixed(2)}×` : "Not available"}</strong></div>
                </div></section>
                <section><h3>Independent model outputs</h3><div className="lender-stat-grid">
                  <div><span>Neural risk score</span><strong>{typeof analysis.neural_scoring?.risk_score === "number" ? `${(analysis.neural_scoring.risk_score * 100).toFixed(1)}%` : "Not available"}<small>{pretty(analysis.components?.neural_scoring?.status)}</small></strong></div>
                  <div><span>Rakshak model score</span><strong>{analysis.lightgbm_scoring?.rakshak_score ?? "Not available"}<small>{analysis.lightgbm_scoring?.risk_tier || pretty(analysis.components?.lightgbm_scoring?.status)}</small></strong></div>
                </div></section>
              </div>
              <div className="lender-component-status"><strong>Component status</strong>{Object.entries(analysis.components || {}).map(([name, component]: [string, any]) => <div key={name}><span>{pretty(name)}</span><b className={`status-${component.status}`}>{component.status}</b><small>{component.message}</small></div>)}</div>
              <div className="lender-analysis-grid">
                <section><h3>Luna · evidence observations</h3>{evidence ? <>
                  <p>{evidence.inflow_regularity_assessment || "No assessment returned."}</p>
                  <ul>{(evidence.missing_evidence_items || []).map((item: string, index: number) => <li key={`m-${index}`}>Missing: {item}</li>)}{(evidence.data_quality_signals || []).map((item: string, index: number) => <li key={`q-${index}`}>Data quality: {item}</li>)}{(evidence.expense_pattern_observations || []).map((item: string, index: number) => <li key={`o-${index}`}>{item}</li>)}</ul>
                  <div className="lender-provenance">{(evidence.source_provenance || []).map((item: any, index: number) => <span key={index}>{item.source_field} · {Math.round(item.confidence * 100)}% confidence</span>)}</div>
                </> : <p>Evidence extraction unavailable or not configured. Review the component status above.</p>}</section>
                <section><h3>Luna · reviewer explanation</h3>{review ? <>
                  <p>{review.summary_narrative}</p>
                  <div className="lender-factor-columns"><div><b>Positive factors</b><ul>{(review.key_positive_factors || []).map((item: string, index: number) => <li key={index}>{item}</li>)}</ul></div><div><b>Risks / evidence gaps</b><ul>{(review.key_risk_factors || []).map((item: string, index: number) => <li key={index}>{item}</li>)}{(review.missing_evidence_items || []).map((item: string, index: number) => <li key={`missing-${index}`}>{item}</li>)}</ul></div></div>
                  <p><b>Affordability:</b> {pretty(review.affordability_verdict)} · <b>Uncertainty:</b> {typeof review.uncertainty_score === "number" ? `${Math.round(review.uncertainty_score * 100)}% (uncalibrated)` : "Not reported"}</p>
                  {(review.underwriter_checklist || []).length > 0 && <><b>Human review checklist</b><ul>{review.underwriter_checklist.map((item: string, index: number) => <li key={index}>{item}</li>)}</ul></>}
                </> : <p>No Luna explanation returned. Do not treat missing output as a clear review.</p>}</section>
              </div>
              <details className="lender-exact-output"><summary>Exact backend scores and warnings</summary><pre>{JSON.stringify({ neural_scoring: analysis.neural_scoring, lightgbm_scoring: analysis.lightgbm_scoring, regulatory_warning: analysis.regulatory_warning }, null, 2)}</pre></details>
            </>}
          </div>
          <div className="lender-review-actions"><div><span className="lp-kicker">HUMAN REVIEW TRACKING</span><small>{record.reviewed_by ? `Last updated by ${record.reviewed_by}${record.reviewed_at ? ` · ${new Date(record.reviewed_at).toLocaleString()}` : ""}` : "No lender review activity recorded."}</small></div><button onClick={() => void setReview(record)} disabled={updatingReview === record.application_id || status === "reviewed"}><ClipboardCheck size={14} />{updatingReview === record.application_id ? "Saving…" : status === "pending" ? "Mark for human review" : status === "in_review" ? "Mark review complete" : "Review complete"}</button></div>
        </article>;
      })}
    </section>
    <footer className="lender-footer">RAKSHAK CREDIT · SYNTHETIC MVP · HUMAN DECISION REQUIRED</footer>
  </main>;
}
