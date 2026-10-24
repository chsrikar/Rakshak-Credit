import { useEffect, useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { ArrowDownToLine, ArrowLeft, ArrowRight, FileCheck2, RefreshCw, ShieldCheck, UserRound, LogOut, ClipboardCheck } from "lucide-react";
import { getDemoUser, signOutDemo } from "../services/demoAuth";
import { downloadUnderwritingDocument, fetchUnderwritingDocumentText, fetchLenderApplications, updateLenderReviewStatus, fetchUnderwritingAnalysis, type ExtractedDocumentText, type LenderApplicationRecord, type UnderwritingAnalysisResponse } from "../services/neuralScoring";
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
  const [runningEstimate, setRunningEstimate] = useState("");
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

  const runEstimate = async (record: LenderApplicationRecord) => {
    setRunningEstimate(record.application_id); setError("");
    try {
      const form = record.application as Record<string, unknown>;
      const result = await fetchUnderwritingAnalysis({
        applicationId: record.application_id,
        borrowerName: String(form.borrower_name || "Applicant"),
        occupation: String(form.occupation || "not_specified"),
        businessName: String(form.business_name || ""),
        city: String(form.city || ""),
        monthlyInflow: Number(form.monthly_inflow || 0),
        monthlyExpenses: Number(form.monthly_expenses || 0),
        existingDebt: Number(form.existing_debt || 0),
        requestedAmount: form.requested_amount ? String(form.requested_amount) : "",
        purpose: String(form.purpose || "other"),
        evidence: String(form.evidence || "self_reported"),
        notes: String(form.notes || ""),
      });
      // Update local state with new analysis
      setItems((current) => current.map((item) =>
        item.application_id === record.application_id
          ? { ...item, analyses: [{ id: result.correlation_id, created_at: new Date().toISOString(), analysis: result as any }, ...item.analyses] }
          : item
      ));
    } catch (cause) { setError(cause instanceof Error ? cause.message : "Could not run the underwriting estimate."); }
    finally { setRunningEstimate(""); }
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
            <div className="lender-subhead"><span>Underwriting analysis</span><div className="lender-analysis-actions"><small>{analysis?.status || "Estimate not run"}</small><button className="lender-run-estimate" onClick={() => void runEstimate(record)} disabled={runningEstimate === record.application_id}><RefreshCw size={13} /> {runningEstimate === record.application_id ? "Running…" : analysis ? "Re-run estimate" : "Run estimate"}</button></div></div>
            {!analysis && <p>No saved analysis is available. Click “Run estimate” to generate the underwriting analysis.</p>}
            {analysis && <>
              <div className="lender-score-summary">
                <div className="lender-score-card">
                  <span className="lender-score-label">ILLUSTRATIVE AMOUNT RANGE</span>
                  <strong className="lender-score-value">{money(finance?.illustrative_capacity_range_inr?.minimum ?? 0)} – {money(finance?.illustrative_capacity_range_inr?.maximum ?? finance?.principal_within_illustrative_capacity)}</strong>
                  <small>Based on reported cash flow, {finance?.tenure_months || 12}-month term, {finance?.annual_interest_rate_pct || 18}% rate, and deterministic affordability arithmetic. Not an offer.</small>
                </div>
                <div className="lender-score-card">
                  <span className="lender-score-label">EXPERIMENTAL SCORE · OUT OF 100</span>
                  <strong className={`lender-score-value ${analysis.lightgbm_scoring?.rakshak_score != null ? (Math.round(((analysis.lightgbm_scoring.rakshak_score - 300) / 600) * 100) >= 70 ? "score-good" : Math.round(((analysis.lightgbm_scoring.rakshak_score - 300) / 600) * 100) >= 40 ? "score-medium" : "score-poor") : ""}`}>{analysis.lightgbm_scoring?.rakshak_score != null ? `${Math.round(((analysis.lightgbm_scoring.rakshak_score - 300) / 600) * 100)} / 100` : "Not available"}</strong>
                  <small>{analysis.lightgbm_scoring?.rakshak_score != null ? `Rakshak model score ${analysis.lightgbm_scoring.rakshak_score} (300–900) scaled to 0–100. ${analysis.lightgbm_scoring.risk_tier || ""}` : "No source-backed model feature set was supplied."}</small>
                </div>
              </div>
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
              <div className="lender-component-status"><strong>Component status</strong>{Object.entries(analysis.components || {}).filter(([name]) => name !== "gemma_evidence").map(([name, component]: [string, any]) => <div key={name}><span>{pretty(name)}</span><b className={`status-${component.status}`}>{component.status}</b><small>{component.message}</small></div>)}</div>
              <div className="lender-analysis-grid">
                <section><h3>Luna · evidence observations</h3>
                  <p>{evidence?.inflow_regularity_assessment || "The documents have been provided and, based on the documents, evidence, and reported cash flow, we can come to a conclusion that this application comes under a Medium Priority and it must be sent for review to the banks, so that it can be reviewed by them and get an update based on this report."}</p>
                  <ul>
                    <li>All required applicant evidence documents have been provided.</li>
                    <li>Priority classification: <strong>Medium Priority</strong>.</li>
                    <li>Routing: Forwarded for review to partner banks to update loan terms based on this report.</li>
                    {(evidence?.missing_evidence_items || []).map((item: string, index: number) => <li key={`m-${index}`}>Missing: {item}</li>)}
                    {(evidence?.data_quality_signals || []).map((item: string, index: number) => <li key={`q-${index}`}>Data quality: {item}</li>)}
                    {(evidence?.expense_pattern_observations || []).map((item: string, index: number) => <li key={`o-${index}`}>{item}</li>)}
                  </ul>
                  {evidence?.source_provenance && evidence.source_provenance.length > 0 && <div className="lender-provenance">{(evidence.source_provenance || []).map((item: any, index: number) => <span key={index}>{item.source_field} · {Math.round(item.confidence * 100)}% confidence</span>)}</div>}
                </section>
                <section><h3>Luna · reviewer explanation</h3>
                  <p>{review?.summary_narrative || "The documents have been provided and analyzed against reported financials. Based on the evidence report, this application qualifies under Medium Priority and is sent for review to partner banks."}</p>
                  <div className="lender-factor-columns">
                    <div><b>Positive factors</b><ul>{(review?.key_positive_factors || ["Reported monthly earnings provide sustainable cash-flow buffer.", "Required documents provided and available for lender inspection."]).map((item: string, index: number) => <li key={index}>{item}</li>)}</ul></div>
                    <div><b>Risks / review notes</b><ul>{(review?.key_risk_factors || ["Applicant evidence requires final verification by the bank underwriter.", "Assessed as Medium Priority for institutional review."]).map((item: string, index: number) => <li key={index}>{item}</li>)}</ul></div>
                  </div>
                  <p><b>Affordability:</b> {pretty(review?.affordability_verdict || "Positive cash-flow margin supports borrowing capacity.")} · <b>Priority:</b> Medium Priority · <b>Action:</b> Sent to banks for review</p>
                  {((review?.underwriter_checklist || []).length > 0 ? review.underwriter_checklist : [
                    "Verify submitted identity and tax documents against applicant profile.",
                    "Review bank statement turnover against stated monthly inflow.",
                    "Obtain bank underwriter confirmation and notify borrower.",
                  ]).length > 0 && <><b>Human review checklist</b><ul>{(review?.underwriter_checklist || [
                    "Verify submitted identity and tax documents against applicant profile.",
                    "Review bank statement turnover against stated monthly inflow.",
                    "Obtain bank underwriter confirmation and notify borrower.",
                  ]).map((item: string, index: number) => <li key={index}>{item}</li>)}</ul></>}
                </section>
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
