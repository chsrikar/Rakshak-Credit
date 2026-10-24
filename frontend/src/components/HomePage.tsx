import { useEffect, useMemo, useState } from "react";
import { Link } from "react-router-dom";
import {
  Activity, ArrowRight, ArrowUpRight, Check, CircleHelp, FileCheck2, Landmark,
  LockKeyhole, Menu, Network, ShieldCheck, Sparkles, Users, X,
} from "lucide-react";
import { fetchDemoApplicants, fetchUnderwritingAnalysis, type DemoApplicant, type UnderwritingAnalysisResponse } from "../services/neuralScoring";
import "../styles/landing-page.css";

const money = (value?: number | null) => value == null || !Number.isFinite(value)
  ? "—"
  : new Intl.NumberFormat("en-IN", { style: "currency", currency: "INR", maximumFractionDigits: 0 }).format(value);

const statusText = (analysis: UnderwritingAnalysisResponse | null, key: string) =>
  analysis?.components?.[key]?.status ?? "Not run";

function applicantToForm(applicant: DemoApplicant) {
  const occupation = /gig/i.test(applicant.occupation) ? "delivery" : "kirana";
  return {
    borrowerName: applicant.name,
    occupation,
    businessName: `${applicant.occupation} demo business`,
    city: applicant.city,
    district: applicant.city,
    state: "India",
    cityTier: "unknown",
    businessAge: "2-5",
    purpose: "working",
    purposeDetails: "Synthetic demo scenario. Loan purpose evidence was not supplied.",
    requestedAmount: Math.max(1000, Math.round(applicant.monthly_income * 1.5)),
    monthlyInflow: applicant.monthly_income,
    monthlyExpenses: applicant.monthly_expenses,
    existingDebt: applicant.monthly_debt,
    tenure: 12,
    annualRate: 18,
    evidence: "self_reported",
    notes: "Synthetic dataset profile selected for a product demonstration. No bank documents are attached.",
  };
}

function Brand() {
  return <Link to="/" className="lp-brand" aria-label="Rakshak Credit home">
    <span className="lp-brand-mark"><ShieldCheck size={16} strokeWidth={1.8} /></span>
    <span>rakshak<span className="lp-brand-light">credit</span></span>
  </Link>;
}

function Metric({ label, value, note }: { label: string; value: string; note: string }) {
  return <div className="lp-metric"><span>{label}</span><strong>{value}</strong><small>{note}</small></div>;
}

export default function HomePage() {
  const [applicants, setApplicants] = useState<DemoApplicant[]>([]);
  const [datasetState, setDatasetState] = useState<"loading" | "ready" | "error">("loading");
  const [selectedId, setSelectedId] = useState("");
  const [analysis, setAnalysis] = useState<UnderwritingAnalysisResponse | null>(null);
  const [analysisError, setAnalysisError] = useState("");
  const [running, setRunning] = useState(false);
  const [menuOpen, setMenuOpen] = useState(false);

  useEffect(() => {
    let active = true;
    fetchDemoApplicants().then((result) => {
      if (!active) return;
      setApplicants(result.items);
      setSelectedId(result.items[0]?.id ?? "");
      setDatasetState("ready");
    }).catch(() => {
      if (active) setDatasetState("error");
    });
    return () => { active = false; };
  }, []);

  const selected = useMemo(() => applicants.find((item) => item.id === selectedId) ?? null, [applicants, selectedId]);
  const runReview = async () => {
    if (!selected || running) return;
    setRunning(true);
    setAnalysisError("");
    try {
      setAnalysis(await fetchUnderwritingAnalysis(applicantToForm(selected)));
    } catch (error) {
      setAnalysisError(error instanceof Error ? error.message : "The underwriting API could not be reached.");
    } finally {
      setRunning(false);
    }
  };

  return <div className="landing-page">
    <header className="lp-header">
      <div className="lp-header-inner">
        <Brand />
        <button className="lp-menu-toggle" onClick={() => setMenuOpen((open) => !open)} aria-label="Toggle navigation">{menuOpen ? <X size={18} /> : <Menu size={18} />}</button>
        <nav className={menuOpen ? "lp-nav is-open" : "lp-nav"}>
          <a href="#platform" onClick={() => setMenuOpen(false)}>Platform</a>
          <a href="#workflow" onClick={() => setMenuOpen(false)}>Workflow</a>
          <a href="#control-room" onClick={() => setMenuOpen(false)}>Control room</a>
          <a href="#evidence" onClick={() => setMenuOpen(false)}>Evidence</a>
        </nav>
        <Link className="lp-nav-cta" to="/signin">Borrower / lender sign in <ArrowUpRight size={12} /></Link>
      </div>
    </header>

    <main>
      <section className="lp-hero lp-wrap">
        <div className="lp-hero-copy">
          <div className="lp-kicker"><span className="lp-kicker-dot" /> Cash-flow credit · Human-led review</div>
          <h1>The lending workflow<br />you can <em>build on.</em></h1>
          <p>One clear view of an application, its reported evidence, deterministic affordability and independent model signals.</p>
          <p className="lp-hero-sub">Luna supports evidence review and explanation. People remain responsible for decisions.</p>
          <div className="lp-actions">
            <a className="lp-button lp-button-primary" href="#control-room">Explore the demo <ArrowRight size={14} /></a>
            <Link className="lp-button lp-button-ghost" to="/underwriting">Open review desk</Link>
          </div>
          <div className="lp-hero-meta"><span>DEMO DATA ONLY</span><span>·</span><span>NO AUTOMATIC APPROVALS</span></div>
        </div>

        <div className="lp-preview-card">
          <div className="lp-card-top"><span>APPLICATION REVIEW</span><span className="lp-live-tag"><i /> {datasetState === "ready" ? "DATA CONNECTED" : datasetState === "loading" ? "CONNECTING" : "API OFFLINE"}</span></div>
          <div className="lp-preview-title">A review built around the evidence.</div>
          <div className="lp-workflow-list">
            <div className="lp-workflow-row"><span className="lp-step-number">01</span><div><strong>Application data</strong><small>{selected ? `${selected.name} · synthetic profile` : "Waiting for demo data"}</small></div><span className="lp-state">{selected ? "Loaded" : "Pending"}</span></div>
            <div className="lp-workflow-row"><span className="lp-step-number">02</span><div><strong>Financial calculations</strong><small>EMI · surplus · illustrative capacity</small></div><span className="lp-state">Deterministic</span></div>
            <div className="lp-workflow-row"><span className="lp-step-number">03</span><div><strong>Luna evidence review</strong><small>Extracts from supplied fields</small></div><span className="lp-state">{analysis ? statusText(analysis, "gemma_evidence") : "Ready"}</span></div>
            <div className="lp-workflow-row"><span className="lp-step-number">04</span><div><strong>Human review</strong><small>Required for every application</small></div><span className="lp-state lp-state-muted">Required</span></div>
          </div>
          <button className="lp-run-button" onClick={runReview} disabled={!selected || running}>
            {running ? <><span className="lp-spinner" /> Running Luna review…</> : <>Run sample review <ArrowRight size={13} /></>}
          </button>
          {analysisError && <div className="lp-inline-error" role="alert">{analysisError}</div>}
          <div className="lp-preview-foot"><LockKeyhole size={11} /> Synthetic example · no source documents attached</div>
        </div>
      </section>

      <section className="lp-trust-strip"><div className="lp-wrap lp-trust-inner"><span>BUILT FOR A CLEARER REVIEW</span><span>Reported cash flow</span><b>·</b><span>Transparent calculations</span><b>·</b><span>Independent model signals</span><b>·</b><span>Human decision</span></div></section>

      <section className="lp-section lp-wrap lp-infrastructure" id="platform">
        <div className="lp-section-copy">
          <div className="lp-kicker">A connected lending workflow</div>
          <h2>Lending infrastructure.<br /><em>Built to connect.</em></h2>
          <p>Bring the application, evidence and review steps into one place. Keep the source and status of each output visible.</p>
          <div className="lp-link-list">
            <a href="#control-room"><span><FileCheck2 size={14} /> Application intake</span><ArrowUpRight size={13} /></a>
            <a href="#evidence"><span><Activity size={14} /> Deterministic affordability</span><ArrowUpRight size={13} /></a>
            <a href="#control-room"><span><Sparkles size={14} /> Luna evidence + explanation</span><ArrowUpRight size={13} /></a>
            <Link to="/workspace"><span><Network size={14} /> Independent score services</span><ArrowUpRight size={13} /></Link>
          </div>
        </div>
        <div className="lp-stack-card">
          <div className="lp-stack-label">REVIEW PIPELINE</div>
          <div className="lp-stack-node"><span className="lp-node-icon"><Users size={15} /></span><div><strong>Application</strong><small>Applicant-entered context</small></div><span className="lp-node-state">01</span></div>
          <div className="lp-stack-connector" />
          <div className="lp-stack-node"><span className="lp-node-icon"><Activity size={15} /></span><div><strong>Financial layer</strong><small>Backend formula · exact values</small></div><span className="lp-node-state">02</span></div>
          <div className="lp-stack-connector" />
          <div className="lp-stack-node lp-stack-node-accent"><span className="lp-node-icon"><Sparkles size={15} /></span><div><strong>Luna AI</strong><small>Evidence extraction + reviewer support</small></div><span className="lp-node-state">API</span></div>
          <div className="lp-stack-connector" />
          <div className="lp-stack-node"><span className="lp-node-icon"><ShieldCheck size={15} /></span><div><strong>Human decision</strong><small>No automated approval or rejection</small></div><span className="lp-node-state">04</span></div>
          <div className="lp-stack-note"><LockKeyhole size={12} /> API keys stay in the backend environment.</div>
        </div>
      </section>

      <section className="lp-experiences" id="workflow"><div className="lp-wrap">
        <div className="lp-kicker">One platform · two experiences</div>
        <h2>Designed for the person applying<br className="lp-desktop-break" /> and the person reviewing.</h2>
        <div className="lp-experience-grid">
          <article className="lp-experience-card"><div className="lp-experience-icon"><Users size={16} /></div><div className="lp-kicker">BORROWER EXPERIENCE</div><h3>A clearer way to apply.</h3><p>Share business context, explain the request and see what evidence is still needed.</p><Link to="/underwriting">Explore the application flow <ArrowRight size={13} /></Link></article>
          <article className="lp-experience-card"><div className="lp-experience-icon"><Landmark size={16} /></div><div className="lp-kicker">LENDER EXPERIENCE</div><h3>Evidence at a glance.</h3><p>Review reported information, calculations, model statuses and Luna’s explanation together.</p><a href="#control-room">Open the control room <ArrowRight size={13} /></a></article>
        </div>
      </div></section>

      <section className="lp-section lp-wrap lp-control" id="control-room">
        <div className="lp-control-heading"><div><div className="lp-kicker">A live product walkthrough</div><h2>The lender’s control room.</h2><p>Connected to the local Rakshak API · all listed profiles are synthetic.</p></div><div className="lp-control-actions"><span className="lp-demo-label"><i /> DEMO WORKSPACE</span><Link to="/underwriting" className="lp-small-button">Review a case <ArrowUpRight size={12} /></Link></div></div>
        <div className="lp-dashboard">
          <div className="lp-dashboard-top"><div><span className="lp-kicker">SYNTHETIC DATASET</span><h3>Portfolio overview</h3></div><span className={`lp-api-state ${datasetState}`}><i />{datasetState === "ready" ? "API connected" : datasetState === "loading" ? "Connecting" : "API unavailable"}</span></div>
          <div className="lp-metrics-grid">
            <Metric label="Demo profiles" value={datasetState === "ready" ? String(applicants.length) : "—"} note="Read from FastAPI" />
            <Metric label="Selected profile" value={selected?.id ?? "—"} note={selected?.occupation ?? "Choose a profile below"} />
            <Metric label="Luna evidence extraction" value={analysis ? statusText(analysis, "gemma_evidence") : "Not run"} note="Evidence + provenance" />
            <Metric label="Decision authority" value="Human" note="No automatic loan decision" />
          </div>

          <div className="lp-dashboard-grid">
            <div className="lp-subpanel">
              <div className="lp-subpanel-head"><div><span className="lp-kicker">SELECTED DEMO APPLICATION</span><h3>{selected ? selected.name : "Synthetic profile"}</h3></div><select aria-label="Choose a synthetic applicant" value={selectedId} onChange={(event) => { setSelectedId(event.target.value); setAnalysis(null); }} disabled={!applicants.length}><option value="">Select a profile</option>{applicants.map((applicant) => <option key={applicant.id} value={applicant.id}>{applicant.name} · {applicant.id}</option>)}</select></div>
              {selected ? <>
                <div className="lp-profile-line"><span>{selected.occupation}</span><span>{selected.city}</span><span>{selected.id}</span></div>
                <div className="lp-data-grid">
                  <Metric label="Monthly inflow" value={money(selected.monthly_income)} note="Synthetic dataset field" />
                  <Metric label="Monthly expenses" value={money(selected.monthly_expenses)} note="Synthetic dataset field" />
                  <Metric label="Existing debt" value={money(selected.monthly_debt)} note="Synthetic dataset field" />
                  <Metric label="Data completeness" value={`${Math.round(selected.data_completeness * 100)}%`} note="Dataset indicator · not verification" />
                </div>
                <div className="lp-assumption-note"><CircleHelp size={13} /> Sample run assumes request = 1.5× monthly inflow, 12 months and 18% annual interest. These are demo inputs, not applicant-provided terms.</div>
                <button className="lp-outline-button" onClick={runReview} disabled={running}>{running ? "Running analysis…" : analysis ? "Run again" : "Run Luna + financial review"}<ArrowRight size={13} /></button>
              </> : <div className="lp-empty">{datasetState === "error" ? "Start the FastAPI backend to load the synthetic demo profiles." : "Loading synthetic profiles from the API…"}</div>}
            </div>

            <aside className="lp-subpanel lp-status-panel">
              <span className="lp-kicker">REVIEW STATUS</span><h3>{analysis ? `Application ${analysis.application_id}` : "My review workspace"}</h3>
              <div className="lp-review-steps">
                <ReviewStep title="Application data" detail={selected ? "Synthetic profile loaded" : "Waiting for profile"} state={selected ? "Loaded" : "Pending"} ok={!!selected} />
                <ReviewStep title="Evidence documents" detail="No document files attached" state="Missing" ok={false} />
                <ReviewStep title="Financial calculations" detail={analysis ? "Backend calculation completed" : "Runs with review"} state={analysis ? statusText(analysis, "financial_calculations") : "Ready"} ok={analysis?.components?.financial_calculations?.status === "success"} />
                <ReviewStep title="Luna evidence extraction" detail={analysis?.gemma_evidence?.model_name ?? "Evidence + provenance"} state={analysis ? statusText(analysis, "gemma_evidence") : "Not run"} ok={analysis?.components?.gemma_evidence?.status === "success"} />
                <ReviewStep title="Luna reviewer support" detail={analysis?.luna_decision_support?.model_name ?? "Explanation + missing evidence"} state={analysis ? statusText(analysis, "luna_decision_support") : "Not run"} ok={analysis?.components?.luna_decision_support?.status === "success"} />
                <ReviewStep title="Human decision" detail="Required · no model can decide" state="Pending" ok={false} />
              </div>
              {analysis && <div className="lp-review-summary"><strong>{analysis.status === "completed" ? "All requested components completed" : "Partial analysis returned"}</strong><span>{analysis.luna_decision_support?.decision_support?.summary_narrative ?? "Check the component statuses and messages above."}</span></div>}
              {analysisError && <div className="lp-inline-error" role="alert">{analysisError}</div>}
            </aside>
          </div>

          {analysis && <div className="lp-analysis-results">
            <div className="lp-result-card"><span className="lp-kicker">BACKEND EMI</span><strong>{money(analysis.financial_calculations.estimated_monthly_emi)}</strong><small>Deterministic calculation · INR</small></div>
            <div className="lp-result-card"><span className="lp-kicker">SURPLUS AFTER EMI</span><strong>{money(analysis.financial_calculations.estimated_monthly_surplus_after_new_loan)}</strong><small>Uses the demo assumptions noted above</small></div>
            <div className="lp-result-card"><span className="lp-kicker">MODEL SCORES</span><strong>{analysis.neural_scoring || analysis.lightgbm_scoring ? "Available" : "Not supplied"}</strong><small>No feature vectors were invented for this demo</small></div>
            {analysis.gemma_evidence?.extraction?.missing_evidence_items?.length ? <div className="lp-result-card lp-result-wide"><span className="lp-kicker">LUNA · EVIDENCE GAPS</span><ul>{analysis.gemma_evidence.extraction.missing_evidence_items.map((item, index) => <li key={index}>{item}</li>)}</ul></div> : null}
            <div className="lp-analysis-warning"><ShieldCheck size={15} /><span>{analysis.regulatory_warning}</span></div>
          </div>}

          <div className="lp-table-wrap">
            <div className="lp-table-heading"><div><span className="lp-kicker">DATA CONNECTED FROM FASTAPI</span><h3>Synthetic applicant records</h3></div><span>{applicants.length} profiles · no credit decisions</span></div>
            <div className="lp-table-scroll"><table className="lp-table"><thead><tr><th>Applicant</th><th>Business type</th><th>Location</th><th>Monthly inflow</th><th>Data status</th><th /></tr></thead><tbody>
              {applicants.slice(0, 5).map((applicant) => <tr key={applicant.id}>
                <td><strong>{applicant.name}</strong><small>{applicant.id}</small></td><td>{applicant.occupation}</td><td>{applicant.city}</td><td>{money(applicant.monthly_income)}</td><td><span className="lp-record-tag">Synthetic · unverified</span></td><td><button className="lp-select-row" onClick={() => { setSelectedId(applicant.id); setAnalysis(null); }}>Select <ArrowUpRight size={11} /></button></td>
              </tr>)}
              {!applicants.length && <tr><td colSpan={6} className="lp-table-empty">{datasetState === "error" ? "API unavailable · start the backend to load demo records." : "Loading demo records…"}</td></tr>}
            </tbody></table></div>
          </div>
        </div>
      </section>

      <section className="lp-proof" id="evidence"><div className="lp-wrap lp-proof-grid">
        <div><div className="lp-kicker">BUILT FOR A BETTER REVIEW</div><h2>Prove the value.<br /><em>Then scale the trust.</em></h2><p>Keep sources, calculations, model limits and missing information visible before a person makes a decision.</p><Link to="/underwriting" className="lp-button lp-button-primary">Explore the full review desk <ArrowRight size={13} /></Link></div>
        <div className="lp-proof-list">
          <div><Check size={13} /><span><strong>Transparent financial calculations</strong><small>Backend values remain canonical.</small></span></div>
          <div><Check size={13} /><span><strong>Luna extraction + explanation</strong><small>Evidence-grounded roles with explicit status.</small></span></div>
          <div><Check size={13} /><span><strong>Independent model outputs</strong><small>Shown only when complete sourced features are supplied.</small></span></div>
          <div><Check size={13} /><span><strong>Human review stays in control</strong><small>No automatic approval or rejection.</small></span></div>
        </div>
      </div></section>
    </main>

    <footer className="lp-footer"><div className="lp-wrap lp-footer-inner"><Brand /><p>Cash-flow credit intelligence · Synthetic demo · Human-reviewed decisions</p><div><Link to="/workspace">Product workspace</Link><Link to="/underwriting">Underwriting desk</Link></div><small>© Rakshak Credit · MVP demonstration</small></div></footer>
  </div>;
}

function ReviewStep({ title, detail, state, ok }: { title: string; detail: string; state: string; ok: boolean }) {
  return <div className="lp-review-step"><span className={ok ? "lp-review-check done" : "lp-review-check"}>{ok ? <Check size={11} /> : <span />}</span><div><strong>{title}</strong><small>{detail}</small></div><em className={ok ? "is-done" : ""}>{state}</em></div>;
}
