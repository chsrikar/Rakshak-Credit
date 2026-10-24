import React, { useMemo, useState } from "react";
import { Link } from "react-router-dom";
import "../styles/rakshak-four-products.css";

const PRODUCTS = [
  { id: "cashflow", letter: "A", title: "Cash-Flow Intelligence", eyebrow: "PRODUCT A", description: "Understand the financial reality behind every application.", icon: "↗" },
  { id: "affordability", letter: "B", title: "Affordability & Credit Capacity", eyebrow: "PRODUCT B", description: "Model sustainable repayments using transparent calculations.", icon: "₹" },
  { id: "underwriting", letter: "C", title: "AI Underwriting Support", eyebrow: "PRODUCT C", description: "Turn evidence and risk signals into review-ready cases.", icon: "◎" },
  { id: "business", letter: "D", title: "Small-Business Intelligence", eyebrow: "PRODUCT D", description: "Add business context to the financial picture.", icon: "▦" },
];

const money = (n) => new Intl.NumberFormat("en-IN", {
  style: "currency", currency: "INR", maximumFractionDigits: 0
}).format(Number.isFinite(n) ? n : 0);

function Metric({ label, value, note }) {
  return <div className="rc-metric"><span>{label}</span><strong>{value}</strong>{note && <small>{note}</small>}</div>;
}

function CashFlow({ form }) {
  const surplus = Math.max(0, Number(form.inflow || 0) - Number(form.expenses || 0) - Number(form.debt || 0));
  const [period, setPeriod] = useState("monthly");
  return <div className="rc-product-grid">
    <section className="rc-panel rc-panel-wide">
      <div className="rc-panel-heading"><div><span className="rc-kicker">CASH-FLOW PROFILE</span><h3>Monthly financial snapshot</h3></div><span className="rc-chip">Illustrative inputs</span></div>
      <div className="rc-metrics">
        <Metric label="Reported inflow" value={money(Number(form.inflow || 0))} note="Borrower-provided" />
        <Metric label="Reported outflow" value={money(Number(form.expenses || 0))} note="Expenses / operating costs" />
        <Metric label="Existing monthly debt" value={money(Number(form.debt || 0))} note="Declared obligations" />
        <Metric label="Cash remaining" value={money(surplus)} note="Before any new loan" />
      </div>
      <div className="rc-chart-placeholder">
        <div className="rc-chart-label"><div><strong>Cash-flow trend</strong><small>Connect verified statement data to populate this chart.</small></div>
          <select value={period} onChange={e => setPeriod(e.target.value)} aria-label="Cash flow period"><option value="monthly">Monthly</option><option value="weekly">Weekly</option></select></div>
        <div className="rc-chart-bars" aria-label="Illustrative chart placeholder">{[38,55,46,68,52,76,61,82,57,72,64,88].map((h,i)=><span key={i} style={{height:`${h}%`}} />)}</div>
        <p className="rc-muted">Preview only — no bank-statement trend is inferred from these placeholder bars.</p>
      </div>
    </section>
    <section className="rc-panel">
      <span className="rc-kicker">LUNA · ANALYSIS</span><h3>Cash-flow observations</h3>
      <div className="rc-empty-state"><span className="rc-empty-icon">✦</span><strong>Analysis will appear here</strong><p>Luna will summarize supplied inflows, volatility, expense patterns and evidence gaps. Source transaction data must be attached by the underwriting workflow.</p></div>
    </section>
    <section className="rc-panel">
      <span className="rc-kicker">DATA QUALITY</span><h3>Evidence checklist</h3>
      {["Income source verified","Transaction period sufficient","Irregular inflows reviewed","Unusual transactions checked"].map((x,i)=><div className="rc-check-row" key={x}><span className="rc-check">{i===0?"○":"—"}</span><span>{x}</span><small>{i===0?"Pending":"Not assessed"}</small></div>)}
    </section>
  </div>;
}

function Affordability({ form, setForm }) {
  const [rate, setRate] = useState(18);
  const [months, setMonths] = useState(12);
  const [bufferPct, setBufferPct] = useState(25);
  const inflow=Number(form.inflow||0), expenses=Number(form.expenses||0), debt=Number(form.debt||0), principal=Number(form.amount||0);
  const surplus=Math.max(0,inflow-expenses-debt);
  const safeEmi=surplus*(1-bufferPct/100);
  const r=rate/1200;
  const emi=principal>0?(r===0?principal/months:principal*r*Math.pow(1+r,months)/(Math.pow(1+r,months)-1)):0;
  return <div className="rc-product-grid">
    <section className="rc-panel rc-panel-wide">
      <div className="rc-panel-heading"><div><span className="rc-kicker">REPAYMENT SIMULATOR</span><h3>What could repayment look like?</h3></div><span className="rc-chip">Deterministic calculation</span></div>
      <div className="rc-metrics"><Metric label="Monthly surplus before new loan" value={money(surplus)} /><Metric label="Illustrative safe EMI ceiling" value={money(safeEmi)} note={`${bufferPct}% surplus retained as buffer`} /><Metric label="Estimated EMI" value={money(emi)} note={`${months} months · ${rate}% annual interest`} /></div>
      <div className={`rc-status ${emi <= safeEmi ? "rc-status-calm" : "rc-status-warn"}`}><strong>{emi <= safeEmi ? "Within the configured EMI ceiling" : "Above the configured EMI ceiling"}</strong><span>This is a scenario indicator, not a credit approval or lending recommendation.</span></div>
      <div className="rc-controls">
        <label>Annual interest rate <b>{rate}%</b><input type="range" min="0" max="36" step="0.5" value={rate} onChange={e=>setRate(Number(e.target.value))}/></label>
        <label>Tenure <b>{months} months</b><input type="range" min="3" max="24" step="1" value={months} onChange={e=>setMonths(Number(e.target.value))}/></label>
        <label>Surplus safety buffer <b>{bufferPct}%</b><input type="range" min="10" max="50" step="5" value={bufferPct} onChange={e=>setBufferPct(Number(e.target.value))}/></label>
      </div>
    </section>
    <section className="rc-panel">
      <span className="rc-kicker">INPUTS</span><h3>Application details</h3>
      <label className="rc-field">Requested loan amount (₹)<input type="number" min="0" value={form.amount} onChange={e=>setForm({...form,amount:e.target.value})}/></label>
      <label className="rc-field">Monthly inflow (₹)<input type="number" min="0" value={form.inflow} onChange={e=>setForm({...form,inflow:e.target.value})}/></label>
      <label className="rc-field">Monthly expenses (₹)<input type="number" min="0" value={form.expenses} onChange={e=>setForm({...form,expenses:e.target.value})}/></label>
      <label className="rc-field">Existing monthly debt (₹)<input type="number" min="0" value={form.debt} onChange={e=>setForm({...form,debt:e.target.value})}/></label>
    </section>
  </div>;
}

function Underwriting({ form }) {
  return <div className="rc-product-grid">
    <section className="rc-panel rc-panel-wide">
      <div className="rc-panel-heading">
        <div><span className="rc-kicker">CASE REVIEW</span><h3>Underwriting evidence summary</h3></div>
        <span className="rc-chip">Model inputs incomplete</span>
      </div>
      <div className="rc-case-summary">
        <div className="rc-avatar">{(form.businessType||"K").slice(0,1).toUpperCase()}</div>
        <div>
          <strong>{form.businessType || "Business"} applicant</strong>
          <p>{form.city || "City not provided"} · {form.history || "Business history not provided"} · Purpose: {form.purpose || "Not provided"}</p>
        </div>
        <span className="rc-review-pill">Awaiting source-backed model inputs</span>
      </div>
      <div className="rc-status rc-status-calm"><strong>No neural score was generated.</strong><span>This screen does not collect all 77 model features. Add a complete, source-backed payload before using the neural endpoint; no placeholder values are substituted.</span></div>

      <div className="rc-three-columns">
        <div>
          <span className="rc-kicker">NEURAL RISK SCORE</span>
          <strong>Not run</strong>
          <p>Complete 77-feature payload required.</p>
        </div>
        <div>
          <span className="rc-kicker">MODEL ARCHITECTURE</span>
          <strong>Saved PyTorch model</strong>
          <p>Preprocessing is applied by the backend when features are supplied.</p>
        </div>
        <div>
          <span className="rc-kicker">UNDERWRITING ACTION</span>
          <strong>Human review remains required</strong>
          <p><Link to="/underwriting" style={{ color: "#34d399", fontWeight: 600 }}>Open Full Underwriting Desk →</Link></p>
        </div>
      </div>
    </section>
    <section className="rc-panel">
      <span className="rc-kicker">LUNA · DECISION SUPPORT</span><h3>Review pathway</h3>
      {["Proceed to evidence review","Request additional information","Escalate for human assessment"].map((t,i)=><div className="rc-path-row" key={t}><span>{String(i+1).padStart(2,"0")}</span><div><strong>{t}</strong><small>{["Only after required inputs are validated","If key documents or data are missing","For uncertainty, conflicts or policy exceptions"][i]}</small></div></div>)}
      <p className="rc-muted">No automated approval or rejection is produced by this UI.</p>
    </section>
    <section className="rc-panel">
      <span className="rc-kicker">AUDITABILITY</span><h3>Required controls</h3>
      {["Record model and prompt versions","Store evidence references","Log reviewer actions","Monitor fairness and calibration"].map(t=><div className="rc-check-row" key={t}><span className="rc-check">○</span><span>{t}</span></div>)}
    </section>
  </div>;
}


function Business({ form, setForm }) {
  return <div className="rc-product-grid">
    <section className="rc-panel rc-panel-wide">
      <div className="rc-panel-heading"><div><span className="rc-kicker">BUSINESS CONTEXT</span><h3>Understand the operating model</h3></div><span className="rc-chip">Evidence-led</span></div>
      <div className="rc-form-grid">
        <label className="rc-field">Business / occupation<select value={form.businessType} onChange={e=>setForm({...form,businessType:e.target.value})}><option>Kirana shop</option><option>Gig worker</option><option>Street vendor</option><option>Small retailer</option><option>Service business</option><option>Other</option></select></label>
        <label className="rc-field">City<input value={form.city} onChange={e=>setForm({...form,city:e.target.value})} placeholder="e.g. Kochi"/></label>
        <label className="rc-field">City tier<select value={form.cityTier} onChange={e=>setForm({...form,cityTier:e.target.value})}><option>Tier 1</option><option>Tier 2</option><option>Tier 3</option><option>Unclassified</option></select></label>
        <label className="rc-field">Business history<select value={form.history} onChange={e=>setForm({...form,history:e.target.value})}><option>Under 1 year</option><option>1–2 years</option><option>2–5 years</option><option>5+ years</option></select></label>
        <label className="rc-field">Loan purpose<select value={form.purpose} onChange={e=>setForm({...form,purpose:e.target.value})}><option>Inventory</option><option>Working capital</option><option>Equipment</option><option>Business expansion</option><option>Emergency liquidity</option><option>Other</option></select></label>
        <label className="rc-field">Evidence available<select><option>Not yet specified</option><option>Sales records</option><option>Inventory records</option><option>Invoices / supplier bills</option><option>Multiple sources</option></select></label>
      </div>
      <div className="rc-status rc-status-calm"><strong>Context is not a credit score.</strong><span>Occupation and city tier are descriptive context only. Credit assessments must rely on relevant, verified evidence and applicable policy.</span></div>
    </section>
    <section className="rc-panel">
      <span className="rc-kicker">LUNA · BUSINESS ANALYSIS</span><h3>Business intelligence report</h3>
      <div className="rc-empty-state"><span className="rc-empty-icon">▦</span><strong>Ready for business evidence</strong><p>When connected, Luna will summarize supplied operating evidence, seasonal needs and evidence gaps without inventing sales or market facts.</p></div>
    </section>
  </div>;
}

export default function RakshakFourProducts() {
  const [active, setActive] = useState("cashflow");
  const [form, setForm] = useState({ businessType:"Kirana shop", city:"Kochi", cityTier:"Tier 2", history:"2–5 years", purpose:"Inventory", amount:"150000", inflow:"75000", expenses:"42000", debt:"8000" });
  const selected = useMemo(() => PRODUCTS.find(p=>p.id===active) || PRODUCTS[0], [active]);
  return <div className="rakshak-four-products">
    <aside className="rc-sidebar">
      <div className="rc-brand"><div className="rc-brand-mark">R</div><div><strong>RAKSHAK</strong><small>CREDIT INTELLIGENCE</small></div></div>
      <div className="rc-workspace"><span className="rc-status-dot" /> DEMO WORKSPACE <span className="rc-chevron">⌄</span></div>
      <div className="rc-nav-label">PRODUCTS</div>
      <nav className="rc-nav">{PRODUCTS.map(p=><button key={p.id} className={`rc-nav-item ${active===p.id?"active":""}`} onClick={()=>setActive(p.id)}><span className="rc-nav-icon">{p.icon}</span><span className="rc-nav-text"><strong>{p.title}</strong><small>{p.eyebrow}</small></span><span className="rc-nav-letter">{p.letter}</span></button>)}</nav>
      <div className="rc-sidebar-bottom"><div className="rc-ai-status"><span className="rc-status-dot" /><div><strong>AI services</strong><small>Availability reported per analysis</small></div></div><small>Luna · extraction and reviewer support</small></div>
    </aside>
    <main className="rc-main">
      <header className="rc-topbar">
        <div className="rc-breadcrumb">Workspace <span>/</span> Products <span>/</span> <strong>{selected.title}</strong></div>
        <div className="rc-top-actions">
          <Link to="/" style={{ marginRight: '14px', color: '#9ee5c1', textDecoration: 'none', fontWeight: 700, fontSize: '12px', display: 'inline-flex', alignItems: 'center', gap: '4px' }}>
            ← Home
          </Link>
          <Link to="/underwriting" style={{ marginRight: '14px', color: '#c8d8d3', textDecoration: 'none', fontWeight: 600, fontSize: '12px' }}>
            Underwriting Desk ↗
          </Link>
          <span className="rc-demo-pill">DEMO DATA</span>
          <div className="rc-user-avatar">RC</div>
        </div>
      </header>
      <div className="rc-content">
        <div className="rc-page-heading"><div><div className="rc-kicker">{selected.eyebrow} <span>·</span> RAKSHAK CREDIT</div><h1>{selected.title}</h1><p>{selected.description}</p></div><button className="rc-secondary-button" onClick={()=>setActive("cashflow")}>Overview ↗</button></div>
        <section className="rc-application-strip">
          <div><span className="rc-kicker">SAMPLE APPLICATION</span><strong>{form.businessType} <span>·</span> {form.city || "City not provided"}</strong><small>Illustrative profile — not a real credit decision</small></div>
          <div className="rc-strip-metric"><small>Requested amount</small><strong>{money(Number(form.amount||0))}</strong></div>
          <div className="rc-strip-metric"><small>Monthly inflow</small><strong>{money(Number(form.inflow||0))}</strong></div>
          <button className="rc-secondary-button" onClick={()=>setActive("business")}>Edit profile ↗</button>
        </section>
        {active==="cashflow" && <CashFlow form={form}/>}
        {active==="affordability" && <Affordability form={form} setForm={setForm}/>}
        {active==="underwriting" && <Underwriting form={form}/>}
        {active==="business" && <Business form={form} setForm={setForm}/>}
        <footer className="rc-footer"><span>RAKSHAK CREDIT <span>·</span> PRODUCT WORKSPACE</span><span>AI-assisted analysis · Human-reviewed decisions</span></footer>
      </div>
    </main>
  </div>;
}
