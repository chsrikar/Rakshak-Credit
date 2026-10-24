import { useEffect, useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { ArrowLeft, ArrowRight, Landmark, ShieldCheck, UserRound } from "lucide-react";
import { getDemoUser, signInDemo, verifyDemoSession, type DemoRole } from "../services/demoAuth";
import "../styles/demo-auth.css";

export default function DemoSignIn() {
  const navigate = useNavigate();
  const [role, setRole] = useState<DemoRole>("borrower");
  const [username, setUsername] = useState("borrower.demo");
  const [password, setPassword] = useState("");
  const [error, setError] = useState("");
  const [submitting, setSubmitting] = useState(false);

  useEffect(() => {
    let active = true;
    verifyDemoSession().then((user) => {
      if (active && user) navigate(user.role === "lender" ? "/lender" : "/underwriting", { replace: true });
    });
    return () => { active = false; };
  }, [navigate]);

  const chooseRole = (next: DemoRole) => {
    setRole(next);
    const existing = getDemoUser();
    setUsername(existing?.role === next ? existing.username : next === "borrower" ? "borrower.demo" : "lender.demo");
    setPassword("");
    setError("");
  };

  const submit = async (event: React.FormEvent) => {
    event.preventDefault();
    setSubmitting(true);
    setError("");
    try {
      const user = await signInDemo(username.trim(), password);
      if (user.role !== role) throw new Error(`This account is a ${user.role} account. Select the matching sign-in role.`);
      navigate(user.role === "lender" ? "/lender" : "/underwriting", { replace: true });
    } catch (submitError) {
      setError(submitError instanceof Error ? submitError.message : "Sign-in failed.");
    } finally {
      setSubmitting(false);
    }
  };

  return <main className="auth-page">
    <header className="auth-header">
      <Link className="auth-brand" to="/"><span><ShieldCheck size={16} /></span>rakshak<span className="auth-brand-muted">credit</span></Link>
      <Link className="auth-back" to="/"><ArrowLeft size={13} /> Overview</Link>
    </header>
    <section className="auth-card">
      <div className="lp-kicker">LOCAL MVP DEMO · ROLE-BASED ACCESS</div>
      <h1>Sign in to your workspace.</h1>
      <p className="auth-intro">Use the borrower workspace to submit an application and evidence. Lenders get a separate review queue.</p>
      <div className="auth-role-tabs" role="tablist" aria-label="Choose demo role">
        <button type="button" className={role === "borrower" ? "is-active" : ""} onClick={() => chooseRole("borrower")} role="tab" aria-selected={role === "borrower"}><UserRound size={15} /> Borrower</button>
        <button type="button" className={role === "lender" ? "is-active" : ""} onClick={() => chooseRole("lender")} role="tab" aria-selected={role === "lender"}><Landmark size={15} /> Lender</button>
      </div>
      <form onSubmit={submit} className="auth-form">
        <label>Username<input autoComplete="username" required value={username} onChange={(event) => setUsername(event.target.value)} /></label>
        <label>Password<input type="password" autoComplete="current-password" required value={password} onChange={(event) => setPassword(event.target.value)} /></label>
        {error && <div className="auth-error" role="alert">{error}</div>}
        <button className="auth-submit" type="submit" disabled={submitting}>{submitting ? "Signing in…" : `Continue as ${role}`} <ArrowRight size={14} /></button>
      </form>
      <div className="auth-demo-note"><strong>Demo account</strong><span>Accounts are provisioned locally by the backend. This is not public signup.</span></div>
      <p className="auth-privacy">Session credentials are held in an HttpOnly cookie. Uploaded sample files are synthetic and visibly marked as demo-only.</p>
    </section>
    <footer className="auth-footer">RAKSHAK CREDIT · SYNTHETIC MVP DEMONSTRATION</footer>
  </main>;
}
