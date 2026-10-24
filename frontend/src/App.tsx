import React, { useEffect, useState } from "react";
import { BrowserRouter, Routes, Route, Navigate } from "react-router-dom";
import HomePage from "./components/HomePage";
import RakshakFourProducts from "./components/RakshakFourProducts";
import BorrowerUnderwriting from "./components/BorrowerUnderwriting";
import DemoSignIn from "./components/DemoSignIn";
import LenderWorkspace from "./components/LenderWorkspace";
import { verifyDemoSession, type DemoRole, type DemoUser } from "./services/demoAuth";

function RequireDemoRole({ role, children }: { role: DemoRole; children: React.ReactNode }) {
  const [user, setUser] = useState<DemoUser | null>(null);
  const [checking, setChecking] = useState(true);

  useEffect(() => {
    let active = true;
    verifyDemoSession().then((verified) => {
      if (active) { setUser(verified); setChecking(false); }
    });
    return () => { active = false; };
  }, []);

  if (checking) return <div className="auth-page"><div className="auth-card">Checking your demo session…</div></div>;
  if (!user) return <Navigate to="/signin" replace />;
  if (user.role !== role) return <Navigate to={user.role === "lender" ? "/lender" : "/underwriting"} replace />;
  return <>{children}</>;
}

export default function App() {
  return <BrowserRouter>
    <Routes>
      <Route path="/" element={<HomePage />} />
      <Route path="/signin" element={<DemoSignIn />} />
      <Route path="/workspace" element={<RakshakFourProducts />} />
      <Route path="/products" element={<RakshakFourProducts />} />
      <Route path="/underwriting" element={<RequireDemoRole role="borrower"><BorrowerUnderwriting /></RequireDemoRole>} />
      <Route path="/lender" element={<RequireDemoRole role="lender"><LenderWorkspace /></RequireDemoRole>} />
      <Route path="*" element={<Navigate to="/" replace />} />
    </Routes>
  </BrowserRouter>;
}
