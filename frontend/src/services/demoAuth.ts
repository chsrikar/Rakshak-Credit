export type DemoRole = "borrower" | "lender";

export interface DemoUser {
  username: string;
  role: DemoRole;
}

const SESSION_KEY = "rakshak_demo_user";
const API_BASES = ["", "http://127.0.0.1:8000", "http://localhost:8000"];

export function getDemoUser(): DemoUser | null {
  try {
    const value = sessionStorage.getItem(SESSION_KEY);
    return value ? JSON.parse(value) as DemoUser : null;
  } catch {
    return null;
  }
}

function saveDemoUser(user: DemoUser) {
  sessionStorage.setItem(SESSION_KEY, JSON.stringify(user));
}

export async function signInDemo(username: string, password: string): Promise<DemoUser> {
  let lastError: unknown;
  for (const base of API_BASES) {
    try {
      const response = await fetch(`${base}/api/v1/auth/login`, {
        method: "POST",
        credentials: "include",
        headers: { "Content-Type": "application/json", Accept: "application/json" },
        body: JSON.stringify({ username, password }),
      });
      const body = await response.json().catch(() => ({}));
      if (!response.ok) throw new Error(body.detail || `Sign-in failed (HTTP ${response.status})`);
      const user = body.user as DemoUser;
      saveDemoUser(user);
      return user;
    } catch (error) {
      lastError = error;
      if (!(error instanceof TypeError) && !(error instanceof Error && error.message.includes("Failed to fetch"))) throw error;
    }
  }
  throw lastError instanceof Error ? lastError : new Error("Could not connect to Rakshak sign-in.");
}

export async function verifyDemoSession(): Promise<DemoUser | null> {
  for (const base of API_BASES) {
    try {
      const response = await fetch(`${base}/api/v1/auth/me`, { credentials: "include", headers: { Accept: "application/json" } });
      if (response.status === 401) {
        sessionStorage.removeItem(SESSION_KEY);
        return null;
      }
      if (!response.ok) continue;
      const user = await response.json() as DemoUser;
      saveDemoUser(user);
      return user;
    } catch {
      // Try the loopback fallback when the development proxy is unavailable.
    }
  }
  return null;
}

export async function signOutDemo(): Promise<boolean> {
  let completed = false;
  try {
    for (const base of API_BASES) {
      try {
        const response = await fetch(`${base}/api/v1/auth/logout`, { method: "POST", credentials: "include" });
        if (response.ok) { completed = true; break; }
        if (response.status !== 404 && response.status < 500) break;
      } catch {
        // Try the loopback API if the current frontend origin has no proxy.
      }
    }
  } finally {
    sessionStorage.removeItem(SESSION_KEY);
  }
  return completed;
}

export function clearDemoUser() {
  sessionStorage.removeItem(SESSION_KEY);
}
