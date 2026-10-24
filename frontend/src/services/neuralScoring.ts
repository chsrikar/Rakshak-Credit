/**
 * Service for Rakshak Credit Neural Scoring (POST /api/v1/neural-score)
 * Calls the Deep Tabular ResNet endpoint with all 77 required features.
 */

export interface NeuralScoreResponse {
  risk_score: number;
  threshold: number;
  review_required: boolean;
  model_name: string;
  model_type: string;
  synthetic_only: boolean;
  decision: "REVIEW_REQUIRED" | "REVIEW_NOT_TRIGGERED";
  warning: string;
}

export interface UnderwritingFormData {
  applicationId?: string;
  borrowerName?: string;
  occupation: string;
  businessName?: string;
  city?: string;
  district?: string;
  state?: string;
  cityTier?: string;
  businessAge?: string;
  purpose?: string;
  purposeDetails?: string;
  requestedAmount?: string | number;
  monthlyInflow: string | number;
  monthlyExpenses: string | number;
  existingDebt: string | number;
  tenure?: string | number;
  annualRate?: string | number;
  evidence?: string;
  notes?: string;
}

/**
 * Executes inference against POST /api/v1/neural-score
 */
export async function fetchNeuralScore(features: Record<string, any>): Promise<NeuralScoreResponse> {
  const endpoints = [
    "/api/v1/neural-score",
    "http://127.0.0.1:8000/api/v1/neural-score",
    "http://localhost:8000/api/v1/neural-score",
  ];

  let lastError: any = null;

  for (const url of endpoints) {
    try {
      const response = await fetch(url, {
        method: "POST",
        credentials: "include",
        headers: {
          "Content-Type": "application/json",
          "Accept": "application/json",
        },
        body: JSON.stringify({ features }),
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.detail || `Server responded with HTTP ${response.status}`);
      }

      const data: NeuralScoreResponse = await response.json();
      return data;
    } catch (err: any) {
      lastError = err;
      // If it's a network error (like failed to fetch relative /api on standalone dev), try next endpoint
      if (err.message && err.message.includes("Failed to fetch")) {
        continue;
      }
      // If server returned 422 or 503, throw immediately
      throw err;
    }
  }

  throw lastError || new Error("Unable to connect to Rakshak Credit neural scoring service.");
}

export interface UnderwritingAnalysisResponse {
  application_id: string;
  correlation_id: string;
  status: "completed" | "partial";
  components: Record<string, { status: string; latency_ms: number; message: string }>;
  financial_calculations: {
    requested_amount_inr: number | null;
    estimated_monthly_emi: number | null;
    estimated_monthly_surplus_before_new_loan: number;
    estimated_monthly_surplus_after_new_loan: number | null;
    estimated_surplus_after_emi_and_buffer_inr: number | null;
    estimated_available_amount_inr: number;
    illustrative_monthly_payment_inr: number;
    illustrative_emi_capacity: number;
    principal_within_illustrative_capacity: number;
    illustrative_capacity_range_inr?: { minimum: number; maximum: number };
    within_illustrative_rule: boolean | null;
    debt_service_coverage_ratio: number;
    annual_interest_rate_pct: number;
    tenure_months: number;
  };
  neural_scoring?: {
    risk_score: number;
    threshold: number;
    review_required: boolean;
    decision: string;
    model_name: string;
    warning: string;
  };
  lightgbm_scoring?: {
    rakshak_score: number;
    risk_probability: number;
    risk_tier: string;
    threshold_used: number;
  };
  gemma_evidence?: {
    status: string;
    model_name: string;
    extraction?: {
      business_type: string;
      verified_monthly_inflow?: number | null;
      inflow_regularity_assessment: string;
      expense_pattern_observations: string[];
      operating_volatility_flag: boolean;
      seasonal_demand_notes: string;
      data_quality_signals: string[];
      missing_evidence_items: string[];
      source_provenance: Array<{ observation: string; source_field: string; confidence: number }>;
      confidence_score: number;
    };
    message?: string;
  };
  luna_decision_support?: {
    status: string;
    model_name: string;
    decision_support?: {
      summary_narrative: string;
      key_positive_factors: string[];
      key_risk_factors: string[];
      affordability_verdict: string;
      uncertainty_score: number;
      recommended_action: string;
      human_review_required: true;
      limitations: string[];
      missing_evidence_items: string[];
      underwriter_checklist: string[];
      exact_scores_preserved: Record<string, any>;
    };
    message?: string;
  };
  synthetic_only: boolean;
  regulatory_warning: string;
}

export interface DemoApplicant {
  id: string;
  name: string;
  occupation: string;
  city: string;
  monthly_income: number;
  monthly_expenses: number;
  monthly_debt: number;
  closing_balance: number;
  transaction_regularity: number;
  data_completeness: number;
  demo_only?: number | boolean;
}

export interface DemoApplicantList {
  count: number;
  items: DemoApplicant[];
  synthetic_only: true;
}

export interface UploadedUnderwritingDocument {
  id: string;
  application_id: string;
  category: string;
  stored_filename: string;
  content_type: string;
  size_bytes: number;
  created_at: string;
}

export interface ExtractedDocumentText {
  application_id: string;
  document_id: string;
  category: string;
  status: "success" | "ocr_unavailable" | "unsupported" | "error";
  text: string;
  message: string;
  warning: string;
}

export interface LenderApplicationRecord {
  application_id: string;
  owner: string | null;
  created_at: string;
  updated_at: string;
  application: Record<string, unknown>;
  documents: UploadedUnderwritingDocument[];
  analyses: Array<{ id: string; created_at: string; analysis: UnderwritingAnalysisResponse }>;
  review_status: "pending" | "in_review" | "reviewed";
  reviewed_by?: string | null;
  reviewed_at?: string | null;
}

export async function fetchLenderApplications(): Promise<LenderApplicationRecord[]> {
  let lastError: unknown;
  for (const base of API_BASES) {
    try {
      const response = await fetch(`${base}/api/v1/lender/applications`, { credentials: "include", headers: { Accept: "application/json" } });
      const body = await response.json().catch(() => ({}));
      if (!response.ok) throw new Error(body.detail || `Could not load applications (HTTP ${response.status})`);
      return body.items as LenderApplicationRecord[];
    } catch (error) {
      lastError = error;
      if (!(error instanceof TypeError) && !(error instanceof Error && error.message.includes("Failed to fetch"))) throw error;
    }
  }
  throw lastError instanceof Error ? lastError : new Error("Could not connect to the lender workspace.");
}

export async function updateLenderReviewStatus(applicationId: string, status: "in_review" | "reviewed") {
  const path = `/api/v1/lender/applications/${encodeURIComponent(applicationId)}/review`;
  let lastError: unknown;
  for (const base of API_BASES) {
    try {
      const response = await fetch(`${base}${path}`, {
        method: "PUT", credentials: "include",
        headers: { "Content-Type": "application/json", Accept: "application/json" },
        body: JSON.stringify({ status }),
      });
      const body = await response.json().catch(() => ({}));
      if (!response.ok) throw new Error(body.detail || `Could not update review status (HTTP ${response.status})`);
      return body as { application_id: string; review_status: "in_review" | "reviewed" };
    } catch (error) {
      lastError = error;
      if (!(error instanceof TypeError) && !(error instanceof Error && error.message.includes("Failed to fetch"))) throw error;
    }
  }
  throw lastError instanceof Error ? lastError : new Error("Could not update human review status.");
}

export async function downloadUnderwritingDocument(applicationId: string, documentId: string, filename: string) {
  const path = `/api/v1/underwriting/applications/${encodeURIComponent(applicationId)}/documents/${encodeURIComponent(documentId)}/download`;
  let lastError: unknown;
  for (const base of API_BASES) {
    try {
      const response = await fetch(`${base}${path}`, { credentials: "include" });
      if (!response.ok) {
        const body = await response.json().catch(() => ({}));
        throw new Error(body.detail || `Document download failed (HTTP ${response.status})`);
      }
      const blobUrl = URL.createObjectURL(await response.blob());
      const link = document.createElement("a");
      link.href = blobUrl;
      link.download = filename;
      link.click();
      URL.revokeObjectURL(blobUrl);
      return;
    } catch (error) {
      lastError = error;
      if (!(error instanceof TypeError) && !(error instanceof Error && error.message.includes("Failed to fetch"))) throw error;
    }
  }
  throw lastError instanceof Error ? lastError : new Error("Could not connect to document storage.");
}

export async function fetchUnderwritingDocumentText(applicationId: string, documentId: string): Promise<ExtractedDocumentText> {
  const path = `/api/v1/underwriting/applications/${encodeURIComponent(applicationId)}/documents/${encodeURIComponent(documentId)}/text`;
  let lastError: unknown;
  for (const base of API_BASES) {
    try {
      const response = await fetch(`${base}${path}`, { credentials: "include", headers: { Accept: "application/json" } });
      const body = await response.json().catch(() => ({}));
      if (!response.ok) throw new Error(body.detail || `Could not extract document text (HTTP ${response.status})`);
      return body as ExtractedDocumentText;
    } catch (error) {
      lastError = error;
      if (!(error instanceof TypeError) && !(error instanceof Error && error.message.includes("Failed to fetch"))) throw error;
    }
  }
  throw lastError instanceof Error ? lastError : new Error("Could not connect to document text extraction.");
}

function underwritingPayload(form: UnderwritingFormData) {
  return {
    ...(form.applicationId ? { application_id: form.applicationId } : {}),
    borrower_name: form.borrowerName || "Applicant",
    occupation: form.occupation || "not_specified",
    business_name: form.businessName || "",
    city: form.city || "",
    district: form.district || "",
    state: form.state || "",
    city_tier: form.cityTier || "unknown",
    business_age: form.businessAge || "unknown",
    purpose: form.purpose || "other",
    purpose_details: form.purposeDetails || "",
    requested_amount: form.requestedAmount ? Number(form.requestedAmount) : null,
    monthly_inflow: Number(form.monthlyInflow),
    monthly_expenses: Number(form.monthlyExpenses),
    existing_debt: Number(form.existingDebt),
    tenure_months: form.tenure ? Number(form.tenure) : 12,
    annual_interest_rate_pct: form.annualRate ? Number(form.annualRate) : 18,
    evidence: form.evidence || "self_reported",
    notes: form.notes || "",
    include_neural: true,
    include_lightgbm: true,
    include_gemma: true,
    include_luna: true,
  };
}

const API_BASES = ["", "http://127.0.0.1:8000", "http://localhost:8000"];

export async function saveUnderwritingApplication(form: UnderwritingFormData, applicationId?: string) {
  const path = applicationId
    ? `/api/v1/underwriting/applications/${encodeURIComponent(applicationId)}`
    : "/api/v1/underwriting/applications";
  let lastError: unknown;
  for (const base of API_BASES) {
    try {
      const response = await fetch(`${base}${path}`, {
        method: applicationId ? "PUT" : "POST",
        credentials: "include",
        headers: { "Content-Type": "application/json", Accept: "application/json" },
        body: JSON.stringify(underwritingPayload(form)),
      });
      const body = await response.json().catch(() => ({}));
      if (!response.ok) throw new Error(body.detail || `Could not save application (HTTP ${response.status})`);
      return body as { application_id: string; status: string; updated_at: string };
    } catch (error) {
      lastError = error;
      if (!(error instanceof TypeError) && !(error instanceof Error && error.message.includes("Failed to fetch"))) throw error;
    }
  }
  throw lastError instanceof Error ? lastError : new Error("Could not connect to the underwriting database.");
}

export async function fetchUnderwritingDocuments(applicationId: string): Promise<UploadedUnderwritingDocument[]> {
  const path = `/api/v1/underwriting/applications/${encodeURIComponent(applicationId)}/documents`;
  let lastError: unknown;
  for (const base of API_BASES) {
    try {
      const response = await fetch(`${base}${path}`, { credentials: "include", headers: { Accept: "application/json" } });
      const body = await response.json().catch(() => ({}));
      if (!response.ok) throw new Error(body.detail || `Could not load documents (HTTP ${response.status})`);
      return body.items as UploadedUnderwritingDocument[];
    } catch (error) {
      lastError = error;
      if (!(error instanceof TypeError) && !(error instanceof Error && error.message.includes("Failed to fetch"))) throw error;
    }
  }
  throw lastError instanceof Error ? lastError : new Error("Could not connect to document storage.");
}

export async function uploadUnderwritingDocument(applicationId: string, category: string, file: File) {
  const path = `/api/v1/underwriting/applications/${encodeURIComponent(applicationId)}/documents`;
  const body = new FormData();
  body.append("category", category);
  body.append("file", file);
  let lastError: unknown;
  for (const base of API_BASES) {
    try {
      const response = await fetch(`${base}${path}`, { method: "POST", body, credentials: "include", headers: { Accept: "application/json" } });
      const data = await response.json().catch(() => ({}));
      if (!response.ok) throw new Error(data.detail || `Upload failed (HTTP ${response.status})`);
      return data as UploadedUnderwritingDocument;
    } catch (error) {
      lastError = error;
      if (!(error instanceof TypeError) && !(error instanceof Error && error.message.includes("Failed to fetch"))) throw error;
    }
  }
  throw lastError instanceof Error ? lastError : new Error("Could not connect to encrypted document storage.");
}

export async function removeUnderwritingDocument(applicationId: string, documentId: string) {
  const path = `/api/v1/underwriting/applications/${encodeURIComponent(applicationId)}/documents/${encodeURIComponent(documentId)}`;
  let lastError: unknown;
  for (const base of API_BASES) {
    try {
      const response = await fetch(`${base}${path}`, { method: "DELETE", credentials: "include", headers: { Accept: "application/json" } });
      const body = await response.json().catch(() => ({}));
      if (!response.ok) throw new Error(body.detail || `Could not remove document (HTTP ${response.status})`);
      return;
    } catch (error) {
      lastError = error;
      if (!(error instanceof TypeError) && !(error instanceof Error && error.message.includes("Failed to fetch"))) throw error;
    }
  }
  throw lastError instanceof Error ? lastError : new Error("Could not connect to document storage.");
}

export async function fetchDemoApplicants(): Promise<DemoApplicantList> {
  const endpoints = [
    "/api/v1/applicants",
    "http://127.0.0.1:8000/api/v1/applicants",
    "http://localhost:8000/api/v1/applicants",
  ];
  let lastError: unknown;
  for (const url of endpoints) {
    try {
      const response = await fetch(url, { headers: { Accept: "application/json" } });
      if (!response.ok) throw new Error(`Demo dataset returned HTTP ${response.status}`);
      return await response.json() as DemoApplicantList;
    } catch (error) {
      lastError = error;
      if (!(error instanceof TypeError) && !(error instanceof Error && error.message.includes("Failed to fetch"))) throw error;
    }
  }
  throw lastError instanceof Error ? lastError : new Error("Could not connect to the demo dataset.");
}

export async function fetchUnderwritingAnalysis(form: UnderwritingFormData): Promise<UnderwritingAnalysisResponse> {
  const endpoints = [
    "/api/v1/underwriting/analyze",
    "http://127.0.0.1:8000/api/v1/underwriting/analyze",
    "http://localhost:8000/api/v1/underwriting/analyze",
  ];

  const payload = underwritingPayload(form);

  let lastError: any = null;

  for (const url of endpoints) {
    try {
      const response = await fetch(url, {
        method: "POST",
        credentials: "include",
        headers: {
          "Content-Type": "application/json",
          "Accept": "application/json",
        },
        body: JSON.stringify(payload),
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        const detailMsg = typeof errorData.detail === "string"
          ? errorData.detail
          : (errorData.detail?.message || (errorData.detail ? JSON.stringify(errorData.detail) : `Server responded with HTTP ${response.status}`));
        throw new Error(detailMsg);
      }

      const data: UnderwritingAnalysisResponse = await response.json();
      return data;
    } catch (err: any) {
      lastError = err;
      if (err.message && err.message.includes("Failed to fetch")) {
        continue;
      }
      throw err;
    }
  }

  throw lastError || new Error("Unable to connect to Rakshak Credit underwriting analysis service.");
}
