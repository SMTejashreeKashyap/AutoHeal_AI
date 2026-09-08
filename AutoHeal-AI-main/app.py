"""FastAPI Web Application and REST API for AutoHeal-AI.

Serves the modern HTML5/JS dashboard and exposes endpoints for chaos injection,
real-time diagnostics, autonomous self-healing, and audit reporting.
"""

import os
from typing import Dict, Any, List, Optional
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, PlainTextResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from autoheal.config import AutoHealConfig
from autoheal.datasets.generator import LoanDatasetGenerator
from autoheal.pipeline.orchestrator import SelfHealingOrchestrator
from autoheal.core.reporter import SelfHealingReporter

# Initialize FastAPI app
app = FastAPI(
    title="AutoHeal-AI: Teaching AI to Fix Itself",
    description="Autonomous MLOps framework that spots data errors, algorithmic biases, and covariate/concept drifts, and repairs them automatically.",
    version="1.0.0"
)

# Global State Container for the Demo/Project
class SystemState:
    def __init__(self):
        self.generator = LoanDatasetGenerator(random_state=42)
        self.orchestrator = SelfHealingOrchestrator(AutoHealConfig())
        self.clean_df = self.generator.generate_clean_dataset(n_samples=1000)
        # Bootstrap baseline model
        self.orchestrator.bootstrap_baseline(self.clean_df, target_col="target")
        # Current working dataset (starts clean, can be corrupted via chaos)
        self.current_df = self.clean_df.copy()
        self.chaos_metadata: Dict[str, Any] = {"status": "clean"}
        self.latest_report = None

state = SystemState()

# Mount static files directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")
TEMPLATES_DIR = os.path.join(BASE_DIR, "templates")

os.makedirs(STATIC_DIR, exist_ok=True)
os.makedirs(TEMPLATES_DIR, exist_ok=True)
os.makedirs(os.path.join(STATIC_DIR, "css"), exist_ok=True)
os.makedirs(os.path.join(STATIC_DIR, "js"), exist_ok=True)

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


# Pydantic Request Models
class ChaosInjectionRequest(BaseModel):
    missing_rate: float = 0.15
    outlier_rate: float = 0.05
    label_noise_rate: float = 0.10
    bias_penalty: float = 0.45
    drift_magnitude: float = 0.35


class ApplicantPredictionRequest(BaseModel):
    income: float = 65000.0
    credit_score: float = 710.0
    debt_to_income: float = 0.28
    loan_amount: float = 25000.0
    employment_length: float = 6.0
    savings_balance: float = 15000.0
    gender: str = "Female"
    age_group: str = "Middle-Aged"


# Routes
@app.get("/", response_class=HTMLResponse)
def serve_index():
    """Serve the single-page frontend application."""
    html_path = os.path.join(TEMPLATES_DIR, "index.html")
    if os.path.exists(html_path):
        with open(html_path, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    return HTMLResponse(content="<h1>AutoHeal-AI is running. index.html loading...</h1>")


@app.get("/api/status")
def get_system_status():
    """Retrieve active champion version, current health metrics, and chaos state."""
    champ = state.orchestrator.registry.get_champion()
    models = state.orchestrator.registry.list_models()
    return {
        "status": "healthy" if state.chaos_metadata.get("status") == "clean" else "degraded",
        "champion": {
            "version": champ.version if champ else "None",
            "model_type": champ.model_type if champ else "None",
            "metrics": champ.metrics if champ else {},
            "thresholds": champ.group_thresholds if champ else {},
            "notes": champ.remediation_notes if champ else "",
        },
        "total_models": len(models),
        "model_history": models,
        "chaos_state": state.chaos_metadata,
        "dataset_rows": len(state.current_df),
        "has_healing_report": state.latest_report is not None,
    }


@app.post("/api/chaos/inject")
def inject_chaos(req: ChaosInjectionRequest):
    """Inject synthetic data corruption, algorithmic bias, and feature drift."""
    df = state.clean_df.copy()

    # 1. Inject data errors
    df, err_meta = state.generator.inject_data_errors(
        df,
        missing_rate=req.missing_rate,
        outlier_rate=req.outlier_rate,
        label_noise_rate=req.label_noise_rate
    )
    # 2. Inject demographic bias against unprivileged subgroup
    df, bias_meta = state.generator.inject_bias(
        df,
        sensitive_col="gender",
        unprivileged_val="Female",
        disparity_penalty=req.bias_penalty
    )
    # 3. Inject macroeconomic covariate drift
    df, drift_meta = state.generator.inject_drift(
        df,
        drift_magnitude=req.drift_magnitude,
        drift_type="covariate"
    )

    state.current_df = df
    state.chaos_metadata = {
        "status": "corrupted",
        "errors": err_meta,
        "bias": bias_meta,
        "drift": drift_meta,
    }
    return {
        "message": "Chaos successfully injected into production stream!",
        "chaos_metadata": state.chaos_metadata,
        "current_sample_count": len(state.current_df)
    }


@app.post("/api/chaos/reset")
def reset_to_clean():
    """Reset data stream back to pristine baseline."""
    state.current_df = state.clean_df.copy()
    state.chaos_metadata = {"status": "clean"}
    return {"message": "Data stream restored to clean baseline state."}


@app.get("/api/diagnose")
def run_diagnostics():
    """Execute diagnostic scanner on current dataset to spot errors, bias, and drift."""
    findings = state.orchestrator.diagnose(state.current_df, target_col="target")
    findings_by_type = {}
    for f in findings:
        t = f.issue_type.value
        findings_by_type.setdefault(t, []).append(f.to_dict())

    # Summary scores
    data_error_count = sum(1 for f in findings if f.issue_type.value in ["MISSING_VALUES", "SCHEMA_VIOLATION", "OUTLIERS", "LABEL_NOISE"])
    bias_count = sum(1 for f in findings if f.issue_type.value in ["DISPARATE_IMPACT", "DEMOGRAPHIC_PARITY", "EQUALIZED_ODDS"])
    drift_count = sum(1 for f in findings if f.issue_type.value in ["COVARIATE_DRIFT", "CONCEPT_DRIFT"])

    return {
        "total_findings": len(findings),
        "data_error_count": data_error_count,
        "bias_count": bias_count,
        "drift_count": drift_count,
        "findings": [f.to_dict() for f in findings],
        "findings_by_type": findings_by_type,
    }


@app.post("/api/heal")
def run_self_healing():
    """Trigger autonomous remediation loop: cleans, debiases, retrains, canary tests, and promotes."""
    report = state.orchestrator.heal(state.current_df, target_col="target")
    state.latest_report = report

    # If promoted, refresh current data with clean holdout
    if report.status == "HEALED_AND_PROMOTED":
        state.chaos_metadata["status"] = "healed"

    return {
        "message": f"Autonomous healing completed with status: {report.status}",
        "report": report.to_dict()
    }


@app.get("/api/report/latest")
def get_latest_report():
    """Retrieve full details of the latest self-healing report."""
    if state.latest_report is None:
        raise HTTPException(status_code=404, detail="No self-healing cycle has been executed yet.")
    return state.latest_report.to_dict()


@app.get("/api/report/markdown", response_class=PlainTextResponse)
def download_markdown_report():
    """Export the latest self-healing incident audit report as Markdown."""
    if state.latest_report is None:
        return PlainTextResponse("# AutoHeal-AI Report\n\nNo self-healing report generated yet.")
    md_content = SelfHealingReporter.generate_markdown(state.latest_report)
    return PlainTextResponse(content=md_content)


@app.post("/api/predict")
def predict_applicant(applicant: ApplicantPredictionRequest):
    """Run real-time inference on a prospective borrower application."""
    champ = state.orchestrator.registry.get_champion()
    if champ is None:
        raise HTTPException(status_code=500, detail="No active champion model registered.")

    import pandas as pd
    input_df = pd.DataFrame([{
        "income": applicant.income,
        "credit_score": applicant.credit_score,
        "debt_to_income": applicant.debt_to_income,
        "loan_amount": applicant.loan_amount,
        "employment_length": applicant.employment_length,
        "savings_balance": applicant.savings_balance,
        "gender": applicant.gender,
        "age_group": applicant.age_group,
    }])

    prob = float(champ.predict_proba(input_df)[0, 1])
    pred = int(champ.predict_with_fair_thresholds(input_df, sensitive_df=input_df)[0])

    applied_threshold = 0.50
    if applicant.gender in champ.group_thresholds.get("gender", {}):
        applied_threshold = champ.group_thresholds["gender"][applicant.gender]

    return {
        "approval_prediction": bool(pred == 1),
        "verdict": "APPROVED" if pred == 1 else "REJECTED",
        "approval_probability": round(prob, 4),
        "applied_fair_threshold": applied_threshold,
        "champion_model_version": champ.version,
    }


def main():
    """CLI entrypoint to run the web application."""
    print("==================================================================")
    print("  AutoHeal-AI: Teaching AI to Fix Itself")
    print("  Frontend UI & FastAPI Server running at: http://127.0.0.1:8000")
    print("==================================================================")
    uvicorn.run(app, host="127.0.0.1", port=8000)


if __name__ == "__main__":
    main()
