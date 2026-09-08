/**
 * AutoHeal-AI: Frontend Controller & Visualization Engine
 * Communicates with FastAPI backend, manages Chart.js graphs, and animates self-healing steps.
 */

// Global Chart Instances
let chartDataErrors = null;
let chartBias = null;
let chartDrift = null;
let chartComparison = null;

// Initialize when DOM is ready
document.addEventListener("DOMContentLoaded", () => {
    setupTabNavigation();
    setupEventListeners();
    refreshSystemStatus();
    runDiagnostics();
});

// Tab Switching
function setupTabNavigation() {
    const navButtons = document.querySelectorAll(".nav-btn");
    navButtons.forEach(btn => {
        btn.addEventListener("click", () => {
            navButtons.forEach(b => b.classList.remove("active"));
            btn.classList.add("active");

            const tabId = btn.getAttribute("data-tab");
            document.querySelectorAll(".tab-content").forEach(tab => {
                tab.classList.remove("active");
            });
            const targetTab = document.getElementById(tabId);
            if (targetTab) {
                targetTab.classList.add("active");
            }
        });
    });
}

// Event Listeners for Buttons and Sliders
function setupEventListeners() {
    // Sliders sync with displayed values
    const sliders = [
        { id: "sliderMissing", valId: "valMissing", suffix: "%", scale: 100 },
        { id: "sliderOutliers", valId: "valOutliers", suffix: "%", scale: 100 },
        { id: "sliderLabelNoise", valId: "valLabelNoise", suffix: "%", scale: 100 },
        { id: "sliderBias", valId: "valBias", suffix: "%", scale: 100 },
        { id: "sliderDrift", valId: "valDrift", suffix: "%", scale: 100 },
    ];

    sliders.forEach(s => {
        const input = document.getElementById(s.id);
        const label = document.getElementById(s.valId);
        if (input && label) {
            input.addEventListener("input", (e) => {
                label.textContent = Math.round(parseFloat(e.target.value) * s.scale) + s.suffix;
            });
        }
    });

    // Chaos Buttons
    document.getElementById("btnInjectChaos")?.addEventListener("click", injectChaos);
    document.getElementById("btnResetChaos")?.addEventListener("click", resetChaos);

    // Diagnostics & Healing
    document.getElementById("btnRunDiagnostics")?.addEventListener("click", runDiagnostics);
    document.getElementById("btnTriggerHeal")?.addEventListener("click", triggerAutonomousHeal);
    document.getElementById("btnTriggerHealBanner")?.addEventListener("click", () => {
        // Switch to heal tab and trigger
        document.querySelector('[data-tab="tab-heal"]')?.click();
        triggerAutonomousHeal();
    });

    // Inference Form
    document.getElementById("formPredict")?.addEventListener("submit", handleInference);

    // Download Report
    document.getElementById("btnDownloadReport")?.addEventListener("click", downloadReport);
}

// 1. Refresh System Status
async function refreshSystemStatus() {
    try {
        const res = await fetch("/api/status");
        const data = await res.json();

        // Update badges
        const badge = document.getElementById("systemStatusBadge");
        if (badge) {
            if (data.status === "healthy") {
                badge.className = "badge badge-healthy";
                badge.textContent = "Status: Healthy";
            } else {
                badge.className = "badge badge-degraded";
                badge.textContent = "Status: Degraded (Drift/Bias Detected)";
            }
        }

        // Update cards
        const champVer = document.getElementById("statChampVersion");
        if (champVer) champVer.textContent = data.champion.version || "None";

        const champF1 = document.getElementById("statChampF1");
        if (champF1) {
            const f1 = data.champion.metrics?.f1;
            champF1.textContent = f1 ? (f1 * 100).toFixed(1) + "%" : "--";
        }

        const champAcc = document.getElementById("statChampAcc");
        if (champAcc) {
            const acc = data.champion.metrics?.accuracy;
            champAcc.textContent = acc ? (acc * 100).toFixed(1) + "%" : "--";
        }

        const modelCount = document.getElementById("statModelCount");
        if (modelCount) modelCount.textContent = data.total_models || 1;

    } catch (err) {
        console.error("Failed to fetch system status:", err);
    }
}

// 2. Inject Chaos
async function injectChaos() {
    const payload = {
        missing_rate: parseFloat(document.getElementById("sliderMissing")?.value || 0.15),
        outlier_rate: parseFloat(document.getElementById("sliderOutliers")?.value || 0.05),
        label_noise_rate: parseFloat(document.getElementById("sliderLabelNoise")?.value || 0.10),
        bias_penalty: parseFloat(document.getElementById("sliderBias")?.value || 0.45),
        drift_magnitude: parseFloat(document.getElementById("sliderDrift")?.value || 0.35),
    };

    const btn = document.getElementById("btnInjectChaos");
    if (btn) btn.innerHTML = '<span class="spinner"></span> Injecting Chaos...';

    try {
        const res = await fetch("/api/chaos/inject", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload)
        });
        const data = await res.json();
        
        alert("Chaos successfully injected into production stream! Run diagnostics to inspect.");
        await refreshSystemStatus();
        await runDiagnostics();
        
        // Auto-switch to Diagnostic Scanner
        document.querySelector('[data-tab="tab-scanner"]')?.click();
    } catch (err) {
        alert("Failed to inject chaos: " + err);
    } finally {
        if (btn) btn.innerHTML = 'Inject Chaos Perturbations';
    }
}

// 3. Reset Chaos
async function resetChaos() {
    try {
        await fetch("/api/chaos/reset", { method: "POST" });
        alert("Stream reset to clean baseline.");
        await refreshSystemStatus();
        await runDiagnostics();
    } catch (err) {
        alert("Failed to reset stream: " + err);
    }
}

// 4. Run Diagnostics & Update Charts
async function runDiagnostics() {
    try {
        const res = await fetch("/api/diagnose");
        const data = await res.json();

        // Update Counter Pills
        const totalBadges = document.getElementById("statTotalFindings");
        if (totalBadges) totalBadges.textContent = data.total_findings;

        document.getElementById("countDataErrors").textContent = data.data_error_count;
        document.getElementById("countBiasFindings").textContent = data.bias_count;
        document.getElementById("countDriftFindings").textContent = data.drift_count;

        // Populate Findings Table
        const tbody = document.getElementById("tableFindingsBody");
        if (tbody) {
            tbody.innerHTML = "";
            if (data.findings.length === 0) {
                tbody.innerHTML = '<tr><td colspan="5" style="text-align:center; color:#9ca3af;">All systems green! No data errors, bias, or drift spotted.</td></tr>';
            } else {
                data.findings.forEach(f => {
                    const row = document.createElement("tr");
                    const sevBadgeClass = f.severity === "CRITICAL" ? "badge-degraded" : "badge-healthy";
                    row.innerHTML = `
                        <td><span class="badge ${sevBadgeClass}">${f.severity}</span></td>
                        <td><code>${f.issue_type}</code></td>
                        <td><strong>${f.affected_target}</strong></td>
                        <td>${f.metric_name}: <strong>${f.metric_value.toFixed(3)}</strong> (thresh ${f.threshold_value})</td>
                        <td style="color:#d1d5db; font-size:0.8rem;">${f.description}</td>
                    `;
                    tbody.appendChild(row);
                });
            }
        }

        // Render Charts
        renderDataErrorsChart(data);
        renderFairnessChart(data);
        renderDriftChart(data);

    } catch (err) {
        console.error("Failed to run diagnostics:", err);
    }
}

// Chart 1: Data Errors
function renderDataErrorsChart(diagData) {
    const ctx = document.getElementById("chartDataErrors")?.getContext("2d");
    if (!ctx) return;

    const counts = {
        "Missing Values": diagData.findings.filter(f => f.issue_type === "MISSING_VALUES").length,
        "Outliers": diagData.findings.filter(f => f.issue_type === "OUTLIERS").length,
        "Schema Bounds": diagData.findings.filter(f => f.issue_type === "SCHEMA_VIOLATION").length,
        "Label Noise": diagData.findings.filter(f => f.issue_type === "LABEL_NOISE").length,
    };

    if (chartDataErrors) chartDataErrors.destroy();

    chartDataErrors = new Chart(ctx, {
        type: "doughnut",
        data: {
            labels: Object.keys(counts),
            datasets: [{
                data: Object.values(counts),
                backgroundColor: ["#f59e0b", "#f43f5e", "#8b5cf6", "#06b6d4"],
                borderColor: "#111827",
                borderWidth: 2,
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { position: "bottom", labels: { color: "#9ca3af" } }
            }
        }
    });
}

// Chart 2: Fairness & Bias
function renderFairnessChart(diagData) {
    const ctx = document.getElementById("chartBias")?.getContext("2d");
    if (!ctx) return;

    // Search for gender metrics in findings metadata
    const genderFinding = diagData.findings.find(f => f.affected_target === "gender");
    let privRate = 0.38;
    let unprivRate = genderFinding ? privRate * genderFinding.metric_value : 0.38;

    if (genderFinding?.metadata?.priv_selection_rate !== undefined) {
        privRate = genderFinding.metadata.priv_selection_rate;
        unprivRate = genderFinding.metadata.unpriv_selection_rate;
    }

    if (chartBias) chartBias.destroy();

    chartBias = new Chart(ctx, {
        type: "bar",
        data: {
            labels: ["Privileged (Male)", "Unprivileged (Female)"],
            datasets: [{
                label: "Approval Selection Rate",
                data: [privRate * 100, unprivRate * 100],
                backgroundColor: ["rgba(6, 182, 212, 0.7)", "rgba(244, 63, 94, 0.7)"],
                borderColor: ["#06b6d4", "#f43f5e"],
                borderWidth: 1.5,
                borderRadius: 4
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                y: {
                    beginAtZero: true,
                    max: 60,
                    grid: { color: "#1f2937" },
                    ticks: { color: "#9ca3af", callback: v => v + "%" }
                },
                x: {
                    grid: { display: false },
                    ticks: { color: "#9ca3af" }
                }
            },
            plugins: {
                legend: { display: false }
            }
        }
    });
}

// Chart 3: Covariate Drift (PSI)
function renderDriftChart(diagData) {
    const ctx = document.getElementById("chartDrift")?.getContext("2d");
    if (!ctx) return;

    const driftFindings = diagData.findings.filter(f => f.issue_type === "COVARIATE_DRIFT");
    const labels = driftFindings.length > 0 ? driftFindings.map(f => f.affected_target) : ["debt_to_income", "credit_score", "loan_amount"];
    const values = driftFindings.length > 0 ? driftFindings.map(f => f.metric_value) : [0.03, 0.02, 0.04];

    if (chartDrift) chartDrift.destroy();

    chartDrift = new Chart(ctx, {
        type: "bar",
        data: {
            labels: labels,
            datasets: [{
                label: "Population Stability Index (PSI)",
                data: values,
                backgroundColor: values.map(v => v >= 0.20 ? "rgba(244, 63, 94, 0.7)" : v >= 0.10 ? "rgba(245, 158, 11, 0.7)" : "rgba(16, 185, 129, 0.7)"),
                borderRadius: 4
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                y: {
                    beginAtZero: true,
                    grid: { color: "#1f2937" },
                    ticks: { color: "#9ca3af" }
                },
                x: {
                    grid: { display: false },
                    ticks: { color: "#9ca3af" }
                }
            },
            plugins: {
                legend: { display: false }
            }
        }
    });
}

// 5. Trigger Autonomous Self-Healing
async function triggerAutonomousHeal() {
    const btn = document.getElementById("btnTriggerHeal");
    const flowContainer = document.getElementById("healingFlowContainer");
    const steps = document.querySelectorAll(".step-node");

    if (btn) btn.innerHTML = '<span class="spinner"></span> Executing Self-Healing Feedback Loop...';
    if (flowContainer) flowContainer.style.display = "block";

    // Progressive animation helper
    const highlightStep = (index) => {
        steps.forEach((s, i) => {
            if (i < index) {
                s.className = "step-node completed";
            } else if (i === index) {
                s.className = "step-node active";
            } else {
                s.className = "step-node";
            }
        });
    };

    highlightStep(0); // Diagnose
    await new Promise(r => setTimeout(r, 600));
    highlightStep(1); // Clean Data
    await new Promise(r => setTimeout(r, 600));
    highlightStep(2); // Sample Reweighting
    await new Promise(r => setTimeout(r, 600));
    highlightStep(3); // Fair Threshold Calibration
    await new Promise(r => setTimeout(r, 600));
    highlightStep(4); // Retrain Candidate
    await new Promise(r => setTimeout(r, 600));
    highlightStep(5); // Canary Gate

    try {
        const res = await fetch("/api/heal", { method: "POST" });
        const data = await res.json();
        const report = data.report;

        highlightStep(6); // Promote Champion
        steps.forEach(s => s.className = "step-node completed");

        // Display Status Banner
        const resultBanner = document.getElementById("healResultBanner");
        if (resultBanner) {
            resultBanner.style.display = "block";
            resultBanner.innerHTML = `
                <div style="background:rgba(16, 185, 129, 0.15); border:1px solid rgba(16, 185, 129, 0.3); border-radius:0.5rem; padding:1rem; margin-top:1rem;">
                    <h3 style="color:#10b981; font-size:1.1rem; margin-bottom:0.3rem;">Self-Healing Succeeded & Promoted!</h3>
                    <p style="color:#d1d5db; font-size:0.875rem;">${report.executive_summary}</p>
                    <p style="color:#9ca3af; font-size:0.8rem; margin-top:0.3rem;">New Champion Version: <code>${report.promoted_version_after}</code> (Canary Gate: <strong>PASSED</strong>)</p>
                </div>
            `;
        }

        // Render Comparison Table
        const compTbody = document.getElementById("tableComparisonBody");
        if (compTbody && report.comparisons) {
            compTbody.innerHTML = "";
            report.comparisons.forEach(c => {
                const tr = document.createElement("tr");
                const deltaColor = c.improved ? "#10b981" : "#9ca3af";
                const badge = c.improved ? '<span class="badge badge-healthy">IMPROVED</span>' : '<span class="badge">STABLE</span>';
                tr.innerHTML = `
                    <td><strong>${c.metric_name}</strong></td>
                    <td>${c.before_value.toFixed(3)}</td>
                    <td><strong style="color:#06b6d4;">${c.after_value.toFixed(3)}</strong></td>
                    <td style="color:${deltaColor}; font-weight:600;">${c.relative_change_pct > 0 ? '+' : ''}${c.relative_change_pct.toFixed(1)}%</td>
                    <td>${badge}</td>
                `;
                compTbody.appendChild(tr);
            });
        }

        // Render Remediation Action Log Table
        const planTbody = document.getElementById("tableRemediationPlanBody");
        if (planTbody && report.remediation_plan) {
            planTbody.innerHTML = "";
            report.remediation_plan.forEach(step => {
                const tr = document.createElement("tr");
                tr.innerHTML = `
                    <td><code>${step.step_id}</code></td>
                    <td><span class="badge badge-healed">${step.action_type}</span></td>
                    <td><strong>${step.target}</strong></td>
                    <td>${step.impact_summary || step.description}</td>
                `;
                planTbody.appendChild(tr);
            });
        }

        // Render Comparison Chart
        renderComparisonChart(report);

        // Update global status
        await refreshSystemStatus();
        await runDiagnostics();

    } catch (err) {
        alert("Failed during self-healing: " + err);
    } finally {
        if (btn) btn.innerHTML = 'Run Autonomous Self-Healing';
    }
}

// Chart 4: Before vs After Comparison
function renderComparisonChart(report) {
    const ctx = document.getElementById("chartComparison")?.getContext("2d");
    if (!ctx) return;

    const metrics = ["ACCURACY", "F1", "ROC_AUC"];
    const beforeVals = [];
    const afterVals = [];

    metrics.forEach(m => {
        const item = report.comparisons.find(c => c.metric_name === m);
        if (item) {
            beforeVals.push(item.before_value * 100);
            afterVals.push(item.after_value * 100);
        } else {
            beforeVals.push(0);
            afterVals.push(0);
        }
    });

    if (chartComparison) chartComparison.destroy();

    chartComparison = new Chart(ctx, {
        type: "bar",
        data: {
            labels: ["Accuracy", "F1-Score", "ROC-AUC"],
            datasets: [
                {
                    label: "Before (Degraded)",
                    data: beforeVals,
                    backgroundColor: "rgba(244, 63, 94, 0.7)",
                    borderRadius: 4
                },
                {
                    label: "After (Healed)",
                    data: afterVals,
                    backgroundColor: "rgba(16, 185, 129, 0.7)",
                    borderRadius: 4
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                y: {
                    beginAtZero: true,
                    max: 100,
                    grid: { color: "#1f2937" },
                    ticks: { color: "#9ca3af", callback: v => v + "%" }
                },
                x: {
                    grid: { display: false },
                    ticks: { color: "#9ca3af" }
                }
            },
            plugins: {
                legend: { position: "top", labels: { color: "#9ca3af" } }
            }
        }
    });
}

// 6. Test Single Applicant Inference
async function handleInference(e) {
    e.preventDefault();
    const resultCard = document.getElementById("inferenceResultCard");
    const payload = {
        income: parseFloat(document.getElementById("infIncome")?.value || 65000),
        credit_score: parseFloat(document.getElementById("infCreditScore")?.value || 710),
        debt_to_income: parseFloat(document.getElementById("infDTI")?.value || 0.28),
        loan_amount: parseFloat(document.getElementById("infLoanAmount")?.value || 25000),
        employment_length: parseFloat(document.getElementById("infEmpLength")?.value || 6),
        savings_balance: parseFloat(document.getElementById("infSavings")?.value || 15000),
        gender: document.getElementById("infGender")?.value || "Female",
        age_group: document.getElementById("infAgeGroup")?.value || "Middle-Aged",
    };

    try {
        const res = await fetch("/api/predict", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload)
        });
        const data = await res.json();

        if (resultCard) {
            resultCard.style.display = "block";
            const verdictColor = data.verdict === "APPROVED" ? "#10b981" : "#f43f5e";
            resultCard.innerHTML = `
                <div style="background:#1f2937; border-left: 4px solid ${verdictColor}; padding:1rem; border-radius:0.375rem;">
                    <div style="font-size:1.25rem; font-weight:700; color:${verdictColor}; margin-bottom:0.25rem;">
                        ${data.verdict}
                    </div>
                    <div style="font-size:0.875rem; color:#9ca3af;">
                        Approval Probability: <strong>${(data.approval_probability * 100).toFixed(1)}%</strong>
                    </div>
                    <div style="font-size:0.8rem; color:#6b7280; margin-top:0.25rem;">
                        Evaluated by <code>${data.champion_model_version}</code> (Applied Threshold: <strong>${data.applied_fair_threshold}</strong>)
                    </div>
                </div>
            `;
        }
    } catch (err) {
        alert("Inference failed: " + err);
    }
}

// 7. Download Markdown Report
function downloadReport() {
    window.location.href = "/api/report/markdown";
}
