# AutoHeal-AI: Teaching AI to Fix Itself

> **Autonomous MLOps Framework for Spotting Data Errors, Algorithmic Biases, and Feature Drifts — and Automatically Self-Healing Production AI Models.**
> 
> *A 3rd-Year Computer Science Capstone & Pair-Programming Project*

---

## 1. Project Abstract & Motivation

Modern Machine Learning models deployed in production inevitably degrade due to **silent failures**:
1. **Data Corruption**: Corrupted schemas, missing entries, extreme statistical outliers, and mislabeled ground truth.
2. **Algorithmic Bias**: Unintended discrimination where models grant fewer favorable outcomes (e.g., loan approvals) to protected groups (e.g., gender, race, age), violating the US EEOC **80% (Four-Fifths) Rule**.
3. **Covariate & Concept Drift**: Changing real-world economic environments that shift input feature distributions ($P(X)$) or mutate the underlying target relationship ($P(Y|X)$).

Traditionally, detecting and fixing these failures requires weeks of human data engineering. **AutoHeal-AI** teaches AI to **fix itself** through a closed-loop autonomous pipeline:

```
[Production Stream]
       │
       ▼
[1. DIAGNOSTIC SCANNER] ────► Spots Data Errors, Bias & Drifts
       │
       ▼
[2. REMEDIATION PLANNER] ───► Formulates Exact Mathematical Cures
       │
       ▼
[3. AUTONOMOUS HEALER] ─────► Cleans Data + Reweights Bias + Retrains Model
       │
       ▼
[4. CANARY SAFETY GATE] ────► Validates Challenger vs Champion on Holdout
       │
       ▼
[5. AUTO-PROMOTION] ────────► Promotes Healed Model with Full Audit Trail
```

---

## 2. Theoretical & Mathematical Foundations (CS Curriculum Alignment)

### Pillar 1: Data Error Detection & Cleansing
* **Missing Value Imputation**: Uses context-aware **K-Nearest Neighbors (KNN)** imputer on numerical manifolds:
  $$\hat{x}_i = \frac{1}{K} \sum_{j \in \mathcal{N}_K(i)} x_j$$
* **Outlier Winsorization**: Identifies anomalies via Interquartile Range ($IQR = Q_3 - Q_1$) and clips extreme values to prevent gradient explosion:
  $$x_{\text{clipped}} = \min(\max(x, Q_1 - 2.5 \cdot IQR), Q_3 + 2.5 \cdot IQR)$$
* **Confident Learning (Label Noise Pruning)**: Identifies flipped ground-truth labels where high-confidence model predictions sharply contradict the record:
  $$\text{Noise Sample if } y = 0 \text{ and } P(\hat{Y}=1|x) \ge \tau_{\text{noise}}$$

---

### Pillar 2: Algorithmic Bias Auditing & Debiasing
* **Disparate Impact (DI)**: Measures selection rate ratio between unprivileged and privileged groups:
  $$DI = \frac{P(\hat{Y}=1 \mid A = \text{unprivileged})}{P(\hat{Y}=1 \mid A = \text{privileged})}$$
  *Standard 80% (Four-Fifths) Rule: $DI \ge 0.80$ is required for legal and ethical compliance.*
* **Kamiran-Calders Fair Sample Reweighting (Pre-Processing)**:
  Forces the sensitive attribute $A$ and label $Y$ to be statistically independent in training:
  $$W(A=a, Y=y) = \frac{P(A=a) \times P(Y=y)}{P(A=a, Y=y)}$$
* **Equalized Odds Fair Threshold Tuning (Post-Processing)**:
  Calibrates subgroup decision thresholds $\tau_a$ to equalize True/False positive rates across demographic slices while maximizing predictive utility.

---

### Pillar 3: Covariate & Concept Drift Detection
* **Two-Sample Kolmogorov-Smirnov (KS) Test**:
  Computes the maximum vertical divergence between the reference and current empirical cumulative distributions:
  $$D = \sup_x |F_{\text{ref}}(x) - F_{\text{curr}}(x)|$$
  *Rejects null hypothesis of identical distributions if $p\text{-value} < 0.05$.*
* **Population Stability Index (PSI)**:
  Measures shift across $B$ quantile buckets:
  $$PSI = \sum_{b=1}^{B} \left( \text{Actual}_b - \text{Expected}_b \right) \times \ln\left( \frac{\text{Actual}_b}{\text{Expected}_b} \right)$$
  * $PSI < 0.10$: Stable (No drift)
  * $0.10 \le PSI < 0.20$: Moderate drift (Warning)
  * $PSI \ge 0.20$: Significant drift (Action Required)

---

## 3. Project Structure

```
Main/
├── autoheal/                        # Core Python Engine
│   ├── config.py                    # Thresholds (missing rate, PSI, DI, canary gates)
│   ├── core/                        # Incident events, model registry & reporter
│   │   ├── events.py                # Findings, steps, and audit log structures
│   │   ├── registry.py              # Champion-Challenger model registry
│   │   └── reporter.py              # Generates Markdown & JSON audit reports
│   ├── detectors/                   # Spotting algorithms
│   │   ├── data_errors.py           # Missingness, schema violations, outliers, label noise
│   │   ├── bias.py                  # Disparate impact, demographic parity, equalized odds
│   │   └── drift.py                 # KS-test, PSI, Wasserstein, Page-Hinkley
│   ├── remediators/                 # Fixing algorithms
│   │   ├── cleaner.py               # Automated KNN imputation, outlier winsorization
│   │   ├── debiasing.py             # Sample reweighting & fair threshold optimizer
│   │   └── retrainer.py             # Self-healing retraining & canary evaluation gate
│   ├── pipeline/
│   │   └── orchestrator.py          # Central Self-Healing Feedback Controller
│   └── datasets/
│       └── generator.py             # Credit risk benchmark dataset + chaos injectors
├── static/                          # Pure Web Frontend (NO Streamlit!)
│   ├── css/style.css                # Dark theme UI stylesheet
│   └── js/main.js                   # REST API client & Chart.js visualizer
├── templates/
│   └── index.html                   # Single-page application template
├── tests/                           # Complete test suite (10/10 passing)
│   ├── test_detectors.py            # Unit tests for detectors
│   ├── test_remediators.py          # Unit tests for cleaning & debiasing
│   └── test_orchestrator.py         # End-to-end self-healing integration test
├── app.py                           # FastAPI Web Server & REST API
├── demo_cli.py                      # 5-second colorful terminal demonstration
└── requirements.txt                 # Clean dependency manifest
```

---

## 4. Quickstart Guide

### Prerequisites
- Python 3.10+ (Tested on Python 3.12)
- Virtual environment (recommended)

### Installation
```bash
pip install -r requirements.txt
```

---

### Running Option A: Interactive Web Dashboard (FastAPI + HTML5/JS)
```bash
python app.py
```
Open your browser and navigate to:
**`http://127.0.0.1:8000`**

#### Interactive Dashboard Features:
1. **Executive Cockpit**: Real-time health score, active champion model, accuracy, and F1.
2. **Chaos Simulator**: Move sliders to inject missing values, outliers, label corruptions, demographic bias, and feature drift in real-time.
3. **Diagnostic Scanner**: Inspect detected anomalies with exact statistical metrics ($p$-values, PSI, Disparate Impact).
4. **Autonomous Self-Healing Studio**: Click **"Run Autonomous Self-Healing"** to watch the animated 7-step feedback loop, verify the before-vs-after comparison table, and observe automatic champion promotion.
5. **Live Borrower Sandbox**: Submit custom loan applications and see instant verdicts with applied fair threshold indicators.
6. **Audit Report Downloader**: Export formal incident reports in Markdown format.

---

### Running Option B: Terminal CLI Demonstration (For Quick Demos / Viva)
```bash
python demo_cli.py
```
Executes the full cycle in the terminal in under 5 seconds:
1. Bootstraps pristine baseline model ($F1 \approx 0.60$).
2. Injects real-world chaos (missing fields, extreme outliers, flipped labels, 45% gender bias, 35% covariate drift).
3. Spots 14 anomalies using the diagnostic engine.
4. Autonomously formulates remediation plan, cleans data, debiases samples, retrains champion ($F1 \approx 0.67$, Accuracy $+12\%$).
5. Promotes the healed model through the Canary Gate with $DI = 1.01$ (perfect fairness).

---

### Running Option C: Automated Test Suite
```bash
python -m unittest discover -s tests -v
```
Output:
```
test_bias_detection ... ok
test_drift_detection ... ok
test_label_noise_detection ... ok
test_missing_values_detection ... ok
test_outlier_detection ... ok
test_full_autonomous_self_healing_cycle ... ok
test_cleaner_imputation_and_clipping ... ok
test_cleaner_label_noise_pruning ... ok
test_fairness_reweighting ... ok
test_retrainer_candidate ... ok

----------------------------------------------------------------------
Ran 10 tests in 8.103s

OK (100% Passing)
```

---

## 5. Viva Voce / Oral Examination Preparation (Q&A)

### Q1: What is the difference between Covariate Drift and Concept Drift?
* **Covariate Drift**: The distribution of input features $P(X)$ changes over time, while the conditional label relationship $P(Y \mid X)$ remains unchanged (e.g., average applicant credit score drops due to inflation, but the criteria for creditworthiness stay the same).
* **Concept Drift**: The conditional probability $P(Y \mid X)$ changes (e.g., a credit score of 700 previously implied low risk, but now implies higher default risk due to mutating fraud techniques).

### Q2: How does the system detect Label Noise?
AutoHeal-AI employs **Confident Learning**. For each sample, it compares the recorded label $y$ against out-of-fold predicted class probabilities $P(\hat{Y} \mid x)$. If $y = 0$ but the model predicts $P(\hat{Y}=1 \mid x) \ge 0.65$ with high confidence, the sample is identified as a flipped label and safely pruned before retraining.

### Q3: Why is accuracy insufficient when evaluating fairness?
A model can achieve 95% accuracy while completely rejecting a minority subgroup (e.g., if a subgroup represents 5% of applicants). AutoHeal-AI enforces **Disparate Impact** and **Demographic Parity** to ensure positive selection rates do not disproportionately disadvantage protected groups.

### Q4: How does the Canary Safety Gate prevent catastrophic failure?
Before any retrained "challenger" model is promoted to production "champion", the Canary Gate tests it against the active champion on a clean holdout validation set. The challenger is only promoted if:
1. F1-Score does not degrade beyond allowable tolerance ($\Delta F1 \ge -0.02$).
2. Disparate Impact meets the legal threshold ($DI \ge 0.80$).
If any check fails, the candidate is safely rejected, preserving system stability.

---

## 6. Academic Credits & License
- **Framework**: Developed for 3rd-Year Computer Science Capstone in Artificial Intelligence & Machine Learning.
- **License**: MIT Open Source License.
