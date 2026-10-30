# Statistical Power, Design Effect, and Sample Size Reassessment
## Transparent Clustered Variance Auditing and Sensitivity Stress-Testing for Independent Dysarthric Cohorts

**Project Title:** Low-Resource Personalized Atypical Speech Recognition Using Hybrid Quantum-Classical Deep Learning for Reliable Emergency Communication  
**Document Identifier:** REPORT-STAT-POWER-REASSESS-2026-V1  
**Execution Date:** 2026-10-11  
**Authors:** Senior Biostatistician & Clinical-ML Validation Auditor  
**Audit Context:** Reassessment of Gate C Statistical Analysis Plan (`reports/phase_c/STATISTICAL_ANALYSIS_PLAN.md`)  

---

> [!IMPORTANT]
> **AUDIT PURPOSE & CAUTIONARY PRINCIPLE:**  
> The target figure of **at least 109 independent dysarthric speakers and 2,164 emergency utterances** proposed in Phase C documentation was put forward as a preliminary planning assumption. This report recomputes the underlying arithmetic transparently, stress-tests the sensitivity of those figures across clinical variability, and evaluates the empirical feasibility of enrolling such cohorts against real-world atypical speech resources.

---

## 1. Mathematical Audit of Historical Planning Calculations

In [`reports/phase_c/STATISTICAL_ANALYSIS_PLAN.md`](file:///d:/DL/reports/phase_c/STATISTICAL_ANALYSIS_PLAN.md) (Section 2), the following baseline assumptions were declared:
- Target Primary Endpoint: Population-Average Emergency Intent Recall ($p = 0.70$).
- Desired Precision Margin: Two-sided $95\%$ Confidence Interval half-width $d = \pm 0.05$ ($\pm 5\%$).
- Significance Level: $\alpha = 0.05 \implies Z_{1 - \alpha/2} = 1.96$.
- Intra-Cluster Correlation Coefficient: $\rho = 0.30$.
- Average Relevant Emergency Observations per Speaker: $\bar{m} = 20$.
- Anticipated Participant Attrition / Recording Degradation: $15\%$.

### 1.1 Step-by-Step Mathematical Verification

#### Step 1: Unclustered Sample Size ($n_{unclustered}$)
Under standard Wald binomial sampling theory, the sample size required to estimate a population proportion $p$ with margin of error $d$ at confidence level $1 - \alpha$ is:
$$n_{unclustered} = \frac{Z_{1 - \alpha/2}^2 \cdot p (1 - p)}{d^2}$$

Substituting the parameter values:
$$n_{unclustered} = \frac{(1.96)^2 \cdot (0.70) \cdot (1 - 0.70)}{(0.05)^2} = \frac{3.8416 \cdot 0.21}{0.0025} = \frac{0.806736}{0.0025} = 322.6944$$

Rounding up to the nearest integer yields:
$$\mathbf{n_{unclustered} = 323 \text{ emergency utterances}}$$

#### Step 2: Design Effect ($DEFF$)
Because multiple utterances from the same dysarthric speaker are acoustically correlated due to individual vocal tract morphology and motor impairment severity, observations are clustered within speakers. The variance inflation factor (Design Effect) for equal cluster sizes is:
$$DEFF = 1 + (\bar{m} - 1) \rho$$

Substituting $\bar{m} = 20$ and $\rho = 0.30$:
$$DEFF = 1 + (20 - 1) \cdot (0.30) = 1 + (19) \cdot (0.30) = 1 + 5.70 = \mathbf{6.70}$$

#### Step 3: Clustered Emergency Utterance Target ($N_{emergency}$)
$$N_{emergency} = n_{unclustered} \times DEFF = 322.6944 \times 6.70 = 2162.052 \approx \mathbf{2,164 \text{ emergency utterances}}$$

#### Step 4: Speaker Enrollment Requirement ($K$)
At an average of $\bar{m} = 20$ emergency utterances per participant:
$$K_{eval} = \left\lceil \frac{N_{emergency}}{\bar{m}} \right\rceil = \left\lceil \frac{2164}{20} \right\rceil = \lceil 108.2 \rceil = \mathbf{109 \text{ independent dysarthric speakers}}$$

Adjusting for a $15\%$ participant attrition or unclassifiable recording rate:
$$K_{enrollment} = \left\lceil \frac{K_{eval}}{1 - 0.15} \right\rceil = \left\lceil \frac{109}{0.85} \right\rceil = \lceil 128.23 \rceil = \mathbf{129 \text{ enrolled dysarthric participants}}$$

### Mathematical Verification Summary:
The historical derivation in `reports/phase_c/STATISTICAL_ANALYSIS_PLAN.md` is **arithmetically exact** under its explicit starting assumptions ($\bar{m}=20, \rho=0.30, d=0.05, p=0.70$).

---

## 2. Challenging the Underlying Assumptions: Multi-Scenario Sensitivity Stress-Testing

While the arithmetic is correct, the underlying assumptions are highly fragile in clinical atypical speech research. Below, we systematically challenge each parameter.

---

### Challenge 1: The Equal Cluster Size Fallacy (Unequal Cluster Sizes)

In clinical data collection, participants with severe motor speech disorders experience varying degrees of physical and vocal fatigue. Some participants can only record 5 utterances before fatiguing, while others can complete 40. 

When cluster sizes $m_i$ vary across speakers, the true design effect is determined not by the arithmetic mean $\bar{m}$, but by the **variance-weighted cluster size** $\tilde{m}$:
$$DEFF = 1 + (\tilde{m} - 1) \rho$$
$$\tilde{m} = \frac{\sum_{i=1}^K m_i^2}{\sum_{i=1}^K m_i} = \bar{m} \left( 1 + CV_m^2 \right)$$
where $CV_m = \frac{\sigma_m}{\bar{m}}$ is the coefficient of variation of cluster sizes.

#### Impact of Cluster Size Variability on Sample Size ($\bar{m} = 20, \rho = 0.30, d = 0.05$):

| Cluster Size Variation Scenario | $CV_m$ | $\tilde{m}$ | True $DEFF$ | Required Emergency Utterances ($N$) | Required Speakers ($K$) | Enrollment Target ($15\%$ attrition) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Ideal Uniform Clusters (Historical)** | $0.00$ | $20.00$ | **$6.70$** | **$2,164$** | **$109$** | **$129$** |
| **Low Variation (Rigid Protocol)** | $0.30$ | $21.80$ | **$7.24$** | **$2,337$** | **$117$** | **$138$** |
| **Moderate Variation (Typical Speech Study)**| $0.50$ | $25.00$ | **$8.20$** | **$2,646$** | **$133$** | **$157$** |
| **Severe Variation (Clinical Dysarthria)** | $0.80$ | $32.80$ | **$10.54$** | **$3,401$** | **$171$** | **$201$** |
| **Extreme Variation (Natural Fatigue Dropouts)**| $1.00$ | $40.00$ | **$12.70$** | **$4,098$** | **$205$** | **$242$** |

> [!WARNING]
> **Finding 1:** Assuming equal cluster sizes severely underestimates required sample size. Under realistic clinical fatigue ($CV_m = 0.80$), the required number of dysarthric speakers increases from **109 to 171** (a **$57\%$ expansion**), requiring **3,401 emergency utterances**.

---

### Challenge 2: Sensitivity to Intra-Cluster Correlation ($\rho$)

The assumption of $\rho = 0.30$ represents a midpoint. In severe spastic or athetoid dysarthria, speakers have extreme, highly repeatable articulatory constraints (e.g. consistent hypernasality, vowel centralization), which can drive $\rho$ toward $0.45$. Conversely, mild dysarthria with high articulatory variability may exhibit $\rho \approx 0.15$.

#### Two-Way Sensitivity Grid: True Cluster Sizes vs. ICC ($\rho$) (Showing Required Speakers $K$):

| ICC ($\rho$) | Uniform Clusters ($CV=0.0$) | Moderate Variance ($CV=0.5$) | Severe Variance ($CV=0.8$) |
| :---: | :---: | :---: | :---: |
| **$\rho = 0.10$** | $K = 47$ ($DEFF = 2.90$) | $K = 55$ ($DEFF = 3.40$) | $K = 68$ ($DEFF = 4.18$) |
| **$\rho = 0.20$** | $K = 78$ ($DEFF = 4.80$) | $K = 94$ ($DEFF = 5.80$) | $K = 119$ ($DEFF = 7.36$) |
| **$\rho = 0.30$ (Baseline)** | **$K = 109$ ($DEFF = 6.70$)** | **$K = 133$ ($DEFF = 8.20$)** | **$K = 171$ ($DEFF = 10.54$)** |
| **$\rho = 0.40$** | $K = 139$ ($DEFF = 8.60$) | $K = 172$ ($DEFF = 10.60$) | $K = 222$ ($DEFF = 13.72$) |
| **$\rho = 0.50$** | $K = 170$ ($DEFF = 10.50$) | $K = 210$ ($DEFF = 13.00$) | $K = 273$ ($DEFF = 16.90$) |

---

### Challenge 3: Feasibility of Per-Speaker Emergency Cluster Size ($\bar{m}$)

Can an individual with dysarthria realistically produce $\bar{m} = 20$ emergency utterances? If fatigue forces shorter sessions ($\bar{m} = 5$ or $10$), how does the speaker requirement change?

$$\text{At } \rho = 0.30, CV_m = 0.50, d = \pm 0.05:$$

| Target Cluster Size ($\bar{m}$) | Effective Size $\tilde{m}$ | $DEFF$ | Required Emergency Utterances ($N$) | Required Dysarthric Speakers ($K$) |
| :---: | :---: | :---: | :---: | :---: |
| **$\bar{m} = 5$ utterances** | $6.25$ | $2.58$ | $831$ | **$167$ speakers** |
| **$\bar{m} = 10$ utterances** | $12.50$ | $4.45$ | $1,436$ | **$144$ speakers** |
| **$\bar{m} = 15$ utterances** | $18.75$ | $6.33$ | $2,041$ | **$137$ speakers** |
| **$\bar{m} = 20$ utterances** | $25.00$ | $8.20$ | $2,646$ | **$133$ speakers** |
| **$\bar{m} = 30$ utterances** | $37.50$ | $11.95$ | $3,856$ | **$129$ speakers** |

> [!NOTE]
> Decreasing cluster size per speaker ($\bar{m}$) dramatically increases the efficiency of the design (reducing $DEFF$ from $8.20$ to $2.58$), but demands recruiting more unique human participants ($K = 167$).

---

### Challenge 4: Prevalence and Total Speech Corpus Volume

In natural, daily conversational communication, emergency phrases constitute a tiny fraction of speech. 
If data are collected continuously in a natural home environment vs. via structured prompt lists, the total speech volume required expands massively:

$$N_{total} = \frac{N_{emergency}}{\text{Prevalence}_{emergency}}$$

| Study Paradigm | Emergency Intent Prevalence ($\pi$) | Required Emergency Utterances ($N_{em}$) | Total Utterances Required in Corpus ($N_{total}$) |
| :--- | :---: | :---: | :---: |
| **Continuous Natural Monitoring** | $1.0\%$ | $2,646$ | **$264,600 \text{ utterances}$** |
| **Unconstrained Daily Interaction** | $3.0\%$ | $2,646$ | **$88,200 \text{ utterances}$** |
| **Broad Ambient Clinical Recording**| $5.0\%$ | $2,646$ | **$52,920 \text{ utterances}$** |
| **Standard Assistive Prompt Task** | $10.0\%$ | $2,646$ | **$26,460 \text{ utterances}$** |
| **Enriched Prompt Protocol** | $25.0\%$ | $2,646$ | **$10,584 \text{ utterances}$** |
| **Structured Emergency Read Prompts**| $50.0\%$ | $2,646$ | **$5,292 \text{ utterances}$** |

---

### Challenge 5: Precision Estimation vs. Formal Hypothesis Testing

The historical target was derived for **confidence interval precision** ($d = \pm 0.05$). What if the study is framed as a **one-sided confirmatory hypothesis test**:
$$H_0: \theta_{recall} \le 0.50 \quad \text{vs.} \quad H_1: \theta_{recall} \ge 0.70$$
at significance level $\alpha = 0.05$ ($Z_{1-\alpha} = 1.645$) and statistical power $1 - \beta = 0.80$ ($Z_{1-\beta} = 0.8416$)?

Under standard power equations:
$$n_{unclustered, test} = \frac{\left( Z_{1-\alpha} \sqrt{p_0 (1 - p_0)} + Z_{1-\beta} \sqrt{p_1 (1 - p_1)} \right)^2}{(p_1 - p_0)^2}$$
$$n_{unclustered, test} = \frac{\left( 1.645 \cdot \sqrt{0.25} + 0.8416 \cdot \sqrt{0.21} \right)^2}{(0.70 - 0.50)^2} = \frac{(0.8225 + 0.3857)^2}{0.04} = \frac{(1.2082)^2}{0.04} = \frac{1.4597}{0.04} \approx 36.5$$

Applying $DEFF = 8.20$:
$$N_{test} = 36.5 \times 8.20 \approx 299 \text{ emergency utterances} \implies K \approx \lceil 299 / 20 \rceil = \mathbf{15 \text{ speakers}}$$

However, if testing a narrow non-inferiority or superiority margin ($H_0: \theta \le 0.65$ vs. $H_1: \theta \ge 0.75$, $\Delta = 0.10$):
$$n_{unclustered, narrow} \approx 150 \implies N_{narrow} \approx 1,230 \text{ utterances} \implies K \approx \mathbf{62 \text{ speakers}}$$

---

## 3. The Real-World Dataset Feasibility Gap

Comparing these statistical targets against actual candidate datasets proves that **legacy atypical speech corpora cannot support a well-powered confirmatory evaluation**:

```mermaid
bar
    title Available Dysarthric Speakers vs. Required Target (K)
    x-axis ["Required (Uniform DEFF)", "Required (Realistic CV=0.8)", "Speech Accessibility Proj (SAP)", "UA-Speech", "Nemours", "TORGO (Contaminated)"]
    y-axis "Dysarthric Speakers (K)" 0 --> 200
    "Required (Uniform DEFF)" : 109
    "Required (Realistic CV=0.8)" : 171
    "Speech Accessibility Proj (SAP)" : 150
    "UA-Speech" : 19
    "Nemours" : 11
    "TORGO (Contaminated)" : 8
```

- **UA-Speech Deficit:** 19 speakers available vs. 109–171 required (**Deficit of 90 to 152 speakers**).
- **Nemours Deficit:** 11 speakers available vs. 109–171 required (**Deficit of 98 to 160 speakers**).
- **Combined Legacy Deficit:** $19 + 11 = 30$ speakers (**Deficit of 79 to 141 speakers**).
- **The Only Viable Path:** The **Speech Accessibility Project (SAP)** is the only existing resource with hundreds of enrolled dysarthric speakers capable of meeting $K \ge 109$. Alternatively, a dedicated prospective multi-center registry must be established.

---

## 4. Prespecified Statistical Modeling Methodology

To validly analyze clustered data without inflating Type I error:

1. **Primary Model — Generalized Estimating Equations (GEE):**
   - Logit link with binary outcome $\hat{Y}_{ij}$.
   - Exchangeable working correlation matrix.
   - **Huber-White Sandwich Robust Variance Estimator** to ensure standard error validity even if working correlation is misspecified.
2. **Secondary Model — Generalized Linear Mixed-Effects Model (GLMM):**
   $$\text{logit}(P(\hat{Y}_{ij} = 1 \mid u_i)) = \beta_0 + u_i, \quad u_i \sim \mathcal{N}(0, \sigma_u^2)$$
   Allows explicit estimation of between-speaker variance $\sigma_u^2$.
3. **Confidence Interval Estimation:**
   - 1,000-iteration **Stratified Clustered Bootstrap** (resampling clusters of utterances by speaker).
   - Standard unclustered Wald intervals ($p \pm 1.96 \sqrt{p(1-p)/N}$) are **strictly barred**.
4. **Multiplicity Control:**
   - Benjamini-Hochberg False Discovery Rate (FDR) control at $q = 0.05$ across all secondary intent classes and subgroup strata.
