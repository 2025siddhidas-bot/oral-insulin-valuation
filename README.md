# 🔬 Oral Insulin: Risk-Adjusted Net Present Value (rNPV) Engine
**A dynamic, institutional-grade valuation model featuring Monte Carlo simulations, clinical stage-gate de-risking, and empirical proxy data for early-stage metabolic assets.**

## Overview
This repository contains a full-scale pharmaceutical valuation engine built in Python and Streamlit. The model evaluates the Risk-Adjusted Net Present Value (rNPV) of a pre-clinical/Phase 1 oral insulin asset targeted at the Type 2 Diabetes (T2D) market. 

To avoid the "false precision" of complex multi-national tax accounting for early-stage assets, the model strictly utilizes an **Unlevered Free Cash Flow (UFCF)** framework. This isolates the valuation to the core commercial economics and clinical viability of the drug.

---

## 📊 1. Epidemiology & Total Addressable Market (TAM)
The patient funnel is grounded in micro-simulation analyses of global insulin dependency.
* The Total Global T2D Population in 2030 is estimated at 511 million adults.
* Of this population, 79.2 million adults with T2D will strictly require insulin to manage their HbA1c levels.
* Based on current healthcare trajectories, only 38 million of these adults will actually have access to and use insulin.
* This establishes a Base Case access rate of 7.43% (38 million accessing / 511 million total).
* The Bull Case access rate is established at 15.49% (79.2 million needed / 511 million total).
* The total global vial volume required for 2030 is projected at 633.7 million.
* Source: Basu S, Yudkin J, Kehlenbrink S et al., Estimation of global insulin use for type 2 diabetes, 2018–30: a microsimulation analysis, The Lancet Diabetes & Endocrinology, 2018.

---

## 📈 2. Market Capture (LRx Proxy Dynamics)
Since oral insulin is an emerging modality, market capture rates are proxied using historical adoption curves of insulin analogs and oral GLP-1s.
* **Bear Case (8%):** Based on the absolute ceiling of adoption rates historically seen for insulin analogs. Sourced from IQVIA's report on insulin market dynamics.
* **Base Case (15%):** Anchored between the 14% GLP-1 market share captured by Rybelsus (PharmaVoice) and the 16.4% oral revenue capture reported in Novo Nordisk's 2023 SEC Annual Results.
* **Bull Case (33%):** Based on IQVIA data demonstrating that Oral Wegovy captured one-third of all new GLP-1 New-to-Brand market demand within just 8 weeks of launch.

---

## 💰 3. Commercial Pricing & Manufacturing Margins
Pricing assumptions balance the standard of care with mathematically justified oral convenience premiums.

**Wholesale Acquisition Cost (WAC):**
* The baseline injectable WAC for Lantus is set at $3,628 per year, sourced from a Value in Health publication.
* **Base Case WAC ($4,789/year):** Utilizes a 1.32x premium over Lantus, justified purely by the convenience of oral delivery.
* **Bull Case WAC ($5,445/year):** Utilizes a 1.5x premium, modeling a scenario where clinical trials prove a statistically significant reduction in hypoglycemic events.
* Source: Premium multiplier calculations are derived from Mixed Logic Model results published in Dove Press regarding patient preferences for T2D medications.

**Cost of Goods Sold (COGS) & Price-to-Stockist (PTS):**
Manufacturing costs are proxied using the launch-time Maximum Retail Price (MRP) of Rybelsus in India to simulate outsourced API production.
* The consumer printed MRP for a pill in India is 315.00 INR.
* Deducting a 12% GST yields a Base Trade Value of 281.25 INR.
* Deducting a 20% Retail Margin yields a Price-to-Retailer of 225.00 INR.
* Deducting a 10% Distributor Margin results in a Price-to-Stockist (PTS) of 204.54 INR, or roughly $2.12 USD per pill.
* The modeled Base COGS is 16.84% (calculated as $2.21 PTS / $13.12 WAC). 
* The modeled Bull COGS is 14.82% (calculated as $2.21 PTS / $14.91 WAC).

**Gross-to-Net (GTN) Rebate:**
* Modeled at a 50% baseline discount (Levy et al., 2018 average VAFSS discount of 48.3% rounded for metabolic markets). 
* Monte Carlo boundaries are set at a 40% floor (IQVIA insulin discount data) and a 65% ceiling (Optum Rx Medicare Part D historical demands).

---

## 🧬 4. Clinical Risk (PTRS) & Discount Rates
The model abandons static, taxonomy-based risk averages in favor of longitudinal, phase-by-phase transition probabilities to dynamically un-risk the asset as it clears clinical hurdles.

**Probability of Technical and Regulatory Success (PTRS):**
* Data is strictly sourced from the MIT Cancer's study: Estimation of clinical trial success rates and related parameters.
* For the Metabolic/Endocrinology therapeutic group, the Overall Probability of Success from Phase 1 to Approval is 19.6%.
* The asset is evaluated using dynamic stage-gate transitions: Phase 1 to 2 (76.2%), Phase 2 to 3 (59.7%), and Phase 3 to Approval (51.6%).

**Weighted Average Cost of Capital (WACC):**
* The model utilizes a baseline WACC of 11%.
* This is constructed using the Damodaran Biotech Industry standard WACC of roughly 9.01%, plus a 2% Company-Specific Risk Premium (CSRP).
* The 2% premium is explicitly applied because an unproven oral insulin modality faces higher-than-average commercial adoption risks compared to traditional injectables.

---

## 🌪️ 5. Monte Carlo & Tornado Sensitivity Analysis
To eliminate the unreliability of single-point estimates, the engine utilizes `numpy` and `matplotlib` to run a 10,000-iteration Monte Carlo simulation. 

**Stochastic Boundaries (Triangular Distributions):**
* **Peak Market Share:** 8% (Bear) | 15% (Base) | 33% (Bull)
* **GTN Rebate:** 40% (Floor) | 50% (Base) | 65% (PBM Stress)
* **Base WAC:** $3,628 (Lantus Parity) | $4,789 (Base) | $5,500 (Premium)
* **Clinical POS:** 18.2% (Min) | 19.6% (Base) | 21.0% (Max) — Bounds calculated dynamically using a 95% Confidence Interval based on the exact 0.7% Standard Error published in the MIT Metabolic/Endocrinology dataset.

The output generates a normal distribution (bell curve) outlining the 90% confidence interval of the asset's value, accompanied by a calculated Tornado Diagram isolating the financial swing of individual variables.

---

## 🚀 6. How to Use the Model

### Option A: Run the Live Web Application
The valuation engine has been deployed as an interactive web interface using Streamlit. 
* **[Insert your Streamlit Cloud URL here]**

### Option B: Run Locally via Python
To run the 10,000-iteration Monte Carlo simulation on your local machine:

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/your-username/oral-insulin-valuation.git](https://github.com/your-username/oral-insulin-valuation.git)
   cd oral-insulin-valuation

2. Install the dependencies:

  Bash
  pip install -r requirements.txt

3. Launch the Streamlit interface:

  Bash
  streamlit run app.py
