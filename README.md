# 🔬 Oral Insulin: Risk-Adjusted Net Present Value (rNPV) Engine
**A dynamic, institutional-grade valuation model featuring Monte Carlo simulations, clinical stage-gate de-risking, and empirical proxy data for early-stage metabolic assets.**

Web Version: https://oral-insulin-valuation-jertplxv7tgdwjvpod2nmb.streamlit.app/

## Overview
This repository contains a full-scale pharmaceutical valuation engine built in Python and Streamlit. The model evaluates the Risk-Adjusted Net Present Value (rNPV) of a pre-clinical/Phase 1 oral insulin asset targeted at the Type 2 Diabetes (T2D) market. 

The model strictly utilizes an **Unlevered Free Cash Flow (UFCF)** framework. This isolates the valuation to the core commercial economics and clinical viability of the drug, independent of capital structure. Furthermore, the engine outputs a dual-valuation: the strict probability-weighted **rNPV** and the **Un-risked Commercial NPV** to highlight the massive asymmetrical upside upon FDA approval.

## 🧮 Mathematical Framework
The engine bifurcates global patient data into two distinct economic markets to account for the massive pricing delta between the US and the Rest of World (ROW):
*   **US Market:** Modeled using higher WAC pricing offset by heavy Pharmacy Benefit Manager (PBM) rebate requirements (GTN).
*   **ROW Market:** Modeled using a 90% list price discount and an additional 20% statutory market access rebate to reflect international, state-negotiated pricing mechanisms.

### 1. The rNPV Equation
The total asset value is calculated by discounting all future R&D outflows and geographically segmented commercial inflows by the Weighted Average Cost of Capital (WACC), adjusted for the clinical Probability of Success (POS):

$$rNPV = \left[ \sum_{t=1}^{T_{pre}} \frac{-R\&D_t}{(1 + WACC)^t} \right] \times POS + \left[ \sum_{t=1}^{20} \frac{ \overbrace{\Big[ Pat_{US,t} \big( WAC(1 - GTN_{US}) - COGS_{ann} \big) \Big]}^{\text{US Net Cash Flow}} + \overbrace{\Big[ Pat_{ROW,t} \big( WAC(0.10)(1 - GTN_{ROW}) - COGS_{ann} \big) \Big]}^{\text{ROW Net Cash Flow}} }{(1 + WACC)^{t + T_{pre}}} \times E_t \right] \times POS$$

### 2. Commercial Cash Inflows & S-Curve Adoption
Cash flows are not realized instantly. The model applies a standard biopharma **S-Curve adoption rate** to simulate the 5-to-7-year ramp required to change physician prescribing habits and secure formulary placement.

*   **US Cash Flow:** $CF_{US,t} = (Patients_{US,t} \times US WAC \times (1 - US GTN)) - (Patients_{US,t} \times COGS)$
*   **ROW Cash Flow:** $CF_{ROW,t} = (Patients_{ROW,t} \times (US WAC \times 0.10) \times (1 - ROW GTN)) - (Patients_{ROW,t} \times COGS)$

---

## 📊 1. Epidemiology & Total Addressable Market (TAM)
The patient funnel leverages a micro-simulation analysis of global insulin dependency, split by macro-geography:
*   **Global T2D Base:** 37.3 Million (United States) and 544.1 Million (Rest of World)...as of 2024 (IDF Global Datasets)
*   **Clinical Access Rate:** The model filters this baseline by a 7.4% (Base) to 15.5% (Bull) access rate, isolating the specific subset of patients eligible for this exact therapy. (Study: Basu S, Yudkin J, Kehlenbrink S et al., Estimation of global insulin use for type 2 diabetes, 2018–30: a microsimulation analysis, The Lancet Diabetes & Endocrinology, 2018; 7, 25-33)
*   Population growth is compounded annually at a 1.47% CAGR (accurately reflecting the IDF's projection of a 46% global increase in diabetes by 2050).

---

## 📈 2. Market Capture (LRx Proxy Dynamics)
Market capture rates are proxied using historical adoption curves of insulin analogs and Gabelli Funds data on oral GLP-1 transitions.
*   **Bear Case (8%):** Absolute floor based on historical slow-adopting analogs. (Source: Source: IQVIA https://www.iqvia.com/insights/the-iqvia-institute/reports-and-publications/reports/understanding-insulin-market-dynamics-in-low-and-middle-income-countries)
*   **Base Case (15%):** A defensible capture of the oral ceiling, targeting approximately half of the projected oral metabolic market. (Source: Source: https://www.pharmavoice.com/news/weight-loss-market-oral-glp1-novo-lilly/717148/)
*   **Bull Case (20%):** Capped at 20% based on peak oral GLP-1 transition estimates, acknowledging competitive dynamics from current injectables.

---

## 💰 3. Commercial Pricing & Manufacturing Margins
Pricing assumptions balance the standard of care with mathematically justified oral convenience premiums.

*   **Wholesale Acquisition Cost (US WAC):** Scaled from a baseline injectable parity of $3,628 up to a $5,445 premium. The Base Case is modeled at $4,789/year. (Source: Value in Health Publication). ROW WAC taken as 10% of US WAC.
*   **Gross-to-Net (GTN) Rebate:** The US market is characterized by severe Pharmacy Benefit Manager (PBM) rebate walls. The model applies a 76% US GTN discount, and 20% ROW GTN discount.
*   **Cost of Goods Sold (COGS):** Calculated utilizing API values from Gotham et al.'s study 'Production costs and potential prices for biosimilars of human insulin and insulin analogues'.

---

## 🧬 4. Clinical Risk (PTRS) & Discount Rates
*   **PTRS:** Data is sourced from the MIT BIO study on clinical trial success rates. The overall Phase 1 to Approval probability is 19.6% for Metabolic/Endocrinology.
*   **WACC:** 11% (Damodaran Biotech Industry standard of ~9% + 2% Company-Specific Risk Premium for novel administration routes).

---

## 🌪️ 5. Monte Carlo & Tornado Sensitivity Analysis
The engine utilizes `numpy` and `matplotlib` to run a 10,000-iteration Monte Carlo simulation, removing single-point estimate flaws.

**Stochastic Boundaries (Triangular Distributions):**
*   **Adoption Rates:** 7.4% to 15.5%
*   **Peak Market Share:** 8% to 20%
*   **Base WAC:** $3,628 to $5,445
*   **COGS:** $1.37 to $1.18 per pill
*   **Clinical POS:** Adjusts dynamically using a 95% Confidence Interval based on a +/- 1.4% Standard Error applied to the active clinical phase.

Outputs include a normal distribution (bell curve) of the rNPV and a Tornado Diagram isolating the financial swing of individual variables.

---

## 🚀 6. How to Use the Model

### Option A: Run the Live Web Application
The valuation engine has been deployed as an interactive web interface using Streamlit. 
*   **https://oral-insulin-valuation-jertplxv7tgdwjvpod2nmb.streamlit.app/**

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
