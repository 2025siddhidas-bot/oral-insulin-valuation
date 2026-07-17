import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(page_title="rNPV Valuation Engine", layout="wide")
st.title("🔬 Oral Insulin: Risk-Adjusted NPV Engine")
st.markdown("Interactive valuation model featuring MIT BIO stage-gate risk, dynamic patent exclusivity, and 10,000-iteration Monte Carlo sensitivity analysis.")

# --- 2. SIDEBAR (USER INTERFACE) ---
st.sidebar.header("1. Commercial Parameters")
target_wac = st.sidebar.slider("Base WAC Price ($)", min_value=3628, max_value=6000, value=4789, step=10)
gtn_rebate = st.sidebar.slider("GTN Rebate (%)", min_value=40, max_value=70, value=50, step=1) / 100
peak_market_share = st.sidebar.slider("Peak Market Share (%)", min_value=5, max_value=35, value=15, step=1) / 100

st.sidebar.header("2. Intellectual Property")
patent_life_years = st.sidebar.slider("Years of Exclusivity", min_value=8, max_value=15, value=12, step=1)
pts_usd = st.sidebar.number_input("Indian PTS Cost ($)", value=2.21)

st.sidebar.header("3. Clinical Risk (MIT Data)")
stage_options = {
    "Phase 1 (19.6% POS)": 0.196,
    "Phase 2 (24.1% POS)": 0.241,
    "Phase 3 (51.6% POS)": 0.516,
    "Pre-Launch (85.0% POS)": 0.850
}
stage_selection = st.sidebar.selectbox("Current Clinical Stage", list(stage_options.keys()))
POS = stage_options[stage_selection]

# --- 3. VALUATION ENGINE (BACKEND LOGIC) ---
# Constants
YEARS = 20
INITIAL_POPULATION = 38000000
POP_CAGR = 0.01
WACC = 0.11
POST_LOE_RETENTION = 0.15
UPTAKE_CURVE = [0.05, 0.15, 0.35, 0.55, 0.75, 0.90, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]

def calculate_rnpv(wac, gtn, share, ptrs, patent_years):
    net_price = wac * (1 - gtn)
    cogs_margin = min(pts_usd / net_price, 0.99) if net_price > 0 else 0.99
    
    total_rnpv = 0
    # Dummy R&D burn for quick web calculation (-$100M over 3 years)
    for i, burn in enumerate([40e6, 60e6]):
        total_rnpv -= burn / ((1 + WACC)**(i+1))
        
    current_pop = INITIAL_POPULATION
    for yr in range(1, YEARS + 1):
        current_pop *= (1 + POP_CAGR)
        patients = current_pop * share * UPTAKE_CURVE[yr-1]
        
        revenue_factor = 1.0 if yr <= patent_years else POST_LOE_RETENTION
        gross_revenue = patients * net_price * revenue_factor
        cash_flow = gross_revenue * (1 - cogs_margin)
        
        discount_factor = (1 + WACC) ** (yr + 2) # Offset by 2 R&D years
        rnpv_yr = (cash_flow / discount_factor) * ptrs
        total_rnpv += rnpv_yr
        
    return total_rnpv

# Calculate Base rNPV
base_rnpv = calculate_rnpv(target_wac, gtn_rebate, peak_market_share, POS, patent_life_years)
st.metric(label="Calculated Base rNPV", value=f"${base_rnpv / 1e9:.2f} Billion")

# --- 4. MONTE CARLO & TORNADO CHARTS ---
st.divider()
col1, col2 = st.columns(2)

# Run 10k Iterations
ITERATIONS = 10000
sim_wac = np.random.triangular(3628, target_wac, 5500, ITERATIONS)
sim_gtn = np.random.triangular(0.40, gtn_rebate, 0.65, ITERATIONS)
sim_share = np.random.triangular(0.08, peak_market_share, 0.33, ITERATIONS)
sim_ptrs = np.random.triangular(0.182, POS, 0.210, ITERATIONS)

# Vectorized simulation
sim_rnpv = np.zeros(ITERATIONS)
for i in range(ITERATIONS):
    sim_rnpv[i] = calculate_rnpv(sim_wac[i], sim_gtn[i], sim_share[i], sim_ptrs[i], patent_life_years)

# Chart 1: Monte Carlo Histogram
with col1:
    st.subheader("Monte Carlo: Probability Distribution")
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(sim_rnpv / 1e9, bins=50, color='#4A6FA5', edgecolor='white')
    ax.axvline(base_rnpv / 1e9, color='goldenrod', linestyle='solid', linewidth=3, label='Mean Valuation')
    ax.axvline(np.percentile(sim_rnpv, 5) / 1e9, color='red', linestyle='dashed', label='5th Percentile')
    ax.axvline(np.percentile(sim_rnpv, 95) / 1e9, color='green', linestyle='dashed', label='95th Percentile')
    ax.set_xlabel("Risk-Adjusted NPV ($ Billions)")
    ax.set_ylabel("Frequency (10,000 Iterations)")
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    st.pyplot(fig)

# Chart 2: Tornado Sensitivity
with col2:
    st.subheader("Tornado Analysis: rNPV Sensitivity")
    
    # Calculate swings
    base_b = base_rnpv / 1e9
    swing_share = (calculate_rnpv(target_wac, gtn_rebate, 0.08, POS, patent_life_years)/1e9, calculate_rnpv(target_wac, gtn_rebate, 0.33, POS, patent_life_years)/1e9)
    swing_gtn = (calculate_rnpv(target_wac, 0.65, peak_market_share, POS, patent_life_years)/1e9, calculate_rnpv(target_wac, 0.40, peak_market_share, POS, patent_life_years)/1e9)
    swing_wac = (calculate_rnpv(3628, gtn_rebate, peak_market_share, POS, patent_life_years)/1e9, calculate_rnpv(5500, gtn_rebate, peak_market_share, POS, patent_life_years)/1e9)
    swing_ptrs = (calculate_rnpv(target_wac, gtn_rebate, peak_market_share, 0.182, patent_life_years)/1e9, calculate_rnpv(target_wac, gtn_rebate, peak_market_share, 0.210, patent_life_years)/1e9)

    variables = ['Peak Market Share (8% - 33%)', 'GTN Rebate (65% - 40%)', 'Base WAC Price ($3,628 - $5,500)', 'Clinical POS (Min/Max CI)']
    swings = [swing_share, swing_gtn, swing_wac, swing_ptrs]
    
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    for i, (low, high) in enumerate(swings):
        ax2.broken_barh([(low, base_b - low)], (i - 0.4, 0.8), facecolors='#E63946')
        ax2.broken_barh([(base_b, high - base_b)], (i - 0.4, 0.8), facecolors='#2A9D8F')
        
    ax2.set_yticks(range(len(variables)))
    ax2.set_yticklabels(variables)
    ax2.axvline(base_b, color='black', linewidth=2)
    ax2.set_xlabel("Asset Valuation ($ Billions)")
    st.pyplot(fig2)