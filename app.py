import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import io

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(page_title="rNPV Valuation Engine", layout="wide")
st.title("🔬 Oral Insulin: Risk-Adjusted NPV Engine")
st.markdown("Interactive valuation model featuring MIT BIO stage-gate risk, geographical pricing segmentation, dynamic patent exclusivity, and 10,000-iteration Monte Carlo analysis.")

# --- 2. SIDEBAR (USER INTERFACE) ---
st.sidebar.header("1. Commercial Parameters")
target_wac = st.sidebar.slider("Base US WAC Price ($)", min_value=3628, max_value=6000, value=4789, step=10)
gtn_rebate = st.sidebar.slider("US GTN Rebate (%)", min_value=50, max_value=90, value=75, step=1) / 100
peak_market_share = st.sidebar.slider("Peak Global Share (%)", min_value=5, max_value=20, value=15, step=1) / 100

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

st.sidebar.header("4. R&D Timeline & Costs")
st.sidebar.markdown("*(Annual burn rate input)*")
p1_yrs = st.sidebar.number_input("Phase 1 Duration (Years)", value=2.0, step=0.5)
p1_burn = st.sidebar.number_input("Phase 1 Annual Burn ($M)", value=20.0, step=1.0) * 1e6
p2_yrs = st.sidebar.number_input("Phase 2 Duration (Years)", value=3.0, step=0.5)
p2_burn = st.sidebar.number_input("Phase 2 Annual Burn ($M)", value=40.0, step=1.0) * 1e6
p3_yrs = st.sidebar.number_input("Phase 3 Duration (Years)", value=4.0, step=0.5)
p3_burn = st.sidebar.number_input("Phase 3 Annual Burn ($M)", value=100.0, step=1.0) * 1e6
p4_yrs = st.sidebar.number_input("NDA / Pre-Launch Duration (Years)", value=2.0, step=0.5)
p4_burn = st.sidebar.number_input("NDA Annual Burn ($M)", value=25.0, step=1.0) * 1e6

# --- 3. VALUATION ENGINE (BACKEND LOGIC) ---
YEARS = 20
US_POPULATION = 30000000 
ROW_POPULATION = 470000000
POP_CAGR = 0.0147
ACCESS_RATE_BASE = 0.07    
ACCESS_RATE_BULL = 0.155   
WACC = 0.11
POST_LOE_RETENTION = 0.15
UPTAKE_CURVE = [0.10, 0.35, 0.65, 0.85, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 0.50, 0.25, 0.15, 0.10, 0.05]

def calculate_rnpv(wac, gtn, share, ptrs, patent_years, access_rate, return_logs=False):
    annual_cogs = pts_usd * 365.0
    us_net_price = wac * (1 - gtn)
    row_net_price = max((wac * 0.10) * 0.80, annual_cogs * 1.15) 
    
    total_rnpv = 0
    current_year = 1
    logs = []
    
    if return_logs:
        logs.append("[1] Processing Clinical Trial Pipeline Cash Flows...")
        
    phases = [(p1_yrs, p1_burn), (p2_yrs, p2_burn), (p3_yrs, p3_burn), (p4_yrs, p4_burn)]
    for duration, annual_burn in phases:
        if duration > 0:
            for _ in range(int(np.ceil(duration))):
                discounted_burn = annual_burn / ((1 + WACC)**current_year)
                rnpv_burn = discounted_burn * ptrs
                total_rnpv -= rnpv_burn
                if return_logs:
                    logs.append(f"  R&D Yr {current_year:02d} | Capital Burn: ${annual_burn/1e6:5.1f}M | rNPV Impact: ${-rnpv_burn/1e6:7.2f}M")
                current_year += 1
                
    launch_year_offset = current_year
    
    if return_logs:
        logs.append(f"\n[2] Processing {YEARS}-Year Commercialization Window (LOE Cliff at Yr {patent_years})...")
        
    current_us_pop = US_POPULATION
    current_row_pop = ROW_POPULATION
    
    for yr in range(1, YEARS + 1):
        current_us_pop *= (1 + POP_CAGR)
        current_row_pop *= (1 + POP_CAGR)
        
        us_patients = current_us_pop * access_rate * share * UPTAKE_CURVE[yr-1]
        row_patients = current_row_pop * access_rate * share * UPTAKE_CURVE[yr-1]
        
        revenue_factor = 1.0 if yr <= patent_years else POST_LOE_RETENTION
        
        gross_revenue = (us_patients * us_net_price + row_patients * row_net_price) * revenue_factor
        total_cogs = (us_patients + row_patients) * annual_cogs * revenue_factor
        
        cash_flow = gross_revenue - total_cogs
        
        discount_factor = (1 + WACC) ** (yr + launch_year_offset - 1)
        rnpv_yr = (cash_flow / discount_factor) * ptrs
        total_rnpv += rnpv_yr
        
        if return_logs and (yr in [1, 5, 10, patent_years, 15, YEARS]):
            cliff_note = " <--- [GENERIC CLIFF EXECUTED]" if yr == patent_years else ""
            rev_str = f"${gross_revenue/1e9:.2f}B" if gross_revenue >= 1e9 else f"${gross_revenue/1e6:.2f}M"
            logs.append(f"  Com. Yr {yr:02d} | Net Revenue: {rev_str:>7} | rNPV: ${rnpv_yr/1e6:7.2f}M{cliff_note}")
            
    if return_logs:
        return total_rnpv, "\n".join(logs)
    return total_rnpv

# --- 4. SCENARIO DASHBOARD ---
bear_rnpv, bear_logs = calculate_rnpv(3628, 0.90, 0.08, POS, patent_life_years, ACCESS_RATE_BASE, return_logs=True)
base_rnpv, base_logs = calculate_rnpv(target_wac, gtn_rebate, peak_market_share, POS, patent_life_years, ACCESS_RATE_BASE, return_logs=True)
bull_rnpv, bull_logs = calculate_rnpv(5500, 0.60, 0.20, POS, patent_life_years, ACCESS_RATE_BULL, return_logs=True)

st.subheader("📊 Scenario Valuations")
col_bear, col_base, col_bull = st.columns(3)

with col_bear:
    st.metric("📉 Bear Case (8% Share, 90% GTN)", f"${bear_rnpv / 1e9:.2f} B")
    st.caption(f"Un-risked Commercial NPV: **${(bear_rnpv / POS) / 1e9:.2f} B**")

with col_base:
    st.metric("🎯 Base Case (Current Inputs)", f"${base_rnpv / 1e9:.2f} B")
    st.caption(f"Un-risked Commercial NPV: **${(base_rnpv / POS) / 1e9:.2f} B**")

with col_bull:
    st.metric("🚀 Bull Case (20% Share, 60% GTN)", f"${bull_rnpv / 1e9:.2f} B")
    st.caption(f"Un-risked Commercial NPV: **${(bull_rnpv / POS) / 1e9:.2f} B**")

with st.expander("🔍 View Detailed Year-by-Year Cash Flows (Terminal Logs)"):
    col_log1, col_log2, col_log3 = st.columns(3)
    with col_log1:
        st.code(f"--- RUNNING VALUATION SCENARIO: BEAR ---\n\n{bear_logs}\n\n{'-'*45}\nFINAL BEAR ASSET rNPV: ${bear_rnpv/1e9:.3f} BILLION\nUN-RISKED COMMERCIAL NPV: ${(bear_rnpv/POS)/1e9:.3f} BILLION")
    with col_log2:
        st.code(f"--- RUNNING VALUATION SCENARIO: BASE ---\n\n{base_logs}\n\n{'-'*45}\nFINAL BASE ASSET rNPV: ${base_rnpv/1e9:.3f} BILLION\nUN-RISKED COMMERCIAL NPV: ${(base_rnpv/POS)/1e9:.3f} BILLION")
    with col_log3:
        st.code(f"--- RUNNING VALUATION SCENARIO: BULL ---\n\n{bull_logs}\n\n{'-'*45}\nFINAL BULL ASSET rNPV: ${bull_rnpv/1e9:.3f} BILLION\nUN-RISKED COMMERCIAL NPV: ${(bull_rnpv/POS)/1e9:.3f} BILLION")

# Download Button for Text Logs
full_report = f"--- BEAR CASE ---\n{bear_logs}\n\n--- BASE CASE ---\n{base_logs}\n\n--- BULL CASE ---\n{bull_logs}"
st.download_button(
    label="📄 Download Detailed Cash Flows (.txt)",
    data=full_report,
    file_name="rNPV_Cash_Flow_Logs.txt",
    mime="text/plain"
)

st.divider()

# --- 5. MONTE CARLO & TORNADO CHARTS ---
col1, col2 = st.columns(2)

ITERATIONS = 10000
sim_wac = np.random.triangular(3628, target_wac, 5500, ITERATIONS)
sim_gtn = np.random.triangular(min(0.60, gtn_rebate), gtn_rebate, max(0.90, gtn_rebate), ITERATIONS)
sim_share = np.random.triangular(min(0.08, peak_market_share), peak_market_share, max(0.20, peak_market_share), ITERATIONS)
sim_ptrs = np.random.triangular(max(0.01, POS - 0.014), POS, min(1.0, POS + 0.014), ITERATIONS)

sim_rnpv = np.zeros(ITERATIONS)
for i in range(ITERATIONS):
    sim_rnpv[i] = calculate_rnpv(sim_wac[i], sim_gtn[i], sim_share[i], sim_ptrs[i], patent_life_years, ACCESS_RATE_BASE)

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
    
    # Download Button for Chart 1
    buf1 = io.BytesIO()
    fig.savefig(buf1, format="png", bbox_inches="tight", dpi=300)
    buf1.seek(0)
    st.download_button(
        label="📥 Download Monte Carlo Chart (.png)",
        data=buf1,
        file_name="Monte_Carlo_Distribution.png",
        mime="image/png"
    )

with col2:
    st.subheader("Tornado Analysis: rNPV Sensitivity")
    base_b = base_rnpv / 1e9
    
    swing_share = (calculate_rnpv(target_wac, gtn_rebate, 0.08, POS, patent_life_years, ACCESS_RATE_BASE)/1e9, calculate_rnpv(target_wac, gtn_rebate, 0.20, POS, patent_life_years, ACCESS_RATE_BASE)/1e9)
    swing_gtn = (calculate_rnpv(target_wac, 0.90, peak_market_share, POS, patent_life_years, ACCESS_RATE_BASE)/1e9, calculate_rnpv(target_wac, 0.60, peak_market_share, POS, patent_life_years, ACCESS_RATE_BASE)/1e9)
    swing_wac = (calculate_rnpv(3628, gtn_rebate, peak_market_share, POS, patent_life_years, ACCESS_RATE_BASE)/1e9, calculate_rnpv(5500, gtn_rebate, peak_market_share, POS, patent_life_years, ACCESS_RATE_BASE)/1e9)
    swing_ptrs = (calculate_rnpv(target_wac, gtn_rebate, peak_market_share, max(0.01, POS - 0.014), patent_life_years, ACCESS_RATE_BASE)/1e9, calculate_rnpv(target_wac, gtn_rebate, peak_market_share, min(1.0, POS + 0.014), patent_life_years, ACCESS_RATE_BASE)/1e9)

    variables = ['Peak Market Share (8% - 20%)', 'US GTN Rebate (90% - 60%)', 'Base US WAC Price ($3,628 - $5,500)', 'Clinical POS (Min/Max CI)']
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
    
    # Download Button for Chart 2
    buf2 = io.BytesIO()
    fig2.savefig(buf2, format="png", bbox_inches="tight", dpi=300)
    buf2.seek(0)
    st.download_button(
        label="📥 Download Tornado Chart (.png)",
        data=buf2,
        file_name="Tornado_Sensitivity.png",
        mime="image/png"
    )

st.divider()

# --- 6. COMPREHENSIVE PDF REPORT GENERATOR ---
st.subheader("📑 Export Complete Report")
st.markdown("Download a full PDF tear-sheet containing your active inputs, clinical parameters, Monte Carlo/Tornado charts, and Base Case cash flow logs.")

try:
    from fpdf import FPDF
    import tempfile
    import os

    # Create the PDF Document
    pdf = FPDF()
    pdf.add_page()
    
    # Title
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(0, 10, "Oral Insulin rNPV - Comprehensive Valuation Report", ln=True, align="C")
    pdf.ln(5)

    # Section 1: User Inputs
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, "1. Commercial & Clinical Parameters Selected:", ln=True)
    pdf.set_font("Arial", '', 10)
    pdf.cell(0, 6, f"Base US WAC Price: ${target_wac:,.0f}", ln=True)
    pdf.cell(0, 6, f"US GTN Rebate: {gtn_rebate*100:.0f}%", ln=True)
    pdf.cell(0, 6, f"Peak Global Share: {peak_market_share*100:.0f}%", ln=True)
    pdf.cell(0, 6, f"Clinical Stage: {stage_selection}", ln=True)
    pdf.cell(0, 6, f"Calculated Base Case rNPV: ${base_rnpv/1e9:.3f} Billion", ln=True)
    pdf.ln(5)
    
    # Save the matplotlib figures to temporary files so FPDF can read them
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as f1, \
         tempfile.NamedTemporaryFile(delete=False, suffix=".png") as f2, \
         tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as fpdf_out:
         
         fig.savefig(f1.name, format="png", bbox_inches="tight")
         fig2.savefig(f2.name, format="png", bbox_inches="tight")
         
         # Section 2 & 3: Charts
         pdf.set_font("Arial", 'B', 12)
         pdf.cell(0, 10, "2. Monte Carlo Distribution:", ln=True)
         pdf.image(f1.name, w=160)
         
         pdf.add_page()
         pdf.cell(0, 10, "3. Tornado Sensitivity Analysis:", ln=True)
         pdf.image(f2.name, w=160)
         
         # Section 4: Cash Flow Logs
         pdf.add_page()
         pdf.cell(0, 10, "4. Base Case Cash Flow Log:", ln=True)
         pdf.set_font("Courier", '', 8)
         
         # Write the logs line by line
         for line in base_logs.split('\n'):
             safe_line = line.encode('latin-1', 'replace').decode('latin-1')
             pdf.multi_cell(0, 4, safe_line)
             
         # Save the finalized PDF to the third temp file
         pdf.output(fpdf_out.name)
         
         # Read the PDF back into bytes for the Streamlit download button
         with open(fpdf_out.name, "rb") as f:
             pdf_bytes = f.read()
             
    # Clean up the temporary files from the server
    os.unlink(f1.name)
    os.unlink(f2.name)
    os.unlink(fpdf_out.name)
    
    # Render the master download button
    st.download_button(
        label="Download Comprehensive PDF Report",
        data=pdf_bytes,
        file_name="Oral_Insulin_Valuation_Report.pdf",
        mime="application/pdf",
        use_container_width=True
    )

except ImportError:
    st.warning("⚠️ PDF generation requires the 'fpdf' library. Please add 'fpdf' to your requirements.txt file on GitHub to enable this button.")
