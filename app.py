import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import io

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(page_title="rNPV Valuation Engine", layout="wide")
st.title("🔬 Oral Insulin: Risk-Adjusted NPV Engine")
st.markdown("Interactive valuation model featuring MIT BIO stage-gate risk (marginal, stage-weighted), "
            "geographical pricing segmentation, dynamic patent exclusivity, and 10,000-iteration Monte Carlo analysis.")

# --- 2. SIDEBAR (USER INTERFACE) ---
st.sidebar.header("1. Commercial Parameters")
target_wac = st.sidebar.slider("Base US WAC Price ($)", min_value=3628, max_value=6000, value=4789, step=10)
gtn_rebate = st.sidebar.slider("US GTN Rebate (%)", min_value=50, max_value=90, value=76, step=1) / 100
peak_market_share = st.sidebar.slider("Base Global Share (%)", min_value=5, max_value=20, value=15, step=1) / 100

st.sidebar.header("2. Intellectual Property")
patent_life_years = st.sidebar.slider("Years of Exclusivity", min_value=8, max_value=15, value=12, step=1)

st.sidebar.header("3. Clinical Risk (MIT Data)")

CUMULATIVE_PTRS_MAP = {
    "1": (0.196, 0.014),  # Phase 1 Start (Overall POS, SE 0.7% * 2)
    "2": (0.241, 0.018),  # Phase 2 Start (POS 2,APP, SE 0.9% * 2)
    "3": (0.516, 0.030),  # Phase 3 Start (POS 3,APP, SE 1.5% * 2)
    "4": (0.850, 0.020)   # NDA / Pre-Launch (Assigned conservative 2.0% bound)
}
STAGE_ORDER = ["1", "2", "3", "4"]
STAGE_LABELS = {
    "1": "Phase 1 (Safety)",
    "2": "Phase 2 (Efficacy)",
    "3": "Phase 3 (Pivotal/CVOT)",
    "4": "NDA / Pre-Launch Supply"
}

stage_options = {
    "Phase 1 (19.6% POS)": ("1", CUMULATIVE_PTRS_MAP["1"][0]),
    "Phase 2 (24.1% POS)": ("2", CUMULATIVE_PTRS_MAP["2"][0]),
    "Phase 3 (51.6% POS)": ("3", CUMULATIVE_PTRS_MAP["3"][0]),
    "Pre-Launch (85.0% POS)": ("4", CUMULATIVE_PTRS_MAP["4"][0])
}
stage_selection = st.sidebar.selectbox("Current Clinical Stage", list(stage_options.keys()))
current_stage, POS = stage_options[stage_selection]

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
POPULATION_CAGR = 0.0147
WACC = 0.11
POST_LOE_RETENTION = 0.40

WAC_PRICES = {"Bear": 3628.0, "Base": target_wac, "Bull": 5445.0}
COGS_PER_PILL = {"Bear": 1.37, "Base": 1.27, "Bull": 1.18}
ADOPTION_RATES = {"Bear": 0.08, "Base": peak_market_share, "Bull": 0.20}
ACCESS_RATES = {"Bear": 0.074, "Base": 0.074, "Bull": 0.155}

UPTAKE_CURVE = [
    0.05, 0.335, 0.65, 0.88, 1.00,
    1.00, 1.00, 1.00, 1.00, 1.00,
    1.00, 1.00, 1.00, 1.00, 1.00,
    0.50, 0.25, 0.15, 0.10, 0.05
]

@st.cache_data
def clean_df(df, target_val_name):
    country_col = 'Location' if 'Location' in df.columns else 'Country_Territory'
    if country_col not in df.columns and 'Country' in df.columns: country_col = 'Country'

    val_col = 'Value' if 'Value' in df.columns else 'Total'
    if val_col not in df.columns and 'Expenditure' in df.columns: val_col = 'Expenditure'

    df = df.rename(columns={country_col: 'Country', val_col: target_val_name})
    if 'Country' in df.columns:
        df['Country'] = df['Country'].astype(str).str.strip()

    exclusions = [
        'World', 'Africa', 'Europe', 'Middle East and North Africa',
        'North America and Caribbean', 'South and Central America',
        'South-East Asia', 'Western Pacific', 'High income',
        'Middle income', 'Low income'
    ]
    df = df[~df['Country'].isin(exclusions)]

    if df[target_val_name].dtype == object:
        df[target_val_name] = df[target_val_name].astype(str).str.replace(',', '', regex=True)

    df[target_val_name] = pd.to_numeric(df[target_val_name], errors='coerce').fillna(0)
    return df[['Country', target_val_name]]

@st.cache_data
def load_data():
    try:
        df_total = pd.read_csv('Estimated total number of adults (20–79 years) with diabetes in 2024.csv')
        df_t1_all = pd.read_csv('People with type 1 diabetes (all age groups) by Country_Territory.csv')
        df_t1_youth = pd.read_csv('People with type 1 diabetes (0-19 y) by Country_Territory.csv')
    except FileNotFoundError:
        return pd.DataFrame({
            'Country': ['United States of America', 'Rest of World'],
            'T2D_Population': [37256319, 544119737]
        })

    df_total = clean_df(df_total, 'Total_Diabetics_20_79')
    df_t1_all = clean_df(df_t1_all, 'Type_1_All_Ages')
    df_t1_youth = clean_df(df_t1_youth, 'Type_1_0_19')

    df_total['Total_Diabetics_20_79'] *= 1000

    master_df = pd.merge(df_total, df_t1_all, on='Country', how='outer')
    master_df = pd.merge(master_df, df_t1_youth, on='Country', how='outer').fillna(0)

    master_df['Type_1_Adults'] = (master_df['Type_1_All_Ages'] - master_df['Type_1_0_19']).clip(lower=0)
    master_df['T2D_Population'] = (master_df['Total_Diabetics_20_79'] - master_df['Type_1_Adults']).clip(lower=0)

    return master_df

market_data = load_data()

clinical_burn_map = []  
for stage_key, duration, annual_burn in [("1", p1_yrs, p1_burn), ("2", p2_yrs, p2_burn),
                                          ("3", p3_yrs, p3_burn), ("4", p4_yrs, p4_burn)]:
    if duration > 0:
        full_years = int(duration)
        remainder = duration - full_years
        
        for _ in range(full_years):
            clinical_burn_map.append((stage_key, annual_burn))
            
        if remainder > 0:
            clinical_burn_map.append((stage_key, annual_burn * remainder))

def calculate_marginal_weights(current_stage, current_pos, ptrs_map=CUMULATIVE_PTRS_MAP, stage_order=STAGE_ORDER):
    weights = {}
    reached_current = False
    for stage in stage_order:
        if stage == current_stage:
            reached_current = True
        weights[stage] = (current_pos / ptrs_map[stage][0]) if reached_current else 0.0
    return weights

# Asterisk added to force keyword arguments to prevent positional mismatching
def calculate_rnpv(*, wac, gtn, cogs_per_pill, share, ptrs, patent_years, access_rate, df, return_logs=False):
    annual_cogs = cogs_per_pill * 365.0
    us_net_price = wac * (1 - gtn)

    row_wac = wac * 0.10
    row_net_price = max(row_wac * 0.80, annual_cogs * 1.15)

    total_rnpv = 0
    current_year = 1
    logs = []

    stage_weights = calculate_marginal_weights(current_stage, ptrs)

    if return_logs:
        logs.append("[1] Processing Clinical Trial Pipeline Cash Flows (Marginal Stage-Weighted)...")

    for annual_burn_year, (stage_key, annual_burn) in enumerate(clinical_burn_map, 1):
        discounted_burn = annual_burn / ((1 + WACC) ** current_year)
        stage_weight = stage_weights[stage_key]
        rnpv_burn = discounted_burn * stage_weight
        total_rnpv -= rnpv_burn
        if return_logs:
            logs.append(f"  R&D Yr {current_year:02d} | {STAGE_LABELS[stage_key]:26s} | Burn: ${annual_burn/1e6:5.1f}M | "
                        f"Stage Weight: {stage_weight*100:5.1f}% | rNPV Impact: ${-rnpv_burn/1e6:7.2f}M")
        current_year += 1

    launch_year_offset = current_year

    if return_logs:
        logs.append(f"\n[2] Processing {YEARS}-Year Commercialization Window (LOE Cliff at Yr {patent_years + 1})...")

    us_base_pool = df[df['Country'] == 'United States of America']['T2D_Population'].sum()
    row_base_pool = df[df['Country'] != 'United States of America']['T2D_Population'].sum()

    for yr in range(1, YEARS + 1):
        us_pool = us_base_pool * ((1 + POPULATION_CAGR) ** yr)
        row_pool = row_base_pool * ((1 + POPULATION_CAGR) ** yr)

        us_patients = us_pool * access_rate * share * UPTAKE_CURVE[yr-1]
        row_patients = row_pool * access_rate * share * UPTAKE_CURVE[yr-1]

        revenue_factor = 1.0 if yr <= patent_years else POST_LOE_RETENTION

        gross_revenue = (us_patients * us_net_price + row_patients * row_net_price) * revenue_factor
        total_cogs = (us_patients + row_patients) * annual_cogs * revenue_factor

        cash_flow = gross_revenue - total_cogs

        discount_factor = (1 + WACC) ** (yr + launch_year_offset - 1)
        rnpv_yr = (cash_flow / discount_factor) * ptrs
        total_rnpv += rnpv_yr

        if return_logs and (yr in [1, 5, 10, patent_years, patent_years+1, 15, YEARS]):
            cliff_note = " <--- [GENERIC/BIOSIMILAR CLIFF EXECUTED]" if yr == patent_years + 1 else ""
            rev_str = f"${gross_revenue/1e9:.2f}B" if gross_revenue >= 1e9 else f"${gross_revenue/1e6:.2f}M"
            logs.append(f"  Com. Yr {yr:02d} | Net Revenue: {rev_str:>7} | rNPV: ${rnpv_yr/1e6:7.2f}M{cliff_note}")

    if return_logs:
        return total_rnpv, "\n".join(logs)
    return total_rnpv

# --- 4. SCENARIO DASHBOARD ---
# Explicit Keyword Arguments applied across all cases
bear_rnpv, bear_logs = calculate_rnpv(wac=WAC_PRICES["Bear"], gtn=gtn_rebate, cogs_per_pill=COGS_PER_PILL["Bear"], share=ADOPTION_RATES["Bear"], ptrs=POS, patent_years=patent_life_years, access_rate=ACCESS_RATES["Bear"], df=market_data, return_logs=True)
base_rnpv, base_logs = calculate_rnpv(wac=WAC_PRICES["Base"], gtn=gtn_rebate, cogs_per_pill=COGS_PER_PILL["Base"], share=ADOPTION_RATES["Base"], ptrs=POS, patent_years=patent_life_years, access_rate=ACCESS_RATES["Base"], df=market_data, return_logs=True)
bull_rnpv, bull_logs = calculate_rnpv(wac=WAC_PRICES["Bull"], gtn=gtn_rebate, cogs_per_pill=COGS_PER_PILL["Bull"], share=ADOPTION_RATES["Bull"], ptrs=POS, patent_years=patent_life_years, access_rate=ACCESS_RATES["Bull"], df=market_data, return_logs=True)

st.subheader("📊 Scenario Valuations")
col_bear, col_base, col_bull = st.columns(3)

with col_bear:
    st.metric(f"📉 Bear Case ({ADOPTION_RATES['Bear']*100:.0f}% Share, $3.6K WAC, $1.37 COGS)", f"${bear_rnpv / 1e9:.2f} B")
    st.caption(f"Un-risked Commercial NPV: **${(bear_rnpv / POS) / 1e9:.2f} B**")

with col_base:
    st.metric(f"🎯 Base Case ({ADOPTION_RATES['Base']*100:.0f}% Share, $4.7K WAC, $1.27 COGS)", f"${base_rnpv / 1e9:.2f} B")
    st.caption(f"Un-risked Commercial NPV: **${(base_rnpv / POS) / 1e9:.2f} B**")

with col_bull:
    st.metric(f"🚀 Bull Case ({ADOPTION_RATES['Bull']*100:.0f}% Share, $5.4K WAC, $1.18 COGS)", f"${bull_rnpv / 1e9:.2f} B")
    st.caption(f"Un-risked Commercial NPV: **${(bull_rnpv / POS) / 1e9:.2f} B**")

with st.expander("🔍 View Detailed Year-by-Year Cash Flows (Terminal Logs)"):
    col_log1, col_log2, col_log3 = st.columns(3)
    with col_log1:
        st.code(f"--- RUNNING VALUATION SCENARIO: BEAR ---\n\n{bear_logs}\n\n{'-'*45}\nFINAL BEAR ASSET rNPV: ${bear_rnpv/1e9:.3f} BILLION\nUN-RISKED COMMERCIAL NPV: ${(bear_rnpv/POS)/1e9:.3f} BILLION")
    with col_log2:
        st.code(f"--- RUNNING VALUATION SCENARIO: BASE ---\n\n{base_logs}\n\n{'-'*45}\nFINAL BASE ASSET rNPV: ${base_rnpv/1e9:.3f} BILLION\nUN-RISKED COMMERCIAL NPV: ${(base_rnpv/POS)/1e9:.3f} BILLION")
    with col_log3:
        st.code(f"--- RUNNING VALUATION SCENARIO: BULL ---\n\n{bull_logs}\n\n{'-'*45}\nFINAL BULL ASSET rNPV: ${bull_rnpv/1e9:.3f} BILLION\nUN-RISKED COMMERCIAL NPV: ${(bull_rnpv/POS)/1e9:.3f} BILLION")

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

np.random.seed(42)

sim_wac = np.random.triangular(WAC_PRICES["Bear"], WAC_PRICES["Base"], WAC_PRICES["Bull"], ITERATIONS)
sim_gtn = np.random.triangular(0.60, gtn_rebate, 0.90, ITERATIONS)
sim_share = np.random.triangular(ADOPTION_RATES["Bear"], ADOPTION_RATES["Base"], ADOPTION_RATES["Bull"], ITERATIONS)
sim_cogs = np.random.triangular(COGS_PER_PILL["Bull"], COGS_PER_PILL["Base"], COGS_PER_PILL["Bear"], ITERATIONS)
sim_access = np.random.triangular(ACCESS_RATES["Base"], ACCESS_RATES["Base"], ACCESS_RATES["Bull"], ITERATIONS)

current_pos, current_se_bound = CUMULATIVE_PTRS_MAP[current_stage]

pos_min = max(0.01, current_pos - current_se_bound)
pos_max = min(1.00, current_pos + current_se_bound)
pos_dist = np.random.triangular(pos_min, current_pos, pos_max, ITERATIONS)

sim_rnpv = np.zeros(ITERATIONS)
for i in range(ITERATIONS):
    sim_rnpv[i] = calculate_rnpv(wac=sim_wac[i], gtn=sim_gtn[i], cogs_per_pill=sim_cogs[i], share=sim_share[i], ptrs=sim_ptrs[i], patent_years=patent_life_years, access_rate=sim_access[i], df=market_data)

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

    swing_share = (calculate_rnpv(wac=target_wac, gtn=gtn_rebate, cogs_per_pill=COGS_PER_PILL["Base"], share=ADOPTION_RATES["Bear"], ptrs=POS, patent_years=patent_life_years, access_rate=ACCESS_RATES["Base"], df=market_data)/1e9,
                   calculate_rnpv(wac=target_wac, gtn=gtn_rebate, cogs_per_pill=COGS_PER_PILL["Base"], share=ADOPTION_RATES["Bull"], ptrs=POS, patent_years=patent_life_years, access_rate=ACCESS_RATES["Base"], df=market_data)/1e9)

    swing_wac = (calculate_rnpv(wac=WAC_PRICES["Bear"], gtn=gtn_rebate, cogs_per_pill=COGS_PER_PILL["Base"], share=ADOPTION_RATES["Base"], ptrs=POS, patent_years=patent_life_years, access_rate=ACCESS_RATES["Base"], df=market_data)/1e9,
                 calculate_rnpv(wac=WAC_PRICES["Bull"], gtn=gtn_rebate, cogs_per_pill=COGS_PER_PILL["Base"], share=ADOPTION_RATES["Base"], ptrs=POS, patent_years=patent_life_years, access_rate=ACCESS_RATES["Base"], df=market_data)/1e9)

    swing_cogs = (calculate_rnpv(wac=target_wac, gtn=gtn_rebate, cogs_per_pill=COGS_PER_PILL["Bear"], share=ADOPTION_RATES["Base"], ptrs=POS, patent_years=patent_life_years, access_rate=ACCESS_RATES["Base"], df=market_data)/1e9,
                  calculate_rnpv(wac=target_wac, gtn=gtn_rebate, cogs_per_pill=COGS_PER_PILL["Bull"], share=ADOPTION_RATES["Base"], ptrs=POS, patent_years=patent_life_years, access_rate=ACCESS_RATES["Base"], df=market_data)/1e9)

    swing_access = (calculate_rnpv(wac=target_wac, gtn=gtn_rebate, cogs_per_pill=COGS_PER_PILL["Base"], share=ADOPTION_RATES["Base"], ptrs=POS, patent_years=patent_life_years, access_rate=ACCESS_RATES["Bear"], df=market_data)/1e9,
                    calculate_rnpv(wac=target_wac, gtn=gtn_rebate, cogs_per_pill=COGS_PER_PILL["Base"], share=ADOPTION_RATES["Base"], ptrs=POS, patent_years=patent_life_years, access_rate=ACCESS_RATES["Bull"], df=market_data)/1e9)

    swing_ptrs = (calculate_rnpv(wac=target_wac, gtn=gtn_rebate, cogs_per_pill=COGS_PER_PILL["Base"], share=ADOPTION_RATES["Base"], ptrs=max(0.01, POS - current_se_bound), patent_years=patent_life_years, access_rate=ACCESS_RATES["Base"], df=market_data)/1e9,
                  calculate_rnpv(wac=target_wac, gtn=gtn_rebate, cogs_per_pill=COGS_PER_PILL["Base"], share=ADOPTION_RATES["Base"], ptrs=min(1.0, POS + current_se_bound), patent_years=patent_life_years, access_rate=ACCESS_RATES["Base"], df=market_data)/1e9)

    swing_gtn = (calculate_rnpv(wac=target_wac, gtn=0.90, cogs_per_pill=COGS_PER_PILL["Base"], share=ADOPTION_RATES["Base"], ptrs=POS, patent_years=patent_life_years, access_rate=ACCESS_RATES["Base"], df=market_data)/1e9,
                 calculate_rnpv(wac=target_wac, gtn=0.60, cogs_per_pill=COGS_PER_PILL["Base"], share=ADOPTION_RATES["Base"], ptrs=POS, patent_years=patent_life_years, access_rate=ACCESS_RATES["Base"], df=market_data)/1e9)

    swings_dict = {
        'Peak Market Share (8% - 20%)': (swing_share, swing_share[1] - swing_share[0]),
        'Base US WAC Price ($3,628 - $5,445)': (swing_wac, swing_wac[1] - swing_wac[0]),
        'Access Rate (7.4% - 15.5%)': (swing_access, swing_access[1] - swing_access[0]),
        'COGS Per Pill ($1.37 - $1.18)': (swing_cogs, swing_cogs[1] - swing_cogs[0]),
        'Clinical POS (Min/Max CI)': (swing_ptrs, swing_ptrs[1] - swing_ptrs[0]),
        'US GTN Rebate (90% - 60%)': (swing_gtn, swing_gtn[1] - swing_gtn[0])
    }

    sorted_swings = dict(sorted(swings_dict.items(), key=lambda item: item[1][1], reverse=False))

    variables = list(sorted_swings.keys())
    mins = [val[0][0] for val in sorted_swings.values()]
    maxs = [val[0][1] for val in sorted_swings.values()]

    fig2, ax2 = plt.subplots(figsize=(8, 5))
    for i, (low, high) in enumerate(zip(mins, maxs)):
        ax2.broken_barh([(low, base_b - low)], (i - 0.4, 0.8), facecolors='#E63946')
        ax2.broken_barh([(base_b, high - base_b)], (i - 0.4, 0.8), facecolors='#2A9D8F')

    ax2.set_yticks(range(len(variables)))
    ax2.set_yticklabels(variables)
    ax2.axvline(base_b, color='black', linewidth=2)
    ax2.set_xlabel("Asset Valuation ($ Billions)")
    st.pyplot(fig2)

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

    pdf = FPDF()
    pdf.add_page()

    pdf.set_font("Arial", 'B', 16)
    pdf.cell(0, 10, "Oral Insulin rNPV - Comprehensive Valuation Report", ln=True, align="C")
    pdf.ln(5)

    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, "1. Commercial & Clinical Parameters Selected:", ln=True)
    pdf.set_font("Arial", '', 10)
    pdf.cell(0, 6, f"Base US WAC Price: ${target_wac:,.0f}", ln=True)
    pdf.cell(0, 6, f"US GTN Rebate: {gtn_rebate*100:.0f}%", ln=True)
    pdf.cell(0, 6, f"Peak Global Share: {peak_market_share*100:.0f}%", ln=True)
    pdf.cell(0, 6, f"Clinical Stage: {stage_selection}", ln=True)
    pdf.cell(0, 6, f"Post-LOE Retention (biologic-adjusted): {POST_LOE_RETENTION*100:.0f}%", ln=True)
    pdf.cell(0, 6, f"Calculated Base Case rNPV: ${base_rnpv/1e9:.3f} Billion", ln=True)
    pdf.ln(5)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as f1, \
         tempfile.NamedTemporaryFile(delete=False, suffix=".png") as f2, \
         tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as fpdf_out:

         fig.savefig(f1.name, format="png", bbox_inches="tight")
         fig2.savefig(f2.name, format="png", bbox_inches="tight")

         pdf.set_font("Arial", 'B', 12)
         pdf.cell(0, 10, "2. Monte Carlo Distribution:", ln=True)
         pdf.image(f1.name, w=160)

         pdf.add_page()
         pdf.cell(0, 10, "3. Tornado Sensitivity Analysis:", ln=True)
         pdf.image(f2.name, w=160)

         pdf.add_page()
         pdf.cell(0, 10, "4. Base Case Cash Flow Log:", ln=True)
         pdf.set_font("Courier", '', 8)

         for line in base_logs.split('\n'):
             safe_line = line.encode('latin-1', 'replace').decode('latin-1')
             pdf.multi_cell(0, 4, safe_line)

         pdf.output(fpdf_out.name)

         with open(fpdf_out.name, "rb") as f:
             pdf_bytes = f.read()

    os.unlink(f1.name)
    os.unlink(f2.name)
    os.unlink(fpdf_out.name)

    st.download_button(
        label="Download Comprehensive PDF Report",
        data=pdf_bytes,
        file_name="Oral_Insulin_Valuation_Report.pdf",
        mime="application/pdf",
        use_container_width=True
    )

except ImportError:
    st.warning("⚠️ PDF generation requires the 'fpdf' library. Please add 'fpdf' to your requirements.txt file on GitHub to enable this button.")
