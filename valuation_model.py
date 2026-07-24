import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

print("\n" + "="*60)
print("--- INSTITUTIONAL rNPV VALUATION ENGINE (MONTE CARLO) ---")
print("="*60)

# ---------------------------------------------------------
# 1. COMMERCIAL & PRICING PARAMETERS 
# ---------------------------------------------------------
print("\n[COMMERCIAL & PRICING PARAMETERS]")
try:
    target_wac = float(input("1. Target Annual US WAC Price (USD, e.g., 4789 for Base): "))
    gtn_rebate = float(input("2. Expected US GTN Rebate % (e.g., 0.76 for 76% RAND benchmark): "))
except ValueError:
    print("\n[!] Invalid commercial input. Defaulting to baseline parameters ($4,789 WAC, 76% GTN)...")
    target_wac, gtn_rebate = 4789.0, 0.76

# The 4 Variables that shift across Scenarios
WAC_PRICES = {
    "Bear": 3628.0,
    "Base": 4789.0,
    "Bull": 5445.0
}
COGS_PER_PILL = {
    "Bear": 1.37,  
    "Base": 1.27,  
    "Bull": 1.18   
}
ADOPTION_RATES = {
    "Bear": 0.08,   
    "Base": 0.15,   
    "Bull": 0.20    
}
ACCESS_RATES = {
    "Bear": 0.074,
    "Base": 0.074,
    "Bull": 0.155
}

ANNUAL_COGS_BASE = COGS_PER_PILL["Base"] * 365.0

print(f"\n-> Calculated US Net Price Per Patient (Base): ${target_wac * (1 - gtn_rebate):,.2f}")
print(f"-> Calculated Annual COGS Per Patient (Base): ${ANNUAL_COGS_BASE:,.2f}")

# ---------------------------------------------------------
# 2. CLINICAL RISK & INTELLECTUAL PROPERTY
# ---------------------------------------------------------
print("\n[CLINICAL RISK & INTELLECTUAL PROPERTY]")

CUMULATIVE_PTRS_MAP = {
    "1": 0.196,  # Phase 1 Start
    "2": 0.241,  # Phase 2 Start
    "3": 0.516,  # Phase 3 Start
    "4": 0.850   # NDA / Pre-Launch
}

print("Current Clinical Stage:")
print("  1 = Phase 1 (First-in-Human)")
print("  2 = Phase 2 (Efficacy)")
print("  3 = Phase 3 (Pivotal/CVOT)")
print("  4 = NDA Submitted / Pre-Launch")
current_stage = input("Enter the number corresponding to the current stage (1-4): ").strip()

POS = CUMULATIVE_PTRS_MAP.get(current_stage, 0.196) 
print(f"-> Risk-Adjustment (Cumulative PTRS) locked at: {POS * 100:.1f}%")

try:
    patent_life_years = int(input("\nExpected Years of Patent Exclusivity post-launch (e.g., 10, 12, 14): "))
except ValueError:
    print("[!] Invalid entry. Defaulting to 12 years of exclusivity.")
    patent_life_years = 12

POST_LOE_RETENTION = 0.15 
print(f"-> Generic Erosion Cliff will trigger in Commercial Year {patent_life_years + 1}.")

# ---------------------------------------------------------
# 3. DYNAMIC PHASE-BY-PHASE R&D INPUTS
# ---------------------------------------------------------
print("\n[CLINICAL PIPELINE R&D PARAMETERS]")
clinical_burn_map = []
phases = ["Phase 1 (Safety)", "Phase 2 (Efficacy)", "Phase 3 (Pivotal/CVOT)", "NDA / Pre-Launch Supply"]

for phase in phases:
    print(f"\n--- {phase} ---")
    try:
        duration = int(input(f"  Duration of this phase in years: "))
        annual_burn = float(input(f"  Annual Burn Rate during this phase (in Millions USD): ")) * 1e6
        for _ in range(duration):
            clinical_burn_map.append(annual_burn)
    except ValueError:
        print(f"  [!] Invalid entry. Defaulting to 0 years for {phase}...")

years_to_launch = len(clinical_burn_map)
print(f"\n-> Total Pre-Launch Pipeline Timeline Constructed: {years_to_launch} Years.")

# ---------------------------------------------------------
# 4. GLOBAL MACRO VALUATION PARAMETERS 
# ---------------------------------------------------------
POPULATION_CAGR = 0.0147   
WACC = 0.11         
YEARS = 20          

UPTAKE_CURVE = [
    0.05, 0.335, 0.65, 0.88, 1.00,
    1.00, 1.00, 1.00, 1.00, 1.00,  
    1.00, 1.00, 1.00, 1.00, 1.00,  
    0.50, 0.25, 0.15, 0.10, 0.05   
]

# ---------------------------------------------------------
# 5. DATA CLEANING & LOADING
# ---------------------------------------------------------
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

def load_data():
    print("\nLoading macro demographic datasets from CSV...")
    try:
        df_total = pd.read_csv('Estimated total number of adults (20–79 years) with diabetes in 2024.csv')
        df_t1_all = pd.read_csv('People with type 1 diabetes (all age groups) by Country_Territory.csv')
        df_t1_youth = pd.read_csv('People with type 1 diabetes (0-19 y) by Country_Territory.csv')
    except FileNotFoundError:
        print("\n[!] WARNING: Could not find files. Using dummy global patient data for simulation purposes.")
        return pd.DataFrame({
            'Country': ['United States of America', 'Rest of World'], 
            'T2D_Population': [30000000, 470000000]
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

# ---------------------------------------------------------
# 6. CORE VALUATION ENGINE (DCF & rNPV)
# ---------------------------------------------------------
def calculate_rnpv(wac, gtn, cogs_per_pill, pos, capture_rate, access_rate, patent_life, df):
    annual_cogs = cogs_per_pill * 365.0
    us_net_price = wac * (1 - gtn)
    
    row_wac = wac * 0.10
    row_net_price = row_wac * (1 - 0.20)
    
    if row_net_price < annual_cogs:
        row_net_price = annual_cogs * 1.15

    total_rnpv = 0
    actual_year = 1
    
    for burn in clinical_burn_map:
        dcf_yr = (-burn) / ((1 + WACC) ** actual_year)
        total_rnpv += dcf_yr * pos
        actual_year += 1

    us_base_pool = df[df['Country'] == 'United States of America']['T2D_Population'].sum()
    row_base_pool = df[df['Country'] != 'United States of America']['T2D_Population'].sum()

    for yr in range(1, YEARS + 1):
        revenue_factor = 1.0 if yr <= patent_life else POST_LOE_RETENTION
        
        us_pool = us_base_pool * ((1 + POPULATION_CAGR) ** yr)
        row_pool = row_base_pool * ((1 + POPULATION_CAGR) ** yr)
        
        captured_us = us_pool * access_rate * capture_rate * UPTAKE_CURVE[yr - 1]
        captured_row = row_pool * access_rate * capture_rate * UPTAKE_CURVE[yr - 1]
        
        gross_us = captured_us * us_net_price * revenue_factor
        gross_row = captured_row * row_net_price * revenue_factor
        total_revenue = gross_us + gross_row
        
        total_cogs = (captured_us + captured_row) * annual_cogs * revenue_factor
        
        cash_flow_yr = total_revenue - total_cogs
        dcf_yr = cash_flow_yr / ((1 + WACC) ** actual_year)
        total_rnpv += dcf_yr * pos
        actual_year += 1
        
    return total_rnpv

def print_valuation(df, active_scenario="Base"):
    print(f"\n" + "="*50)
    print(f"--- RUNNING VALUATION SCENARIO: {active_scenario.upper()} ---")
    print("="*50)
    
    access_rate = ACCESS_RATES[active_scenario]
    capture_rate = ADOPTION_RATES[active_scenario]
    cogs_per_pill = COGS_PER_PILL[active_scenario]
    wac = WAC_PRICES[active_scenario]
    
    annual_cogs = cogs_per_pill * 365.0
    us_net_price = wac * (1 - gtn_rebate)
    row_net_price = max((wac * 0.10) * 0.80, annual_cogs * 1.15)
    
    total_rnpv = 0
    actual_year = 1
    
    if len(clinical_burn_map) > 0:
        print("\n  [1] Processing Clinical Trial Pipeline Cash Flows...")
        for pre_yr, burn in enumerate(clinical_burn_map, 1):
            dcf_yr = (-burn) / ((1 + WACC) ** actual_year)
            rnpv_yr = dcf_yr * POS 
            total_rnpv += rnpv_yr
            actual_year += 1
            print(f"    R&D Yr {pre_yr:02d} | Capital Burn: ${burn/1e6:5.1f}M | rNPV Impact: ${rnpv_yr/1e6:7.2f}M")

    us_base_pool = df[df['Country'] == 'United States of America']['T2D_Population'].sum()
    row_base_pool = df[df['Country'] != 'United States of America']['T2D_Population'].sum()

    print(f"\n  [2] Processing 20-Year Commercialization Window (LOE Cliff at Yr {patent_life_years + 1})...")
    for yr in range(1, YEARS + 1):
        revenue_factor = 1.0 if yr <= patent_life_years else POST_LOE_RETENTION
        
        captured_us = us_base_pool * ((1 + POPULATION_CAGR) ** yr) * access_rate * capture_rate * UPTAKE_CURVE[yr - 1]
        captured_row = row_base_pool * ((1 + POPULATION_CAGR) ** yr) * access_rate * capture_rate * UPTAKE_CURVE[yr - 1]
        
        total_revenue = (captured_us * us_net_price + captured_row * row_net_price) * revenue_factor
        total_cogs = (captured_us + captured_row) * annual_cogs * revenue_factor
        
        dcf_yr = (total_revenue - total_cogs) / ((1 + WACC) ** actual_year)
        rnpv_yr = dcf_yr * POS
        
        total_rnpv += rnpv_yr
        actual_year += 1
        
        if yr % 5 == 0 or yr == 1 or yr == patent_life_years + 1:
            cliff_marker = " <-- [GENERIC CLIFF EXECUTED]" if yr == patent_life_years + 1 else ""
            print(f"    Com. Yr {yr:02d} | Net Revenue: ${total_revenue / 1e9:5.2f}B | rNPV: ${rnpv_yr / 1e6:7.2f}M{cliff_marker}")

    unrisked_npv = total_rnpv / POS

    print("\n" + "-" * 55)
    print(f"  FINAL {active_scenario.upper()} ASSET rNPV:     ${total_rnpv / 1e9:7.3f} BILLION")
    print(f"  UN-RISKED COMMERCIAL NPV: ${unrisked_npv / 1e9:7.3f} BILLION")
    print("-" * 55)
    return total_rnpv

# ---------------------------------------------------------
# 7. MONTE CARLO & TORNADO DIAGRAM GENERATOR
# ---------------------------------------------------------
def run_monte_carlo(df):
    print("\n[+] INITIALIZING MONTE CARLO SIMULATION (10,000 ITERATIONS)...")
    
    gtn_dist = np.random.triangular(0.60, gtn_rebate, 0.90, 10000)
    wac_dist = np.random.triangular(WAC_PRICES["Bear"], WAC_PRICES["Base"], WAC_PRICES["Bull"], 10000)
    share_dist = np.random.triangular(ADOPTION_RATES["Bear"], ADOPTION_RATES["Base"], ADOPTION_RATES["Bull"], 10000)
    cogs_dist = np.random.triangular(COGS_PER_PILL["Bull"], COGS_PER_PILL["Base"], COGS_PER_PILL["Bear"], 10000)
    access_dist = np.random.triangular(ACCESS_RATES["Bear"], ACCESS_RATES["Base"], ACCESS_RATES["Bull"], 10000)
    
    pos_min = max(0.01, POS - 0.014)
    pos_max = min(1.00, POS + 0.014)
    pos_dist = np.random.triangular(pos_min, POS, pos_max, 10000)
    
    results = []
    for i in range(10000):
        val = calculate_rnpv(wac_dist[i], gtn_dist[i], cogs_dist[i], pos_dist[i], share_dist[i], access_dist[i], patent_life_years, df)
        results.append(val / 1e9) 
        
    results = np.array(results)
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    axes[0].hist(results, bins=50, color='#1e3a8a', alpha=0.8, edgecolor='white')
    axes[0].axvline(np.percentile(results, 5), color='#dc2626', linestyle='dashed', linewidth=2, label='5th Percentile')
    axes[0].axvline(np.percentile(results, 95), color='#16a34a', linestyle='dashed', linewidth=2, label='95th Percentile')
    axes[0].axvline(np.mean(results), color='#eab308', linestyle='solid', linewidth=3, label='Mean Valuation')
    axes[0].set_title('Monte Carlo Simulation: rNPV Probability Distribution', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Risk-Adjusted Net Present Value ($ Billions)', fontsize=10)
    axes[0].set_ylabel('Frequency (10,000 Iterations)', fontsize=10)
    axes[0].grid(axis='y', alpha=0.3)
    axes[0].legend()

    print("[+] SIMULATION COMPLETE. CALCULATING SENSITIVITY TORNADO...")
    
    base_rnpv = calculate_rnpv(target_wac, gtn_rebate, COGS_PER_PILL["Base"], POS, ADOPTION_RATES["Base"], ACCESS_RATES["Base"], patent_life_years, df) / 1e9
                 
    wac_swing = (calculate_rnpv(WAC_PRICES["Bear"], gtn_rebate, COGS_PER_PILL["Base"], POS, ADOPTION_RATES["Base"], ACCESS_RATES["Base"], patent_life_years, df) / 1e9,
                 calculate_rnpv(WAC_PRICES["Bull"], gtn_rebate, COGS_PER_PILL["Base"], POS, ADOPTION_RATES["Base"], ACCESS_RATES["Base"], patent_life_years, df) / 1e9)
                 
    share_swing = (calculate_rnpv(target_wac, gtn_rebate, COGS_PER_PILL["Base"], POS, ADOPTION_RATES["Bear"], ACCESS_RATES["Base"], patent_life_years, df) / 1e9,
                   calculate_rnpv(target_wac, gtn_rebate, COGS_PER_PILL["Base"], POS, ADOPTION_RATES["Bull"], ACCESS_RATES["Base"], patent_life_years, df) / 1e9)

    access_swing = (calculate_rnpv(target_wac, gtn_rebate, COGS_PER_PILL["Base"], POS, ADOPTION_RATES["Base"], ACCESS_RATES["Bear"], patent_life_years, df) / 1e9,
                    calculate_rnpv(target_wac, gtn_rebate, COGS_PER_PILL["Base"], POS, ADOPTION_RATES["Base"], ACCESS_RATES["Bull"], patent_life_years, df) / 1e9)

    cogs_swing = (calculate_rnpv(target_wac, gtn_rebate, COGS_PER_PILL["Bear"], POS, ADOPTION_RATES["Base"], ACCESS_RATES["Base"], patent_life_years, df) / 1e9,
                  calculate_rnpv(target_wac, gtn_rebate, COGS_PER_PILL["Bull"], POS, ADOPTION_RATES["Base"], ACCESS_RATES["Base"], patent_life_years, df) / 1e9)

    pos_swing = (calculate_rnpv(target_wac, gtn_rebate, COGS_PER_PILL["Base"], pos_min, ADOPTION_RATES["Base"], ACCESS_RATES["Base"], patent_life_years, df) / 1e9,
                 calculate_rnpv(target_wac, gtn_rebate, COGS_PER_PILL["Base"], pos_max, ADOPTION_RATES["Base"], ACCESS_RATES["Base"], patent_life_years, df) / 1e9)

    # Added GTN to the Tornado to show investors its extreme sensitivity, even though the base scenarios fix it at 76%
    gtn_swing = (calculate_rnpv(target_wac, 0.90, COGS_PER_PILL["Base"], POS, ADOPTION_RATES["Base"], ACCESS_RATES["Base"], patent_life_years, df) / 1e9,
                 calculate_rnpv(target_wac, 0.60, COGS_PER_PILL["Base"], POS, ADOPTION_RATES["Base"], ACCESS_RATES["Base"], patent_life_years, df) / 1e9)

    swings = {
        'Peak Market Share (8% to 20%)': (share_swing, share_swing[1] - share_swing[0]),
        'Base WAC Price ($3,628 to $5,445)': (wac_swing, wac_swing[1] - wac_swing[0]),
        'Access Rate (7.4% to 15.5%)': (access_swing, access_swing[1] - access_swing[0]),
        'COGS Per Pill ($1.37 to $1.18)': (cogs_swing, cogs_swing[1] - cogs_swing[0]),
        'Clinical POS (Min/Max Conf. Interval)': (pos_swing, pos_swing[1] - pos_swing[0]),
        'US GTN Rebate (90% to 60%)': (gtn_swing, gtn_swing[1] - gtn_swing[0])
    }
    
    sorted_swings = dict(sorted(swings.items(), key=lambda item: item[1][1], reverse=False))
    
    labels = list(sorted_swings.keys())
    mins = [val[0][0] for val in sorted_swings.values()]
    maxs = [val[0][1] for val in sorted_swings.values()]
    
    y_pos = np.arange(len(labels))
    
    axes[1].barh(y_pos, [base_rnpv - m for m in mins], left=mins, color='#ef4444', label='Bear Case Impact')
    axes[1].barh(y_pos, [m - base_rnpv for m in maxs], left=base_rnpv, color='#22c55e', label='Bull Case Impact')
    
    axes[1].axvline(base_rnpv, color='black', linestyle='solid', linewidth=1.5)
    axes[1].set_yticks(y_pos)
    axes[1].set_yticklabels(labels, fontsize=9)
    axes[1].set_title('Tornado Analysis: rNPV Sensitivity ($ Billions)', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Asset Valuation ($ Billions)', fontsize=10)
    axes[1].legend(loc='lower right')
    
    plt.tight_layout()
    plt.show()

# ---------------------------------------------------------
# 8. MAIN EXECUTION
# ---------------------------------------------------------
if __name__ == "__main__":
    market_data = load_data()
    
    print_valuation(market_data, active_scenario="Bear")
    print_valuation(market_data, active_scenario="Base")
    print_valuation(market_data, active_scenario="Bull")
    
    run_monte_carlo(market_data)
