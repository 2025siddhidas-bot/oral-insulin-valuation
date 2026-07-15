import pandas as pd
import numpy as np
import os

print("--- ORAL INSULIN rNPV VALUATION ENGINE ---")
print("Initializing Market Parameters...\n")

# ---------------------------------------------------------
# 1. GLOBAL VALUATION PARAMETERS (Anchored by Proxy Data)
# ---------------------------------------------------------

# Clinical Need / Insulin-Dependency Rate (The Lancet Microsimulation)
ACCESS_RATE_BASE = 0.07   # Base access to insulin (Updated to 7%)
ACCESS_RATE_BULL = 0.155  # Universal clinical need (15.5%)

# Market Capture Scenarios (IQVIA NBRx & Corporate Financials)
ADOPTION_RATES = {
    "Bear": 0.08,   # 8% - Biosimilar analog ceiling
    "Base": 0.15,   # 15% - Rybelsus audited revenue baseline
    "Bull": 0.33    # 33% - Oral Wegovy launch velocity
}

# Financial Mechanics
POS = 0.135         # 13.5% (Yamaguchi: Early-stage biologic agonist)
COGS = 0.33         # 33% (Bottom-up bioprocess estimate for oral peptides)
WACC = 0.11         # 11% (Damodaran Biotech + 2% CSRP)
YEARS = 20          # 20-year patent/exclusivity lifecycle

# Standard Biotech Launch Curve (S-Curve to Peak Sales)
UPTAKE_CURVE = [
    0.10, 0.35, 0.65, 0.85, 1.00,  # Years 1-5 (Launch Ramp)
    1.00, 1.00, 1.00, 1.00, 1.00,  # Years 6-10 (Peak Sales)
    1.00, 1.00, 1.00, 1.00, 1.00,  # Years 11-15 (Peak Sales)
    0.50, 0.25, 0.15, 0.10, 0.05   # Years 16-20 (Patent Expiry / Generic Erosion)
]

# ---------------------------------------------------------
# 2. DATA CLEANING & LOADING
# ---------------------------------------------------------

def clean_df(df, target_val_name):
    """
    Helper function to dynamically handle messy CSV headers and 
    convert string numbers with commas into pure floats for math.
    """
    # Dynamically find the country column whether it's named 'Location' or 'Country_Territory'
    country_col = 'Location' if 'Location' in df.columns else 'Country_Territory'
    if country_col not in df.columns and 'Country' in df.columns:
        country_col = 'Country'
        
    # Dynamically find the value column
    val_col = 'Value' if 'Value' in df.columns else 'Total'
    if val_col not in df.columns and 'Expenditure' in df.columns:
        val_col = 'Expenditure'
        
    df = df.rename(columns={country_col: 'Country', val_col: target_val_name})
    
    # Strip invisible trailing/leading spaces from country names so they merge perfectly
    if 'Country' in df.columns:
        df['Country'] = df['Country'].astype(str).str.strip()
    
    # Clean the numbers: remove commas and convert to float so Pandas can do math
    if df[target_val_name].dtype == object:
        df[target_val_name] = df[target_val_name].astype(str).str.replace(',', '', regex=True)
        
    df[target_val_name] = pd.to_numeric(df[target_val_name], errors='coerce').fillna(0)
        
    return df[['Country', target_val_name]]

def load_data():
    print("Loading macro datasets from CSV...")
    
    try:
        # Load the raw CSV files
        df_total = pd.read_csv('Estimated total number of adults (20–79 years) with diabetes in 2024.csv')
        df_t1_all = pd.read_csv('People with type 1 diabetes (all age groups) by Country_Territory.csv')
        df_t1_youth = pd.read_csv('People with type 1 diabetes (0-19 y) by Country_Territory.csv')
        df_expenditure = pd.read_csv('Diabetes-related health expenditure per person, USD by Country_Territory.csv')
    except FileNotFoundError as e:
        print(f"\n[!] ERROR: Could not find file: {e.filename}")
        print("Please ensure your terminal is opened directly inside the 'oral-insulin-valuation' folder.")
        exit()
    
    # Standardize headers and clean numeric data using our helper function
    df_total = clean_df(df_total, 'Total_Diabetics_20_79')
    df_t1_all = clean_df(df_t1_all, 'Type_1_All_Ages')
    df_t1_youth = clean_df(df_t1_youth, 'Type_1_0_19')
    df_expenditure = clean_df(df_expenditure, 'Expenditure_Per_Person')
    
    # --- THE CRITICAL FIX: SCALING MATCH ---
    # Total Diabetics is reported in 1,000s (e.g., 1932.8 = 1,932,800)
    df_total['Total_Diabetics_20_79'] = df_total['Total_Diabetics_20_79'] * 1000
    
    # Type 1 Diabetics (All Ages & 0-19) are reported as ABSOLUTE exact numbers (e.g., 13600 = 13,600)
    # Expenditure is also exact USD. We DO NOT multiply these by 1000!

    # Merge them together into one master dataframe
    master_df = pd.merge(df_total, df_t1_all, on='Country', how='outer')
    master_df = pd.merge(master_df, df_t1_youth, on='Country', how='outer')
    master_df = pd.merge(master_df, df_expenditure, on='Country', how='outer')
    
    # Clean up missing data
    master_df = master_df.fillna(0)
    
    # The Age-Matched Subtraction Logic
    # Step A: Isolate Type 1 Adults (Subtracting 0-19 from All Ages)
    master_df['Type_1_Adults'] = master_df['Type_1_All_Ages'] - master_df['Type_1_0_19']
    master_df['Type_1_Adults'] = master_df['Type_1_Adults'].clip(lower=0) 
    
    # Step B: Calculate accurate Type 2 Adult Population
    master_df['T2D_Population'] = master_df['Total_Diabetics_20_79'] - master_df['Type_1_Adults']
    master_df['T2D_Population'] = master_df['T2D_Population'].clip(lower=0) 
    
    print("Data successfully loaded, age-matched, and merged.")
    
    # DIAGNOSTIC CHECK: Prove the data is fixed before running math
    global_t2d = master_df['T2D_Population'].sum()
    print(f"[!] DIAGNOSTIC: Total Global Adult T2D Pool calculated at {global_t2d:,.0f} patients.")
    
    return master_df

# ---------------------------------------------------------
# 3. VALUATION ENGINE
# ---------------------------------------------------------

def run_valuation(df, active_scenario="Base", use_bull_clinical_need=False):
    """
    Runs the 20-year rNPV loop based on selected scenarios.
    """
    print(f"\n--- RUNNING SCENARIO: {active_scenario.upper()} ---")
    
    # Step 1: Apply the Lancet Clinical Dependency Rate (Base 7% vs Bull 15.5%)
    clinical_rate = ACCESS_RATE_BULL if use_bull_clinical_need else ACCESS_RATE_BASE
    df['Target_Patients'] = df['T2D_Population'] * clinical_rate
    
    # Step 2: Apply the IQVIA / Novo Nordisk Market Capture Rate (8% / 15% / 33%)
    capture_rate = ADOPTION_RATES[active_scenario]
    df['Peak_Captured_Patients'] = df['Target_Patients'] * capture_rate
    
    # Step 3: Calculate Peak Gross Revenue per country
    df['Peak_Gross_Revenue'] = df['Peak_Captured_Patients'] * df['Expenditure_Per_Person']
    
    total_peak_revenue = df['Peak_Gross_Revenue'].sum()
    print(f"Total Peak Annual Gross Revenue (Undiscounted): ${total_peak_revenue / 1e9:.2f} Billion")
    
    # Step 4: The 20-Year Discounted Cash Flow (DCF) & rNPV Loop
    total_rnpv = 0
    
    for year in range(1, YEARS + 1):
        # Apply the launch uptake multiplier for this specific year
        uptake_multiplier = UPTAKE_CURVE[year - 1]
        
        # Gross Revenue for the year
        gross_revenue_yr = total_peak_revenue * uptake_multiplier
        
        # Deduct COGS (33%) to get Operating Cash Flow
        cash_flow_yr = gross_revenue_yr * (1 - COGS)
        
        # Discount to Present Value using WACC (11%)
        discount_factor = (1 + WACC) ** year
        dcf_yr = cash_flow_yr / discount_factor
        
        # Risk-Adjust using Probability of Success (13.5%)
        rnpv_yr = dcf_yr * POS
        
        total_rnpv += rnpv_yr
        
        # A snapshot of every 5 years
        if year % 5 == 0 or year == 1:
            print(f"  Year {year:02d} | Uptake: {uptake_multiplier*100:3.0f}% | rNPV Contrib: ${rnpv_yr / 1e6:6.2f} M")

    print("-" * 50)
    print(f"FINAL PROJECT rNPV ({active_scenario} Case): ${total_rnpv / 1e9:.2f} BILLION")
    print("-" * 50)
    return total_rnpv

# ---------------------------------------------------------
# 4. MAIN EXECUTION BLOCK
# ---------------------------------------------------------
if __name__ == "__main__":
    
    # Load dataset
    market_data = load_data()
    
    # Run the valuation for all three scenarios
    run_valuation(market_data, active_scenario="Bear", use_bull_clinical_need=False)
    run_valuation(market_data, active_scenario="Base", use_bull_clinical_need=False)
    
    # For the Bull case, we use both the 33% capture rate AND the Bull Clinical Need (15.5%)
    run_valuation(market_data, active_scenario="Bull", use_bull_clinical_need=True)