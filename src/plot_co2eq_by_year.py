#!/usr/bin/env python3
"""
Extract and plot cumulative CO2eq from Epoch AI dataset by year.

CO2eq calculation:
- CO2eq (kg) = power_draw (W) * training_time (h) / 1000 (kWh) * carbon_intensity (gCO2/kWh) / 1000

If power_draw is not available:
- power_draw (W) = hardware_quantity * hardware_power (kW) * 1000 * PUE * server_overhead

PUE (Power Usage Effectiveness) decays from 1.23 (2008) to 1.08 (2025)
Server overhead: 1.0 for single GPU, 1.82 for multi-GPU
"""
import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import re


def get_pue(year: int) -> float:
    """Get Power Usage Effectiveness for a given year (decays over time)."""
    if year is None:
        return 1.15  # Default
    pue_2008 = 1.23
    pue_2025 = 1.08
    years_span = 2025 - 2008
    progress = min(1.0, max(0.0, (year - 2008) / years_span))
    return pue_2008 - progress * (pue_2008 - pue_2025)


def get_server_overhead(hw_quantity: float) -> float:
    """Get server overhead based on hardware quantity."""
    if hw_quantity is None or hw_quantity <= 1:
        return 1.0
    return 1.82  # Multi-GPU setup


def extract_year(date_str: str) -> int | None:
    """Extract year from publication date string."""
    if pd.isna(date_str) or not date_str:
        return None
    match = re.search(r'(\d{4})', str(date_str))
    if match:
        return int(match.group(1))
    return None


def get_country_carbon_intensity(country_str: str, country_lookup: dict) -> float:
    """Get average carbon intensity for a country string (may contain multiple countries)."""
    if pd.isna(country_str) or not country_str:
        return 475.0  # World average
    
    countries = [c.strip() for c in str(country_str).split(',')]
    intensities = []
    
    for country in countries:
        if country in country_lookup:
            intensities.append(country_lookup[country])
    
    if intensities:
        return np.mean(intensities)
    return 475.0  # World average fallback


def load_data(db_path: str = "data/epoch.db"):
    """Load all required data from the database."""
    conn = sqlite3.connect(db_path)
    
    # Load epoch data
    epoch_df = pd.read_sql("""
        SELECT 
            id,
            Model,
            Publication_date,
            Training_power_draw__W,
            Training_time__hours,
            Training_hardware,
            Hardware_quantity,
            Country__of_organization
        FROM epoch
    """, conn)
    
    # Load country carbon intensity
    country_df = pd.read_sql("SELECT name, carbon_intensity FROM country", conn)
    country_lookup = dict(zip(country_df['name'], country_df['carbon_intensity']))
    
    # Load hardware power
    hardware_df = pd.read_sql("SELECT name, power FROM hardware", conn)
    # Power is in kW, create lookup
    hardware_lookup = {}
    for _, row in hardware_df.iterrows():
        if row['name'] and not pd.isna(row['power']):
            hardware_lookup[row['name'].lower().strip()] = row['power']  # kW
    
    conn.close()
    
    return epoch_df, country_lookup, hardware_lookup


def match_hardware_power(hw_name: str, hardware_lookup: dict) -> float | None:
    """Try to match hardware name to get power in kW."""
    if pd.isna(hw_name) or not hw_name:
        return None
    
    hw_clean = str(hw_name).lower().strip()
    
    # Exact match
    if hw_clean in hardware_lookup:
        return hardware_lookup[hw_clean]
    
    # Try substring matching
    for name, power in hardware_lookup.items():
        if name in hw_clean or hw_clean in name:
            return power
    
    # Try matching key terms (A100, H100, etc.)
    key_terms = ['a100', 'h100', 'h800', 'v100', 'tpu', 'mi300', 'mi250', 'a10']
    for term in key_terms:
        if term in hw_clean:
            for name, power in hardware_lookup.items():
                if term in name:
                    return power
    
    return None


def calculate_co2eq(row, country_lookup: dict, hardware_lookup: dict) -> float | None:
    """Calculate CO2eq for a single model."""
    year = extract_year(row['Publication_date'])
    
    # Get training time in hours
    training_time = row['Training_time__hours']
    if pd.isna(training_time) or training_time is None or training_time <= 0:
        return None
    
    # Get power draw in Watts
    power_draw = row['Training_power_draw__W']
    
    if pd.isna(power_draw) or power_draw is None or power_draw <= 0:
        # Calculate from hardware info
        hw_quantity = row['Hardware_quantity']
        hw_name = row['Training_hardware']
        
        if pd.isna(hw_quantity) or hw_quantity is None or hw_quantity <= 0:
            return None
        
        hw_power_kw = match_hardware_power(hw_name, hardware_lookup)
        if hw_power_kw is None:
            return None
        
        # Calculate power draw
        pue = get_pue(year)
        server_overhead = get_server_overhead(hw_quantity)
        power_draw = hw_quantity * hw_power_kw * 1000 * pue * server_overhead  # Watts
    
    # Get carbon intensity
    carbon_intensity = get_country_carbon_intensity(row['Country__of_organization'], country_lookup)
    
    # Calculate CO2eq
    # power_draw (W) * time (h) = Wh -> / 1000 = kWh
    # kWh * carbon_intensity (gCO2/kWh) = gCO2 -> / 1000 = kgCO2 -> / 1000 = tCO2
    energy_kwh = power_draw * training_time / 1000
    co2eq_kg = energy_kwh * carbon_intensity / 1000
    co2eq_tonnes = co2eq_kg / 1000
    
    return co2eq_tonnes


def plot_co2eq_by_year(co2_by_year: pd.DataFrame, output_path: Path):
    """Create a beautiful plot of annual CO2eq by year (English labels)."""
    plt.style.use('default')
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    years = co2_by_year['year'].values
    total_co2 = co2_by_year['total_co2eq_tonnes'].values
    n_models = co2_by_year['n_models'].values
    
    # Bar chart for annual CO2
    bars = ax.bar(years, total_co2, color='#3498db', alpha=0.8, width=0.8, edgecolor='#2980b9')
    
    # Labels
    ax.set_xlabel('Year', fontsize=12, fontweight='bold')
    ax.set_ylabel('Annual CO₂eq (tonnes)', fontsize=12, fontweight='bold')
    
    # Title
    ax.set_title('Annual CO₂eq Emissions from ML Models (Epoch AI Dataset)', 
                 fontsize=14, fontweight='bold', pad=20)
    
    # X-axis ticks
    ax.set_xticks(years)
    ax.set_xticklabels(years, rotation=45, ha='right')
    
    # Grid
    ax.grid(True, axis='y', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)
    
    # Transparent background
    fig.patch.set_alpha(0.0)
    ax.patch.set_alpha(0.0)
    
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, transparent=True, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_co2eq_stacked_by_country(co2_by_country_year: pd.DataFrame, output_path: Path):
    """Create a stacked bar chart by country (English labels)."""
    # Get top 8 countries by total CO2
    country_totals = co2_by_country_year.groupby('country')['co2eq_tonnes'].sum().sort_values(ascending=False)
    top_countries = country_totals.head(8).index.tolist()
    
    # Pivot data
    pivot_data = co2_by_country_year.pivot_table(
        index='year', 
        columns='country', 
        values='co2eq_tonnes', 
        aggfunc='sum',
        fill_value=0
    )
    
    # Group others
    others = pivot_data[[c for c in pivot_data.columns if c not in top_countries]].sum(axis=1)
    pivot_data = pivot_data[top_countries]
    pivot_data['Others'] = others
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Colors
    colors = plt.cm.tab10(np.linspace(0, 1, len(pivot_data.columns)))
    
    pivot_data.plot(kind='bar', stacked=True, ax=ax, color=colors, width=0.8)
    
    ax.set_xlabel('Year', fontsize=12, fontweight='bold')
    ax.set_ylabel('CO₂eq (tonnes)', fontsize=12, fontweight='bold')
    ax.set_title('Annual CO₂eq Emissions by Country of Origin', fontsize=14, fontweight='bold', pad=20)
    
    ax.legend(title='Country', bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=9)
    ax.set_xticklabels([int(x) for x in pivot_data.index], rotation=45, ha='right')
    
    ax.grid(True, axis='y', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)
    
    fig.patch.set_alpha(0.0)
    ax.patch.set_alpha(0.0)
    
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, transparent=True, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {output_path}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Plot CO2eq by year from Epoch AI dataset")
    parser.add_argument("--output-dir", type=str, default="results_epoch/co2eq",
                        help="Output directory for plots")
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("Loading data...")
    epoch_df, country_lookup, hardware_lookup = load_data()
    
    print(f"Total models: {len(epoch_df)}")
    
    # Calculate CO2eq for each model
    results = []
    for _, row in epoch_df.iterrows():
        year = extract_year(row['Publication_date'])
        if year is None or year < 2010:
            continue
        
        co2eq = calculate_co2eq(row, country_lookup, hardware_lookup)
        if co2eq is not None and co2eq > 0:
            # Get primary country
            country_str = row['Country__of_organization']
            if pd.isna(country_str) or not country_str:
                country = "Unknown"
            else:
                country = str(country_str).split(',')[0].strip()
            
            results.append({
                'year': year,
                'model': row['Model'],
                'co2eq_tonnes': co2eq,
                'country': country
            })
    
    results_df = pd.DataFrame(results)
    print(f"Models with calculated CO2eq: {len(results_df)}")
    
    if len(results_df) == 0:
        print("No data to plot!")
        return
    
    # Save raw data
    results_df.to_csv(output_dir / "co2eq_by_model.csv", index=False)
    print(f"Saved: {output_dir / 'co2eq_by_model.csv'}")
    
    # Aggregate by year
    co2_by_year = results_df.groupby('year').agg({
        'co2eq_tonnes': 'sum',
        'model': 'count'
    }).reset_index()
    co2_by_year.columns = ['year', 'total_co2eq_tonnes', 'n_models']
    co2_by_year['cumulative_co2eq_tonnes'] = co2_by_year['total_co2eq_tonnes'].cumsum()
    
    print("\n" + "="*60)
    print("CO2eq par année:")
    print("="*60)
    print(co2_by_year.to_string(index=False))
    print(f"\nTotal cumulé: {co2_by_year['cumulative_co2eq_tonnes'].iloc[-1]:,.0f} tonnes CO2eq")
    
    # Save aggregated data
    co2_by_year.to_csv(output_dir / "co2eq_by_year.csv", index=False)
    
    # Plot
    plot_co2eq_by_year(co2_by_year, output_dir / "co2eq_by_year.png")
    
    # Plot by country
    co2_by_country_year = results_df.groupby(['year', 'country'])['co2eq_tonnes'].sum().reset_index()
    plot_co2eq_stacked_by_country(co2_by_country_year, output_dir / "co2eq_by_country.png")
    
    # Top 10 models by CO2
    print("\n" + "="*60)
    print("Top 10 modèles par CO2eq:")
    print("="*60)
    top10 = results_df.nlargest(10, 'co2eq_tonnes')[['model', 'year', 'country', 'co2eq_tonnes']]
    top10['co2eq_tonnes'] = top10['co2eq_tonnes'].apply(lambda x: f"{x:,.1f}")
    print(top10.to_string(index=False))


if __name__ == "__main__":
    main()

