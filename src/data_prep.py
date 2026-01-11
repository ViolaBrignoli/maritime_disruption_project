"""
MARITIME DISRUPTION PROJECT - Data Preparation Pipeline

Reproducible, executable script for data processing and cleaning.
It fixes NameErrors, KeyErrors, and ensures robust path finding for all data.

Reads: data/raw/
Writes: data/processed/.
"""

import pandas as pd
import numpy as np
import warnings
import re
import gc
from pathlib import Path

# Configuration and Paths

ROOT_DIR = Path(__file__).resolve().parent.parent 
RAW_DIR = ROOT_DIR / 'data' / 'raw'

PROCESSED_DIR = ROOT_DIR / 'data' / 'processed' 
PROCESSED_DIR.mkdir(parents=True, exist_ok=True) 

# Helper Functions 

def clean_column_names(df):
    """Standardize column names to snake_case."""
    df.columns = df.columns.str.strip().str.lower()
    df.columns = df.columns.str.replace(r'[^a-z0-9_]+', '', regex=True)
    df.columns = df.columns.str.replace(r'_+', '_', regex=True).str.strip('_')
    return df

def safe_to_numeric(series, dtype=np.float32, round_digits=3):
    """Convert a Series to a numeric dtype, coercing non-numeric values to 0 and downcasting."""
    s = pd.to_numeric(series, errors='coerce').fillna(0)
    if round_digits is not None:
        s = s.round(round_digits)
    return s.astype(dtype)

def clean_dates_to_iso(series):
    """Robustly parse and format date strings to YYYY-MM-DD."""
    parsed = pd.to_datetime(series, errors='coerce', utc=True)
    out = parsed.dt.strftime('%Y-%m-%d')
    return out.replace({'NaT': np.nan, 'None': np.nan})

def normalize_time_string(s):
    """Return NaN for null/unknown inputs; parse strings like '1:23pm' into 24-hour 'HH:MM' format and otherwise return the stripped original string."""
    if pd.isna(s) or str(s).strip().lower() in ('nan', 'unknown'):
        return np.nan
    st = str(s).strip()
    m2 = re.search(r'(\d{1,2}):(\d{2})([ap]m)', st, flags=re.IGNORECASE)
    if m2:
        hh = int(m2.group(1)); mm = int(m2.group(2)); ap = m2.group(3).lower()
        if 'p' in ap and hh < 12: hh += 12
        if 'a' in ap and hh == 12: hh = 0
        return f"{hh:02d}:{mm:02d}"
    return st

def parse_coordinates_unlocode(coord_str):
    """Parse LOCODE coordinate format to lat/lon."""
    if not isinstance(coord_str, str) or coord_str.upper() == 'UNKNOWN':
        return np.nan, np.nan
    try:
        coord_str = coord_str.replace(' ', '')
        match = re.match(r'(\d{4}[NS])(\d{5}[EW])', coord_str)
        if match:
            lat_str, lon_str = match.groups()
            lat_val = float(lat_str[:-1]) / 100
            if lat_str[-1] == 'S': lat_val *= -1
            lon_val = float(lon_str[:-1]) / 100
            if lon_str[-1] == 'W': lon_val *= -1
            return round(lat_val, 6), round(lon_val, 6)
        return np.nan, np.nan
    except:
        return np.nan, np.nan


# Data Processing Functions

def process_unlocode_ports():
    """
    Dataset: UN/LOCODE Ports; Node coordinates and names.
    """
    print("- Processing UN/LOCODE Ports...")
    raw_path = RAW_DIR / 'unlocode_ports/unlocode_ports.csv'
    if not raw_path.exists():
        print(f"  [ERROR] Raw file not found: {raw_path}. Skipping.")
        return
        
    df = pd.read_csv(raw_path, low_memory=False)
    df = clean_column_names(df)
    
    # Drop and rename 
    df = df.drop(columns=['iata', 'change', 'remarks'], errors='ignore')
    df = df.rename(columns={
        'country': 'country_code', 'location': 'location_code', 'namewodiacritics': 'port_name_clean',
        'name': 'port_name', 'subdivision': 'region_code', 'status': 'port_status',
        'function': 'port_function_code', 'date': 'date_updated', 'coordinates': 'raw_coordinates'
    })
    
    df = df.dropna(subset=['location_code', 'port_name'])
    df = df.fillna({'country_code': 'Unknown', 'region_code': 'Unknown', 'port_status': 'Unknown', 
                    'date_updated': 'Unknown', 'raw_coordinates': 'Unknown'})

    df[['lat', 'lon']] = df['raw_coordinates'].apply(lambda x: pd.Series(parse_coordinates_unlocode(x)))
    df = df.dropna(subset=['lat', 'lon'])
    
    df.to_csv(PROCESSED_DIR / 'df_UNLOCODE_Ports_standardized.csv', index=False)
    print(f"  ✓ Saved df_UNLOCODE_Ports_standardized.csv (Shape: {df.shape})")
    del df; gc.collect()


def process_imf_ports():
    """
    Dataset: IMF Ports Master List; Node features (vessel calls, industry).
    """
    print("- Processing IMF Ports Master List...")
    raw_path = RAW_DIR / 'imf_portwatch/ports_imf/ports_imf.csv'
    if not raw_path.exists():
        print(f"  [ERROR] Raw file not found: {raw_path}. Skipping.")
        return
        
    df = pd.read_csv(raw_path, low_memory=False)
    df = clean_column_names(df)
    
    for col in ['industry_top1', 'industry_top2', 'industry_top3']:
        df[col] = df[col].fillna('Unknown')
    
    df = df.drop(columns=['locode', 'objectid', 'pageid', 'countrynoaccents', 'x', 'y'], errors='ignore')
    
    vessel_cols = [c for c in df.columns if c.startswith('vessel_count')]
    share_cols = [c for c in df.columns if c.startswith('share_country')]
    
    for c in vessel_cols: 
        df[c] = safe_to_numeric(df[c], dtype=np.int32, round_digits=None)
    for c in share_cols: 
        df[c] = safe_to_numeric(df[c], dtype=np.float32, round_digits=3)

    df.to_csv(PROCESSED_DIR / 'df_Ports_IMF_standardized.csv', index=False)
    print(f"  ✓ Saved df_Ports_IMF_standardized.csv (Shape: {df.shape})")
    del df; gc.collect()


def process_imf_chokepoints_master():
    """
    Dataset: IMF Chokepoints Master List; Chokepoint node metrics.
    """
    print("- Processing IMF Chokepoints Master List...")
    raw_path = RAW_DIR / 'imf_portwatch/chokepoints_imf/Chokepoints.csv'
    if not raw_path.exists():
        print(f"  [ERROR] Raw file not found: {raw_path}. Skipping.")
        return
        
    df = pd.read_csv(raw_path, low_memory=False)
    df = clean_column_names(df)
    
    for col in ['industry_top1', 'industry_top2', 'industry_top3']:
        df[col] = df[col].fillna('Unknown')
    
    df = df.rename(columns={'portid': 'port_id', 'portname': 'port_name', 'iso3': 'iso_code'})
    
    df = df.drop(columns=[c for c in df.columns if c in ['country', 'iso_code', 'continent', 'locode', 'objectid', 'pageid', 'countrynoaccents', 'x', 'y', 'share_country_maritime_import', 'share_country_maritime_export']], errors='ignore')
    
    vessel_cols = [c for c in df.columns if c.startswith('vessel_count')]
    for c in vessel_cols: 
        df[c] = safe_to_numeric(df[c], dtype=np.int32, round_digits=None)
    
    df.to_csv(PROCESSED_DIR / 'df_Chokepoints_standardized.csv', index=False)
    print(f"  ✓ Saved df_Chokepoints_standardized.csv (Shape: {df.shape})")
    del df; gc.collect()


def process_imf_disruptions_db():
    """
    Dataset: IMF Master Disruption Database; Provides event labels.
    """
    print("- Processing IMF Disruptions Database...")
    raw_path = RAW_DIR / 'imf_portwatch/disruptions_imf/portwatch_disruptions_database_-3602226124776604501.csv'
    
    if not raw_path.exists():
        print(f"  [ERROR] Raw file not found: {raw_path}. Skipping.")
        return

    df = pd.read_csv(raw_path, low_memory=False)
    df = clean_column_names(df) 
    
    # Rename using clean column names 
    df = df.rename(columns={
        'eventid': 'event_id', 'eventtype': 'event_type', 'eventname': 'event_name', 
        'fromdate': 'from_date', 'todate': 'to_date', 'objectid': 'object_id',
        'lat': 'latitude', 'long': 'longitude', 
        'severitytext': 'severity_text', 'editdate': 'edit_date_raw'
    }, errors='ignore')
    
    # Standardize date columns 
    for col in ['from_date', 'to_date', 'edit_date_raw']: 
        if col in df.columns:
            df[col] = clean_dates_to_iso(df[col])
    
    # Fill remaining NaNs with 'Unknown' or 0
    string_cols = ['country', 'severity_text', 'affectedports', 'affectedpopulation', 'pageid']
    for col in string_cols:
        if col in df.columns:
             df[col] = df[col].fillna('Unknown').astype(str)
        
    df = df.drop(columns=[c for c in df.columns if c.startswith('shape__')], errors='ignore')

    df.to_csv(PROCESSED_DIR / 'df_Portwatch_Disruptions_Database_standardized.csv', index=False)
    print(f"  ✓ Saved df_Portwatch_Disruptions_Database_standardized.csv (Shape: {df.shape})")
    del df; gc.collect()


def process_imf_disruptions_with_ports():
    """
    Dataset: IMF Disruptions with Ports; Links event features to specific affected port nodes.
    """
    print("- Processing IMF Disruptions with Ports...")
    raw_path = RAW_DIR / 'imf_portwatch/disruptions_with_ports/disruptions_with_ports.csv'
    if not raw_path.exists():
        print(f"  [ERROR] Raw file not found: {raw_path}. Skipping.")
        return
        
    df = pd.read_csv(raw_path, low_memory=False)
    df = clean_column_names(df)
    
    # Rename 
    df = df.rename(columns={
        'portid': 'port_id', 'portname': 'port_name', 'long': 'lon', 
        'eventid': 'event_id', 'eventname': 'event_name', 
        'fromdate': 'from_date', 'todate': 'to_date', 'objectid': 'object_id'
    })
    
    # Standardize date columns
    for col in ['from_date', 'to_date']:
        if col in df.columns:
             df[col] = clean_dates_to_iso(df[col])
        
    # Fill NaNs and standardize numerics
    df['country'] = df['country'].fillna('Unknown').astype(str)
    df['lat'] = safe_to_numeric(df['lat'], round_digits=6)
    df['lon'] = safe_to_numeric(df['lon'], round_digits=6)
    df['distance_km'] = safe_to_numeric(df['distance_km'], round_digits=3)

    df.to_csv(PROCESSED_DIR / 'df_Disruptions_With_Ports_standardized.csv', index=False)
    print(f"  ✓ Saved df_Disruptions_With_Ports_standardized.csv (Shape: {df.shape})")
    del df; gc.collect()


def process_unctad_lsci():
    """
    Dataset: UNCTAD LSCI (Country Level); Country-level connectivity features.
    """
    print("- Processing UNCTAD LSCI (Country Level)...")
    raw_path = RAW_DIR / 'unctad/lsci/US.LSCI_20251119_130938.csv'
    if not raw_path.exists():
        print(f"  [ERROR] Raw file not found: {raw_path}. Skipping.")
        return

    df = pd.read_csv(raw_path, low_memory=False)
    df = clean_column_names(df)
    
    df = df.drop(columns=[c for c in df.columns if c.endswith(('footnote', 'missingvalue'))], errors='ignore')
    
    df = df.rename(columns={
        'economy_label': 'country_name',
        'quarter_label': 'quarter_date',
        'index_average_q1_2023_100_value': 'lsci_index'
    })
    
    df['year'] = df['quarter_date'].str.extract(r'(\d{4})').astype('Int32')
    df['quarter'] = df['quarter_date'].str.extract(r'Q(\d)').astype('Int32')
    df = df[df['year'].between(2015, 2024)]
    
    df['lsci_index'] = safe_to_numeric(df['lsci_index'], round_digits=3)
    
    df.to_csv(PROCESSED_DIR / 'df_Liner_Shipping_Connectivity_standardized.csv', index=False)
    print(f"  ✓ Saved df_Liner_Shipping_Connectivity_standardized.csv (Shape: {df.shape})")
    del df; gc.collect()


def process_unctad_plsci():
    """
    Dataset: UNCTAD PLSCI (Port Level); Time-series connectivity features for each port node.
    """
    print("- Processing UNCTAD PLSCI (Port Level)...")
    raw_path = RAW_DIR / 'unctad/plsci/US.PLSCI_20251115_142158.csv'
    if not raw_path.exists():
        print(f"  [ERROR] Raw file not found: {raw_path}. Skipping.")
        return

    df = pd.read_csv(raw_path, low_memory=False)
    df = clean_column_names(df)
    
    df = df.drop(columns=[c for c in df.columns if c.endswith(('footnote', 'missingvalue'))], errors='ignore')
    
    df = df.rename(columns={
        'port_label': 'port_name',
        'quarter_label': 'quarter_date',
        'index_average_q1_2023_100_value': 'plsci_index'
    })
    
    df['year'] = df['quarter_date'].str.extract(r'(\d{4})').astype('Int32')
    df['quarter'] = df['quarter_date'].str.extract(r'Q(\d)').astype('Int32')
    df = df[df['year'].between(2015, 2024)]
    
    df['plsci_index'] = safe_to_numeric(df['plsci_index'], round_digits=3)

    df.to_csv(PROCESSED_DIR / 'df_Port_Level_Shipping_Connectivity_standardized.csv', index=False)
    print(f"  ✓ Saved df_Port_Level_Shipping_Connectivity_standardized.csv (Shape: {df.shape})")
    del df; gc.collect()


def process_unctad_bilateral_connectivity():
    """
    Dataset: UNCTAD Bilateral Connectivity Index (Combined); Time-series **weighted edges**.
    """
    print("- Processing UNCTAD Bilateral Connectivity (Combined)...")
    
    dataframes = []
    # Mapping based on the file naming convention 
    date_mapping = {
        '140113': '2015 Q1', '140129': '2015 Q2', '140144': '2015 Q3', '140153': '2015 Q4',
        '140206': '2016 Q1', '140216': '2016 Q2', '140225': '2016 Q3', '140232': '2016 Q4',
        '140306': '2017 Q1', '140315': '2017 Q2', '140326': '2017 Q3', '140334': '2017 Q4',
        '140343': '2018 Q1', '140353': '2018 Q2', '140402': '2018 Q3', '140410': '2018 Q4',
        '140417': '2019 Q1', '140427': '2019 Q2', '140435': '2019 Q3', '140443': '2019 Q4',
        '140457': '2020 Q1', '140506': '2020 Q2', '140514': '2020 Q3', '140521': '2020 Q4',
        '140528': '2021 Q1', '140535': '2021 Q2', '140553': '2021 Q3', '140600': '2021 Q4',
    }
    
    base_dir = RAW_DIR / 'unctad/linear_shipping_bilateral_connectivity_index'
    if not base_dir.is_dir():
        print(f"  [ERROR] Raw directory not found: {base_dir}. Skipping.")
        return

    for file_path in base_dir.glob('US.LSBCI_*.csv'):
        df_temp = pd.read_csv(file_path, low_memory=False)
        file_code = file_path.stem.split('_')[-1]
        
        if file_code in date_mapping:
            df_temp['Period'] = date_mapping[file_code]
            dataframes.append(df_temp)
        
    if not dataframes:
        print("  [WARNING] No bilateral connectivity files loaded.")
        return

    df = pd.concat(dataframes, ignore_index=True)
    df = clean_column_names(df)
    
    df = df.drop(columns=[c for c in df.columns if c.endswith(('footnote', 'missingvalue'))], errors='ignore')
    
    df = df.rename(columns={'economy_label': 'from_country', 'partner_label': 'to_country', 'index_value': 'connectivity_index', 'period': 'period'})
    
    df['year'] = df['period'].str.extract(r'(\d{4})').astype('Int32')
    df['quarter'] = df['period'].str.extract(r'Q(\d)').astype('Int32')
    df = df[df['year'].between(2015, 2024)]
    
    df['connectivity_index'] = safe_to_numeric(df['connectivity_index'], round_digits=3)
    df = df.dropna(subset=['connectivity_index'])

    df.to_csv(PROCESSED_DIR / 'df_Combined_Connectivity_standardized.csv', index=False)
    print(f"  ✓ Saved df_Combined_Connectivity_standardized.csv (Shape: {df.shape})")
    del df; gc.collect()


def process_worldbank_traffic():
    """
    Dataset: World Bank Container Port Traffic; Secondary measure of port activity/throughput.
    """
    print("- Processing World Bank Container Port Traffic...")
    raw_path = RAW_DIR / 'container_port_traffic_worldbank/API_IS.SHP.GOOD.TU_DS2_en_csv_v2_6398.csv'

    if not raw_path.exists():
        print(f"  [ERROR] Raw file not found: {raw_path}. Skipping.")
        return
        
    df = pd.read_csv(raw_path, skiprows=4, low_memory=False)
    df = clean_column_names(df) 
    
    # Rename the cleaned and compressed column names
    df = df.rename(columns={
        'countryname': 'country_name', 'countrycode': 'country_code', 
        'indicatorname': 'indicator_name', 'indicatorcode': 'indicator_code'
    }, errors='ignore')
    
    df = df.drop(columns=['unnamed_69'], errors='ignore')
    
    id_vars = ['country_name', 'country_code', 'indicator_name', 'indicator_code'] 
    year_cols = [c for c in df.columns if c not in id_vars and c.isdigit()]
    
    df = pd.melt(df, id_vars=id_vars, value_vars=year_cols, var_name='year', value_name='container_traffic_teu')
    
    df['year'] = safe_to_numeric(df['year'], dtype=np.int32, round_digits=None)
    df['container_traffic_teu'] = df['container_traffic_teu'].replace('Unknown', np.nan)
    df['container_traffic_teu'] = safe_to_numeric(df['container_traffic_teu'], dtype=np.float64, round_digits=0)
    
    df = df[df['year'].between(2015, 2024)]
    df = df.dropna(subset=['container_traffic_teu'])

    df.to_csv(PROCESSED_DIR / 'df_WorldBank_Container_Traffic_standardized.csv', index=False)
    print(f"  ✓ Saved df_WorldBank_Container_Traffic_standardized.csv (Shape: {df.shape})")
    del df; gc.collect()


def process_worldbank_metadata():
    """
    Dataset: World Bank Country Metadata; Enrich nodes with macro-level features. 
    """
    print("- Processing World Bank Country Metadata...")
    raw_path = RAW_DIR / 'container_port_traffic_worldbank/Metadata_Country_API_IS.SHP.GOOD.TU_DS2_en_csv_v2_6398.csv'
    
    if not raw_path.exists():
        print(f"  [ERROR] Raw file not found: {raw_path}. Skipping.")
        return
        
    df = pd.read_csv(raw_path, low_memory=False)
    df = clean_column_names(df) 
    
    df = df.drop(columns=['unnamed_5'], errors='ignore')
    
    # Rename using clean column names
    df = df.rename(columns={'countrycode': 'country_code', 'incomegroup': 'income_group',
                            'specialnotes': 'special_notes', 'tablename': 'country_name', 
                            'region': 'region'})
    
    df['region'] = df['region'].fillna('Unknown')
    df['income_group'] = df['income_group'].fillna('Unknown')
    df['special_notes'] = df['special_notes'].fillna('No special notes')

    df.to_csv(PROCESSED_DIR / 'df_WorldBank_Country_Metadata_standardized.csv', index=False)
    print(f"  ✓ Saved df_WorldBank_Country_Metadata_standardized.csv (Shape: {df.shape})")
    del df; gc.collect()


def process_seaborne_trade():
    """
    Dataset: UNCTAD Seaborne Trade by Cargo Type; Cargo volume features.
    """
    print("- Processing UNCTAD Seaborne Trade by Cargo Type...")
    raw_path = RAW_DIR / 'unctad/seaborne_trade_cargotype/US.SeaborneTrade_20251115_150628.csv'
    if not raw_path.exists():
        print(f"  [ERROR] Raw file not found: {raw_path}. Skipping.")
        return
        
    df = pd.read_csv(raw_path, low_memory=False)
    df = clean_column_names(df)
    
    df = df.drop(columns=[c for c in df.columns if c.endswith(('footnote', 'missingvalue'))], errors='ignore')
    
    df = df.rename(columns={
        'economy_label': 'economy',
        'cargotype_label': 'cargo_type',
        'metric_tons_in_thousands_value': 'cargo_volume_thousand_tonnes'
    })
    
    df['year'] = safe_to_numeric(df['year'], dtype=np.int32, round_digits=None)
    df['cargo_volume_thousand_tonnes'] = safe_to_numeric(df['cargo_volume_thousand_tonnes'], round_digits=0)
    
    df['cargo_volume_tonnes'] = df['cargo_volume_thousand_tonnes'] * 1000
    df['cargo_volume_tonnes'] = df['cargo_volume_tonnes'].round(0)

    df = df[df['year'].between(2015, 2024)]

    df.to_csv(PROCESSED_DIR / 'df_Seaborne_Trade_Cargo_standardized.csv', index=False)
    print(f"  ✓ Saved df_Seaborne_Trade_Cargo_standardized.csv (Shape: {df.shape})")
    del df; gc.collect()


def process_trade_transport_costs():
    """
    Dataset: UNCTAD Trade and Transport Costs; Economic features for edge weights.
    """
    print("- Processing UNCTAD Trade and Transport Costs...")
    raw_path = RAW_DIR / 'unctad/trade-and-transport/US.TransportCosts_20251115_145318.csv'
    if not raw_path.exists():
        print(f"  [ERROR] Raw file not found: {raw_path}. Skipping.")
        return

    df = pd.read_csv(raw_path, low_memory=False)
    df = clean_column_names(df)
    
    df = df.drop(columns=[c for c in df.columns if c.endswith('missingvalue')], errors='ignore')
    
    df = df.rename(columns={
        'destination_label': 'destination',
        'perunit_freight_rate_uskg_value': 'freight_rate_usd_per_kg',
        'perunit_freight_rate_uskg_footnote': 'data_quality_note'
    })
    
    df['year'] = safe_to_numeric(df['year'], dtype=np.int32, round_digits=None)
    df['freight_rate_usd_per_kg'] = safe_to_numeric(df['freight_rate_usd_per_kg'], round_digits=3)
    
    df['freight_rate_usd_per_tonne'] = df['freight_rate_usd_per_kg'] * 1000
    df['freight_rate_usd_per_tonne'] = df['freight_rate_usd_per_tonne'].round(2)

    df.to_csv(PROCESSED_DIR / 'df_Trade_Transport_Costs_standardized.csv', index=False)
    print(f"  ✓ Saved df_Trade_Transport_Costs_standardized.csv (Shape: {df.shape})")
    del df; gc.collect()


def process_kaggle_pirate_attacks():
    """
    Dataset: Pirate Attacks; Source of disruption events for ML model training.
    """
    print("- Processing Kaggle Pirate Attacks...")
    raw_path = RAW_DIR / 'global_maritime_pirate_attacks/pirate_attacks.csv'
    if not raw_path.exists():
        print(f"  [ERROR] Raw file not found: {raw_path}. Skipping.")
        return
        
    df = pd.read_csv(raw_path, low_memory=False)
    df = clean_column_names(df)
    
    df = df.drop(columns=['attack_description', 'vessel_name', 'vessel_type'], errors='ignore')
    
    df = df.rename(columns={
        'location_description': 'location_desc', 'nearest_country': 'nearest_country_code',
        'eez_country': 'eez_country_code', 'shore_distance': 'shore_distance_km',
        'shore_longitude': 'nearest_shore_lon', 'shore_latitude': 'nearest_shore_lat',
        'vessel_status': 'vessel_status',
    })
    
    df['date'] = clean_dates_to_iso(df['date'])
    df['time'] = df['time'].apply(normalize_time_string)
    
    string_cols = ['attack_type', 'location_desc', 'nearest_country_code', 'eez_country_code', 'vessel_status', 'data_source']
    for col in string_cols:
        df[col] = df[col].fillna('Unknown').astype(str)

    df.to_csv(PROCESSED_DIR / 'df_Pirate_Attacks_standardized.csv', index=False)
    print(f"  ✓ Saved df_Pirate_Attacks_standardized.csv (Shape: {df.shape})")
    del df; gc.collect()


def process_imf_daily_port_activity():
    """
    Dataset: IMF Daily Port Activity; High-frequency time-series features (vessel calls, trade volume).
    """
    print("- Processing IMF Daily Port Activity (Chunked)...")
    raw_path = RAW_DIR / 'imf_portwatch/daily_port_activity_imf/Daily_Port_Activity_Data_and_Trade_Estimates (2).csv'
    output_path = PROCESSED_DIR / 'df_Daily_Port_Activity_standardized.csv'
    
    if not raw_path.exists():
        print(f"  [ERROR] Raw file not found: {raw_path}. Skipping.")
        return

    CHUNK_SIZE = 500_000 
    first_write = True
    rows_processed = 0

    RENAME_MAP = {
        'portid': 'port_id', 'portname': 'port_name', 'iso3': 'iso_code',
        'portcalls_container': 'vessel_calls_container', 'portcalls_dry_bulk': 'vessel_calls_dry_bulk',
        'portcalls_general_cargo': 'vessel_calls_general_cargo', 'portcalls_roro': 'vessel_calls_roro',
        'portcalls_tanker': 'vessel_calls_tanker', 'portcalls_cargo': 'vessel_calls_cargo',
        'portcalls': 'vessel_calls_total',
        'import': 'import_volume_total_tonnes',
        'export': 'export_volume_total_tonnes'
    }

    FINAL_COLS = [
        'date', 'year', 'month', 'day', 'port_id', 'port_name', 'country', 'iso_code', 
        'vessel_calls_container', 'vessel_calls_dry_bulk', 'vessel_calls_general_cargo', 'vessel_calls_roro', 'vessel_calls_tanker', 'vessel_calls_total',
        'import_volume_total_tonnes', 'export_volume_total_tonnes', 
        'total_trade_volume_tonnes', 'import_export_ratio', 'avg_cargo_per_call_tonnes'
    ]

    for chunk_i, chunk in enumerate(pd.read_csv(raw_path, chunksize=CHUNK_SIZE, low_memory=False)):
        
        chunk.columns = chunk.columns.str.lower()
        chunk = chunk.rename(columns=RENAME_MAP, errors='ignore')

        chunk['date'] = clean_dates_to_iso(chunk['date'])
        
        vessel_call_cols = [c for c in chunk.columns if c.startswith('vessel_calls')]
        
        for c in vessel_call_cols: 
            chunk[c] = safe_to_numeric(chunk[c], dtype=np.int32, round_digits=None)
        
        # Feature Engineering (Volume/Trade Ratios)
        chunk['total_trade_volume_tonnes'] = (
            chunk['import_volume_total_tonnes'] + chunk['export_volume_total_tonnes']
        ).round(2).astype(np.float32)
        
        chunk['import_export_ratio'] = (
            (chunk['import_volume_total_tonnes'] + 1.0) / (chunk['export_volume_total_tonnes'] + 1.0)
        ).round(2).astype(np.float32)
        
        chunk['avg_cargo_per_call_tonnes'] = (
            (chunk['total_trade_volume_tonnes'] + 1.0) / (chunk['vessel_calls_total'] + 1.0)
        ).round(2).astype(np.float32)
        
        chunk = chunk.reindex(columns=FINAL_COLS)

        if first_write:
            chunk.to_csv(output_path, index=False, mode='w')
            first_write = False
        else:
            chunk.to_csv(output_path, index=False, mode='a', header=False)

        rows_processed += len(chunk)
        del chunk; gc.collect()

    print(f"  ✓ Saved df_Daily_Port_Activity_standardized.csv (Total Rows: {rows_processed})")


def process_imf_daily_chokepoints():
    """
    Dataset: IMF Daily Chokepoints Transit Calls & Trade Volumes; Chunked processing like daily port activity but for chokepoints.
    """
    print("- Processing IMF Daily Chokepoints Transit Calls & Trade Volumes (Chunked)...")
    raw_path = RAW_DIR / 'imf_portwatch/daily_chokepoints_transitcalls_tradevolumes/Daily_Chokepoint_Transit_Calls_and_Trade_Volume_Estimates (1).csv'
    output_path = PROCESSED_DIR / 'df_Daily_Chokepoints_standardized.csv'

    if not raw_path.exists():
        print(f"  [ERROR] Raw file not found: {raw_path}. Skipping.")
        return

    CHUNK_SIZE = 500_000
    first_write = True
    rows_processed = 0

    RENAME_MAP = {
        'portid': 'port_id',
        'portname': 'port_name',
        'n_container': 'vessel_calls_container',
        'n_dry_bulk': 'vessel_calls_dry_bulk',
        'n_general_cargo': 'vessel_calls_general_cargo',
        'n_roro': 'vessel_calls_roro',
        'n_tanker': 'vessel_calls_tanker',
        'n_cargo': 'vessel_calls_cargo',
        'n_total': 'vessel_calls_total',
        'capacity_container': 'capacity_container',
        'capacity_dry_bulk': 'capacity_dry_bulk',
        'capacity_general_cargo': 'capacity_general_cargo',
        'capacity_roro': 'capacity_roro',
        'capacity_tanker': 'capacity_tanker',
        'capacity_cargo': 'capacity_cargo',
        'capacity': 'capacity_total',
        'objectid': 'object_id'
    }

    FINAL_COLS = [
        'date', 'year', 'month', 'day',
        'port_id', 'port_name',
        'vessel_calls_container', 'vessel_calls_dry_bulk', 'vessel_calls_general_cargo',
        'vessel_calls_roro', 'vessel_calls_tanker', 'vessel_calls_cargo', 'vessel_calls_total',
        'capacity_container', 'capacity_dry_bulk', 'capacity_general_cargo',
        'capacity_roro', 'capacity_tanker', 'capacity_cargo', 'capacity_total',
        'avg_capacity_per_call'
    ]

    for chunk in pd.read_csv(raw_path, chunksize=CHUNK_SIZE, low_memory=False):
        # Normalize Columns and Rename
        chunk = clean_column_names(chunk)
        chunk = chunk.rename(columns=RENAME_MAP, errors='ignore')

        # Dates
        if 'date' in chunk.columns:
            chunk['date'] = clean_dates_to_iso(chunk['date'])

        # Numeric Conversions for Vessel Call Counts
        vessel_call_cols = [c for c in chunk.columns if c.startswith('vessel_calls')]
        for c in vessel_call_cols:
            chunk[c] = safe_to_numeric(chunk[c], dtype=np.int32, round_digits=None)

        
        capacity_cols = [c for c in chunk.columns if c.startswith('capacity')]
        for c in capacity_cols:
            chunk[c] = safe_to_numeric(chunk[c], dtype=np.float32, round_digits=3)

       
        if 'capacity_total' not in chunk.columns or chunk['capacity_total'].isna().all():
            present_capacity_cols = [c for c in capacity_cols if c in chunk.columns]
            if present_capacity_cols:
                chunk['capacity_total'] = chunk[present_capacity_cols].sum(axis=1)

        
        if 'vessel_calls_total' not in chunk.columns:
            chunk['vessel_calls_total'] = chunk[[c for c in vessel_call_cols if c in chunk.columns]].sum(axis=1)

      
        chunk['avg_capacity_per_call'] = (
            (chunk.get('capacity_total', 0).astype(float) / (chunk.get('vessel_calls_total', 0).astype(float) + 1.0))
        ).round(3).astype(np.float32)

        
        chunk = chunk.reindex(columns=[c for c in FINAL_COLS if c in chunk.columns or c == 'avg_capacity_per_call'])

        
        if first_write:
            chunk.to_csv(output_path, index=False, mode='w')
            first_write = False
        else:
            chunk.to_csv(output_path, index=False, mode='a', header=False)

        rows_processed += len(chunk)
        del chunk; gc.collect()

    print(f"  ✓ Saved df_Daily_Chokepoints_standardized.csv (Total Rows: {rows_processed})")

def process_unctad_container_throughput():
    """
    Dataset: UNCTAD Container Port Throughput; Standardize yearly TEU throughput by economy.
    """
    print("- Processing UNCTAD Container Port Throughput...")
    raw_path = RAW_DIR / 'unctad/container_port_throughput/US.ContPortThroughput_20251115_150326.csv'
    output_path = PROCESSED_DIR / 'df_Container_Port_Throughput_standardized.csv'

    if not raw_path.exists():
        print(f"  [ERROR] Raw file not found: {raw_path}. Skipping.")
        return

    df = pd.read_csv(raw_path, low_memory=False)
    df = clean_column_names(df)

    # Drop Footnote
    df = df.drop(columns=[c for c in df.columns if c.endswith(('footnote', 'missingvalue'))], errors='ignore')

    # Rename Canonical Columns 
    df = df.rename(columns={
        'economy_label': 'country',
        'year': 'year',
        'teu_twenty_foot_equivalent_unit_value': 'teu_20ft_equivalent'
    }, errors='ignore')

    # Ensure Year is Integer and TEU is Numeric
    if 'year' in df.columns:
        df['year'] = safe_to_numeric(df['year'], dtype=np.int32, round_digits=None)

    if 'teu_20ft_equivalent' in df.columns:
        df['teu_20ft_equivalent'] = df['teu_20ft_equivalent'].replace('Unknown', np.nan)
        df['teu_20ft_equivalent'] = safe_to_numeric(df['teu_20ft_equivalent'], dtype=np.float64, round_digits=0)

    # Drop Rows Missing TEU
    if 'year' in df.columns:
        df = df[df['year'].between(2015, 2024)]
    df = df.dropna(subset=['teu_20ft_equivalent'])

    df.to_csv(output_path, index=False)
    print(f"  ✓ Saved df_Container_Port_Throughput_standardized.csv (Shape: {df.shape})")
    del df; gc.collect()


# Orchestrator 

def run_data_pipeline():
    """
    Orchestrates the execution of all data processing functions.
    """
    print("\n" + "-" * 100)
    print("STARTING MARITIME DATA PIPELINE")
    print("-" * 100)
    
    # 1. Process Core Port/Node Data
    process_unlocode_ports()
    process_imf_ports()
    process_worldbank_metadata()
    process_imf_chokepoints_master()
    
    # 2. Process Network Edge Data and External Features
    process_unctad_bilateral_connectivity()
    process_unctad_plsci()
    process_unctad_lsci()
    process_worldbank_traffic()
    process_seaborne_trade()
    process_trade_transport_costs()
    process_unctad_container_throughput()
    
    # 3. Process Disruption/Event Data
    process_imf_disruptions_db()
    process_imf_disruptions_with_ports()
    process_kaggle_pirate_attacks()
    
    # 4. Process Large Time-Series Data 
    process_imf_daily_port_activity()
    process_imf_daily_chokepoints()
    
    print("\n" + "-" * 100)
    print(f"DATA PIPELINE COMPLETE. All standardized files saved to: {PROCESSED_DIR}")
    print("-" * 100)

if __name__ == "__main__":
    run_data_pipeline()
