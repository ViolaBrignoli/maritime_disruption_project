"""
MARITIME DISRUPTION PROJECT - Port-Year Panel Loader

Builds a clean, analysis-ready port-year panel from standardized CSV artifacts located
in data/processed/. The loader discovers available source files (via FILE_ALIASES),
aggregates high-frequency sources to yearly port-level features, maps country- and
network-level attributes, imputes and sanitizes numeric/categorical fields, separates
static vs dynamic features, and returns train/test splits plus a fitted preprocessing
pipeline for downstream modeling.

Reads:  data/processed/* (aliases defined in FILE_ALIASES)
Returns: X_train, X_test, y_train, y_test, df_panel, feature_names, preprocessor

Key behaviours:
- Robust file discovery and flexible column aliasing for heterogeneous input schemas.
- Safe date parsing (quarter handling), aggregation helpers for activity, chokepoints,
  events, piracy, traffic and country-level data.
- Network feature coercion and mapping to join_key (ensures numeric consistency).
- Leakage protection (drops target/join/year-like fields), controlled OHE selection
  (cardinality-based), and reproducible preprocessor fitting.
"""
from pathlib import Path
import re
import warnings
import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional, List, Any
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

# Warning filtering
warnings.filterwarnings("ignore", message="Could not infer format")


# Configuration and Paths
PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"

FILE_ALIASES = {
"ports_master": ["df_Ports_IMF_standardized.csv", "ports_data4.csv"],
"unlocode_ports": ["df_UNLOCODE_Ports_standardized.csv", "ports_data5.csv"],
"chokepoints": ["df_Chokepoints_standardized.csv"],
"daily_chokepoints": ["df_Daily_Chokepoints_standardized.csv"],
"daily_activity": ["df_Daily_Port_Activity_standardized.csv", "port_calls_data.csv"],
"port_traffic": ["df_Port_Level_Shipping_Connectivity_standardized.csv"],
"port_events": ["df_Disruptions_With_Ports_standardized.csv", "port_events_2019.csv"],
"disasters": ["df_Portwatch_Disruptions_Database_standardized.csv", "natural_disasters.csv"],
"plsci_port": ["df_Port_Level_Shipping_Connectivity_standardized.csv", "algeria_data_analysis.csv"],
"lsci_country": ["df_Liner_Shipping_Connectivity_standardized.csv", "country_data_quarters.csv"],
"combined_conn": ["df_Combined_Connectivity_standardized.csv", "connectivity_data2.csv"],
"container_traffic": ["df_WorldBank_Container_Traffic_standardized.csv", "container_traffic2.csv"],
"wb_metadata": ["df_WorldBank_Country_Metadata_standardized.csv"],
"network_features": ["df_Network_Centrality_Features.csv"],
"piracy": ["df_Pirate_Attacks_standardized.csv", "piracy_data_1993.csv"],
"container_world": ["df_Container_Port_Throughput_standardized.csv", "shipping_data.csv"],
"trade_costs": ["df_Trade_Transport_Costs_standardized.csv", "freight_rates_data.csv"],
}

_QUARTER_RE = re.compile(r"Q([1-4])\s*[,\- ]*\s*(\d{4})", flags=re.IGNORECASE)


# Helpers
def _find_existing_file(candidates: List[str]) -> Optional[Path]:
    for fn in candidates:
        p = PROCESSED_DIR / fn
        if p.exists():
            return p
    return None


def _safe_parse_dates(series: pd.Series) -> pd.Series:
    if series is None or series.empty:
        return pd.Series(pd.NaT, index=series.index if series is not None else [])
    try:
        return pd.to_datetime(series, errors="coerce")
    except Exception:
        return series.apply(lambda x: pd.to_datetime(x, errors="coerce") if pd.notna(x) else pd.NaT)


def _quarter_to_date(qs: pd.Series) -> pd.Series:
    def q2d(x):
        if pd.isna(x):
            return pd.NaT
        m = _QUARTER_RE.search(str(x))
        if m:
            q = int(m.group(1)); y = int(m.group(2))
        else:
            m2 = re.search(r"(\d{4}).*Q\s*([1-4])", str(x))
            if m2:
                y = int(m2.group(1)); q = int(m2.group(2))
            else:
                y_match = re.search(r"(\d{4})", str(x))
                if y_match:
                    return pd.to_datetime(f"{int(y_match.group(1))}-12-31", errors="coerce")
                return pd.NaT
        if q == 1:
            return pd.to_datetime(f"{y}-03-31", errors="coerce")
        if q == 2:
            return pd.to_datetime(f"{y}-06-30", errors="coerce")
        if q == 3:
            return pd.to_datetime(f"{y}-09-30", errors="coerce")
        return pd.to_datetime(f"{y}-12-31", errors="coerce")
    return qs.apply(q2d)


# Load datasets 
def load_all_processed_data() -> Dict[str, pd.DataFrame]:
    datasets = {}
    for key, candidates in FILE_ALIASES.items():
        p = _find_existing_file(candidates)
        if p:
            try:
                datasets[key] = pd.read_csv(p, low_memory=False)
            except Exception:
                datasets[key] = pd.read_csv(p, low_memory=False, dtype=str)
    return datasets


# Ports master 
def build_ports_master(datasets: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    parts = []
    if "ports_master" in datasets:
        parts.append(datasets["ports_master"].copy())
    if "unlocode_ports" in datasets:
        parts.append(datasets["unlocode_ports"].copy())
    if "chokepoints" in datasets:
        df = datasets["chokepoints"].copy()
        if "fullname" in df.columns and "port_name" not in df.columns:
            df = df.rename(columns={"fullname": "port_name"})
        parts.append(df)
    if not parts:
        raise FileNotFoundError("No port master files found in data/processed via aliases.")

    normalized = []
    for df in parts:
        df2 = df.copy()
        if "port_name" not in df2.columns:
            for cand in ("portname", "port", "name", "port_name_clean", "fullname"):
                if cand in df2.columns:
                    df2 = df2.rename(columns={cand: "port_name"})
                    break
        if "port_id" not in df2.columns:
            for cand in ("portid", "port_id", "location_code", "locationid", "id"):
                if cand in df2.columns:
                    df2 = df2.rename(columns={cand: "port_id"})
                    break
        if "iso3" not in df2.columns:
            for cand in ("iso3", "iso_code", "country_code"):
                if cand in df2.columns:
                    df2 = df2.rename(columns={cand: "iso3"})
                    break
        if "country" not in df2.columns:
            for cand in ("country", "country_name", "countryname"):
                if cand in df2.columns:
                    df2 = df2.rename(columns={cand: "country"})
                    break
        normalized.append(df2)

    big = pd.concat(normalized, ignore_index=True, sort=False)
    if "port_name" not in big.columns:
        big["port_name"] = big.iloc[:, 0].astype(str)
    big["port_name"] = big["port_name"].astype(str)
    big["join_key"] = big["port_name"].str.strip().str.lower()
    keep_cols = ["join_key", "port_name"]
    if "port_id" in big.columns:
        keep_cols.append("port_id")
    if "country" in big.columns:
        keep_cols.append("country")
    if "iso3" in big.columns:
        keep_cols.append("iso3")
    for c in ("lat", "lon", "vessel_count_total", "vessel_count_container", "industry_top1", "industry_top2", "industry_top3"):
        if c in big.columns:
            keep_cols.append(c)
    master = big[keep_cols].drop_duplicates(subset=["join_key"]).reset_index(drop=True)
    return master


# Aggregation helpers 
def _aggregate_daily_activity_to_yearly(df_daily: pd.DataFrame, port_map_name: Dict[str, str], port_map_id: Dict[str, str]) -> pd.DataFrame:
    if df_daily is None or df_daily.empty:
        return pd.DataFrame()
    date_col = next((c for c in ("date", "day", "timestamp", "from_date") if c in df_daily.columns), None)
    port_col = next((c for c in ("port_id", "portid", "port_name", "portname") if c in df_daily.columns), None)
    if port_col is None or date_col is None:
        return pd.DataFrame()
    df = df_daily.copy()
    df["_date"] = _safe_parse_dates(df[date_col])
    df = df.dropna(subset=["_date"])
    df["_year"] = df["_date"].dt.year
    if port_col in ("port_id", "portid", "location_code"):
        df["_join_key"] = df[port_col].astype(str).str.strip().map(port_map_id)
    else:
        df["_join_key"] = df[port_col].astype(str).str.strip().str.lower().map(port_map_name)
    df = df.dropna(subset=["_join_key"])
    numeric_cols = [c for c in df.columns if c.startswith("vessel_calls") or c.endswith("tonnes") or c.startswith("import_") or c.startswith("export_") or c.startswith("total_trade")]
    fallback = ["vessel_calls_total", "vessel_calls_container", "import_volume_total_tonnes", "export_volume_total_tonnes", "total_trade_volume_tonnes", "avg_cargo_per_call_tonnes"]
    for c in fallback:
        if c in df.columns and c not in numeric_cols:
            numeric_cols.append(c)
    if not numeric_cols:
        grp_counts = df.groupby(["_join_key", "_year"]).size().rename("activity__n_activity_days").reset_index().rename(columns={"_join_key": "join_key", "_year": "year"}).set_index(["join_key", "year"])
        return grp_counts
    agg_dict = {}
    for c in numeric_cols:
        agg_dict[c] = ["sum", "mean"]
    grouped = df.groupby(["_join_key", "_year"]).agg(agg_dict)
    grouped.columns = ["__".join(col).strip() for col in grouped.columns.values]
    grouped = grouped.reset_index().rename(columns={"_join_key": "join_key", "_year": "year"}).set_index(["join_key", "year"])
    grouped = grouped.rename(columns={col: f"activity__{col}" for col in grouped.columns})
    return grouped


def _aggregate_daily_chokepoints_to_yearly(df_chkp: pd.DataFrame, port_map_name: Dict[str, str], port_map_id: Dict[str, str]) -> pd.DataFrame:
    if df_chkp is None or df_chkp.empty:
        return pd.DataFrame()
    date_col = next((c for c in ("date", "day", "timestamp", "from_date") if c in df_chkp.columns), None)
    port_col = next((c for c in ("port_id", "portid", "port_name", "portname") if c in df_chkp.columns), None)
    if port_col is None or date_col is None:
        return pd.DataFrame()
    df = df_chkp.copy()
    df["_date"] = _safe_parse_dates(df[date_col])
    df = df.dropna(subset=["_date"])
    df["_year"] = df["_date"].dt.year
    if port_col in ("port_id", "portid"):
        df["_join_key"] = df[port_col].astype(str).str.strip().map(port_map_id)
    else:
        df["_join_key"] = df[port_col].astype(str).str.strip().str.lower().map(port_map_name)
    df = df.dropna(subset=["_join_key"])
    numeric_candidates = [c for c in ("capacity_total", "avg_capacity_per_call", "vessel_calls_total") if c in df.columns]
    if not numeric_candidates:
        return df.groupby(["_join_key", "_year"]).size().rename("chokepoints__n_chokepoint_days").reset_index().rename(columns={"_join_key": "join_key", "_year": "year"}).set_index(["join_key", "year"])
    agg = df.groupby(["_join_key", "_year"])[numeric_candidates].agg(["sum", "mean"])
    agg.columns = ["__".join(col).strip() for col in agg.columns.values]
    agg = agg.reset_index().rename(columns={"_join_key": "join_key", "_year": "year"}).set_index(["join_key", "year"])
    agg = agg.rename(columns={col: f"chokepoints__{col}" for col in agg.columns})
    return agg


def _aggregate_plsci_to_yearly(df_plsci: pd.DataFrame, port_map_name: Dict[str, str]) -> pd.DataFrame:
    if df_plsci is None or df_plsci.empty:
        return pd.DataFrame()
    if "quarter_date" in df_plsci.columns:
        parsed = _quarter_to_date(df_plsci["quarter_date"])
    else:
        date_col = next((c for c in ("period", "date", "year") if c in df_plsci.columns), None)
        if date_col is None:
            return pd.DataFrame()
        parsed = _safe_parse_dates(df_plsci[date_col])
    df = df_plsci.copy()
    df["_date"] = parsed
    df = df.dropna(subset=["_date"])
    df["_year"] = df["_date"].dt.year
    port_col = next((c for c in ("port_name", "portname", "port") if c in df.columns), None)
    if port_col is None:
        return pd.DataFrame()
    cleaned = df[port_col].astype(str).apply(lambda x: x.replace('"', '').split(",")[-1].strip().lower())
    df["_join_key"] = cleaned.map(port_map_name)
    df = df.dropna(subset=["_join_key"])
    if "plsci_index" in df.columns:
        grp = df.groupby(["_join_key", "_year"])["plsci_index"].mean().reset_index().rename(columns={"_join_key": "join_key", "_year": "year", "plsci_index": "plsci__plsci_index_mean"}).set_index(["join_key", "year"])
        return grp
    return pd.DataFrame()


def _aggregate_port_traffic_to_yearly(df_traffic: pd.DataFrame, port_map_name: Dict[str, str], port_map_id: Dict[str, str]) -> pd.DataFrame:
    if df_traffic is None or df_traffic.empty:
        return pd.DataFrame()
    date_col = next((c for c in ("year", "date", "period", "quarter_date") if c in df_traffic.columns), None)
    port_col = next((c for c in ("port_name", "port", "portid", "port_id") if c in df_traffic.columns), None)
    if date_col is None or port_col is None:
        return pd.DataFrame()
    df = df_traffic.copy()
    if date_col == "year":
        parsed = _safe_parse_dates(df[date_col])
    elif "quarter" in str(date_col):
        parsed = _quarter_to_date(df[date_col])
    else:
        parsed = _safe_parse_dates(df[date_col])
    df["_date"] = parsed
    df = df.dropna(subset=["_date"])
    df["_year"] = df["_date"].dt.year
    if port_col in ("port_id", "portid"):
        df["_join_key"] = df[port_col].astype(str).str.strip().map(port_map_id)
    else:
        df["_join_key"] = df[port_col].astype(str).str.strip().str.lower().map(port_map_name)
    df = df.dropna(subset=["_join_key"])
    teu_col = next((c for c in ("container_teu", "container_traffic_teu", "container_teu_sum") if c in df.columns), None)
    if teu_col is None:
        numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        if numeric_cols:
            teu_col = numeric_cols[0]
        else:
            return pd.DataFrame()
    grp = df.groupby(["_join_key", "_year"])[teu_col].sum().reset_index().rename(columns={"_join_key": "join_key", "_year": "year", teu_col: "port_traffic__container_teu_sum"}).set_index(["join_key", "year"])
    return grp


def _aggregate_piracy_to_yearly(df_piracy: pd.DataFrame, port_map_name: Dict[str, str]) -> pd.DataFrame:
    if df_piracy is None or df_piracy.empty:
        return pd.DataFrame()
    if "date" not in df_piracy.columns:
        return pd.DataFrame()
    df = df_piracy.copy()
    df["_date"] = _safe_parse_dates(df["date"])
    df = df.dropna(subset=["_date"])
    df["_year"] = df["_date"].dt.year
    loccol = "location_desc" if "location_desc" in df.columns else (next((c for c in ("location", "location_name") if c in df.columns), None))
    if loccol is None:
        return pd.DataFrame()
    mapped = df[loccol].astype(str).str.strip().str.lower().map(port_map_name)
    df["_join_key"] = mapped
    df = df.dropna(subset=["_join_key"])
    grp = df.groupby(["_join_key", "_year"]).size().reset_index().rename(columns={0: "piracy__attacks_count", "_join_key": "join_key", "_year": "year"}).set_index(["join_key", "year"])
    return grp


def _aggregate_events_to_yearly(df_events: pd.DataFrame, port_map_name: Dict[str, str], port_map_id: Dict[str, str]) -> pd.DataFrame:
    if df_events is None or df_events.empty:
        return pd.DataFrame()
    records = []
    for _, r in df_events.iterrows():
        from_col = next((c for c in ("from_date", "date", "start_date", "year") if c in df_events.columns), None)
        to_col = next((c for c in ("to_date", "date", "end_date", "year") if c in df_events.columns), None)
        if from_col is None and to_col is None:
            continue
        from_dt = _safe_parse_dates(pd.Series([r.get(from_col)]))[0] if from_col is not None else pd.NaT
        to_dt = _safe_parse_dates(pd.Series([r.get(to_col)]))[0] if to_col is not None else pd.NaT
        if pd.isna(from_dt) and not pd.isna(to_dt):
            from_dt = to_dt
        if pd.isna(to_dt) and not pd.isna(from_dt):
            to_dt = from_dt
        if pd.isna(from_dt) and pd.isna(to_dt):
            if "year" in df_events.columns:
                try:
                    y = int(r.get("year"))
                    from_dt = pd.to_datetime(f"{y}-01-01")
                    to_dt = pd.to_datetime(f"{y}-12-31")
                except Exception:
                    continue
            else:
                continue
        aff = r.get("affectedports") if "affectedports" in df_events.columns else None
        if pd.notna(aff):
            parts = re.split(r"[;,]\s*", str(aff))
            for p in parts:
                p = p.strip()
                if not p:
                    continue
                jk = port_map_id.get(p) or port_map_name.get(p.lower())
                if jk:
                    records.append((jk, from_dt, to_dt))
        else:
            if "port_name" in df_events.columns and pd.notna(r.get("port_name")):
                p = str(r.get("port_name")).strip().lower()
                jk = port_map_name.get(p)
                if jk:
                    records.append((jk, from_dt, to_dt))
            elif "port_id" in df_events.columns and pd.notna(r.get("port_id")):
                p = str(r.get("port_id")).strip()
                jk = port_map_id.get(p)
                if jk:
                    records.append((jk, from_dt, to_dt))
    rows = []
    for jk, fdt, tdt in records:
        if pd.isna(fdt) or pd.isna(tdt):
            continue
        start_year = int(fdt.year)
        end_year = int(tdt.year)
        for y in range(start_year, end_year + 1):
            rows.append((jk, y))
    if not rows:
        return pd.DataFrame()
    df_rows = pd.DataFrame(rows, columns=["join_key", "year"])
    df_rows["events__n_events"] = 1
    agg = df_rows.groupby(["join_key", "year"])["events__n_events"].sum().rename("events__n_events").reset_index().set_index(["join_key", "year"])
    agg["events__disrupted_flag"] = 1
    return agg


def _aggregate_country_level_to_yearly(datasets: Dict[str, pd.DataFrame], df_master: pd.DataFrame) -> pd.DataFrame:
    country_frames = []
    if "container_traffic" in datasets:
        df = datasets["container_traffic"]
        country_col = next((c for c in ("country_name", "country") if c in df.columns), None)
        if country_col and "year" in df.columns:
            dfc = df[[country_col, "year"] + [c for c in df.columns if "container" in c.lower() or "teu" in c.lower()]].copy()
            dfc = dfc.rename(columns={country_col: "country_key"})
            dfc["country_key"] = dfc["country_key"].astype(str).str.strip().str.lower()
            dfc["year"] = pd.to_numeric(dfc["year"], errors="coerce").astype("Int64")
            country_frames.append(dfc)
    if "lsci_country" in datasets:
        df = datasets["lsci_country"]
        country_col = next((c for c in ("country_name", "country") if c in df.columns), None)
        date_col = next((c for c in ("quarter_date", "period", "year") if c in df.columns), None)
        if country_col and date_col:
            dfc = df.copy()
            if "quarter" in str(date_col):
                dfc["_date"] = _quarter_to_date(dfc[date_col])
            else:
                dfc["_date"] = _safe_parse_dates(dfc[date_col])
            dfc = dfc.dropna(subset=["_date"])
            dfc["year"] = dfc["_date"].dt.year
            dfc = dfc[[country_col, "year"] + [c for c in dfc.columns if "lsci" in c.lower()]].copy()
            dfc = dfc.rename(columns={country_col: "country_key"})
            dfc["country_key"] = dfc["country_key"].astype(str).str.strip().str.lower()
            country_frames.append(dfc)
    if "combined_conn" in datasets:
        df = datasets["combined_conn"]
        country_col = next((c for c in ("from_country", "to_country") if c in df.columns), None)
        date_col = next((c for c in ("period", "quarter", "year", "date") if c in df.columns), None)
        if country_col and date_col and "connectivity_index" in df.columns:
            dfc = df.copy()
            if "quarter" in str(date_col) or "period" in str(date_col):
                dfc["_date"] = _quarter_to_date(dfc[date_col]) if "quarter" in str(date_col) else _safe_parse_dates(dfc[date_col])
            else:
                dfc["_date"] = _safe_parse_dates(dfc[date_col])
            dfc = dfc.dropna(subset=["_date"])
            dfc["year"] = dfc["_date"].dt.year
            dfc = dfc[[country_col, "year", "connectivity_index"]].rename(columns={country_col: "country_key"})
            dfc["country_key"] = dfc["country_key"].astype(str).str.strip().str.lower()
            country_frames.append(dfc)
    if not country_frames:
        return pd.DataFrame()
    country_all = pd.concat(country_frames, ignore_index=True, sort=False)
    numeric_cols = [c for c in country_all.columns if c not in ("country_key", "year", "_date")]
    country_all = country_all.dropna(subset=["year"])
    if numeric_cols:
        country_grouped = country_all.groupby(["country_key", "year"])[numeric_cols].agg("last").reset_index()
    else:
        return pd.DataFrame()
    map_iso = {}
    map_country = {}
    if "iso3" in df_master.columns:
        for _, r in df_master.iterrows():
            iso = r.get("iso3")
            if pd.notna(iso):
                map_iso.setdefault(str(iso).upper(), []).append(r["join_key"])
    if "country" in df_master.columns:
        for _, r in df_master.iterrows():
            c = r.get("country")
            if pd.notna(c):
                map_country.setdefault(str(c).strip().lower(), []).append(r["join_key"])
    rows = []
    for _, r in country_grouped.iterrows():
        ck = r["country_key"]
        y = int(r["year"])
        assigned_jks = []
        if isinstance(ck, str) and len(ck) == 3 and ck.isupper():
            assigned_jks = map_iso.get(ck.upper(), [])
        if not assigned_jks:
            assigned_jks = map_country.get(ck, [])
        if not assigned_jks:
            for cname, jks in map_country.items():
                if cname and ck and (ck in cname or cname in ck):
                    assigned_jks = jks
                    break
        for jk in assigned_jks:
            d = {"join_key": jk, "year": y}
            for col in numeric_cols:
                d[col] = r.get(col)
            rows.append(d)
    if not rows:
        return pd.DataFrame()
    df_port_country = pd.DataFrame(rows).set_index(["join_key", "year"])
    df_port_country = df_port_country.rename(columns={c: f"country__{c}" for c in df_port_country.columns})
    return df_port_country


def _map_network_features_yearly(df_net: pd.DataFrame, df_master: pd.DataFrame) -> pd.DataFrame:
    if df_net is None or df_net.empty:
        return pd.DataFrame()
    df = df_net.copy()
    
    # Coerce all numeric columns explicitly to numeric (excluding identifiers)
    non_id_cols = [c for c in df.columns if c not in ("country_name", "iso3", "join_key", "year")]
    for col in non_id_cols:
        # Use pd.to_numeric with errors='coerce' and then fill NaNs with 0
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0) 
        
    
    if "join_key" in df.columns:
        if "year" in df.columns:
            df2 = df.set_index(["join_key", "year"])
            # ensure unique index. Aggregate duplicates (mean) if necessary
            if not df2.index.is_unique:
                df2 = df2.groupby(level=list(range(df2.index.nlevels))).mean()
            df2 = df2.rename(columns={c: f"network__{c}" for c in df2.columns})
            return df2
        else:
            df2 = df.set_index("join_key")
            if not df2.index.is_unique:
                df2 = df2.groupby(level=list(range(df2.index.nlevels))).mean()
            df2 = df2.rename(columns={c: f"network__{c}" for c in df2.columns})
            return df2
    
    # Check for identity columns to merge on
    # Build helper country. join_key mapping from master
    map_country = {}
    if "country" in df_master.columns:
        for _, r in df_master.iterrows():
            c = r.get("country")
            if pd.notna(c):
                key = str(c).strip().lower()
                map_country.setdefault(key, []).append(r["join_key"])

    if "iso3" in df.columns and "iso3" in df_master.columns:
        # Use only the columns that are now numeric
        cols_to_map = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        df_iso = df.set_index("iso3")[cols_to_map]
        if not df_iso.index.is_unique:
            df_iso = df_iso.groupby(level=list(range(df_iso.index.nlevels))).mean()
        return df_iso.rename(columns={c: f"network__{c}" for c in cols_to_map})

    if "country_name" in df.columns and map_country:
        # Use country_name to expand to join_key(s)
        cols_to_copy = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) or c == "year"]
        rows = []
        for _, r in df.iterrows():
            raw_country = r.get("country_name")
            if pd.isna(raw_country):
                continue
            ckey = str(raw_country).strip().lower()
            assigned_jks = map_country.get(ckey, [])
            # fallback substring match
            if not assigned_jks:
                for mc, jks in map_country.items():
                    if mc and ckey and (ckey in mc or mc in ckey):
                        assigned_jks = jks
                        break
            if not assigned_jks:
                continue
            for jk in assigned_jks:
                d = {"join_key": jk}
                for col in cols_to_copy:
                    d[col] = r.get(col)
                rows.append(d)
        if rows:
            df_rows = pd.DataFrame(rows)
            if "year" in df_rows.columns:
                df_rows = df_rows.set_index(["join_key", "year"])
            else:
                df_rows = df_rows.set_index("join_key")
            # ensure unique index. Aggregate duplicates (mean) if necessary
            if not df_rows.index.is_unique:
                df_rows = df_rows.groupby(level=list(range(df_rows.index.nlevels))).mean()
            df_rows = df_rows.rename(columns={c: f"network__{c}" for c in df_rows.columns})
            return df_rows

    # If country_name exists and master doesn't have mapping or mapping failed, return country_name indexed data for debugging or alternative merge
    if "country_name" in df.columns:
        cols_to_map = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        if "year" in df.columns:
            df_cn = df.set_index(["country_name", "year"])[cols_to_map]
            if not df_cn.index.is_unique:
                df_cn = df_cn.groupby(level=list(range(df_cn.index.nlevels))).mean()
            return df_cn.rename(columns={c: f"network__{c}" for c in cols_to_map})
        df_cn = df.set_index("country_name")[cols_to_map]
        if not df_cn.index.is_unique:
            df_cn = df_cn.groupby(level=list(range(df_cn.index.nlevels))).mean()
        return df_cn.rename(columns={c: f"network__{c}" for c in cols_to_map})
    
    return pd.DataFrame()


# Final imputation helper 
def _final_impute_panel(df_panel: pd.DataFrame) -> pd.DataFrame:
    """
    Make df_panel NaN-free:
    - numeric columns: fill with median, if column all-missing => fill 0.
    - categorical/object columns: fill 'Unknown'
    """
    df = df_panel.copy()
    # numeric
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    for c in num_cols:
        if df[c].isna().all():
            df[c] = df[c].fillna(0)
        else:
            med = df[c].median(skipna=True)
            if pd.isna(med):
                med = 0
            df[c] = df[c].fillna(med)
    # categorical
    obj_cols = [c for c in df.columns if c not in num_cols]
    for c in obj_cols:
        df[c] = df[c].fillna("Unknown")
    return df


# Build panel 
def build_port_year_panel(datasets: Dict[str, pd.DataFrame], restrict_to_observed_ports: bool = True) -> Tuple[pd.DataFrame, Dict]:
    provenance = {}
    master = build_ports_master(datasets)
    port_map_name = {r["port_name"].strip().lower(): r["join_key"] for _, r in master.iterrows() if pd.notna(r.get("port_name"))}
    port_map_id = {}
    if "port_id" in master.columns:
        for _, r in master.iterrows():
            pid = r.get("port_id")
            if pd.notna(pid):
                port_map_id[str(pid).strip()] = r["join_key"]

    # aggregate sources
    df_daily_activity = datasets.get("daily_activity")
    yearly_activity = _aggregate_daily_activity_to_yearly(df_daily_activity, port_map_name, port_map_id)
    provenance["daily_activity_ports"] = 0 if yearly_activity.empty else int(yearly_activity.reset_index()["join_key"].nunique())

    df_daily_chkp = datasets.get("daily_chokepoints")
    yearly_chkp = _aggregate_daily_chokepoints_to_yearly(df_daily_chkp, port_map_name, port_map_id)
    provenance["daily_chokepoints_ports"] = 0 if yearly_chkp.empty else int(yearly_chkp.reset_index()["join_key"].nunique())

    df_plsci = datasets.get("plsci_port")
    yearly_plsci = _aggregate_plsci_to_yearly(df_plsci, port_map_name)
    provenance["plsci_ports"] = 0 if yearly_plsci.empty else int(yearly_plsci.reset_index()["join_key"].nunique())

    df_port_traffic = datasets.get("port_traffic")
    yearly_port_traffic = _aggregate_port_traffic_to_yearly(df_port_traffic, port_map_name, port_map_id)
    provenance["port_traffic_ports"] = 0 if yearly_port_traffic.empty else int(yearly_port_traffic.reset_index()["join_key"].nunique())

    df_piracy = datasets.get("piracy")
    yearly_piracy = _aggregate_piracy_to_yearly(df_piracy, port_map_name)
    provenance["piracy_ports"] = 0 if yearly_piracy.empty else int(yearly_piracy.reset_index()["join_key"].nunique())

    df_port_events = datasets.get("port_events")
    df_disasters = datasets.get("disasters")
    yearly_port_events = _aggregate_events_to_yearly(df_port_events, port_map_name, port_map_id)
    yearly_disasters = _aggregate_events_to_yearly(df_disasters, port_map_name, port_map_id)
    if not yearly_port_events.empty and not yearly_disasters.empty:
        yearly_events = yearly_port_events.join(yearly_disasters, how="outer", lsuffix="_pe", rsuffix="_dis")
        yearly_events = yearly_events.fillna(0)
        n_events_cols = [c for c in yearly_events.columns if c.startswith("events__n_events")]
        if n_events_cols:
            yearly_events["events__n_events"] = yearly_events[n_events_cols].sum(axis=1)
        disrupted_cols = [c for c in yearly_events.columns if "disrupted_flag" in c]
        if disrupted_cols:
            yearly_events["events__disrupted_flag"] = (yearly_events[disrupted_cols].sum(axis=1) > 0).astype(int)
        yearly_events = yearly_events[[c for c in yearly_events.columns if c.startswith("events__")]]
    else:
        yearly_events = yearly_port_events.combine_first(yearly_disasters)
    provenance["events_ports"] = 0 if yearly_events.empty else int(yearly_events.reset_index()["join_key"].nunique())

    yearly_country = _aggregate_country_level_to_yearly(datasets, master)
    provenance["country_mapped_ports"] = 0 if yearly_country.empty else int(yearly_country.reset_index()["join_key"].nunique())

    df_net = datasets.get("network_features")
    net_mapped = _map_network_features_yearly(df_net, master) if df_net is not None else pd.DataFrame()
    provenance["network_features_present"] = False if net_mapped.empty else True

    # union of observed (join_key, year)
    indices = []
    for src in (yearly_activity, yearly_chkp, yearly_plsci, yearly_port_traffic, yearly_piracy, yearly_events, yearly_country):
        if src is None or src.empty:
            continue
        src_idx = src.reset_index()[["join_key", "year"]]
        indices.append(src_idx)
    if not indices:
        raise ValueError("No date-bearing information found to build panel. Check your data files.")
    union_idx = pd.concat(indices, ignore_index=True).drop_duplicates().reset_index(drop=True)

    if restrict_to_observed_ports:
        observed_ports = set(union_idx["join_key"].unique())
        master = master.loc[master["join_key"].isin(observed_ports)].reset_index(drop=True)

    min_year = int(union_idx["year"].min())
    max_year = int(union_idx["year"].max())
    ports = master["join_key"].unique().tolist()

    # build full panel rows
    panel_rows = []
    for jk in ports:
        for y in range(min_year, max_year + 1):
            panel_rows.append({"join_key": jk, "year": y})
    df_panel = pd.DataFrame(panel_rows).set_index(["join_key", "year"])

    # join sources
    to_merge = {
        "activity": yearly_activity,
        "chokepoints": yearly_chkp,
        "plsci": yearly_plsci,
        "port_traffic": yearly_port_traffic,
        "piracy": yearly_piracy,
        "events": yearly_events,
        "country": yearly_country,
    }
    for name, dfagg in to_merge.items():
        if dfagg is None or dfagg.empty:
            continue
        dfagg2 = dfagg.copy()
        if dfagg2.index.names != ["join_key", "year"]:
            dfagg2 = dfagg2.reset_index().set_index(["join_key", "year"])
        df_panel = df_panel.join(dfagg2, how="left")

    # merge master static metadata
    master_indexed = master.set_index("join_key")
    static_cols = [c for c in master.columns if c not in ("port_name", "join_key", "port_id")]
    if static_cols:
        master_static = master_indexed[static_cols].rename(columns={c: f"meta__{c}" for c in static_cols})
        df_panel = df_panel.join(master_static, on="join_key", how="left")

    # merge network features
    if not net_mapped.empty:
        # All columns in net_mapped are guaranteed to be numeric
        
        # If the network features are time-series (indexed by join_key and year)
        if set(net_mapped.index.names) == {"join_key", "year"}:
            nm = net_mapped.copy().rename(columns={c: f"network__{c}" for c in net_mapped.columns})
            df_panel = df_panel.join(nm, how="left")
        
        # If the network features are static (indexed by join_key)
        elif "join_key" in net_mapped.index.names:
            nm = net_mapped.copy().rename(columns={c: f"network__{c}" for c in net_mapped.columns})
            df_panel = df_panel.join(nm, how="left")
            
        # If the network features are mapped by ISO3
        elif "iso3" in net_mapped.index.names and "iso3" in master_indexed.columns:
            net_by_iso = net_mapped.copy().rename(columns={c: f"network__{c}" for c in net_mapped.columns})
            jk_iso = master_indexed["iso3"].to_dict()
            for col in net_by_iso.columns:
                iso_series = pd.Series(df_panel.index.get_level_values(0)).map(jk_iso).values
                vals = []
                for iso in iso_series:
                    try:
                        vals.append(net_by_iso.at[iso, col] if (iso in net_by_iso.index and col in net_by_iso.columns) else np.nan)
                    except Exception:
                        vals.append(np.nan)
                df_panel[col] = vals

    # Separate Static vs Dynamic

    # 1. Reset index to access columns
    df_panel = df_panel.reset_index().sort_values(["join_key", "year"]).reset_index(drop=True)

    # 2. Identify Column Types
    all_numeric = [c for c in df_panel.select_dtypes(include=[np.number]).columns.tolist()
                   if "disrupted" not in c and c != "year"]

    # 3. Exclusion Step: Ensure network features are in the numeric list.
    # Define the 7 expected features explicitly (from Network Graph script)
    EXPECTED_NETWORK_COLS = [
        "network__net_degree", 
        "network__net_load_centrality", 
        "network__net_closeness", 
        "network__net_eccentricity_inv", 
        "network__net_pagerank", 
        "network__net_clustering",
        "network__net_community"
    ]

    # Add any column starting with 'network__' that is in the EXPECTED list
    network_numeric_cols = [c for c in df_panel.columns if c in EXPECTED_NETWORK_COLS]
    
    # We use a set to efficiently combine and remove duplicates from all numeric columns 
    unique_numeric_cols = set(all_numeric)
    unique_numeric_cols.update(network_numeric_cols)
    
    # We use list comprehension to ensure we only include columns present in df_panel
    all_numeric = [c for c in df_panel.columns if c in unique_numeric_cols]


    static_features = [c for c in all_numeric if c.startswith("meta__") or c in ["lat", "lon"]]

    dynamic_features = [c for c in all_numeric if c not in static_features]

    # 4. Apply Fill Strategies 
    if static_features:
        for c in static_features:
            df_panel[c] = pd.to_numeric(df_panel[c], errors="coerce")
        df_panel[static_features] = df_panel.groupby("join_key")[static_features].transform(lambda g: g.ffill().bfill())
        df_panel[static_features] = df_panel[static_features].fillna(df_panel[static_features].median())

    if dynamic_features:
        for c in dynamic_features:
            df_panel[c] = pd.to_numeric(df_panel[c], errors="coerce")

        df_panel[dynamic_features] = df_panel.groupby("join_key")[dynamic_features].ffill()
        df_panel[dynamic_features] = df_panel[dynamic_features].fillna(0)

        for c in dynamic_features:
            if df_panel[c].max() > 100:
                df_panel[c] = np.log1p(df_panel[c])

    # 5. Clean up Extremes 
    df_panel[all_numeric] = df_panel[all_numeric].replace([np.inf, -np.inf], 0)
    df_panel[all_numeric] = df_panel[all_numeric].clip(lower=-1, upper=1e6)

    # 6. Build Target Variable 
    disrupted_cols = [c for c in df_panel.columns if "disrupted_flag" in c]
    if disrupted_cols:
        df_panel[disrupted_cols] = df_panel[disrupted_cols].fillna(0)
        df_panel["target_is_disrupted_in_year"] = (df_panel[disrupted_cols].sum(axis=1) > 0).astype(int)
    else:
        df_panel["target_is_disrupted_in_year"] = 0

    # 7. Final Imputation for Categorical
    df_panel = _final_impute_panel(df_panel)

    # 8. Final Sanity: Fill any leftover NaNs with 0 
    df_panel = df_panel.fillna(0)

    # Restore multiindex for internal consistency, but return reset_index() for easier downstream utilization
    df_panel = df_panel.set_index(["join_key", "year"]).sort_index()

    provenance.update({
        "min_year": min_year,
        "max_year": max_year,
        "n_ports_in_panel": len(ports),
        "panel_shape": df_panel.shape,
    })
    return df_panel.reset_index(), provenance


# Preprocessing and split
def _fit_preprocessor(X_train_df: pd.DataFrame, numeric_features: List[str], categorical_features: List[str]):
    numeric_transformer = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
    # Set handle_unknown="ignore" to prevent errors on unseen categories in the test set
    categorical_transformer = Pipeline([("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")), ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))])
    
    transformers = []
    if numeric_features:
        transformers.append(("num", numeric_transformer, numeric_features))
    
    if categorical_features:
        # Only apply OHE to the restricted list
        transformers.append(("cat", categorical_transformer, categorical_features))
    
    preprocessor = ColumnTransformer(transformers=transformers, verbose_feature_names_out=False)
    preprocessor.fit(X_train_df)
    preprocessor.input_feature_names_ = list(numeric_features) + list(categorical_features)
    return preprocessor


def load_and_split_panel(
    year_cutoff: int = 2020,
    restrict_to_observed_ports: bool = True,
    force_categorical: Optional[List[str]] = None,
) -> Tuple[np.ndarray, np.ndarray, pd.Series, pd.Series, pd.DataFrame, List[str], Any]:
    """
    Returns:
    X_train, X_test, y_train, y_test, df_panel, feature_names, preprocessor
    """
    datasets = load_all_processed_data()
    df_panel, provenance = build_port_year_panel(datasets, restrict_to_observed_ports=restrict_to_observed_ports)

    print("Panel provenance summary:")
    for k, v in provenance.items():
        print(f" - {k}: {v}")

    df_panel["year"] = pd.to_numeric(df_panel["year"], errors="coerce").fillna(0).astype(int)

    unique_years = sorted(df_panel["year"].unique())
    print(f"\n[DEBUG] Unique Years in DataFrame: {unique_years}")
    print(f"[DEBUG] Cutoff Year: {year_cutoff}")

    # Hunt for Leakage
    df_panel_cols = df_panel.columns.tolist()

    drop_patterns = ["target_is_disrupted_in_year", "join_key", "year", "port_name", "port_id", "disrupted"]

    candidate_cols = []
    for c in df_panel_cols:
        if any(p in c for p in drop_patterns):
            continue
        candidate_cols.append(c)

    print(f"[DEBUG] Dropped potential leakage columns. Retained features: {len(candidate_cols)}")

    # Identify Types 
    # Start with all numeric features identified by pandas
    numeric_features = [c for c in candidate_cols if pd.api.types.is_numeric_dtype(df_panel[c])]
    
    # Identify non-numeric features
    all_string_features = [c for c in candidate_cols if not pd.api.types.is_numeric_dtype(df_panel[c])]

    # STRUCTURAL FILTER: Define which categorical features are safe for OHE
    # SAFE features are those with low cardinality or specific domain relevance (e.g. industry sectors).
    # MUST ensure no highly unique IDs make it through.
    SAFE_CATEGORICAL_PATTERNS = ["meta__industry_", "meta__country", "meta__iso3", "meta__region"] 
    HIGH_CARDINALITY_THRESHOLD = 100 # Max unique values allowed for OHE

    categorical_features = []
    
    for c in all_string_features:
        # Check if the feature contains a safe pattern or has low cardinality
        is_safe_pattern = any(p in c for p in SAFE_CATEGORICAL_PATTERNS)
        
        # Calculate cardinality
        cardinality = df_panel[c].nunique()
        is_low_cardinality = cardinality < HIGH_CARDINALITY_THRESHOLD
        
        # If it matches a safe pattern, or if it's generally low cardinality, keep it for OHE.
        if is_safe_pattern and cardinality > 1: # Only OHE if it has more than one value
            categorical_features.append(c)
        elif is_low_cardinality and cardinality > 1:
            categorical_features.append(c)
        else:
            # Dropping high-cardinality IDs
            # Goal is to make sure the final OHE count is minimal 
            pass


    if force_categorical:
        for c in force_categorical:
            if c in df_panel.columns and c not in categorical_features:
                categorical_features.append(c)
            if c in numeric_features:
                numeric_features.remove(c)

    # 9. Apply Split 
    train_mask = df_panel["year"] <= year_cutoff
    test_mask = df_panel["year"] >= (year_cutoff + 1)

    df_train = df_panel.loc[train_mask].copy().reset_index(drop=True)
    df_test = df_panel.loc[test_mask].copy().reset_index(drop=True)

    if df_train.empty:
        raise ValueError(f"Train set is empty (no years <= {year_cutoff}).")
    if df_test.empty:
        raise ValueError(f"Test set is empty (no years >= {year_cutoff+1}).")

    X_train_df = df_train[numeric_features + categorical_features].copy()
    X_test_df = df_test[numeric_features + categorical_features].copy()

    preprocessor = _fit_preprocessor(X_train_df, numeric_features, categorical_features)
    X_train = preprocessor.transform(X_train_df)
    X_test = preprocessor.transform(X_test_df)

    y_train = df_train["target_is_disrupted_in_year"].astype(int).reset_index(drop=True)
    y_test = df_test["target_is_disrupted_in_year"].astype(int).reset_index(drop=True)

    try:
        feature_names = list(preprocessor.get_feature_names_out())
    except Exception:
        feature_names = [f"Feature_{i}" for i in range(X_train.shape[1])]

    print(f"Panel built with {df_panel['join_key'].nunique()} ports.")
    print(f"Train rows: {len(df_train)} (years <= {year_cutoff}), Test rows: {len(df_test)} (years >= {year_cutoff+1})")
    print(f"Selected numeric features: {len(numeric_features)}, retained categorical features: {len(categorical_features)}")

    return X_train, X_test, y_train, y_test, df_panel, feature_names, preprocessor


# CLI test 
if __name__ == "__main__":
    print("--- Panel Data Loader Test ---")
    X_tr, X_te, y_tr, y_te, panel, feat_names, prep = load_and_split_panel(year_cutoff=2020, restrict_to_observed_ports=True, force_categorical=["meta__country", "meta__industry_top1"])
    print("X_train shape:", X_tr.shape)
    print("X_test shape:", X_te.shape)
    print("y_train shape:", y_tr.shape)
    print("Sample panel rows:")
    print(panel.head(8).to_string(index=False))