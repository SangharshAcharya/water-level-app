"""
Water Level Conversion — S4W-Nepal
=================================
Upload raw sensor files (.txt or .csv) — the app auto-detects the format,
applies the correct pressure → water-level formula, and delivers interactive
charts, per-station results, and downloadable CSV / Excel files.

Formats supported
-----------------
• OpenOBS (.txt)   — Unix-timestamp CSV, two firmware variants auto-detected
• Hobo U20L (.csv) — HOBOware-exported absolute pressure CSV

Atmospheric pressure
--------------------
• Default Kathmandu Valley value (~86 kPa)
• Upload METER ATMOS 41 Atmospheric_pressure.csv for time-matched correction
"""

import io
import re
from datetime import datetime
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
DEFAULT_ATM_KPA = 86.0    # Kathmandu Valley approximate atmospheric pressure
GRAVITY_KPA     = 9.80665  # kPa per metre of water column (rho=1000, g=9.80665)
GRAVITY_OBS     = 9.80665e3  # for OBS mbar formula

OBS_SENSOR_HEIGHT_DEFAULTS = {
    "303": 0.08, "469": 0.10, "470": 0.10,
    "300": 0.10, "301": 0.10, "304": 0.10, "455": 0.19, "467": 0.10,
}
# Hobo sensors: default calibration offset is 0.0 m — user can adjust to correct systematic bias.
HOBO_SENSOR_HEIGHT_DEFAULT = 0.0

PALETTE = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
]

# OBS firmware detection threshold:
#   first raw pressure reading > 50 000  →  old firmware, ÷100 to get mbar  (~85 000 range)
#   first raw pressure reading ≤ 50 000  →  new firmware, ÷10  to get mbar  (~8 500  range)
OBS_FW_PRESSURE_THRESHOLD = 50_000

# ─────────────────────────────────────────────────────────────────────────────
# Format detection
# ─────────────────────────────────────────────────────────────────────────────

def detect_format(file_bytes: bytes, filename: str) -> str:
    """
    Returns one of: 'obs_txt' | 'hobo_csv' | 'unknown'

    Priority:
      1. Contains 'time,' + 'ambient_light' header → 'obs_txt'
      2. Contains 'Date Time' + 'Abs Pres' → 'hobo_csv'
      3. .txt with 'time,' header (minimal OBS firmware) → 'obs_txt'
      4. Fallback → 'unknown'
    """
    try:
        text = file_bytes[:3000].decode("utf-8", errors="replace")
    except Exception:
        return "unknown"

    if re.search(r"time\s*,\s*ambient_light", text, re.IGNORECASE):
        return "obs_txt"

    if re.search(r"Date\s*Time", text, re.IGNORECASE) and re.search(
        r"Abs.?Pres", text, re.IGNORECASE
    ):
        return "hobo_csv"

    ext = filename.rsplit(".", 1)[-1].lower()
    if ext == "txt" and re.search(r"^\s*time\s*,", text, re.IGNORECASE | re.MULTILINE):
        return "obs_txt"

    return "unknown"


# ─────────────────────────────────────────────────────────────────────────────
# OBS firmware version detection
# ─────────────────────────────────────────────────────────────────────────────

def detect_obs_firmware(file_bytes: bytes) -> tuple[str, str | None]:
    """
    Returns ('old'|'new', firmware_date_str|None).

    Detection rule: inspect the first pressure reading in the data.
      • first_pressure > OBS_FW_PRESSURE_THRESHOLD (50 000)  →  'old'  (÷100 to mbar)
      • first_pressure ≤ OBS_FW_PRESSURE_THRESHOLD           →  'new'  (÷10  to mbar)

    Old firmware: raw pressure ~85 000  (Pa-equivalent units)
    New firmware: raw pressure ~8 500   (0.1-mbar units)
    """
    text = file_bytes.decode("utf-8", errors="replace")

    # Parse 'Firmware updated: YYYY/MM/DD' from the comment header (display only)
    fw_match    = re.search(r"Firmware\s+updated:\s*(\d{4}/\d{2}/\d{2})", text, re.IGNORECASE)
    fw_date_str = fw_match.group(1) if fw_match else None

    # Find data header row, then read only the first data line
    lines      = text.splitlines()
    header_idx = next(
        (i for i, ln in enumerate(lines)
         if re.match(r"^\s*time\s*,", ln, re.IGNORECASE)),
        None,
    )
    first_pressure: float | None = None
    if header_idx is not None:
        try:
            headers = [c.strip().lower() for c in lines[header_idx].split(",")]
            if "pressure" in headers:
                p_col = headers.index("pressure")
                for ln in lines[header_idx + 1:]:
                    parts = ln.split(",")
                    if len(parts) > p_col and parts[p_col].strip():
                        first_pressure = float(parts[p_col].strip())
                        break
        except Exception:
            pass

    firmware = (
        "old" if (first_pressure is not None and first_pressure > OBS_FW_PRESSURE_THRESHOLD)
        else "new"
    )
    return firmware, fw_date_str


def extract_obs_sn(file_bytes: bytes) -> str:
    """
    Parse 'OpenOBS SN:XXX' from the file header comment block.
    Returns the SN string, or 'Unknown' if not found.
    """
    text = file_bytes.decode("utf-8", errors="replace")
    m = re.search(r"OpenOBS\s+SN\s*:\s*(\w+)", text, re.IGNORECASE)
    return m.group(1) if m else "Unknown"


# ─────────────────────────────────────────────────────────────────────────────
# OBS parser & water-level formula
# ─────────────────────────────────────────────────────────────────────────────

def parse_obs_txt(
    file_bytes: bytes, filename: str
) -> tuple[pd.DataFrame | None, str | None]:
    """
    Returns (raw_df, error_msg).
    raw_df columns: Date, Time_unix, Ambient_light, Backscatter, Pressure,
                    Water_temp, Battery, Source_file
    """
    text  = file_bytes.decode("utf-8", errors="replace")
    lines = text.splitlines()

    header_idx = next(
        (i for i, ln in enumerate(lines) if re.match(r"^\s*time\s*,", ln, re.IGNORECASE)),
        None,
    )
    if header_idx is None:
        return None, f"No `time,` header found in **{filename}**"

    try:
        df = pd.read_csv(io.StringIO("\n".join(lines[header_idx:])))
    except Exception as e:
        return None, f"CSV parse error in {filename}: {e}"

    df.columns = [c.strip().lower() for c in df.columns]
    df.rename(columns={
        "time":         "Time_unix",
        "ambient_light":"Ambient_light",
        "backscatter":  "Backscatter",
        "pressure":     "Pressure",
        "water_temp":   "Water_temp",
        "battery":      "Battery",
    }, inplace=True)

    if "Time_unix" not in df.columns or "Pressure" not in df.columns:
        return None, f"Missing required columns in {filename}"

    df["Date"] = pd.to_datetime(df["Time_unix"], unit="s", origin="unix", errors="coerce")
    df.dropna(subset=["Date"], inplace=True)

    for c in ["Ambient_light", "Backscatter", "Pressure", "Water_temp", "Battery"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    df["Source_file"] = filename
    ordered = ["Date", "Time_unix", "Ambient_light", "Backscatter", "Pressure",
               "Water_temp", "Battery", "Source_file"]
    return df[[c for c in ordered if c in df.columns]], None


def calc_water_level_obs(
    df: pd.DataFrame,
    sensor_height: float,
    baro: pd.DataFrame | None,
    default_atm_kpa: float,
    firmware: str = "new",
) -> pd.DataFrame:
    """
    OBS water-level formula — two firmware variants.

    New firmware (post ~Jul-2025, pressure ~8 600 units = 0.1 mbar each):
        hydroP_mbar   = Pressure / 10   - Atm_kPa * 10

    Old firmware (pre ~Jul-2025, pressure ~85 000 units ≈ Pa / 0.01 mbar each):
        hydroP_mbar   = Pressure / 100  - Atm_kPa * 10

    Then (both variants):
        Water_level_m = hydroP_mbar / 1000 / 9806.65 * 1e5 + h_sensor
    """
    divisor = 100.0 if firmware == "old" else 10.0
    df = df.copy().sort_values("Date")
    df = _merge_baro(df, baro, default_atm_kpa)
    df["hydroP_mbar"]   = df["Pressure"] / divisor - (df["Atm_kPa"] * 10.0)
    df["Water_level_m"] = df["hydroP_mbar"] / 1000.0 / GRAVITY_OBS * 1e5 + sensor_height
    df["Sensor_height_m"] = sensor_height
    df["OBS_firmware"]    = firmware
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Hobo CSV parser & water-level formula
# ─────────────────────────────────────────────────────────────────────────────

def parse_hobo_csv(
    file_bytes: bytes, filename: str
) -> tuple[pd.DataFrame | None, str | None]:
    """
    Returns (raw_df, error_msg).
    raw_df columns: Date, Abs_Pres_kPa, Temp_C, Source_file
    """
    text  = file_bytes.decode("utf-8", errors="replace")
    lines = text.splitlines()

    header_idx = next(
        (i for i, ln in enumerate(lines)
         if re.search(r"Date\s*Time", ln, re.IGNORECASE)),
        None,
    )
    if header_idx is None:
        return None, f"No 'Date Time' header found in **{filename}**"

    try:
        df = pd.read_csv(io.StringIO("\n".join(lines[header_idx:])), on_bad_lines="skip")
    except Exception as e:
        return None, f"CSV parse error in {filename}: {e}"

    df.columns = [c.strip() for c in df.columns]

    date_col = next((c for c in df.columns if "Date Time" in c), None)
    pres_col = next(
        (c for c in df.columns
         if re.search(r"abs.?pres", c, re.IGNORECASE)
         or re.search(r"pressure", c, re.IGNORECASE)),
        None,
    )
    temp_col = next(
        (c for c in df.columns
         if re.search(r"temp", c, re.IGNORECASE) and "Date" not in c),
        None,
    )

    if date_col is None:
        return None, f"No 'Date Time' column found in {filename}"
    if pres_col is None:
        return None, (
            f"No absolute pressure column ('Abs Pres, kPa') found in {filename}. "
            "Make sure you exported a HOBO U20L Water Level CSV from HOBOware."
        )

    # ── Parse raw pressure values first, then detect unit ──────────────────
    df["Date"] = pd.to_datetime(df[date_col], errors="coerce")
    df.dropna(subset=["Date"], inplace=True)
    _raw_pres = pd.to_numeric(df[pres_col], errors="coerce")
    _median_val = float(_raw_pres.dropna().median()) if not _raw_pres.dropna().empty else 0.0

    # ── Step 1: try to read unit from column name ───────────────────────────
    #    HOBOware encodes unit as: "Abs Pres, kPa (LGR S/N: …)"
    #                          or: "Abs Pres, mbar (LGR S/N: …)"  etc.
    _unit_from_name: str | None = None
    _unit_match = re.search(
        r"(?:Abs.?Pres|Pressure)[^,]*,\s*(kPa|psi|mbar|hPa|bar|Pa|cmH2O|inH2O)",
        pres_col, re.IGNORECASE,
    )
    if not _unit_match:
        # also handle parenthesis format: "Abs Pres (kPa)"
        _unit_match = re.search(
            r"(?:Abs.?Pres|Pressure)[^(]*\(\s*(kPa|psi|mbar|hPa|bar|Pa|cmH2O|inH2O)\s*\)",
            pres_col, re.IGNORECASE,
        )
    if _unit_match:
        _unit_from_name = _unit_match.group(1).lower()

    # ── Step 2: numeric-range heuristic (primary trust source) ─────────────
    #    At Kathmandu Valley (~1400 m elevation):
    #      kPa  → absolute pressure ≈ 84–200 kPa
    #              (upper bound extended to 200 to handle deeply-deployed or
    #               calibration-drifted sensors that read up to ~10 m head above atm)
    #      mbar → absolute pressure ≈ 840–1150 mbar
    #      hPa  → same numeric range as mbar
    #      psi  → absolute pressure ≈ 12–17 psi
    #      Pa   → absolute pressure ≈ 84 000–115 000 Pa
    if 60.0 <= _median_val <= 200.0:
        _unit_from_range = "kpa"
    elif 600.0 <= _median_val <= 1300.0:
        _unit_from_range = "mbar"   # mbar / hPa identical numerically
    elif 9.0 <= _median_val <= 20.0:
        _unit_from_range = "psi"
    elif 60_000.0 <= _median_val <= 130_000.0:
        _unit_from_range = "pa"
    else:
        _unit_from_range = None   # can't determine from range alone

    # ── Step 3: reconcile ───────────────────────────────────────────────────
    #    If both agree → use that.  If only one known → use it.
    #    If they disagree → trust the numeric range (more reliable than
    #    column-name parsing differences across HOBOware versions).
    _UNIT_TO_KPA = {
        "kpa":   1.0,
        "mbar":  0.1,       # 1 mbar = 0.1 kPa
        "hpa":   0.1,       # 1 hPa  = 0.1 kPa (same as mbar)
        "pa":    0.001,     # 1 Pa   = 0.001 kPa
        "psi":   6.89476,   # 1 psi  = 6.89476 kPa
        "bar":   100.0,
        "cmh2o": 0.0980665,
        "inh2o": 0.249089,
    }
    if _unit_from_range is not None:
        _pres_unit = _unit_from_range          # numeric range is most reliable
    elif _unit_from_name is not None:
        _pres_unit = _unit_from_name           # fall back to column name
    else:
        _pres_unit = "kpa"                     # last resort default
    _conv = _UNIT_TO_KPA.get(_pres_unit, 1.0)

    df["Abs_Pres_kPa"] = _raw_pres * _conv   # always stored as kPa internally
    df["_pres_unit"]   = _pres_unit           # keep for diagnostics
    df["_pres_conv"]   = _conv
    df["_unit_from_name"]  = _unit_from_name  if _unit_from_name  else "?"
    df["_unit_from_range"] = _unit_from_range if _unit_from_range else "?"
    df["Temp_C"]       = pd.to_numeric(df[temp_col], errors="coerce") if temp_col else np.nan
    df["Source_file"]  = filename

    result = df[["Date", "Abs_Pres_kPa", "_pres_unit", "_pres_conv",
                 "_unit_from_name", "_unit_from_range", "Temp_C", "Source_file"]].dropna(subset=["Abs_Pres_kPa"])
    return result.sort_values("Date").reset_index(drop=True), None


def calc_water_level_hobo(
    df: pd.DataFrame,
    sensor_height: float,
    baro: pd.DataFrame | None,
    default_atm_kpa: float,
) -> pd.DataFrame:
    """
    Hobo U20L formula:
        Hydro_kPa     = Abs_Pres_kPa - Atm_kPa
        Water_level_m = Hydro_kPa / 9.80665 + h_sensor
    """
    df = df.copy().sort_values("Date")
    df = _merge_baro(df, baro, default_atm_kpa)
    df["Hydro_kPa"]     = df["Abs_Pres_kPa"] - df["Atm_kPa"]
    df["Water_level_m"] = df["Hydro_kPa"] / GRAVITY_KPA + sensor_height
    df["Sensor_height_m"] = sensor_height
    return df


# ─────────────────────────────────────────────────────────────────────────────
# OBS edge-row trimmer
# ─────────────────────────────────────────────────────────────────────────────

def trim_obs_edges(
    df: pd.DataFrame, n: int = 6
) -> tuple[pd.DataFrame, list[int]]:
    """
    Inspect the first and last `n` rows of an OBS DataFrame.
    A row is dropped if its Pressure OR Ambient_light value is a statistical
    outlier relative to the bulk of the series:
        |value - median| > 3 * IQR   (where IQR is from the middle 90 % of data)
    Returns (trimmed_df, list_of_dropped_indices).
    Skips trimming when fewer than (2*n + 4) rows are present.
    """
    dropped: list[int] = []
    if len(df) < 2 * n + 4:
        return df, dropped

    inner = df.iloc[n:-n]   # bulk of series, used to compute reference stats

    def is_outlier(series: pd.Series, value: float) -> bool:
        q25, q75 = series.quantile([0.25, 0.75])
        iqr = q75 - q25
        if iqr == 0:
            return False
        return abs(value - series.median()) > 3 * iqr

    edge_indices = list(df.index[:n]) + list(df.index[-n:])
    for idx in edge_indices:
        row = df.loc[idx]
        for col in ["Pressure", "Ambient_light"]:
            if col in df.columns:
                val = row.get(col)
                if pd.notna(val) and is_outlier(inner[col].dropna(), val):
                    dropped.append(idx)
                    break   # already marking this row — no need to check other cols

    return df.drop(index=dropped).reset_index(drop=True), dropped


# ─────────────────────────────────────────────────────────────────────────────
# Atmospheric pressure loader
# ─────────────────────────────────────────────────────────────────────────────

def load_baro_csv(file_obj) -> pd.DataFrame | None:
    """Load METER ATMOS 41 Atmospheric_pressure.csv (2-row header)."""
    try:
        content = file_obj.read()
        baro    = pd.read_csv(io.BytesIO(content), skiprows=2, parse_dates=["Timestamps"])
        baro.columns = [c.strip() for c in baro.columns]
        baro.rename(columns={
            "Timestamps": "DateTime",
            "kPa Atmospheric Pressure": "Atm_kPa",
        }, inplace=True)
        if "Atm_kPa" not in baro.columns:
            st.error("'kPa Atmospheric Pressure' column not found in barometric CSV.")
            return None
        return baro[["DateTime", "Atm_kPa"]].dropna().sort_values("DateTime")
    except Exception as e:
        st.error(f"Error loading barometric pressure file: {e}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Shared utilities
# ─────────────────────────────────────────────────────────────────────────────

def _merge_baro(
    df: pd.DataFrame, baro: pd.DataFrame | None, default_kpa: float
) -> pd.DataFrame:
    if baro is not None:
        df = pd.merge_asof(
            df,
            baro.rename(columns={"DateTime": "Date", "Atm_kPa": "_Atm"}),
            on="Date", direction="nearest",
        )
        df["Atm_kPa"] = df["_Atm"].fillna(default_kpa)
        df.drop(columns=["_Atm"], inplace=True)
    else:
        df["Atm_kPa"] = default_kpa
    return df


def to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


def to_excel_bytes(df: pd.DataFrame) -> bytes:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as w:
        df.to_excel(w, index=False)
    return buf.getvalue()


def download_pair(
    label: str,
    df: pd.DataFrame,
    stem: str,
    key: str,
):
    """Render CSV + Excel download buttons side by side."""
    c1, c2 = st.columns(2)
    with c1:
        st.download_button(
            f"⬇ {label} — CSV",
            data=to_csv_bytes(df),
            file_name=f"{stem}.csv",
            mime="text/csv",
            key=f"csv_{key}",
        )
    with c2:
        st.download_button(
            f"⬇ {label} — Excel",
            data=to_excel_bytes(df),
            file_name=f"{stem}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            key=f"xlsx_{key}",
        )


def wl_timeseries_fig(
    df: pd.DataFrame,
    file_col: str,
    heights: dict,
    offset: int = 0,
) -> go.Figure:
    """Interactive water-level timeseries, one trace per file."""
    files = df[file_col].unique().tolist()
    fig   = go.Figure()
    for i, fn in enumerate(files):
        seg = df[df[file_col] == fn]
        sh  = heights.get(fn, 0.10)
        fig.add_trace(go.Scatter(
            x=seg["Date"], y=seg["Water_level_m"],
            mode="lines",
            name=f"{fn}  (h={sh:.3f} m)",
            line=dict(color=PALETTE[(offset + i) % len(PALETTE)], width=1.5),
            hovertemplate="<b>%{x|%Y-%m-%d %H:%M}</b><br>WL: %{y:.4f} m<extra></extra>",
        ))
    fig.update_layout(
        xaxis_title="Date / Time", yaxis_title="Water Level (m)",
        height=420, hovermode="x unified", template="plotly_white",
        margin=dict(t=30, b=40),
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Water Level Conversion — S4W-Nepal",
    page_icon="💧",
    layout="wide",
)
st.title("💧 Water Level Conversion — S4W-Nepal")
st.caption(
    "Upload **OBS** (`.txt`) and/or **Hobo** (HOBOware-exported `.csv`, or raw `.hobo`) files. "
    "File types are detected automatically and the correct formula is applied to each."
)

# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Configuration")
    st.subheader("Atmospheric Pressure")

    atm_mode = st.radio(
        "Source",
        ["Default Kathmandu value", "Upload atmospheric pressure CSV"],
        index=0,
    )
    default_atm_kpa = st.number_input(
        "Default pressure (kPa)",
        min_value=70.0, max_value=110.0,
        value=DEFAULT_ATM_KPA, step=0.1,
        help="Used when no CSV is provided, or as fallback for gaps in CSV coverage.",
    )

    baro_df: pd.DataFrame | None = None
    if atm_mode == "Upload atmospheric pressure CSV":
        baro_file = st.file_uploader(
            "Atmospheric_pressure.csv",
            type=["csv"],
            help="METER ATMOS 41 export — must contain 'kPa Atmospheric Pressure' column.",
        )
        if baro_file:
            baro_df = load_baro_csv(baro_file)
            if baro_df is not None:
                st.success(
                    f"✅ {len(baro_df):,} baro records  \n"
                    f"{baro_df['DateTime'].min().date()} → "
                    f"{baro_df['DateTime'].max().date()}"
                )

    st.divider()
    st.markdown(
        """
**Accepted file types**

| File | Sensor |
|------|--------|
| `.txt` | OpenOBS logger |
| `.csv` | Hobo U20L (HOBOware export) |

Mixed uploads are fine — formats are detected automatically.

> **Hobo users:** connect the logger via the optical USB coupler, open HOBOware → *Device → Readout*, then **Export → Text/CSV** and upload the `.csv` here.
        """
    )

    st.divider()
    st.link_button(
        "🌐 More about us",
        "https://s4w-nepal.smartphones4water.org/",
        use_container_width=True,
    )

# ─────────────────────────────────────────────────────────────────────────────
# File uploader
# ─────────────────────────────────────────────────────────────────────────────
uploaded = st.file_uploader(
    "Upload raw sensor files",
    type=["txt", "csv"],
    accept_multiple_files=True,
    help="Drag and drop one or more `.txt` (OpenOBS) or `.csv` (HOBOware Hobo) files. "
         "Format is detected automatically — you can mix file types freely.",
)

if not uploaded:
    st.info("👆 Upload one or more sensor files to begin.")
    st.stop()

# ─────────────────────────────────────────────────────────────────────────────
# Auto-detect formats
# ─────────────────────────────────────────────────────────────────────────────
file_cache:  dict[str, bytes] = {}
file_format: dict[str, str]   = {}

for uf in uploaded:
    raw = uf.read()
    file_cache[uf.name]  = raw
    file_format[uf.name] = detect_format(raw, uf.name)

obs_names  = [n for n, fmt in file_format.items() if fmt == "obs_txt"]
hobo_names = [n for n, fmt in file_format.items() if fmt == "hobo_csv"]
unk_names  = [n for n, fmt in file_format.items() if fmt == "unknown"]

# Detect OBS firmware version and serial number for each OBS file
file_obs_firmware: dict[str, tuple[str, str | None]] = {}
file_sn: dict[str, str] = {}
for _n in obs_names:
    file_obs_firmware[_n] = detect_obs_firmware(file_cache[_n])
    file_sn[_n] = extract_obs_sn(file_cache[_n])
# For Hobo files use the stem as identifier
for _n in hobo_names:
    file_sn[_n] = _n.rsplit(".", 1)[0]

# Detection summary
fmt_labels = {
    "obs_txt":  "✅ OBS logger (.txt)",
    "hobo_csv": "✅ Hobo HOBOware CSV",
    "unknown":  "❌ Unknown — will be skipped",
}
det_rows: list[dict] = []
for _n in [u.name for u in uploaded]:
    _row: dict = {
        "File":         _n,
        "Detected as":  fmt_labels.get(file_format[_n], file_format[_n]),
        "Sensor SN":    file_sn.get(_n, "—"),
        "OBS Firmware": "—",
    }
    if file_format[_n] == "obs_txt":
        _fw, _fw_date = file_obs_firmware.get(_n, ("new", None))
        _fw_tag = "🔴 Old (÷100)" if _fw == "old" else "🟢 New (÷10)"
        if _fw_date:
            _fw_tag += f"  FW: {_fw_date}"
        _row["OBS Firmware"] = _fw_tag
    det_rows.append(_row)
det_df = pd.DataFrame(det_rows)
with st.expander("� Uploaded files", expanded=False):
    st.table(det_df)
    if any(file_obs_firmware.get(_n, ("new",))[0] == "old" for _n in obs_names):
        st.warning(
            "⚠️ One or more OBS files detected as **old firmware** "
            "(first pressure reading > 50 000). "
            "Pressure divisor set to **100** (~85 000 Pa units → mbar). "
            "You can override the firmware version per-file in the Sensor Settings table below."
        )

if unk_names:
    st.warning("Skipping unrecognised files: " + ", ".join(f"`{f}`" for f in unk_names))

# ─────────────────────────────────────────────────────────────────────────────
# Station name mapping
# ─────────────────────────────────────────────────────────────────────────────
parseable = obs_names + hobo_names
if not parseable:
    st.info("No recognisable sensor files found. Upload `.txt` (OpenOBS) or `.csv` (HOBOware Hobo) files.")
    st.stop()

st.markdown("---")
st.subheader("📍 Station Name Assignment")
st.caption(
    "Serial numbers detected from the uploaded files are listed below. "
    "Give each one a meaningful station name — plots and downloads will be grouped by this name. "
    "Multiple offload files from the same sensor are automatically merged under one station."
)

unique_sns = sorted(set(file_sn.values()))
_sn_key    = tuple(unique_sns)
if st.session_state.get("_sn_key") != _sn_key:
    st.session_state["_sn_key"]          = _sn_key
    st.session_state["sn_station_map"]   = {sn: sn for sn in unique_sns}

sn_map_df = pd.DataFrame({
    "Sensor_SN":    unique_sns,
    "Station_name": [st.session_state["sn_station_map"].get(sn, sn) for sn in unique_sns],
})
edited_sn = st.data_editor(
    sn_map_df,
    column_config={
        "Sensor_SN":    st.column_config.TextColumn("Sensor SN",    disabled=True),
        "Station_name": st.column_config.TextColumn(
            "Station Name",
            help="Type the location name for this sensor, e.g. 'Kasan', 'Nakkhudole'. "
                 "Files with the same SN are automatically grouped under this name.",
        ),
    },
    hide_index=True,
    use_container_width=True,
    key="sn_station_editor",
)
st.session_state["sn_station_map"] = dict(zip(edited_sn["Sensor_SN"], edited_sn["Station_name"]))
sn_station_map = st.session_state["sn_station_map"]

st.markdown("---")
st.subheader("📏 Sensor Settings")
st.caption(
    "Set the sensor height / calibration offset in metres. "
    "**OBS**: physical height from sensor face to channel bed. "
    "**Hobo**: WL calibration offset — use a negative value (e.g. −8) to correct a systematic over-reading. "
    "Use the batch setters to apply a value to all files at once, "
    "or edit individual rows in the table below. "
    "The Firmware column is auto-detected from the raw pressure value — override it here if needed."
)

# ── Build default lists ────────────────────────────────────────────────────────
default_h_list: list[float] = []
for fn in parseable:
    stem = fn.rsplit(".", 1)[0]
    if file_format[fn] == "obs_txt":
        sid = next((k for k in OBS_SENSOR_HEIGHT_DEFAULTS if k in stem), None)
        default_h_list.append(OBS_SENSOR_HEIGHT_DEFAULTS.get(sid, 0.10))
    else:
        # Hobo sensors: default offset is 0.0 m; user can adjust to correct bias
        default_h_list.append(HOBO_SENSOR_HEIGHT_DEFAULT)

default_fw_list: list[str] = [
    file_obs_firmware.get(fn, ("new",))[0] if file_format[fn] == "obs_txt" else "—"
    for fn in parseable
]

# ── Session-state backed heights (reset whenever file list changes) ────────────
_file_key = tuple(sorted(parseable))
if st.session_state.get("_sensor_files_key") != _file_key:
    st.session_state["_sensor_files_key"]  = _file_key
    st.session_state["sensor_heights"]     = dict(zip(parseable, default_h_list))
    st.session_state["obs_fw_overrides"]   = dict(zip(parseable, default_fw_list))
    st.session_state.pop("results", None)   # reset results when files change

current_heights  = st.session_state["sensor_heights"]
fw_overrides     = st.session_state["obs_fw_overrides"]

# ── Batch height / offset setters ───────────────────────────────────────────────
bc1, bc2, _bc3 = st.columns([2, 1, 3])
with bc1:
    batch_h = st.number_input(
        "Set OBS sensor heights to (m):",
        min_value=0.0, max_value=5.0,
        value=0.10, step=0.005, format="%.3f",
        key="batch_height_input",
        help="Applies the same height to every **OBS** file at once.",
    )
with bc2:
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("✅ Apply to OBS", key="apply_batch_heights"):
        updated = dict(st.session_state["sensor_heights"])
        for fn in parseable:
            if file_format[fn] == "obs_txt":
                updated[fn] = batch_h
        st.session_state["sensor_heights"] = updated
        st.rerun()

hobo_files = [fn for fn in parseable if file_format[fn] == "hobo_csv"]
if hobo_files:
    hb1, hb2, _hb3 = st.columns([2, 1, 3])
    with hb1:
        batch_hobo_offset = st.number_input(
            "Set Hobo WL offset to (m):",
            min_value=-50.0, max_value=50.0,
            value=0.0, step=0.1, format="%.3f",
            key="batch_hobo_offset_input",
            help="Applies the same calibration offset to every **Hobo** file at once. "
                 "Use a **negative** value (e.g. −8.5) when the Hobo reads much higher than "
                 "nearby sensors — this corrects systematic pressure bias.",
        )
    with hb2:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("✅ Apply to Hobo", key="apply_batch_hobo_offset"):
            updated = dict(st.session_state["sensor_heights"])
            for fn in hobo_files:
                updated[fn] = batch_hobo_offset
            st.session_state["sensor_heights"] = updated
            st.rerun()
    with _hb3:
        st.markdown(
            "<br><small style='color:grey'>Applies to: " +
            ", ".join(hobo_files) +
            "</small>",
            unsafe_allow_html=True,
        )

# ── Per-SN height setter ───────────────────────────────────────────────────────
obs_files = [fn for fn in parseable if file_format[fn] == "obs_txt"]
obs_sns = list(dict.fromkeys(file_sn.get(fn, "—") for fn in obs_files))  # unique, ordered
if obs_sns:
    with st.expander("🔢 Set height by Serial Number (SN)", expanded=False):
        st.caption(
            "Set sensor height for all files sharing the same SN at once. "
            "Each SN corresponds to one physical sensor."
        )
        for sn_val in obs_sns:
            sn_files = [fn for fn in obs_files if file_sn.get(fn, "—") == sn_val]
            sn_label = sn_station_map.get(sn_val, sn_val)
            cur_h = current_heights.get(sn_files[0], 0.10) if sn_files else 0.10
            sc1, sc2, sc3 = st.columns([1, 1, 2])
            with sc1:
                sn_h = st.number_input(
                    f"**SN {sn_val}** ({sn_label})  —  {len(sn_files)} file(s)",
                    min_value=0.0, max_value=5.0,
                    value=float(cur_h), step=0.005, format="%.3f",
                    key=f"sn_height_input_{sn_val}",
                )
            with sc2:
                st.markdown("<br>", unsafe_allow_html=True)
                if st.button(f"✅ Apply to SN {sn_val}", key=f"apply_sn_{sn_val}"):
                    updated = dict(st.session_state["sensor_heights"])
                    for fn in sn_files:
                        updated[fn] = sn_h
                    st.session_state["sensor_heights"] = updated
                    st.rerun()
            with sc3:
                st.markdown(
                    "<br><small style='color:grey'>" +
                    ", ".join(fn for fn in sn_files) +
                    "</small>",
                    unsafe_allow_html=True,
                )

# ── Editable settings table ────────────────────────────────────────────────────
height_df = pd.DataFrame({
    "File":            parseable,
    "Sensor_SN":       [file_sn.get(n, "—") for n in parseable],
    "Station":         [sn_station_map.get(file_sn.get(n, "—"), file_sn.get(n, "—")) for n in parseable],
    "Type":            [
        "OBS" if file_format[n] == "obs_txt" else "Hobo CSV"
        for n in parseable
    ],
    "Sensor_height_m": [
        current_heights.get(fn, h)
        for fn, h in zip(parseable, default_h_list)
    ],
    "OBS_firmware":    [fw_overrides.get(fn, fw) for fn, fw in zip(parseable, default_fw_list)],
})
edited = st.data_editor(
    height_df,
    column_config={
        "File":       st.column_config.TextColumn("File",       disabled=True),
        "Sensor_SN":  st.column_config.TextColumn("Sensor SN",  disabled=True),
        "Station":    st.column_config.TextColumn("Station",    disabled=True),
        "Type":       st.column_config.TextColumn("Type",       disabled=True),
        "Sensor_height_m": st.column_config.NumberColumn(
            "Height / Offset (m)", min_value=-50.0, max_value=50.0,
            step=0.005, format="%.3f",
            help="**OBS**: physical sensor height above channel bed (≥ 0). "
                 "**Hobo**: calibration offset — set negative (e.g. −8.5) to correct "
                 "systematic over-reading caused by pressure bias.",
        ),
        "OBS_firmware": st.column_config.SelectboxColumn(
            "OBS Firmware",
            options=["new", "old", "—"],
            help="'new' ÷10 (post July 2025).  'old' ÷100 (pre July 2025).  '—' for Hobo.",
        ),
    },
    hide_index=True,
    use_container_width=True,
    key="height_editor",
)
# Persist edits back to session state (Hobo offsets are now user-editable)
st.session_state["sensor_heights"] = {
    fn: h
    for fn, h in zip(edited["File"], edited["Sensor_height_m"])
}
st.session_state["obs_fw_overrides"] = dict(zip(edited["File"], edited["OBS_firmware"]))
file_heights      = st.session_state["sensor_heights"]
file_fw_overrides = st.session_state["obs_fw_overrides"]

# ─────────────────────────────────────────────────────────────────────────────
# Process button
# ─────────────────────────────────────────────────────────────────────────────
if st.button("▶ Process All Files", type="primary"):
    results_list: list[dict] = []
    prog = st.progress(0, text="Initialising…")

    for idx, fname in enumerate(parseable):
        prog.progress(idx / len(parseable), text=f"Processing {fname}…")
        sh  = file_heights.get(fname, 0.10)
        fmt = file_format[fname]
        fb  = file_cache[fname]

        sn      = file_sn.get(fname, "Unknown")
        station = sn_station_map.get(sn, sn)

        if fmt == "obs_txt":
            raw, err = parse_obs_txt(fb, fname)
            if err:
                st.error(f"**{fname}**: {err}")
                continue
            raw, dropped_idx = trim_obs_edges(raw)
            if dropped_idx:
                st.info(
                    f"\u2702\ufe0f **{fname}**: {len(dropped_idx)} edge row(s) removed "
                    f"(outlier Pressure / Ambient\_light at start or end of deployment)."
                )
            fw_val = file_fw_overrides.get(fname, "new")
            if fw_val not in ("old", "new"):
                fw_val = file_obs_firmware.get(fname, ("new",))[0]
            processed = calc_water_level_obs(raw, sh, baro_df, default_atm_kpa, firmware=fw_val)
        else:
            raw, err = parse_hobo_csv(fb, fname)
            if err:
                st.error(f"**{fname}**: {err}")
                continue
            # ── Unit diagnostics ──────────────────────────────────────────
            _unit        = raw["_pres_unit"].iloc[0]        if "_pres_unit"        in raw.columns else "kpa"
            _conv        = float(raw["_pres_conv"].iloc[0]) if "_pres_conv"        in raw.columns else 1.0
            _name_unit   = raw["_unit_from_name"].iloc[0]   if "_unit_from_name"   in raw.columns else "?"
            _range_unit  = raw["_unit_from_range"].iloc[0]  if "_unit_from_range"  in raw.columns else "?"
            _raw_min = (raw["Abs_Pres_kPa"] / _conv).min()
            _raw_max = (raw["Abs_Pres_kPa"] / _conv).max()
            _kpa_min = raw["Abs_Pres_kPa"].min()
            _kpa_max = raw["Abs_Pres_kPa"].max()
            _match_icon = "✅" if _name_unit == _range_unit else "⚠️"
            if _unit != "kpa":
                st.warning(
                    f"⚠️ **{fname}**: Hobo pressure unit auto-detected as **{_unit.upper()}** "
                    f"→ converted ×{_conv} to kPa.  "
                    f"Raw: {_raw_min:.1f}–{_raw_max:.1f} {_unit.upper()}  →  "
                    f"{_kpa_min:.2f}–{_kpa_max:.2f} kPa.  "
                    f"[column-name: {_name_unit}, value-range: {_range_unit}]"
                )
            elif not (75.0 <= _kpa_min and _kpa_max <= 160.0):
                # Check for calibration-offset sensor: high baseline, small variation
                _wl_min_est = (_kpa_min - default_atm_kpa) / 9.80665
                _wl_range   = (_kpa_max - _kpa_min) / 9.80665
                if _kpa_min > 130.0 and _wl_range < 5.0:
                    st.warning(
                        f"⚠️ **{fname}**: Abs Pres baseline is consistently high "
                        f"({_kpa_min:.1f}–{_kpa_max:.1f} kPa, variation only {_wl_range:.2f} m). "
                        f"This is a **sensor calibration offset** — the zero-point is shifted by "
                        f"~+{_kpa_min - default_atm_kpa:.1f} kPa (~{_wl_min_est:.1f} m above expected). "
                        f"Set the **Hobo WL offset** for this file to approximately "
                        f"**{-_wl_min_est:.1f} m** to bring the baseline to zero."
                    )
                else:
                    st.warning(
                        f"⚠️ **{fname}**: Abs Pres range {_kpa_min:.2f}–{_kpa_max:.2f} kPa "
                        f"is outside the expected 75–160 kPa for Kathmandu. "
                        f"[column-name: {_name_unit}, value-range: {_range_unit}] "
                        "If values are ~10× too large, re-export from HOBOware with **kPa** units selected."
                    )
            else:
                st.info(
                    f"{_match_icon} **{fname}**: Abs Pres {_kpa_min:.2f}–{_kpa_max:.2f} kPa "
                    f"(detected: {_unit.upper()}  |  column-name: {_name_unit}, range: {_range_unit})"
                )
            processed = calc_water_level_hobo(raw, sh, baro_df, default_atm_kpa)

        processed["Offload_file"] = fname
        processed["Station"]      = station
        processed["Sensor_SN"]    = sn
        raw["Offload_file"]       = fname
        raw["Station"]            = station
        raw["Sensor_SN"]          = sn
        results_list.append({
            "name": fname, "fmt": fmt, "station": station, "sn": sn,
            "raw": raw, "processed": processed,
        })

    prog.progress(1.0, text="Done.")

    if not results_list:
        st.error("No data could be processed from the uploaded files.")
        st.stop()

    st.session_state["results"] = results_list
    total = sum(len(r["processed"]) for r in results_list)
    st.success(f"✅ {total:,} records processed from {len(results_list)} file(s).")

# ─────────────────────────────────────────────────────────────────────────────
# Results
# ─────────────────────────────────────────────────────────────────────────────
if "results" not in st.session_state:
    st.stop()

results_list: list[dict] = st.session_state["results"]

all_proc = pd.concat([r["processed"] for r in results_list], ignore_index=True).sort_values("Date")
raw_obs  = pd.concat(
    [r["raw"] for r in results_list if r["fmt"] == "obs_txt"], ignore_index=True
) if any(r["fmt"] == "obs_txt" for r in results_list) else pd.DataFrame()
raw_hobo = pd.concat(
    [r["raw"] for r in results_list if r["fmt"] == "hobo_csv"], ignore_index=True
) if any(r["fmt"] == "hobo_csv" for r in results_list) else pd.DataFrame()

# ── Global date-range filter ─────────────────────────────────────────────────
st.markdown("---")
fc1, fc2 = st.columns(2)
with fc1:
    start_d = st.date_input(
        "Start date", value=all_proc["Date"].min().date(),
        min_value=all_proc["Date"].min().date(),
        max_value=all_proc["Date"].max().date(),
    )
with fc2:
    end_d = st.date_input(
        "End date", value=all_proc["Date"].max().date(),
        min_value=all_proc["Date"].min().date(),
        max_value=all_proc["Date"].max().date(),
    )

mask = (all_proc["Date"].dt.date >= start_d) & (all_proc["Date"].dt.date <= end_d)
view = all_proc[mask].copy()

if view.empty:
    st.warning("No data in selected date range.")
    st.stop()

date_tag     = f"{start_d}_{end_d}"
stations_all = sorted(view["Station"].unique().tolist()) if "Station" in view.columns else ["All"]

# ── Global overview metrics ──────────────────────────────────────────────────
m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Records (filtered)", f"{len(view):,}")
m2.metric("Max WL",   f"{view['Water_level_m'].max():.3f} m")
m3.metric("Min WL",   f"{view['Water_level_m'].min():.3f} m")
m4.metric("Mean WL",  f"{view['Water_level_m'].mean():.3f} m")
m5.metric("Stations", str(len(stations_all)))

# ── Overview — all stations on one plot ───────────────────────────────────────
st.markdown("### 📈 All Stations — Overview")
fig_all = go.Figure()
for i, r in enumerate(results_list):
    seg = view[view["Offload_file"] == r["name"]]
    if seg.empty:
        continue
    sh = file_heights.get(r["name"], 0.10)
    fig_all.add_trace(go.Scatter(
        x=seg["Date"], y=seg["Water_level_m"],
        mode="lines",
        name=f"{r.get('station', r['name'])}  [{r['name']}]",
        line=dict(color=PALETTE[i % len(PALETTE)], width=1.5),
        hovertemplate="<b>%{x|%Y-%m-%d %H:%M}</b><br>WL: %{y:.4f} m<extra></extra>",
    ))
fig_all.update_layout(
    xaxis_title="Date / Time", yaxis_title="Water Level (m)",
    height=420, hovermode="x unified", template="plotly_white",
    margin=dict(t=30, b=40),
)
st.plotly_chart(fig_all, use_container_width=True)

# ── Atmospheric pressure overlay ──────────────────────────────────────────────
if baro_df is not None:
    with st.expander("🌡️ Atmospheric Pressure (uploaded CSV)", expanded=False):
        bv = baro_df[
            (baro_df["DateTime"].dt.date >= start_d) &
            (baro_df["DateTime"].dt.date <= end_d)
        ]
        fig_b = go.Figure(go.Scatter(
            x=bv["DateTime"], y=bv["Atm_kPa"],
            mode="lines", line=dict(color="orange", width=1),
        ))
        fig_b.update_layout(
            xaxis_title="Date / Time", yaxis_title="kPa",
            height=240, template="plotly_white", margin=dict(t=20, b=30),
        )
        st.plotly_chart(fig_b, use_container_width=True)

st.markdown("---")

# ─────────────────────────────────────────────────────────────────────────────
# Manual Value Editor
# ─────────────────────────────────────────────────────────────────────────────
if "manual_edit_log" not in st.session_state:
    st.session_state["manual_edit_log"] = []

with st.expander("✏️ Manual Value Editor — Override Water Level in a Date Range", expanded=False):
    st.caption(
        "Set all **Water_level_m** values within a chosen date/time range to a specific value. "
        "Use this to correct spikes, flatten erroneous periods, or apply known readings. "
        "After clicking **Apply**, adjust the range and value and apply again as many times as needed. "
        "Edits are reflected in all plots and downloads on this page. "
        "**Re-processing the files will reset all manual edits.**"
    )

    _me_c1, _me_c2 = st.columns([2, 2])
    with _me_c1:
        _file_opts = ["— All files —"] + [r["name"] for r in results_list]
        me_file = st.selectbox(
            "Apply to file",
            options=_file_opts,
            key="me_file_select",
            help="Choose a specific offload file, or leave as '— All files —' to apply to every file.",
        )
    with _me_c2:
        me_value = st.number_input(
            "New Water_level_m value (m)",
            min_value=-100.0, max_value=100.0,
            value=0.0, step=0.001, format="%.4f",
            key="me_value_input",
            help="All rows in the selected date/time range will have their Water_level_m set to this value.",
        )

    _dt_min = all_proc["Date"].min()
    _dt_max = all_proc["Date"].max()

    _me_dr1, _me_dr2, _me_dr3, _me_dr4 = st.columns(4)
    with _me_dr1:
        me_start_date = st.date_input(
            "Start date",
            value=_dt_min.date(),
            min_value=_dt_min.date(), max_value=_dt_max.date(),
            key="me_start_date",
        )
    with _me_dr2:
        me_start_time = st.time_input(
            "Start time",
            value=_dt_min.time(),
            key="me_start_time",
            step=60,
        )
    with _me_dr3:
        me_end_date = st.date_input(
            "End date",
            value=_dt_max.date(),
            min_value=_dt_min.date(), max_value=_dt_max.date(),
            key="me_end_date",
        )
    with _me_dr4:
        me_end_time = st.time_input(
            "End time",
            value=_dt_max.time(),
            key="me_end_time",
            step=60,
        )

    me_start_dt = pd.Timestamp(datetime.combine(me_start_date, me_start_time))
    me_end_dt   = pd.Timestamp(datetime.combine(me_end_date,   me_end_time))

    _apply_col, _preview_col = st.columns([1, 3])
    with _apply_col:
        _do_apply = st.button("✅ Apply Range Edit", key="apply_manual_edit", type="primary")
    with _preview_col:
        if me_start_dt < me_end_dt:
            _prev_mask = (all_proc["Date"] >= me_start_dt) & (all_proc["Date"] <= me_end_dt)
            if me_file != "— All files —":
                _prev_mask = _prev_mask & (all_proc["Offload_file"] == me_file)
            _preview_n = int(_prev_mask.sum())
            st.info(f"**{_preview_n}** row(s) will be set to **{me_value:.4f} m** on Apply.")
        else:
            st.warning("⚠️ Start must be before end.")

    if _do_apply:
        if me_start_dt >= me_end_dt:
            st.error("Start must be before end — no changes made.")
        else:
            _target_files = (
                [r["name"] for r in results_list]
                if me_file == "— All files —"
                else [me_file]
            )
            _new_results = []
            _total_changed = 0
            for _r in st.session_state["results"]:
                if _r["name"] in _target_files:
                    _df = _r["processed"].copy()
                    _m  = (_df["Date"] >= me_start_dt) & (_df["Date"] <= me_end_dt)
                    _total_changed += int(_m.sum())
                    _df.loc[_m, "Water_level_m"] = float(me_value)
                    _r = dict(_r)
                    _r["processed"] = _df
                _new_results.append(_r)
            st.session_state["results"] = _new_results
            st.session_state["manual_edit_log"].append({
                "File":         me_file,
                "Start":        str(me_start_dt),
                "End":          str(me_end_dt),
                "Value (m)":    round(float(me_value), 4),
                "Rows changed": _total_changed,
            })
            st.rerun()

    if st.session_state["manual_edit_log"]:
        st.markdown("**Edit history (this session):**")
        st.dataframe(
            pd.DataFrame(st.session_state["manual_edit_log"]),
            use_container_width=True, hide_index=True,
        )
        if st.button("🗑️ Clear history log (data edits remain)", key="clear_edit_log"):
            st.session_state["manual_edit_log"] = []
            st.rerun()

st.markdown("---")

# ─────────────────────────────────────────────────────────────────────────────
# Per-station results (one tab per station)
# ─────────────────────────────────────────────────────────────────────────────
st.subheader("📍 Per-Station Results")

# Group result dicts by station name
station_to_results: dict[str, list[dict]] = {}
for _r in results_list:
    _st = _r.get("station", _r["name"])
    station_to_results.setdefault(_st, []).append(_r)

station_list = sorted(station_to_results.keys())
tab_containers = st.tabs(station_list) if len(station_list) > 1 else [st.container()]

for tab_c, station in zip(tab_containers, station_list):
    with tab_c:
        st_results = station_to_results[station]
        st_files   = [r["name"] for r in st_results]
        st_fmt     = st_results[0]["fmt"]

        st_proc = pd.concat([r["processed"] for r in st_results], ignore_index=True).sort_values("Date")
        st_raw  = pd.concat([r["raw"]       for r in st_results], ignore_index=True).sort_values("Date")
        st_filt = st_proc[
            (st_proc["Date"].dt.date >= start_d) &
            (st_proc["Date"].dt.date <= end_d)
        ].copy()

        if st_filt.empty:
            st.info(f"No data for **{station}** in the selected date range.")
            continue

        sn_display = st_proc["Sensor_SN"].iloc[0] if "Sensor_SN" in st_proc.columns else "—"
        st.markdown(
            f"**Station:** `{station}`  ·  "
            f"**Sensor SN:** `{sn_display}`  ·  "
            f"**Offload files:** {len(st_files)}"
        )

        # Station metrics
        sm1, sm2, sm3, sm4 = st.columns(4)
        sm1.metric("Records", f"{len(st_filt):,}")
        sm2.metric("Max WL",  f"{st_filt['Water_level_m'].max():.3f} m")
        sm3.metric("Min WL",  f"{st_filt['Water_level_m'].min():.3f} m")
        sm4.metric("Mean WL", f"{st_filt['Water_level_m'].mean():.3f} m")

        # Water level timeseries — all offloads on one plot, colored per offload file
        fig_wl = go.Figure()
        for i, fn in enumerate(st_files):
            seg = st_filt[st_filt["Offload_file"] == fn]
            if seg.empty:
                continue
            sh = file_heights.get(fn, 0.10)
            fig_wl.add_trace(go.Scatter(
                x=seg["Date"], y=seg["Water_level_m"],
                mode="lines",
                name=f"{fn}  (h={sh:.3f} m)",
                line=dict(color=PALETTE[i % len(PALETTE)], width=1.5),
                hovertemplate="<b>%{x|%Y-%m-%d %H:%M}</b><br>WL: %{y:.4f} m<extra></extra>",
            ))
        fig_wl.update_layout(
            title=f"Water Level — {station}",
            xaxis_title="Date / Time", yaxis_title="Water Level (m)",
            height=420, hovermode="x unified", template="plotly_white",
            margin=dict(t=50, b=40),
        )
        st.plotly_chart(fig_wl, use_container_width=True)

        # Sub-panels (OBS or Hobo)
        if st_fmt == "obs_txt":
            with st.expander(f"📊 {station} — Pressure / Backscatter / Temperature", expanded=False):
                fig_sub = make_subplots(
                    rows=4, cols=1, shared_xaxes=True,
                    subplot_titles=("Water Level (m)", "Raw Pressure", "Backscatter", "Water Temp (raw)"),
                    vertical_spacing=0.06,
                )
                for i, fn in enumerate(st_files):
                    seg = st_filt[st_filt["Offload_file"] == fn]
                    if seg.empty:
                        continue
                    c = PALETTE[i % len(PALETTE)]
                    for row, ycol in enumerate(["Water_level_m", "Pressure", "Backscatter", "Water_temp"], 1):
                        if ycol in seg.columns:
                            fig_sub.add_trace(go.Scatter(
                                x=seg["Date"], y=seg[ycol], mode="lines",
                                name=fn, line=dict(color=c, width=1.2),
                                showlegend=(row == 1),
                            ), row=row, col=1)
                for row, lbl in enumerate(["WL (m)", "Pressure (raw)", "Backscatter", "Temp (raw)"], 1):
                    fig_sub.update_yaxes(title_text=lbl, row=row, col=1)
                fig_sub.update_xaxes(title_text="Date / Time", row=4, col=1)
                fig_sub.update_layout(height=900, template="plotly_white",
                                      hovermode="x unified", margin=dict(t=40, b=40))
                st.plotly_chart(fig_sub, use_container_width=True)

        elif st_fmt == "hobo_csv":
            with st.expander(f"📊 {station} — Absolute Pressure / Temperature", expanded=False):
                fig_sub = make_subplots(
                    rows=3, cols=1, shared_xaxes=True,
                    subplot_titles=("Water Level (m)", "Absolute Pressure (kPa)", "Water Temp (°C)"),
                    vertical_spacing=0.07,
                )
                for i, fn in enumerate(st_files):
                    seg = st_filt[st_filt["Offload_file"] == fn]
                    if seg.empty:
                        continue
                    c = PALETTE[i % len(PALETTE)]
                    for row, ycol in enumerate(["Water_level_m", "Abs_Pres_kPa", "Temp_C"], 1):
                        if ycol in seg.columns:
                            fig_sub.add_trace(go.Scatter(
                                x=seg["Date"], y=seg[ycol], mode="lines",
                                name=fn, line=dict(color=c, width=1.2),
                                showlegend=(row == 1),
                            ), row=row, col=1)
                for row, lbl in enumerate(["WL (m)", "Abs Pres (kPa)", "Temp (°C)"], 1):
                    fig_sub.update_yaxes(title_text=lbl, row=row, col=1)
                fig_sub.update_xaxes(title_text="Date / Time", row=3, col=1)
                fig_sub.update_layout(height=720, template="plotly_white",
                                      hovermode="x unified", margin=dict(t=40, b=40))
                st.plotly_chart(fig_sub, use_container_width=True)

        # Data previews
        with st.expander(f"🗃️ {station} — Processed Data (preview)", expanded=False):
            proc_cols = [
                "Date", "Station", "Sensor_SN", "Offload_file",
                "Sensor_height_m", "Atm_kPa", "Water_level_m",
                "Pressure", "hydroP_mbar", "Ambient_light", "Backscatter", "Water_temp", "Battery",
                "Abs_Pres_kPa", "Hydro_kPa", "Temp_C",
            ]
            avail = [c for c in proc_cols if c in st_filt.columns]
            st.dataframe(st_filt[avail].head(500), use_container_width=True, height=280)

        with st.expander(f"🗃️ {station} — Raw Data (preview)", expanded=False):
            st.caption("Direct sensor output — no water-level conversion.")
            st.dataframe(st_raw.head(500), use_container_width=True, height=260)

        # Per-station downloads
        st.markdown(f"**⬇ Downloads — {station}**")
        _safe = station.replace(" ", "_")
        dl1, dl2 = st.columns(2)
        with dl1:
            download_pair(
                "Raw (all offloads)", st_raw,
                f"{_safe}_raw", f"st_raw_{_safe}"
            )
        with dl2:
            download_pair(
                "Processed WL", st_filt,
                f"{_safe}_WaterLevel_{date_tag}", f"st_proc_{_safe}"
            )
        st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# Combined downloads — all stations
# ─────────────────────────────────────────────────────────────────────────────
st.subheader("💾 Combined Downloads — All Stations")
download_pair("All Stations · Processed WL", view, f"WaterLevel_all_{date_tag}", "all_proc")
st.markdown("---")
if not raw_obs.empty:
    st.markdown("#### OBS Raw — All Stations")
    download_pair("OBS Raw", raw_obs, "OBS_raw_all", "obs_raw_combined")
if not raw_hobo.empty:
    st.markdown("#### Hobo Raw — All Stations")
    download_pair("Hobo Raw", raw_hobo, "Hobo_raw_all", "hobo_raw_combined")

# ─────────────────────────────────────────────────────────────────────────────
# Comparison Explorer
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("---")
st.subheader("🔀 Comparison Explorer")
st.caption(
    "Compare water levels or other parameters across stations over the same period, "
    "or compare different time periods at the same station."
)

comp_mode = st.radio(
    "Comparison mode",
    ["📍 Multiple stations — same period", "📅 Same station — different periods"],
    horizontal=True,
    label_visibility="collapsed",
    key="comp_mode_radio",
)

# ── Mode A: multiple stations, same period ────────────────────────────────────────────
if comp_mode == "📍 Multiple stations — same period":
    cmp_c1, cmp_c2, cmp_c3 = st.columns([3, 1, 1])
    cmp_stations = cmp_c1.multiselect(
        "Stations to compare",
        options=station_list,
        default=station_list[:min(len(station_list), 4)],
        key="cmp_a_stations",
    )
    cmp_start = cmp_c2.date_input(
        "From",
        value=all_proc["Date"].min().date(),
        min_value=all_proc["Date"].min().date(),
        max_value=all_proc["Date"].max().date(),
        key="cmp_a_start",
    )
    cmp_end = cmp_c3.date_input(
        "To",
        value=all_proc["Date"].max().date(),
        min_value=all_proc["Date"].min().date(),
        max_value=all_proc["Date"].max().date(),
        key="cmp_a_end",
    )

    param_options: dict[str, str] = {"Water Level (m)": "Water_level_m"}
    if "Temp_C" in all_proc.columns:
        param_options["Water Temperature (°C)"] = "Temp_C"
    if "Water_temp" in all_proc.columns:
        param_options["Water Temp — raw (OBS)"] = "Water_temp"
    if "Abs_Pres_kPa" in all_proc.columns:
        param_options["Absolute Pressure (kPa)"] = "Abs_Pres_kPa"

    cmp_param_label = st.selectbox("Parameter", list(param_options.keys()), key="cmp_a_param")
    cmp_param       = param_options[cmp_param_label]

    if not cmp_stations:
        st.info("Select at least one station above.")
    else:
        cmp_mask = (
            (all_proc["Date"].dt.date >= cmp_start) &
            (all_proc["Date"].dt.date <= cmp_end) &
            (all_proc["Station"].isin(cmp_stations))
        )
        cmp_data = all_proc[cmp_mask]
        if cmp_data.empty or cmp_param not in cmp_data.columns:
            st.warning("No data available for the selected stations and period.")
        else:
            fig_cmp = go.Figure()
            for i, stn in enumerate(cmp_stations):
                seg = cmp_data[cmp_data["Station"] == stn]
                if seg.empty:
                    continue
                fig_cmp.add_trace(go.Scatter(
                    x=seg["Date"], y=seg[cmp_param],
                    mode="lines", name=stn,
                    line=dict(color=PALETTE[i % len(PALETTE)], width=1.5),
                    hovertemplate=f"<b>%{{x|%Y-%m-%d %H:%M}}</b><br>{stn}: %{{y:.4f}}<extra></extra>",
                ))
            fig_cmp.update_layout(
                xaxis_title="Date / Time", yaxis_title=cmp_param_label,
                height=420, hovermode="x unified", template="plotly_white",
                margin=dict(t=30, b=40),
            )
            st.plotly_chart(fig_cmp, use_container_width=True)
            out_cols = [c for c in ["Date", "Station", "Sensor_SN", cmp_param] if c in cmp_data.columns]
            download_pair(
                f"Comparison — {cmp_param_label}",
                cmp_data[out_cols],
                f"comparison_stations_{cmp_start}_{cmp_end}",
                "cmp_a_dl",
            )

# ── Mode B: same station, different periods ────────────────────────────────────────────
else:
    bp_c1, bp_c2 = st.columns([3, 1])
    cmp_station = bp_c1.selectbox("Station", options=station_list, key="cmp_b_station")
    n_periods   = int(bp_c2.select_slider("Periods", options=[2, 3, 4], value=2, key="cmp_b_nperiods"))

    align_mode = st.radio(
        "Time axis",
        ["Actual dates (calendar overlay)", "Relative (hours from each period start)"],
        horizontal=True,
        key="cmp_b_align",
    )

    st_data_all = all_proc[all_proc["Station"] == cmp_station]
    date_min_s  = st_data_all["Date"].min()
    date_max_s  = st_data_all["Date"].max()
    chunk_days  = max(1, int((date_max_s - date_min_s).days / n_periods))

    period_cols = st.columns(n_periods)
    periods: list[tuple] = []
    for i, col in enumerate(period_cols):
        default_ps = (date_min_s + pd.Timedelta(days=i * chunk_days)).date()
        default_pe = min(
            date_max_s.date(),
            (date_min_s + pd.Timedelta(days=(i + 1) * chunk_days - 1)).date(),
        )
        with col:
            ps = st.date_input(
                f"Period {i + 1} — start",
                value=default_ps,
                min_value=date_min_s.date(), max_value=date_max_s.date(),
                key=f"cmp_b_ps_{i}",
            )
            pe = st.date_input(
                f"Period {i + 1} — end",
                value=default_pe,
                min_value=date_min_s.date(), max_value=date_max_s.date(),
                key=f"cmp_b_pe_{i}",
            )
            periods.append((ps, pe))

    x_title  = "Hours from period start" if "Relative" in align_mode else "Date / Time"
    fig_cmp2 = go.Figure()
    cmp_b_frames: list[pd.DataFrame] = []
    has_cmp_data = False
    for i, (ps, pe) in enumerate(periods):
        seg = st_data_all[
            (st_data_all["Date"].dt.date >= ps) &
            (st_data_all["Date"].dt.date <= pe)
        ].copy()
        if seg.empty:
            continue
        has_cmp_data = True
        plabel = f"{ps} → {pe}"
        if "Relative" in align_mode:
            seg["_x"] = (seg["Date"] - seg["Date"].min()).dt.total_seconds() / 3600
        else:
            seg["_x"] = seg["Date"]
        seg["Period"] = plabel
        fig_cmp2.add_trace(go.Scatter(
            x=seg["_x"], y=seg["Water_level_m"],
            mode="lines", name=plabel,
            line=dict(color=PALETTE[i % len(PALETTE)], width=1.5),
            hovertemplate=f"{plabel}<br>%{{x}}<br>WL: %{{y:.4f}} m<extra></extra>",
        ))
        cmp_b_frames.append(seg[["Date", "Period", "Water_level_m"]].copy())

    if has_cmp_data:
        fig_cmp2.update_layout(
            title=f"Water Level — {cmp_station}",
            xaxis_title=x_title, yaxis_title="Water Level (m)",
            height=420, hovermode="x unified", template="plotly_white",
            margin=dict(t=50, b=40),
        )
        st.plotly_chart(fig_cmp2, use_container_width=True)
        if cmp_b_frames:
            download_pair(
                f"Comparison — {cmp_station}",
                pd.concat(cmp_b_frames, ignore_index=True),
                f"comparison_{cmp_station.replace(' ', '_')}_periods",
                "cmp_b_dl",
            )
    else:
        st.warning("No data in any of the selected periods for this station.")

# ─────────────────────────────────────────────────────────────────────────────
# About
# ─────────────────────────────────────────────────────────────────────────────
with st.expander("ℹ️ About — Formulas & Data Sources", expanded=False):
    st.markdown(
        r"""
### OBS Sensor (OpenOBS `.txt`)
Raw pressure is a dimensionless ADC integer — **two firmware variants exist**:

**New firmware** (post ~July 2025, raw pressure ~8 600 units, divisor = 10):

$$P_{hydro}\,(\text{mbar}) = \frac{P_{raw}}{10} - P_{atm}\,(\text{kPa}) \times 10$$

**Old firmware** (pre ~July 2025, raw pressure ~85 000 units ≈ Pa, divisor = 100):

$$P_{hydro}\,(\text{mbar}) = \frac{P_{raw}}{100} - P_{atm}\,(\text{kPa}) \times 10$$

Both variants then:

$$H\,(\text{m}) = \frac{P_{hydro}}{1000 \times 9806.65} \times 10^5 + h_{sensor}$$

Auto-detection uses the **first raw pressure reading**: values above 50\u202f000 → old firmware (÷100); at or below → new firmware (÷10).
Override the detected firmware in the *Sensor Settings* table if needed.

---

### Hobo U20L (HOBOware-exported `.csv`)
The HOBO U20L stores **absolute** pressure (water + atmosphere) in kPa:

$$P_{hydro}\,(\text{kPa}) = P_{abs} - P_{atm}$$

$$H\,(\text{m}) = \frac{P_{hydro}}{9.80665} + h_{sensor}$$

($\rho_{water} = 1000\,\text{kg/m}^3$, $g = 9.80665\,\text{m/s}^2$)

---

### Sensor Inventory
| Watershed | OBS Sensors | Hobo Sites |
|-----------|-------------|------------|
| Hanumante | 303, 469, 470 | Gonsal, Maheshwari, RadheRadhe |
| Nakkhu    | 300, 301, 304, 455, 467 | — |
        """
    )
