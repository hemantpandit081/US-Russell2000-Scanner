import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import requests
import io
import json
import os
from datetime import datetime
from zoneinfo import ZoneInfo
import streamlit.components.v1 as components


# =========================================================
# PAGE
# =========================================================

st.set_page_config(
    page_title="Russell 2000 Momentum Scanner",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# =========================================================
# CSS
# =========================================================

st.markdown("""
<style>

.block-container {
    padding-top: 0.5rem;
    padding-left: 0.6rem;
    padding-right: 0.6rem;
    padding-bottom: 0rem;
}

[data-testid="stSidebar"] {
    width: 280px;
}

div[data-testid="column"] {
    padding-left: 3px;
    padding-right: 3px;
}

.stock-row {
    font-size: 12px;
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# SETTINGS
# =========================================================

SETTINGS_FILE = "scanner_settings.json"

DEFAULT_SETTINGS = {
    "min_price": 1.0,
    "max_price": 100.0,
    "min_volume": 100000,
    "min_rvol": 1.5,
    "min_change": 1.0,
    "min_dollar_volume": 1000000,
    "repeat_tolerance": 0.90,
    "refresh_seconds": 60,
    "auto_scan": True,
    "chart_interval": "1",
    "first_stage_limit": 200,
    "final_results": 50
}


def load_settings():

    if os.path.exists(SETTINGS_FILE):

        try:
            with open(SETTINGS_FILE, "r") as f:
                saved = json.load(f)

            result = DEFAULT_SETTINGS.copy()
            result.update(saved)

            return result

        except:
            pass

    return DEFAULT_SETTINGS.copy()


def save_settings():

    with open(SETTINGS_FILE, "w") as f:
        json.dump(settings, f, indent=4)


settings = load_settings()


# =========================================================
# SESSION STATE
# =========================================================

if "selected_symbol" not in st.session_state:
    st.session_state.selected_symbol = "AAPL"

if "previous_volumes" not in st.session_state:
    st.session_state.previous_volumes = {}

if "scan_results" not in st.session_state:
    st.session_state.scan_results = pd.DataFrame()


# =========================================================
# TIME
# =========================================================

NY_TZ = ZoneInfo("America/New_York")


def market_open():

    now = datetime.now(NY_TZ)

    if now.weekday() >= 5:
        return False

    minutes = now.hour * 60 + now.minute

    return 570 <= minutes <= 960


# =========================================================
# RUSSELL 2000 UNIVERSE
# =========================================================

@st.cache_data(ttl=86400, show_spinner=False)
def load_russell2000():

    url = (
        "https://www.ishares.com/us/products/239710/"
        "ishares-russell-2000-etf/"
        "1467271812596.ajax?"
        "fileType=csv&fileName=IWM_holdings&dataType=fund"
    )

    try:

        response = requests.get(
            url,
            timeout=30,
            headers={
                "User-Agent": "Mozilla/5.0"
            }
        )

        response.raise_for_status()

        text = response.text

        # iShares CSV normally contains several information
        # rows before the actual holdings table.
        lines = text.splitlines()

        header_index = None

        for i, line in enumerate(lines):

            if line.startswith("Ticker,"):
                header_index = i
                break

        if header_index is None:
            return []

        csv_text = "\n".join(
            lines[header_index:]
        )

        df = pd.read_csv(
            io.StringIO(csv_text)
        )

        if "Ticker" not in df.columns:
            return []

        symbols = []

        for ticker in df["Ticker"].astype(str):

            ticker = ticker.strip()

            if ticker == "nan":
                continue

            # Remove common special characters
            ticker = ticker.replace(".", "-")

            # Skip non-equity rows
            if ticker in [
                "",
                "-",
                "USD",
                "CASH",
                "N/A"
            ]:
                continue

            # Keep normal US ticker format
            if len(ticker) <= 6:
                symbols.append(ticker)

        symbols = list(dict.fromkeys(symbols))

        return symbols

    except Exception:

        return []


# =========================================================
# FALLBACK SYMBOLS
# =========================================================

FALLBACK_SYMBOLS = [
    "MARA", "RIOT", "IONQ", "RKLB", "SOUN",
    "RGTI", "ACHR", "JOBY", "OPEN", "DNA",
    "PLUG", "CHPT", "LCID", "FSR", "BBAI",
    "AI", "LUNR", "ASTS", "GRAB", "HIMS",
    "UPST", "AFRM", "DKNG", "CLOV", "GME"
]


# =========================================================
# BATCH DOWNLOAD
# =========================================================

def download_batch(symbols):

    try:

        data = yf.download(
            symbols,
            period="5d",
            interval="5m",
            auto_adjust=False,
            prepost=False,
            progress=False,
            threads=True,
            group_by="column"
        )

        return data

    except Exception:

        return None


# =========================================================
# EXTRACT SYMBOL DATA
# =========================================================

def extract_symbol_data(data, symbol):

    try:

        if data is None or data.empty:
            return None

        if isinstance(data.columns, pd.MultiIndex):

            if symbol not in data.columns.get_level_values(1):
                return None

            close = data["Close"][symbol]
            volume = data["Volume"][symbol]

        else:

            close = data["Close"]
            volume = data["Volume"]

        df = pd.DataFrame({
            "Close": close,
            "Volume": volume
        }).dropna()

        if df.empty:
            return None

        return df

    except Exception:

        return None


# =========================================================
# DAILY DATA
# =========================================================

@st.cache_data(ttl=300, show_spinner=False)
def get_daily_batch(symbols):

    try:

        data = yf.download(
            symbols,
            period="20d",
            interval="1d",
            auto_adjust=False,
            progress=False,
            threads=True,
            group_by="column"
        )

        return data

    except Exception:

        return None


# =========================================================
# CALCULATE CANDIDATE
# =========================================================

def calculate_candidate(
    symbol,
    intraday,
    daily
):

    try:

        if intraday is None or intraday.empty:
            return None

        price = float(
            intraday["Close"].iloc[-1]
        )

        session_volume = float(
            intraday["Volume"].fillna(0).sum()
        )

        if daily is not None and not daily.empty:

            if len(daily) >= 2:

                previous_close = float(
                    daily["Close"].iloc[-2]
                )

            else:

                previous_close = price

            if len(daily) >= 6:

                avg_volume = float(
                    daily["Volume"].iloc[-6:-1].mean()
                )

            else:

                avg_volume = float(
                    daily["Volume"].mean()
                )

        else:

            previous_close = price
            avg_volume = 0

        if previous_close > 0:

            change = (
                (price - previous_close)
                / previous_close
                * 100
            )

        else:

            change = 0

        if avg_volume > 0:

            rvol = session_volume / avg_volume

        else:

            rvol = 0

        dollar_volume = (
            price * session_volume
        )

        # Last 5-minute volume
        last_bar_volume = float(
            intraday["Volume"].iloc[-1]
        )

        if len(intraday) >= 6:

            average_recent_bar_volume = float(
                intraday["Volume"].iloc[-6:-1].mean()
            )

        else:

            average_recent_bar_volume = 0

        if average_recent_bar_volume > 0:

            volume_spike = (
                last_bar_volume
                / average_recent_bar_volume
            )

        else:

            volume_spike = 0

        # High of day
        high_of_day = float(
            intraday["Close"].max()
        )

        if high_of_day > 0:

            distance_from_high = (
                (high_of_day - price)
                / high_of_day
                * 100
            )

        else:

            distance_from_high = 0

        return {
            "Symbol": symbol,
            "Price": price,
            "Change": change,
            "RVOL": rvol,
            "Volume": session_volume,
            "Dollar": dollar_volume,
            "Spike": volume_spike,
            "HOD": distance_from_high
        }

    except:

        return None


# =========================================================
# REPEAT VOLUME
# =========================================================

def repeat_volume(
    symbol,
    current_volume
):

    previous = (
        st.session_state
        .previous_volumes
        .get(symbol)
    )

    signal = False

    if previous is not None and previous > 0:

        ratio = (
            current_volume
            / previous
        )

        if ratio >= settings[
            "repeat_tolerance"
        ]:

            signal = True

    st.session_state.previous_volumes[
        symbol
    ] = current_volume

    return signal


# =========================================================
# FIRST STAGE
# =========================================================

def first_stage_scan(symbols):

    candidates = []

    chunk_size = 100

    progress = st.progress(0)

    total_chunks = max(
        1,
        int(
            np.ceil(
                len(symbols) / chunk_size
            )
        )
    )

    for chunk_number, start in enumerate(
        range(0, len(symbols), chunk_size)
    ):

        chunk = symbols[
            start:start + chunk_size
        ]

        data = download_batch(chunk)

        if data is None:
            continue

        for symbol in chunk:

            intraday = extract_symbol_data(
                data,
                symbol
            )

            if intraday is None:
                continue

            try:

                price = float(
                    intraday["Close"].iloc[-1]
                )

                volume = float(
                    intraday["Volume"]
                    .fillna(0)
                    .sum()
                )

                previous_close = (
                    float(
                        intraday["Close"]
                        .iloc[0]
                    )
                )

                if previous_close > 0:

                    change = (
                        (price - previous_close)
                        / previous_close
                        * 100
                    )

                else:

                    change = 0

                dollar_volume = (
                    price * volume
                )

                if price < settings[
                    "min_price"
                ]:
                    continue

                if price > settings[
                    "max_price"
                ]:
                    continue

                if volume < settings[
                    "min_volume"
                ]:
                    continue

                if change < settings[
                    "min_change"
                ]:
                    continue

                if dollar_volume < settings[
                    "min_dollar_volume"
                ]:
                    continue

                candidates.append({
                    "Symbol": symbol,
                    "Price": price,
                    "Change": change,
                    "Volume": volume,
                    "Dollar": dollar_volume
                })

            except:
                continue

        progress.progress(
            min(
                1.0,
                (chunk_number + 1)
                / total_chunks
            )
        )

    progress.empty()

    if not candidates:
        return []

    df = pd.DataFrame(candidates)

    df = df.sort_values(
        ["Change", "Dollar"],
        ascending=False
    )

    limit = int(
        settings["first_stage_limit"]
    )

    return df.head(limit)[
        "Symbol"
    ].tolist()


# =========================================================
# DETAILED SCAN
# =========================================================

def detailed_scan(symbols):

    results = []

    if not symbols:
        return pd.DataFrame()

    daily = get_daily_batch(symbols)

    chunk_size = 50

    for start in range(
        0,
        len(symbols),
        chunk_size
    ):

        chunk = symbols[
            start:start + chunk_size
        ]

        data = download_batch(chunk)

        if data is None:
            continue

        for symbol in chunk:

            intraday = extract_symbol_data(
                data,
                symbol
            )

            if intraday is None:
                continue

            # Daily data for symbol
            symbol_daily = None

            try:

                if daily is not None:

                    if isinstance(
                        daily.columns,
                        pd.MultiIndex
                    ):

                        symbol_daily = pd.DataFrame({
                            "Close":
                                daily["Close"][symbol],
                            "Volume":
                                daily["Volume"][symbol]
                        }).dropna()

                    else:

                        symbol_daily = daily[
                            ["Close", "Volume"]
                        ].dropna()

            except:
                symbol_daily = None

            result = calculate_candidate(
                symbol,
                intraday,
                symbol_daily
            )

            if result is None:
                continue

            if result["RVOL"] < settings[
                "min_rvol"
            ]:
                continue

            if result["Change"] < settings[
                "min_change"
            ]:
                continue

            if result["Volume"] < settings[
                "min_volume"
            ]:
                continue

            if result["Dollar"] < settings[
                "min_dollar_volume"
            ]:
                continue

            result["Repeat"] = repeat_volume(
                symbol,
                result["Volume"]
            )

            # Momentum score
            score = 0

            score += min(
                max(result["Change"], 0),
                20
            )

            score += min(
                max(result["RVOL"], 0) * 2,
                20
            )

            score += min(
                max(result["Spike"], 0) * 2,
                20
            )

            if result["HOD"] <= 2:
                score += 20

            if result["Repeat"]:
                score += 20

            result["Score"] = score

            results.append(result)

    if not results:
        return pd.DataFrame()

    df = pd.DataFrame(results)

    df = df.sort_values(
        [
            "Score",
            "RVOL",
            "Change"
        ],
        ascending=False
    )

    final_limit = int(
        settings["final_results"]
    )

    return df.head(
        final_limit
    )


# =========================================================
# FULL SCAN
# =========================================================

def scan_russell():

    symbols = load_russell2000()

    if not symbols:

        symbols = FALLBACK_SYMBOLS

    candidates = first_stage_scan(
        symbols
    )

    return detailed_scan(
        candidates
    )


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.title("⚙️ Scanner Filters")

    settings["min_price"] = st.number_input(
        "Minimum Price",
        value=float(
            settings["min_price"]
        )
    )

    settings["max_price"] = st.number_input(
        "Maximum Price",
        value=float(
            settings["max_price"]
        )
    )

    settings["min_volume"] = st.number_input(
        "Minimum Volume",
        value=int(
            settings["min_volume"]
        ),
        step=10000
    )

    settings["min_rvol"] = st.number_input(
        "Minimum RVOL",
        value=float(
            settings["min_rvol"]
        ),
        step=0.1
    )

    settings["min_change"] = st.number_input(
        "Minimum % Change",
        value=float(
            settings["min_change"]
        ),
        step=0.5
    )

    settings["min_dollar_volume"] = st.number_input(
        "Minimum Dollar Volume",
        value=int(
            settings["min_dollar_volume"]
        ),
        step=100000
    )

    settings["repeat_tolerance"] = st.slider(
        "Repeat Volume",
        0.50,
        1.00,
        float(
            settings["repeat_tolerance"]
        ),
        0.01
    )

    settings["first_stage_limit"] = st.number_input(
        "Detailed Scan Candidates",
        50,
        500,
        int(
            settings["first_stage_limit"]
        ),
        50
    )

    settings["final_results"] = st.number_input(
        "Stocks to Display",
        10,
        100,
        int(
            settings["final_results"]
        ),
        10
    )

    settings["refresh_seconds"] = st.number_input(
        "Refresh Seconds",
        30,
        3600,
        int(
            settings["refresh_seconds"]
        ),
        30
    )

    settings["chart_interval"] = st.selectbox(
        "Chart Interval",
        ["1", "5", "15", "30", "60", "D"],
        index=[
            "1", "5", "15", "30", "60", "D"
        ].index(
            settings["chart_interval"]
        )
    )

    settings["auto_scan"] = st.checkbox(
        "Auto Scan",
        value=bool(
            settings["auto_scan"]
        )
    )

    if st.button(
        "💾 Save Filters"
    ):

        save_settings()

        st.success(
            "Saved"
        )


# =========================================================
# HEADER
# =========================================================

header1, header2, header3 = st.columns(
    [5, 2, 2]
)

with header1:

    st.markdown(
        "## 📈 Russell 2000 Momentum Scanner"
    )

with header2:

    if market_open():

        st.success(
            "🟢 Market Open"
        )

    else:

        st.info(
            "⚪ Market Closed"
        )

with header3:

    if st.button(
        "🔄 Scan Now",
        use_container_width=True
    ):

        st.session_state.scan_results = (
            scan_russell()
        )


# =========================================================
# FIRST SCAN
# =========================================================

if st.session_state.scan_results.empty:

    with st.spinner(
        "Scanning Russell 2000..."
    ):

        st.session_state.scan_results = (
            scan_russell()
        )


df = st.session_state.scan_results


# =========================================================
# LAYOUT
# =========================================================

left, right = st.columns(
    [35, 65],
    gap="small"
)


# =========================================================
# STOCK LIST
# =========================================================

with left:

    st.markdown(
        "### 📋 Momentum Stocks"
    )

    if df.empty:

        st.info(
            "No stocks match the filters."
        )

    else:

        h1, h2, h3, h4, h5 = st.columns(
            [1.5, 1.1, 1, 1, 1.2]
        )

        h1.caption("Symbol")
        h2.caption("LTP")
        h3.caption("%")
        h4.caption("RVOL")
        h5.caption("Score")

        for _, row in df.iterrows():

            symbol = row["Symbol"]

            c1, c2, c3, c4, c5 = st.columns(
                [1.5, 1.1, 1, 1, 1.2]
            )

            with c1:

                if row["Repeat"]:

                    label = f"■ {symbol}"

                else:

                    label = symbol

                if st.button(
                    label,
                    key=f"select_{symbol}",
                    use_container_width=True
                ):

                    st.session_state.selected_symbol = (
                        symbol
                    )

            with c2:

                st.caption(
                    f"${row['Price']:.2f}"
                )

            with c3:

                st.caption(
                    f"{row['Change']:.1f}%"
                )

            with c4:

                st.caption(
                    f"{row['RVOL']:.1f}x"
                )

            with c5:

                st.caption(
                    f"{row['Score']:.0f}"
                )


# =========================================================
# TRADINGVIEW
# =========================================================

with right:

    symbol = (
        st.session_state.selected_symbol
    )

    interval = settings[
        "chart_interval"
    ]

    st.markdown(
        f"### 📊 {symbol}"
    )

    tradingview_url = (
        "https://www.tradingview.com/widgetembed/"
        "?frameElementId=tradingview_chart"
        f"&symbol=NASDAQ%3A{symbol}"
        f"&interval={interval}"
        "&hide_side_toolbar=0"
        "&allow_symbol_change=1"
        "&save_image=1"
        "&hide_volume=0"
        "&theme=dark"
        "&style=1"
        "&timezone=America%2FNew_York"
        "&withdateranges=1"
        "&hide_legend=0"
        "&locale=en"
    )

    html = f"""
    <iframe
        id="tradingview_chart"
        src="{tradingview_url}"
        style="
            width:100%;
            height:700px;
            border:0;
        "
        allowtransparency="true"
        frameborder="0"
        scrolling="no">
    </iframe>
    """

    components.html(
        html,
        height=720,
        scrolling=False
    )


# =========================================================
# AUTO REFRESH
# =========================================================

if settings["auto_scan"]:

    seconds = int(
        settings["refresh_seconds"]
    )

    st.markdown(
        f"""
        <script>
        setTimeout(function() {{
            window.parent.location.reload();
        }}, {seconds * 1000});
        </script>
        """,
        unsafe_allow_html=True
    )
