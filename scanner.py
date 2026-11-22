from datetime import datetime, timedelta, timezone
import html
import json
import os
import numpy as np
import pandas as pd
import yfinance as yf

GOOGLE_SHEET_CSV_URL = os.getenv("SHEET_CSV_URL", "").strip()
STATE_FILE = "state.json"
HTML_OUTPUT_FILE = "public/index.html"
IST = timezone(timedelta(hours=5, minutes=30))


def load_tickers() -> list[str]:
    if not GOOGLE_SHEET_CSV_URL:
        raise ValueError("SHEET_CSV_URL secret is empty.")
    df_raw = pd.read_csv(GOOGLE_SHEET_CSV_URL)
    col = df_raw.iloc[:, 0].dropna().astype(str).str.strip()
    tickers = [t for t in col if t and t.upper() not in ("NSECODE", "NSE CODE")]
    return list(dict.fromkeys(tickers))


def to_yahoo_symbol(nse_code: str) -> str:
    symbol = nse_code.split(":", 1)[-1].strip()
    return f"{symbol}.NS"


def compute_car_rating(high_series: pd.Series, close_series: pd.Series) -> str:
    high_clean = high_series.dropna()
    close_clean = close_series.dropna()

    if close_clean.empty:
        return "TICKER NOT FOUND"

    if not high_clean.empty:
        max_high_val = high_clean.max()
        high_date = high_clean[high_clean == max_high_val].index[-1]
        closes_from_high = close_clean.loc[high_date:]
    else:
        fallback_start = close_clean.index[-1] - timedelta(days=30)
        closes_from_high = close_clean.loc[fallback_start:]

    if len(closes_from_high) < 10:
        closes_from_high = close_clean.tail(10)

    prices = closes_from_high.to_numpy(dtype=float)
    count_rows = len(prices)
    if count_rows < 10:
        return "Short History"

    cum_avgs = np.cumsum(prices) / np.arange(1, count_rows + 1)
    last_10_newest_first = cum_avgs[-10:][::-1]
    latest_avg = last_10_newest_first[0]

    peers = np.append(last_10_newest_first[1:], 0.0)
    check = int(np.sum(latest_avg > peers))
    return "Buy/Average Out" if check == 10 else "Avoid/Hold"


def load_previous_state() -> dict:
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def save_current_state(state: dict) -> None:
    with open(STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)
def evaluate_car_from_52w_high(close_series: pd.Series, cmp_price: float):
    """
    1. Find the 52-week (252-trading-day) high.
    2. Compute daily Cumulative Average (CAR) from that 52-week high date to today.
    3. Verify CAR increased on EVERY single one of the last 10 market days
       (not even 1 flat or down day allowed) and CMP < current CAR.
    """
    recent_1y = close_series.dropna().tail(252)
    if len(recent_1y) < 11:
        return None, False

    # Position of the most recent 52-week high within the 252-day window
    peak_pos = int(recent_1y.values.argmax())
    post_peak_closes = recent_1y.iloc[peak_pos:]

    # Need the peak day + at least 10 subsequent market days (11 points total)
    if len(post_peak_closes) < 11:
        return None, False

    # Cumulative average from the 52-week high date forward
    car_series = post_peak_closes.expanding().mean()
    current_car = float(car_series.iloc[-1])

    # Strictly increasing across all 10 most recent daily transitions
    last_11_car = car_series.tail(11)
    strictly_rising_10d = bool((last_11_car.diff().dropna() > 0).all())

    # Buy / Average Out: price below CAR AND CAR strictly rising for 10 market days
    car_buy_signal = strictly_rising_10d and (cmp_price < current_car)
    return round(current_car, 2), car_buy_signal

def run_scanner(tickers: list[str]) -> dict:
    now_ist = datetime.now(IST)
    today = now_ist.date()
    start_date = today - timedelta(days=520)
    yf_symbols = [to_yahoo_symbol(t) for t in tickers]

    daily_data = yf.download(
        tickers=yf_symbols,
        start=start_date.isoformat(),
        end=(today + timedelta(days=1)).isoformat(),
        interval="1d",
        group_by="ticker",
        auto_adjust=False,
        progress=False,
        threads=True,
    )

    intraday_data = yf.download(
        tickers=yf_symbols,
        period="1d",
        interval="5m",
        group_by="ticker",
        auto_adjust=False,
        progress=False,
        threads=True,
    )

    prev_state = load_previous_state()
    prev_stocks = prev_state.get("stocks", {})
    curr_stocks = {}

    fresh_entries = []
    fresh_crossovers = []
    final_list = []

    for nse_code, yf_sym in zip(tickers, yf_symbols):
        try:
            if isinstance(daily_data.columns, pd.MultiIndex):
                if yf_sym not in daily_data.columns.get_level_values(0):
                    continue
                stock_df = daily_data[yf_sym].dropna(how="all")
            else:
                stock_df = daily_data.dropna(how="all")

            if stock_df.empty or "Close" not in stock_df:
                continue

            closes = stock_df["Close"].dropna().copy()
            highs = stock_df["High"].dropna().copy() if "High" in stock_df else closes.copy()
            if closes.empty:
                continue

            live_cmp = float(closes.iloc[-1])
            try:
                if isinstance(intraday_data.columns, pd.MultiIndex) and yf_sym in intraday_data.columns.get_level_values(0):
                    intra_df = intraday_data[yf_sym].dropna(how="all")
                elif not isinstance(intraday_data.columns, pd.MultiIndex):
                    intra_df = intraday_data.dropna(how="all")
                else:
                    intra_df = pd.DataFrame()

                if not intra_df.empty and "Close" in intra_df and not intra_df["Close"].dropna().empty:
                    live_cmp = float(intra_df["Close"].dropna().iloc[-1])
                    closes.iloc[-1] = live_cmp
                    if live_cmp > float(highs.iloc[-1]):
                        highs.iloc[-1] = live_cmp
            except Exception:
                pass

            if len(closes) < 200:
                continue

            dma_50 = float(closes.tail(50).mean())
            dma_100 = float(closes.tail(100).mean())
            dma_200 = float(closes.tail(200).mean())
            diff_200 = ((live_cmp - dma_200) / dma_200) * 100.0

            if (live_cmp > dma_50) and (live_cmp > dma_100) and (live_cmp > dma_200) and (0.01 <= diff_200 <= 10.0):
                output_status = "In Bull Run"
            elif (live_cmp < dma_50) and (live_cmp < dma_100) and (live_cmp < dma_200) and (-10.0 <= diff_200 <= -0.01):
                output_status = "In Bear Run"
            else:
                output_status = "Unconfirmed"

            _, is_car_buy = evaluate_car_from_52w_high(closes, live_cmp)

            qualified = (output_status == "In Bull Run") and is_car_buy
            above_50 = live_cmp > dma_50
            above_200 = live_cmp > dma_200

            short_sym = nse_code.split(":", 1)[-1]
            prev_info = prev_stocks.get(nse_code, {})

            is_new_buy = bool(prev_stocks) and qualified and not prev_info.get("qualified", False)
            crossed_200 = bool(prev_stocks) and above_200 and not prev_info.get("above_200", True) and (0.01 <= diff_200 <= 10.0)
            crossed_50 = bool(prev_stocks) and above_50 and not prev_info.get("above_50", True) and above_200 and (0.01 <= diff_200 <= 10.0)

            row_item = {
                "symbol": short_sym,
                "cmp": round(live_cmp, 2),
                "diff_200": round(diff_200, 2),
                "new_buy": is_new_buy,
            }

            if is_new_buy:
                fresh_entries.append(row_item)

            if crossed_200 or crossed_50:
                fresh_crossovers.append({
                    "symbol": short_sym,
                    "cmp": round(live_cmp, 2),
                    "diff_200": round(diff_200, 2),
                    "tag": "X-200D" if crossed_200 else "X-50D",
                })

            if qualified:
                final_list.append(row_item)

            curr_stocks[nse_code] = {
                "qualified": qualified,
                "above_50": above_50,
                "above_200": above_200,
                "cmp": round(live_cmp, 2),
            }
        except Exception:
            continue

    fresh_entries.sort(key=lambda x: x["cmp"])
    fresh_crossovers.sort(key=lambda x: x["cmp"])
    final_list.sort(key=lambda x: x["cmp"])

    save_current_state({
        "updated_ist": now_ist.strftime("%Y-%m-%d %H:%M:%S IST"),
        "stocks": curr_stocks,
    })

    return {
        "updated_ist": now_ist.strftime("%d-%b %H:%M IST"),
        "fresh_entries": fresh_entries,
        "fresh_crossovers": fresh_crossovers,
        "final_list": final_list,
    }
    
def render_nokia_html(report: dict) -> str:
    updated = html.escape(report["updated_ist"])
    fresh_entries = report["fresh_entries"]
    fresh_crossovers = report["fresh_crossovers"]
    final_list = report["final_list"]

    def make_rows(items: list[dict], mode: str) -> str:
        if not items:
            return '<tr><td colspan="3">None right now</td></tr>'
        out = []
        for i, item in enumerate(items, start=1):
            sym = html.escape(item["symbol"])
            cmp_s = f'{item["cmp"]:.1f}'
            if mode == "final":
                badge = " <b>[NEW]</b>" if item.get("new_buy") else ""
                c3 = f'+{item["diff_200"]:.1f}%{badge}'
            elif mode == "fresh":
                c3 = f'+{item["diff_200"]:.1f}%'
            else:
                c3 = f'{html.escape(item["tag"])} (+{item["diff_200"]:.1f}%)'
            out.append(f"<tr><td>{i}.{sym}</td><td>{cmp_s}</td><td>{c3}</td></tr>")
        return "\n".join(out)

    alert_msg = (
        f"ALERT: {len(fresh_entries)} NEW BUY!"
        if fresh_entries
        else "No brand-new buys in last 10m"
    )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>NSE 10m Alerts</title>
<style>
body{{font-family:sans-serif;font-size:14px;margin:4px;padding:0;background:#fff;color:#000}}
h1{{font-size:16px;margin:2px 0}}
h2{{font-size:14px;margin:8px 0 2px;background:#eee;padding:2px}}
p{{margin:3px 0}}
table{{width:100%;border-collapse:collapse;font-size:13px}}
th,td{{border:1px solid #999;padding:2px 3px;text-align:left}}
th{{background:#ddd}}
</style>
</head>
<body>
<h1>NSE Scanner ({len(final_list)})</h1>
<p><b>{html.escape(alert_msg)}</b></p>
<p>{updated} | <a href="./">Refresh</a></p>

<h2>1. NEW BUY ALERTS ({len(fresh_entries)})</h2>
<table>
<tr><th>Stock</th><th>CMP</th><th>200D%</th></tr>
{make_rows(fresh_entries, "fresh")}
</table>

<h2>2. NEW DMA CROSS ({len(fresh_crossovers)})</h2>
<table>
<tr><th>Stock</th><th>CMP</th><th>Signal</th></tr>
{make_rows(fresh_crossovers, "cross")}
</table>

<h2>3. FULL BULL+CAR ({len(final_list)})</h2>
<table>
<tr><th>Stock</th><th>CMP</th><th>200D%</th></tr>
{make_rows(final_list, "final")}
</table>
</body>
</html>
"""


if __name__ == "__main__":
    tickers = load_tickers()
    report = run_scanner(tickers)
    html_page = render_nokia_html(report)
    os.makedirs(os.path.dirname(HTML_OUTPUT_FILE), exist_ok=True)
    with open(HTML_OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write(html_page)
