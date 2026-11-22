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


def evaluate_car_from_52w_high(high_series: pd.Series, close_series: pd.Series):
    """
    1. Find the 52-week (up to 252 trading days) high using High prices.
    2. Compute daily Cumulative Average (CAR) of Close prices from that peak forward.
    3. Return True only if CAR increased strictly on every single one of the last
       10 market days (not even a single flat or down day).
    """
    high_1y = high_series.dropna().tail(252)
    close_clean = close_series.dropna()
    if high_1y.empty or close_clean.empty:
        return None, False

    peak_val = high_1y.max()
    peak_date = high_1y[high_1y == peak_val].index[-1]

    post_peak_closes = close_clean.loc[close_clean.index >= peak_date]
    # Need the peak day + at least 10 subsequent market days (11 points total)
    if len(post_peak_closes) < 11:
        return None, False

    car_series = post_peak_closes.expanding().mean()
    current_car = float(car_series.iloc[-1])

    last_11_car = car_series.tail(11)
    strictly_rising_10d = bool((last_11_car.diff().dropna() > 0).all())

    return round(current_car, 2), strictly_rising_10d
    
def run_scanner(tickers: list[str]) -> dict:
    now_ist = datetime.now(IST)
    today = now_ist.date()
    today_str = today.isoformat()
    time_str = now_ist.strftime("%H:%M")
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

    bull_car_today = []
    bull_today = []
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

            _, is_car_buy = evaluate_car_from_52w_high(highs, closes)

            is_bull = output_status == "In Bull Run"
            qualified = is_bull and is_car_buy
            above_50 = live_cmp > dma_50
            above_200 = live_cmp > dma_200

            short_sym = nse_code.split(":", 1)[-1]
            prev_info = prev_stocks.get(nse_code, {})

            # Track when In Bull Run occurred
            prev_was_bull = prev_info.get("is_bull", False)
            if is_bull:
                if not prev_was_bull or not prev_info.get("bull_date"):
                    bull_date = today_str
                    bull_time = time_str
                else:
                    bull_date = prev_info.get("bull_date")
                    bull_time = prev_info.get("bull_time", time_str)
            else:
                bull_date = None
                bull_time = None

            # Track when Bull+CAR qualified occurred
            prev_was_qual = prev_info.get("qualified", False)
            if qualified:
                if not prev_was_qual or not prev_info.get("bull_car_date"):
                    bull_car_date = today_str
                    bull_car_time = time_str
                else:
                    bull_car_date = prev_info.get("bull_car_date")
                    bull_car_time = prev_info.get("bull_car_time", time_str)
            else:
                bull_car_date = None
                bull_car_time = None

            is_new_buy = bool(prev_stocks) and qualified and not prev_was_qual

            base_item = {
                "symbol": short_sym,
                "cmp": round(live_cmp, 2),
                "dma_50": round(dma_50, 2),
                "dma_100": round(dma_100, 2),
                "dma_200": round(dma_200, 2),
                "diff_200": round(diff_200, 2),
                "new_buy": is_new_buy,
            }

            if qualified and bull_car_date == today_str:
                bull_car_today.append({
                    **base_item,
                    "triggered_at": bull_car_time,
                })

            if is_bull and bull_date == today_str:
                bull_today.append({
                    **base_item,
                    "triggered_at": bull_time,
                })

            if qualified:
                final_list.append(base_item)

            curr_stocks[nse_code] = {
                "qualified": qualified,
                "is_bull": is_bull,
                "bull_date": bull_date,
                "bull_time": bull_time,
                "bull_car_date": bull_car_date,
                "bull_car_time": bull_car_time,
                "above_50": above_50,
                "above_200": above_200,
                "cmp": round(live_cmp, 2),
            }
        except Exception:
            continue

    bull_car_today.sort(key=lambda x: x["cmp"])
    bull_today.sort(key=lambda x: x["cmp"])
    final_list.sort(key=lambda x: x["cmp"])

    save_current_state({
        "updated_ist": now_ist.strftime("%Y-%m-%d %H:%M:%S IST"),
        "stocks": curr_stocks,
    })

    return {
        "updated_ist": now_ist.strftime("%d-%b %H:%M IST"),
        "bull_car_today": bull_car_today,
        "bull_today": bull_today,
        "final_list": final_list,
    }
            
def render_nokia_html(report: dict) -> str:
    updated = html.escape(report["updated_ist"])
    bull_car_today = report["bull_car_today"]
    bull_today = report["bull_today"]
    final_list = report["final_list"]

    def make_triggered_rows(items: list[dict]) -> str:
        if not items:
            return '<tr><td colspan="3">None today</td></tr>'
        out = []
        for i, item in enumerate(items, start=1):
            sym = html.escape(item["symbol"])
            cmp_s = f'{item["cmp"]:.1f}'
            d50 = f'{item["dma_50"]:.1f}'
            d100 = f'{item["dma_100"]:.1f}'
            d200 = f'{item["dma_200"]:.1f}'
            trig = html.escape(item.get("triggered_at", ""))
            out.append(
                f"<tr>"
                f"<td>{i}.{sym}<br><small>@{trig}</small></td>"
                f"<td>{cmp_s}</td>"
                f"<td><small>50:{d50}<br>100:{d100}<br>200:{d200}</small></td>"
                f"</tr>"
            )
        return "\n".join(out)

    def make_final_rows(items: list[dict]) -> str:
        if not items:
            return '<tr><td colspan="3">None right now</td></tr>'
        out = []
        for i, item in enumerate(items, start=1):
            sym = html.escape(item["symbol"])
            cmp_s = f'{item["cmp"]:.1f}'
            badge = " <b>[NEW]</b>" if item.get("new_buy") else ""
            c3 = f'+{item["diff_200"]:.1f}%{badge}'
            out.append(f"<tr><td>{i}.{sym}</td><td>{cmp_s}</td><td>{c3}</td></tr>")
        return "\n".join(out)

    alert_msg = (
        f"BULL+CAR TODAY: {len(bull_car_today)}"
        if bull_car_today
        else "No new Bull+CAR today"
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
th,td{{border:1px solid #999;padding:2px 3px;text-align:left;vertical-align:top}}
th{{background:#ddd}}
</style>
</head>
<body>
<h1>NSE Scanner ({len(final_list)})</h1>
<p><b>{html.escape(alert_msg)}</b></p>
<p>{updated} | <a href="./">Refresh</a></p>

<h2>1. BULL+CAR TODAY ({len(bull_car_today)})</h2>
<table>
<tr><th>Stock/Time</th><th>CMP</th><th>DMAs</th></tr>
{make_triggered_rows(bull_car_today)}
</table>

<h2>2. BULL TODAY ({len(bull_today)})</h2>
<table>
<tr><th>Stock/Time</th><th>CMP</th><th>DMAs</th></tr>
{make_triggered_rows(bull_today)}
</table>

<h2>3. FULL BULL+CAR ({len(final_list)})</h2>
<table>
<tr><th>Stock</th><th>CMP</th><th>200D%</th></tr>
{make_final_rows(final_list)}
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
