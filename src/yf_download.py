# yf_download.py
"""
Downloads and maintains daily equity bar data for all tickers in the S&P 500, Nasdaq 100, and Russell 1000.

- Builds the ticker universe by scraping Wikipedia and saves the lists to ticker_lists/.
- For each ticker, fetches new daily OHLC, Volume, etc via yfinance, starting from the last saved date (or 1997-01-01 if no saved data exists).
- Organizes raw data into a 'bronze' layer: parquet files partitioned by ticker/year/month.
- Append daily MarketCap (computed manually via share counts) and writes to a 'silver' layer with the same partitioning.
- Tracks download progress in a SQLite database (data_lake/progress.db) so reruns only fetch incremental updates.
"""
import re
import sqlite3
import datetime as dt
import yfinance as yf
from pathlib import Path
import pandas as pd
from tqdm import tqdm
from build_gold_csv import main as build_gold_main

# Configs
START_DATE = "1997-01-01"
DATA_ROOT  = Path("data_lake")
BRONZE     = DATA_ROOT / "bronze"
SILVER     = DATA_ROOT / "silver"
META_DB    = Path("data_lake/progress.db")

for p in (BRONZE, SILVER, Path("ticker_lists")):
    p.mkdir(parents=True, exist_ok=True)

def _canon(ticker: str) -> str:
    """
    For each symbol string,
    1. replace any dot with a hyphen
    2. strip out leading/trailing whitespace
    3. make everything uppercase
    """
    return re.sub(r"\.", "-", ticker.strip().upper())

def sp500():
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    t   = pd.read_html(url, match="CIK")[0]
    t["Ticker"] = t["Symbol"].apply(_canon)
    return t[["Ticker"]]

def nasdaq100():
    url = "https://en.wikipedia.org/wiki/Nasdaq-100"
    t   = pd.read_html(url, match="Ticker")[0]
    t["Ticker"] = t["Ticker"].apply(_canon)
    return t[["Ticker"]]

def russell1000():
    url = "https://en.wikipedia.org/wiki/Russell_1000_Index"
    t   = pd.read_html(url, match="Symbol")[0]
    t["Ticker"] = t["Symbol"].apply(_canon)
    return t[["Ticker"]]

def universe() -> list[str]:
    """
    Save and return the lists of assets in the directory "ticker_lists/
    """
    dfs = [sp500(), nasdaq100(), russell1000()]
    for name, df in zip(
        ["sp500", "nasdaq100", "russell1000"], dfs, strict=False
    ):
        df.to_csv(f"ticker_lists/{name}_current.csv", index=False)
    return sorted(pd.concat(dfs)["Ticker"].unique())

def _conn():
    """
    Open or create the SQLite database file at META_DB.
    Ensures a table `progress(ticker, last)` exists, where `last` is a date string, and `ticker` is just ticker string.
    Return a connection object `c` which is used to execute SQL commands.
    """
    c = sqlite3.connect(META_DB)
    c.execute("""
              CREATE TABLE IF NOT EXISTS progress (
                ticker TEXT PRIMARY KEY, 
                last TEXT
            )
        """
    )
    return c

def last_date(tkr: str):
    """
    Retrieve the value of 'last' for the row where the 'ticker' column matches the ticker string, from the 'progress' table.
    Returns it as a datetime.date object.
    """
    with _conn() as c:
        row = c.execute("""
                        SELECT last FROM progress WHERE ticker=?
                        """,
                        (tkr,)
                        ).fetchone()
    return dt.date.fromisoformat(row[0]) if row else None

def set_last(tkr: str, d: dt.date):
    """
    Add a new row (INSERT) or replace an existing row if the primary key, 'tkr' already exist.
    Convert the datetime.date object into a string in ISO format (YYYY-MM-DD).
    """
    with _conn() as c:
        c.execute("""
                        INSERT OR REPLACE INTO progress VALUES (?,?)
                        """,
                  (tkr, d.isoformat())
                  )

def _save(df: pd.DataFrame, base: Path, tkr: str):
    df = df.assign(
        year=df.Date.dt.year,
        month=df.Date.dt.month
    )

    for (y, m), sub in df.groupby(["year", "month"], sort=False, group_keys=False):
        chunk = sub.drop(columns=["year", "month"]).copy()

        path = base / f"ticker={tkr}" / f"year={y}" / f"month={m:02d}"
        path.mkdir(parents=True, exist_ok=True)
        fn = path / "part.parquet"

        if fn.exists():
            old = pd.read_parquet(fn)
            chunk = (
                pd.concat([old, chunk], ignore_index=True)
                  .drop_duplicates("Date")
                  .sort_values("Date")
            )
        chunk.to_parquet(fn, index=False)

def _shares(tkr: str, start: str):
    """
    Returns a time series of a stock's daily total outstanding share count.
    """
    try:
        s = yf.Ticker(tkr).get_shares_full(start=start)
        if s is None or s.empty:
            return None
        s.index = pd.to_datetime(s.index).tz_localize(None)
        return s.resample("1D").ffill().squeeze().rename("Shares")
    except Exception:
        return None

def with_mcap(df: pd.DataFrame, tkr: str):
    sh = _shares(tkr, df.Date.min().strftime("%Y-%m-%d"))
    if sh is not None:
        df = df.join(sh, on="Date")
        df["MarketCap"] = df["Close"] * df["Shares"]
        return df
    so = yf.Ticker(tkr).info.get("sharesOutstanding")
    df["MarketCap"] = df["Close"] * so if so else pd.NA
    return df

def _flatten(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [
            c[0] if isinstance(c, tuple) else c
            for c in df.columns
        ]
    return df

def refresh(tkr: str):
    start = (last_date(tkr) + dt.timedelta(days=1)).strftime("%Y-%m-%d") \
            if last_date(tkr) else START_DATE

    # `auto_adjust=False` to get both raw and adjusted close prices
    df = yf.download(tkr, start=start, auto_adjust=False, actions=True, progress=False, threads=False)

    if df.empty:
        return

    df.reset_index(inplace=True)
    df.rename(columns={
        "index": "Date",
        "Open": "Open",
        "High": "High",
        "Low": "Low",
        "Close": "Close", # Raw close price
        "Adj Close": "AdjClose", # Adjusted close price
        "Volume": "Volume"
    }, inplace=True)
    df = _flatten(df)
    df.insert(0, "Ticker", tkr)

    # Bronze layer contains raw data including 'AdjClose'
    _save(df, BRONZE, tkr)

    # Silver layer calculation for MarketCap is now correct because df['Close'] is the raw price
    _save(with_mcap(df, tkr), SILVER, tkr)
    set_last(tkr, df.Date.max().date())

if __name__ == "__main__":
    tickers = universe()
    print(f"Universe size = {len(tickers)}")
    for tkr in tqdm(tickers, desc="Downloading"):
        try:
            refresh(tkr)
        except Exception as e:
            print(f"Error for {tkr}: {e}")

    build_gold_main()