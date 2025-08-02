# build_gold_csv.py
def main():
    import duckdb
    import pandas as pd
    from pathlib import Path

    SILVER_ROOT = Path("data_lake/silver")
    GOLD_ROOT   = Path("data_lake/gold")
    GOLD_ROOT.mkdir(parents=True, exist_ok=True)

    UNIVERSES = {
        "sp500"      : "ticker_lists/sp500_current.csv",
        "nasdaq100"  : "ticker_lists/nasdaq100_current.csv",
        "russell1000": "ticker_lists/russell1000_current.csv",
    }

    con = duckdb.connect(":memory:")

    # --- Loop through each universe and build a targeted file list ---
    for name, lst_path in UNIVERSES.items():
        tickers = pd.read_csv(lst_path)["Ticker"].tolist()

        # 1. Build a list of all parquet files ONLY for the tickers in this universe
        file_list = []
        for ticker in tickers:
            ticker_path = SILVER_ROOT / f"ticker={ticker}"
            if ticker_path.exists():
                file_list.extend([str(f) for f in ticker_path.rglob("*.parquet")])

        if not file_list:
            print(f"✗ {name:11}  No data files found. Skipping.")
            continue

        # 2. Create a view from this specific list of files
        con.execute(f"""
            CREATE OR REPLACE TEMP VIEW current_universe AS
            SELECT
                Date          AS date,
                Ticker,
                Open, High, Low,
                Close,        -- Raw Close Price
                AdjClose,     -- Adjusted Close Price
                Volume,
                Dividends,
                "Stock Splits" AS StockSplits,
                MarketCap
            FROM read_parquet({file_list}, hive_partitioning=1)
        """)

        # 3. Fetch the data for the current universe
        long_df = con.execute(f"""
            SELECT * FROM current_universe
            ORDER BY date, Ticker
        """).fetch_df()

        out_dir = GOLD_ROOT / name
        out_dir.mkdir(parents=True, exist_ok=True)
        long_df.to_csv(out_dir / "daily_long.csv", index=False)

        # Wide form dataframes
        (long_df
            .pivot(index="date", columns="Ticker", values="AdjClose")
            .sort_index()
            .to_csv(out_dir / "adjclose_wide.csv")
        )

        (long_df
            .pivot(index="date", columns="Ticker", values="MarketCap")
            .sort_index()
            .to_csv(out_dir / "marketcap_wide.csv")
        )

        print(f"{name:11}  {len(tickers):>4} tickers  to  {out_dir}")

    con.close()

if __name__ == "__main__":
    main()