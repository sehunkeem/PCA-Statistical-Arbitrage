# main.py
import argparse
from pathlib import Path
import pandas as pd
from tqdm.auto import tqdm
from typing import Tuple

from AvellanedaLee.src.algorithm import AvellanedaLee, ALConfig
from AvellanedaLee.src.portfolio import Portfolio
from AvellanedaLee.src.performance import perf_metrics
import matplotlib.pyplot as plt

def load_data(close_path: Path, marketcap_path: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load close prices and market cap data from CSV files.

    Args:
        close_path (Path): Path to the close prices CSV file.
        marketcap_path (Path): Path to the market cap CSV file.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: DataFrames of close prices and market cap.
    """
    close = pd.read_csv(close_path, index_col=0, parse_dates=True)
    rets = (close/close.shift(1) -1).dropna(how="all")
    marketcap = pd.read_csv(marketcap_path, index_col=0, parse_dates=True)
    marketcap, rets = marketcap.align(rets, join='inner', axis=0)
    return rets, marketcap

def load_crypto_data(csv_path: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(csv_path)
    df['time'] = pd.to_datetime(df['time'])
    df = df.set_index('time').sort_index()
    drop = {"BTCDOMUSDT", "DEFIUSDT", "USDCUSDT"}
    cols_to_drop = [c for c in df.columns if c in drop]
    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)
    return df, pd.DataFrame()

def backtest_loop(rets: pd.DataFrame, mc: pd.DataFrame, cfg: ALConfig):
    algo = AvellanedaLee(cfg)
    port = Portfolio(rets.index, rets.columns, cfg.start_capital, cfg.t_cost)

    for i in tqdm(range(cfg.pca_win, len(rets)), desc="backtest loop"):
        # d0: yesterday, d1: today
        d0, d1 = rets.index[i - 1], rets.index[i]
        r_t = rets.loc[d1]
        # Initialize equity on the first valid day if it's NaN
        if pd.isna(port.equity.loc[d0]):
            port.equity.loc[d0] = cfg.start_capital

        core_mt, hedge_mt, eq_base = port.mark_to_market(d1, r_t)


        pca_window_data = rets.iloc[i - cfg.pca_win + 1 : i + 1]

        target_core, target_hedge, new_active_hedges = algo.signals(
            date=d1,
            pca_window_rets=pca_window_data,
            mc_row=mc.loc[d1],
            equity_base=eq_base,
            core_mt=core_mt,
            active_hedges=port.active_hedges,
        )
        port.active_hedges = new_active_hedges
        port.rebalance(d1, core_mt, hedge_mt, target_core, target_hedge, eq_base)

    return port

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--end", type=str, default="2007-12-31")
    ap.add_argument("--close", type=str, default="../../data_lake/gold/russell1000/adjclose_wide.csv")
    ap.add_argument("--mcap", type=str, default="../../data_lake/gold/russell1000/marketcap_wide.csv")
    args = ap.parse_args()

    # Load wide format returns and market cap dataframes
    rets, mc = load_data(Path(args.close), Path(args.mcap))
    rets = rets.loc[:args.end]
    mc = mc.loc[:args.end]

    # Config from the Paper
    cfg = ALConfig()
    port = backtest_loop(rets, mc, cfg)
    perf = perf_metrics(port.equity.dropna())

    plt.figure(figsize=(12, 6))
    plt.plot((perf['daily_return']+1).cumprod(), label='Cumulative Return')
    plt.title('Cumulative Returns of Avellaneda-Lee Strategy')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Return')
    plt.legend()
    plt.grid()
    plt.show()

if __name__ == "__main__":
    main()