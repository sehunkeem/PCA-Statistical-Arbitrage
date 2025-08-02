# main.py
import pandas as pd
from typing import Dict

class Portfolio:
    def __init__(self, dates: pd.DatetimeIndex, tickers: pd.Index, start_capital: float, t_cost: float):
        """
        Initializes the portfolio state.
        - core: Dollar positions from the alpha strategy.
        - hedge: Combined dollar positions from hedging vectors.
        - equity: Time series of the portfolio's total equity.
        - active_hedges: Dictionary tracking individual hedge vectors for each core position. We must keep track of these
        since the paper liquidates both the mark-to-market(mtm) core and mtm hedge positions when the close signal for each stock (core) is triggered.
        """
        self.core = pd.DataFrame(0.0, index=dates, columns=tickers)
        self.hedge = pd.DataFrame(0.0, index=dates, columns=tickers)
        self.equity = pd.Series(index=dates, dtype=float)
        self.active_hedges: Dict[str, pd.Series] = {}
        self.t_cost = t_cost

    def mark_to_market(self, date:pd.Timestamp, r_t: pd.Series) -> tuple[pd.Series, pd.Series, float]:
        """
        Marks the portfolio to market for a given date and returns the core and hedge positions
        after applying the daily returns, and mtm equity base.
        """
        # idx: today's index, d0: yesterday's timestamp
        idx = self.core.index.get_loc(date)
        d0 = self.core.index[idx - 1]
        # Find dollar value of core positions, and hedging positions
        core_d0 = self.core.loc[d0]
        hedge_d0 = self.hedge.loc[d0]

        # Mark-to-market positions; update the value of existing positions with today's returns.
        core_mt = core_d0 * (1 + r_t)
        hedge_mt = hedge_d0 * (1 + r_t)

        # Calculate PnL; combined daily pnl of existing positions
        book_d0 = core_d0 + hedge_d0
        book_mt = core_mt + hedge_mt
        pnl = (book_mt - book_d0).sum()
        equity_base = self.equity.loc[d0] + pnl

        for k, v in self.active_hedges.items(): # update individual hedge positions for each core position
            self.active_hedges[k] = v.mul(1 + r_t, fill_value=0.0)

        return core_mt, hedge_mt, equity_base

    def rebalance(self, date: pd.Timestamp, core_mt: pd.Series, hedge_mt: pd.Series, target_core: pd.Series, target_hedge: pd.Series, equity_base: float) -> None:
        """
        Turn the mtm book into a target book, applying transaction costs.
        """
        target_book = target_core + target_hedge
        mt_book = core_mt + hedge_mt

        turnover = (target_book - mt_book).abs().sum()
        self.equity.loc[date] = equity_base - (turnover * self.t_cost)
        self.core.loc[date] = target_core
        self.hedge.loc[date] = target_hedge

    @property
    def book(self) -> pd.DataFrame:
        return self.core + self.hedge
