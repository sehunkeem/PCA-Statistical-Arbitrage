# algorithm.py
"""
Signal engine for the Avellaneda-Lee PCA stat arb strategy.
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm
from dataclasses import dataclass
from typing import Dict, Tuple, List

@dataclass
class ALConfig:
    pca_win: int = 252
    lookback_ou: int = 60
    n_pcs: int = 15
    s_bo: float = 1.25 # buy to open long
    s_so: float = 1.25 # sell to open short
    s_sc: float = 0.50 # sell to close long
    s_bc: float = 0.75 # buy to close short
    b_max: float = 0.9672 # this is equivalent to kappa > 8.4 or speed of mean reversion less than 30 days
    min_universe: int = 70
    leverage_level: float = 0.02 # On each day, put 2% of the equity into each position
    t_cost: float = 0.0005 # 5 bps transaction cost per side
    start_capital: float = 1e5
    mc_floor: float = 1e9 # Minimum market cap of the stocks in the universe


class AvellanedaLee:
    """
    Generates $ orders at each close.
    """
    def __init__(self, cfg: ALConfig):
        self.cfg = cfg

    def _pca(self, z: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray]:
        """
        Perform PCA on standardized return data to extract eigenportfolios
        and their corresponding factor returns (i.e., projection of returns onto the space spanned by eigenportfolios).

        Args:
            z (pd.DataFrame): DataFrame of shape (T, N), where T is the number of time steps and
                              N is the number of assets. The input should be raw returns data.

        Returns:
            Tuple[pd.DataFrame, np.ndarray]:
                - eigenportfolios (pd.DataFrame): DataFrame of shape (N, K), where each column is a
                  normalized eigenvector corresponding to the top K principal components.
                - factor_returns (np.ndarray): Array of shape (T, K), representing the time series of
                  returns of each eigenportfolio (factor).
        """
        mu, sig = z.mean(), z.std()
        z_std = (z - mu) / sig
        cov = z_std.cov()
        evals, evecs = np.linalg.eigh(cov)
        idx = np.argsort(evals)[::-1][: self.cfg.n_pcs]
        # Eigenportfolios are the top K eigenvectors, normalized by their standard deviations
        W = pd.DataFrame(evecs[:, idx] / sig.values[:, None],
                         index=z.columns,
                         )
        f_ret = z.values @ W.values # Eigenportfolio (factor) returns are raw returns projected onto the eigenportfolios
        return W, f_ret

    def signals(self,
                date: pd.Timestamp,
                pca_window_rets: pd.DataFrame,
                mc_row: pd.Series,
                equity_base: float,
                core_mt: pd.Series,
                active_hedges: Dict[str, pd.Series]
                ) -> Tuple[pd.Series, pd.Series, Dict[str, pd.Series]]:
        """
        Compute target dollar positions for new trades on `date`.
        Args:
            date (pd.Timestamp): The date for which to compute signals.
            pca_window_rets (pd.DataFrame): DataFrame of shape (T, N) with raw returns data.
            mc_row (pd.Series): Series containing market cap data for the universe on `date`.
            equity_base (float): The base equity amount to use for position sizing.
            core_mt: Mark-to-market core positions from previous day.
            active_hedges: Dictionary of active hedge vectors.

        Returns:
            A tuple containing:
            - core_new (pd.Series): The target dollar positions for the core book.
            - hedge_new (pd.Series): The target dollar positions for the hedge book.
            - active_hedges (dict): The updated dictionary of active hedge vectors.
        """
        cfg = self.cfg

        # Filter universe based on market cap
        live_mask = mc_row > cfg.mc_floor
        if live_mask.sum() < cfg.min_universe:
            return core_mt, pd.Series(dtype=float), active_hedges

        win = pca_window_rets.loc[:, live_mask].dropna(axis=1)
        win = win.loc[:, win.isna().mean() == 0.0]  # Filter out assets with more than 10% with NaN
        win = win.loc[:, (win != 0).mean() > 0.1]  # Keep assets with at least 10% non-zero returns

        if win.shape[1] < cfg.min_universe:
            return core_mt, pd.Series(dtype=float), active_hedges

        W, f_ret = self._pca(win) # eigenportfolios and factor returns

        # Cross-sectional regressions for each asset; raw retruns ~ eigenportfolio factor returns
        ou_r = win.iloc[-cfg.lookback_ou:]
        ou_f = pd.DataFrame(f_ret[-cfg.lookback_ou:], index=ou_r.index)
        X = sm.add_constant(ou_f.values)
        y = ou_r.values
        coeff = np.linalg.lstsq(X, y, rcond=None)[0]
        B = coeff[1:] # shape (M x N), Factor Exposures
        resid = y - X @ coeff
        x = resid.cumsum(axis=0)

        # AR1 Fit on cumulative residuals. beta=cov/var, alpha=y bar - beta * x bar
        X_ar, y_ar = x[:-1], x[1:]
        mu_X = X_ar.mean(axis=0)
        mu_y = y_ar.mean(axis=0)

        cov_ar = ((X_ar * y_ar).mean(axis=0) - mu_X * mu_y)
        var_ar = X_ar.var(axis=0)

        tol=1e-12
        zero_var = pd.Series(var_ar[var_ar < tol], index=win.columns[var_ar < tol])

        if not zero_var.empty:
            print(">>> Zero var for assets:", zero_var.index.tolist())


        b_i = cov_ar / var_ar
        a_i = mu_y - b_i * mu_X
        var_xi = (y_ar - (a_i + b_i * X_ar)).var(axis=0)
        mr_mask = (np.abs(b_i) < 1) & (var_xi > 1e-12)
        if not mr_mask.any():
            return core_mt, pd.Series(dtype=float), active_hedges

        # AR1 parameters to OU parameters. For the formulas, refer to appendix of the paper.
        m = a_i[mr_mask] / (1 - b_i[mr_mask])
        sigma_eq = np.sqrt(var_xi[mr_mask] / (1 - b_i[mr_mask] ** 2))
        s = -(m - m.mean()) / sigma_eq
        fast_reversion_mask = b_i[mr_mask] < cfg.b_max
        tick = win.columns[mr_mask][fast_reversion_mask]
        if tick.empty:
            return core_mt, pd.Series(dtype=float), active_hedges

        s_today = pd.Series(s[fast_reversion_mask], index=tick)

        core_new = core_mt.copy()

        # Determine previous position signs to correctly apply open/close logic
        prev_signs = np.sign(core_mt.reindex(s_today.index).fillna(0))

        # Force liquidate assets that are no longer in the universe - we don't have s-scores for these assets, but core positions still exist.
        assets_with_positions = core_mt.index[(core_mt != 0.0) & (~core_mt.isna())]
        today_universe = s_today.index
        removed_assets = assets_with_positions.difference(today_universe)

        # Define open/close conditions
        open_long = (s_today < -cfg.s_bo) & (prev_signs == 0)
        open_short = (s_today > cfg.s_so) & (prev_signs == 0)
        close_long = (s_today > -cfg.s_sc) & (prev_signs == 1)
        close_short = (s_today < cfg.s_bc) & (prev_signs == -1)

        # Initialize new hedge book from previous day's MTM hedge book
        hedge_new = pd.Series(0.0, index=core_mt.index)
        for tkr, hedge_vec in active_hedges.items():
            hedge_new = hedge_new.add(hedge_vec, fill_value=0.0)

        # Force liquidate assets that are no longer in the universe
        for tkr in removed_assets:
            core_new[tkr] = 0.0
            old_h = active_hedges.pop(tkr, None)
            if old_h is not None:
                hedge_new = hedge_new.sub(old_h, fill_value=0.0)

        # Process closing trades
        to_close = s_today.index[close_long | close_short]
        for tkr in to_close:
            core_new[tkr] = 0.0
            if tkr in active_hedges:
                hedge_new = hedge_new.sub(active_hedges.pop(tkr), fill_value=0.0)

        # Process opening trades
        to_open = s_today.index[open_long | open_short]
        for tkr in to_open:
            sign = 1 if open_long[tkr] else -1
            alpha = sign * equity_base * cfg.leverage_level
            core_new[tkr] = alpha

            # Calculate and add new hedge
            k = win.columns.get_loc(tkr)
            h_fac = -B[:, k] * alpha
            h_stk = pd.Series(W.values @ h_fac, index=win.columns)
            hedge_new = hedge_new.add(h_stk, fill_value=0.0)
            active_hedges[tkr] = h_stk

        return core_new.fillna(0), hedge_new.fillna(0), active_hedges
