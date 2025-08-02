# performance.py
import numpy as np
import pandas as pd

def perf_metrics(equity_curve: pd.Series) -> pd.DataFrame:
    daily_r = equity_curve.pct_change().dropna()
    daily_r.replace([np.inf, -np.inf], 0.0, inplace=True)

    cum_ret = (1 + daily_r).cumprod()

    out = pd.DataFrame(
        {
            "equity": equity_curve,
            "daily_return": daily_r,
            "cumulative_return": cum_ret
        }
    )

    sharpe = daily_r.mean() / daily_r.std() * np.sqrt(252) if daily_r.std() != 0 else 0.0
    mdd = (cum_ret / cum_ret.cummax() - 1).min()

    out.attrs["sharpe"] = sharpe
    out.attrs["mdd"] = mdd

    print(f"\n--- Performance (Equity-Based) ---")
    print(f"Sharpe Ratio: {sharpe:.2f}")
    print(f"Maximum Drawdown: {mdd:.2%}")

    return out