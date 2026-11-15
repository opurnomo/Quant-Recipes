"""
Portfolio Optimization with Pandas and NumPy
Author: Owen Purnomo | Date: 10/18/2026
"""

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from scipy.optimize import minimize


# ------------------------------- Settings --------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "Data" / "Portfolio-Optimization Data.csv"
FIGURE_PATH = BASE_DIR / "Visualization" / "Project 1" / "efficient_frontier.png"
PERIODS_PER_YEAR = 12
RISK_FREE_RATE = 0.0
N_PORTFOLIOS = 20_000
SEED = 42
TRAIN_FRACTION = 0.8
REPORT_WIDTH = 60

# ------------------------------- Data Loading ----------------------------
def load_data(path: Path) -> pd.DataFrame:
    """Read the CSV with date as a sorted DatetimeIndex"""

    return pd.read_csv(path, index_col="Date", parse_dates=True).sort_index()


def to_log_returns(data: pd.DataFrame, is_price: bool = False) -> pd.DataFrame:
    """Log returns from prices, or simple returns if is_price is False"""
    if is_price:
        return np.log(data / data.shift(1)).dropna()
    return np.log1p(data)


# ----------------------------- Metrics -----------------------------------
def annualize(log_returns: pd.DataFrame, 
              period: int = PERIODS_PER_YEAR) -> tuple[pd.Series, pd.Series, pd.DataFrame]:
    """Annualise mean return, volatility, and covariance matrix"""
    mean_returns = log_returns.mean() * period
    volatilities = log_returns.std() * np.sqrt(period)
    covariance_matrix = log_returns.cov() * period
    return mean_returns, volatilities, covariance_matrix


# ----------------------------- Optimization --------------------------------
def random_weights(n_portfolios: int, n_assets: int, seed: int = SEED) -> np.ndarray:
    """Random long-only weights; each row is one portfolio and sum to 1"""
    rng = np.random.default_rng(seed)
    weights = rng.random((n_portfolios, n_assets))
    return weights / weights.sum(axis=1, keepdims=True)


def portfolio_performance(weights: np.ndarray, mean_returns: np.ndarray, 
                          cov_matrix: np.ndarray, risk_free_rate: float = RISK_FREE_RATE
                          ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return, Volatility, and Sharpe Ratio for each portfolio (1/row)"""
    port_returns = weights @ mean_returns
    port_vols = np.sqrt(((weights @ cov_matrix)*weights).sum(axis=1)) #USE @ to make matrix operation faster
    sharpe_ratios = (port_returns - risk_free_rate) / port_vols
    return port_returns, port_vols, sharpe_ratios


# ----------------------------- Visualization -----------------------------
def plot_frontier(portfolios: pd.DataFrame, min_vol_idx: int, max_sharpe_idx: int,
                  path: Path = FIGURE_PATH) -> None:
    """Scatter all portfolios coloured by Sharpe ratio."""
    fig, ax = plt.subplots(figsize=(10, 6))
    points = ax.scatter(portfolios["Volatility"], portfolios["Return"],
                        c=portfolios["Sharpe"], cmap="magma", s=4, alpha=0.6)
    fig.colorbar(points, ax=ax, label="Sharpe Ratio")

    key_points = [(min_vol_idx, "Minimum Volatility", "red", (15, -25)),
                  (max_sharpe_idx, "Maximum Sharpe", "orange", (15, 15))]

    for idx, label, color, offset in key_points:
        vol, ret = portfolios.loc[idx, ["Volatility", "Return"]]
        ax.scatter(vol, ret, marker="*", s=350, color=color,
                   edgecolor="black", label=label, zorder=3)
        ax.annotate(f"{label}\nVol {vol:.2%} | Return {ret:.2%}", (vol, ret),
                    xytext=offset, textcoords="offset points", fontsize=9,
                    arrowprops=dict(arrowstyle="-", color="gray"))

    ax.xaxis.set_major_formatter(PercentFormatter(1.0))
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_title(f"Efficient Frontier: {len(portfolios):,} Random Portfolios")
    ax.set_xlabel("Annualized Volatility")
    ax.set_ylabel("Annualized Return")
    ax.legend(loc="lower right")

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.show()


# ----------------------------- Backtest ---------------------------------------
def split_train_test(log_returns: pd.DataFrame,
                     train_fraction: float = TRAIN_FRACTION) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Chronological split: earliest rows train, latest rows test (no shuffling)."""
    cutoff = int(len(log_returns) * train_fraction)
    return log_returns.iloc[:cutoff], log_returns.iloc[cutoff:]

def backtest_min_vol(log_returns: pd.DataFrame, n_portfolios: int = N_PORTFOLIOS) -> pd.DataFrame:
    """Pick min-vol weights on training data, then compare them with 1/N on test data."""
    train, test = split_train_test(log_returns)

    # Train by find the min-vol portfolio using the training data
    train_mean, _, train_cov = annualize(train)
    weights = random_weights(n_portfolios, len(train_mean))
    _, train_vols, _ = portfolio_performance(weights, train_mean.to_numpy(), train_cov.to_numpy())
    min_vol_weights = weights[np.argmin(train_vols)]

    # Benchmark by equal weight, 1/N in every asset
    equal_weights = np.full(len(train_mean), 1 / len(train_mean))

    # score portfolios on data never seen by the model
    test_mean, _, test_cov = annualize(test)
    rets, vols, sharpes = portfolio_performance(np.vstack([min_vol_weights, equal_weights]),
                                                test_mean.to_numpy(), test_cov.to_numpy())

    return pd.DataFrame({"Return": rets, "Volatility": vols, "Sharpe": sharpes},
                        index=["Min Volatility (trained)", "Equal Weight"])

# ----------------------------- Numerical Optimization -----------------------------
def portfolio_volatility(weights: np.ndarray, cov_matrix: np.ndarray) -> float:
    """Volatility of ONE portfolio: sqrt(w' Σ w)."""
    return np.sqrt(weights @ cov_matrix @ weights)


def negative_sharpe(weights: np.ndarray, mean_returns: np.ndarray, cov_matrix: np.ndarray,
                    risk_free_rate: float = RISK_FREE_RATE) -> float:
    """Sharpe ratio times -1, because scipy can only minimize."""
    return -(weights @ mean_returns - risk_free_rate) / portfolio_volatility(weights, cov_matrix)

def optimize_weights(objective, n_assets: int, args: tuple) -> np.ndarray:
    """Long-only weights summing to 1 that minimize `objective`."""
    result = minimize(objective,
                      x0=np.full(n_assets, 1 / n_assets),
                      args=args,
                      method="SLSQP",
                      bounds=[(0.0, 1.0)] * n_assets,
                      constraints={"type": "eq", "fun": lambda w: w.sum() - 1})
    if not result.success:
        raise RuntimeError(f"Optimization failed: {result.message}")
    return result.x


# ----------------------------- Report Printing --------------------------------
def print_header(title: str) -> None:
    """Print a section banner framed by '=' lines."""
    print("=" * REPORT_WIDTH)
    print(title.upper().center(REPORT_WIDTH))
    print("=" * REPORT_WIDTH)
    print()


def print_table(title: str, table: pd.DataFrame, decimals: int = 4) -> None:
    """Print a titled table between '-' lines, rounded for display."""
    print("-" * REPORT_WIDTH)
    print(title)
    print("-" * REPORT_WIDTH)
    print(table.round(decimals), end="\n\n")

# ----------------------------- Main Function --------------------------------
def main() -> None:
    """Run the full Project 1 pipeline."""
    print_header("Portfolio Optimization")

    # data
    raw = load_data(DATA_PATH)
    log_returns = to_log_returns(raw)

    # metrics
    mean_returns, volatility, cov_matrix = annualize(log_returns)
    summary = pd.DataFrame({"Annual Return": mean_returns, "Annual Volatility": volatility})

    print_header("Data and Asset Metrics")
    print(f"{len(log_returns)} months x {log_returns.shape[1]} assets: "
          f"{log_returns.index[0].date()} to {log_returns.index[-1].date()}\n")
    print_table("Annualized Return and Volatility", summary)
    print_table("Annualized Covariance Matrix", cov_matrix, decimals=5)

    # random portfolios
    weights = random_weights(N_PORTFOLIOS, len(mean_returns))
    port_returns, port_vols, sharpe = portfolio_performance(
        weights, mean_returns.to_numpy(), cov_matrix.to_numpy())
    portfolios = pd.DataFrame({"Return": port_returns,
                               "Volatility": port_vols,
                               "Sharpe": sharpe})

    print_header("Random Portfolio Simulation")
    print_table(f"Summary of {N_PORTFOLIOS:,} Random Portfolios", portfolios.describe())

    # key portfolios
    min_vol_idx = portfolios["Volatility"].idxmin()
    max_sharpe_idx = portfolios["Sharpe"].idxmax()
    key_labels = ["Min Volatility", "Max Sharpe"]
    key_weights = pd.DataFrame({"Min Volatility": weights[min_vol_idx],
                                "Max Sharpe": weights[max_sharpe_idx]},
                               index=mean_returns.index)

    print_header("Key Portfolios")
    print_table("Performance", portfolios.loc[[min_vol_idx, max_sharpe_idx]].set_axis(key_labels))
    print_table("Weights", key_weights)

    # out-of-sample backtest
    print_header("Out-of-Sample Backtest")
    print_table("Train 2004-2019, Test 2020-2023", backtest_min_vol(log_returns))

    #  exact optimization vs. random search
    mu, cov = mean_returns.to_numpy(), cov_matrix.to_numpy()
    opt_weights = np.vstack([optimize_weights(portfolio_volatility, len(mu), args=(cov,)),
                             optimize_weights(negative_sharpe, len(mu), args=(mu, cov))])
    opt_returns, opt_vols, opt_sharpes = portfolio_performance(opt_weights, mu, cov)
    opt_labels = ["Optimal Min Volatility", "Optimal Max Sharpe"]

    print_header("Numerical Optimization")
    print_table("Performance", pd.DataFrame({"Return": opt_returns, "Volatility": opt_vols,
                                             "Sharpe": opt_sharpes}, index=opt_labels))
    print_table("Weights", pd.DataFrame(opt_weights.T, index=mean_returns.index,
                                        columns=opt_labels))

    plot_frontier(portfolios, min_vol_idx, max_sharpe_idx)
    print(f"Chart saved to {FIGURE_PATH}")


if __name__ == "__main__":
    main()