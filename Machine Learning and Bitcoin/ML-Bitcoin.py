"""
Statistical and Machine Learning Models in Bitcoin Forecasting
Author: Owen Purnomo | Date: 11/12/2026 | MF 703
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             mean_absolute_error, mean_squared_error,
                             precision_score, r2_score, recall_score,
                             roc_auc_score)
from sklearn.model_selection import (GridSearchCV, TimeSeriesSplit,
                                     train_test_split)
from sklearn.preprocessing import StandardScaler


# ------------------------------- Settings ------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "Data" / "BTC-USD.csv"
FIGURE_DIR = BASE_DIR / "Visualization" / "Project 2"
PRICE_COLUMNS = ["Open", "High", "Low", "Close", "Adj Close"]
DAYS_PER_YEAR = 365   # Bitcoin trades every day
MAX_LAGS = 20
ARIMA_ORDERS = range(4)   # p and q in {0, 1, 2, 3}
LJUNG_BOX_LAGS = [10, 20, 30]
FORECAST_DAYS = 7
HISTORY_DAYS = 90
RETURN_LAGS = [1, 2, 3, 5]
TRAIN_FRACTION = 0.8
RSI_WINDOW = 14
SEED = 42
RF_GRID = {"max_depth": [3, 5, None], "min_samples_leaf": [20, 50, 100]}
REPORT_WIDTH = 72


# ------------------------- Part 1.1: Ingestion ---------------------------
def load_data(path: Path) -> pd.DataFrame:
    """Read the price CSV and return it indexed by date."""
    return pd.read_csv(path, index_col="Date", parse_dates=True)


def check_integrity(prices: pd.DataFrame) -> dict:
    """Return the Part 1.1 integrity checks: ordering, gaps, bad values."""
    index = prices.index
    calendar = pd.date_range(index.min(), index.max(), freq="D")
    return {
        "chronological": index.is_monotonic_increasing,
        "duplicate dates": index.duplicated().sum(),
        "missing values": prices.isna().sum().sum(),
        "prices <= 0": (prices[PRICE_COLUMNS] <= 0).sum().sum(),
        "missing days": len(calendar.difference(index)),
    }


def clean_prices(prices: pd.DataFrame) -> pd.DataFrame:
    """Sort by date; drop duplicate dates, NaN rows and non-positive prices."""
    prices = prices.sort_index()
    prices = prices[~prices.index.duplicated()]
    prices = prices.dropna()
    return prices[(prices[PRICE_COLUMNS] > 0).all(axis=1)]


# ------------------------- Part 1.2: Log returns -------------------------
def log_returns(prices: pd.Series) -> pd.Series:
    """Daily log returns r_t = ln(P_t) - ln(P_t-1); drops the first NaN."""
    return np.log(prices).diff().dropna()


# ------------------------- Part 1.3: Moments -----------------------------
def summary_stats(returns: pd.Series,
                  periods: int = DAYS_PER_YEAR) -> pd.Series:
    """Moments of daily returns, annualized with the 365-day convention."""
    jb_stat, jb_pvalue = stats.jarque_bera(returns)
    return pd.Series({
        "Mean (daily)": returns.mean(),
        "Mean (annual)": returns.mean() * periods,
        "Std (daily)": returns.std(),
        "Volatility (annual)": returns.std() * np.sqrt(periods),
        "Skewness": returns.skew(),
        "Excess kurtosis": returns.kurt(),
        "Jarque-Bera stat": jb_stat,
        "Jarque-Bera p-value": jb_pvalue,
    })


# ------------------------- Part 1.4: Stationarity ------------------------
def adf_test(series: pd.Series, name: str) -> pd.Series:
    """Augmented Dickey-Fuller test; H0: unit root (non-stationary)."""
    stat, pvalue, lags, _, critical, _ = adfuller(series, autolag="AIC")
    return pd.Series({"ADF statistic": stat,
                      "p-value": pvalue,
                      "Lags used": lags,
                      "Critical 1%": critical["1%"],
                      "Critical 5%": critical["5%"],
                      "Critical 10%": critical["10%"]}, name=name)


# ------------------------- Part 1.5: Visualizations ------------------
def save_figure(fig: plt.Figure, name: str, folder: Path = FIGURE_DIR):
    """Saves figure as PNG"""
    folder.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(folder / name, dpi=150)
    plt.close(fig)


def plot_prices(prices: pd.Series):
    """Price history on linear and logarithmic scales"""
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    for ax, scale in zip(axes, ["linear", "log"]):
        ax.plot(prices.index, prices, color="tab:blue", linewidth=1)
        ax.set_yscale(scale)
        ax.set_ylabel(f"Price (USD, {scale} scale)")
        ax.grid(True, alpha=0.3)
    axes[0].set_title("Bitcoin (BTC-USD) daily closing price")
    save_figure(fig, "price_history.png")


def plot_returns(returns: pd.Series):
    """Daily log returns over time, showing volatility clustering"""
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(returns.index, returns, color="tab:blue", linewidth=0.6)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title("Bitcoin daily log returns")
    ax.set_ylabel("Log return")
    ax.grid(True, alpha=0.3)
    save_figure(fig, "daily_returns.png")


def plot_distribution(returns: pd.Series):
    """Histogram and KDE of standardized returns against N(0, 1)"""
    z = (returns - returns.mean()) / returns.std()
    grid = np.linspace(-6, 6, 400)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(z, bins=150, density=True, alpha=0.4, color="tab:blue",
            label="Histogram")
    ax.plot(grid, stats.gaussian_kde(z)(grid), color="tab:blue",
            linewidth=2, label="KDE of returns")
    ax.plot(grid, stats.norm.pdf(grid), color="tab:red", linewidth=2,
            linestyle="--", label="Standard normal")
    ax.set_xlim(-6, 6)
    ax.set_title("Distribution of standardized daily log returns")
    ax.set_xlabel("Standardized return (z-score)")
    ax.set_ylabel("Density")
    ax.legend()
    ax.grid(True, alpha=0.3)
    save_figure(fig, "return_distribution.png")


# ------------------------- Part 2.2: Order identification -----------
def plot_correlograms(returns: pd.Series, lags: int = MAX_LAGS):
    """ACF and PACF of returns with the 95% band around zero"""
    fig, axes = plt.subplots(2, 1, figsize=(10, 7))
    plot_acf(returns, lags=lags, ax=axes[0], title="ACF of daily returns")
    plot_pacf(returns, lags=lags, ax=axes[1], title="PACF of daily returns")
    for ax in axes:
        ax.set_ylim(-0.1, 0.1)
    save_figure(fig, "acf_pacf.png")


def arima_grid_search(returns: pd.Series,
                      orders: range = ARIMA_ORDERS) -> pd.DataFrame:
    """Fit ARIMA(p, 0, q) for every p, q in orders; return AIC and BIC"""
    rows = []
    for p in orders:
        for q in orders:
            result = ARIMA(returns, order=(p, 0, q)).fit()
            rows.append({"p": p, "q": q, "AIC": result.aic, "BIC": result.bic})
    return pd.DataFrame(rows)


# ------------------------- Part 2.3: Estimation and diagnostics ---------
def fit_arima(returns: pd.Series, order: tuple):
    """Fit ARIMA(p, d, q) to the returns and return the results object."""
    return ARIMA(returns, order=order).fit()


def residual_diagnostics(result, order: tuple,
                         lags: list = LJUNG_BOX_LAGS) -> pd.DataFrame:
    """Residual ACF and Q-Q plot; return the Ljung-Box test table."""
    resid = result.resid
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    plot_acf(resid, lags=MAX_LAGS, ax=axes[0], title="ACF of ARIMA residuals")
    axes[0].set_ylim(-0.1, 0.1)
    stats.probplot(resid / resid.std(), dist="norm", plot=axes[1])
    axes[1].set_title("Q-Q plot of standardized residuals")
    save_figure(fig, "arima_residuals.png")
    # degrees of freedom are reduced by the p + q fitted ARMA terms
    return acorr_ljungbox(resid, lags=lags, model_df=order[0] + order[2])


# ------------------------- Part 2.4: Forecasting -------------------------
def forecast_prices(result, last_price: float,
                    steps: int = FORECAST_DAYS) -> pd.DataFrame:
    """Return forecast with 95% bands, compounded into price paths."""
    table = result.get_forecast(steps=steps).summary_frame(alpha=0.05)
    out = pd.DataFrame({"return": table["mean"],
                        "return_lower": table["mean_ci_lower"],
                        "return_upper": table["mean_ci_upper"]})
    for col in ["return", "return_lower", "return_upper"]:
        out[col.replace("return", "price")] = (last_price
                                               * np.exp(out[col].cumsum()))
    return out


def plot_forecast(returns: pd.Series, prices: pd.Series, fc: pd.DataFrame,
                  history: int = HISTORY_DAYS) -> None:
    """Two panels: recent returns + forecast, recent prices + price bands."""
    fig, axes = plt.subplots(2, 1, figsize=(10, 8))
    axes[0].plot(returns.index[-history:], returns.iloc[-history:],
                 color="tab:blue", linewidth=1, label="Actual")
    axes[0].plot(fc.index, fc["return"], color="tab:red", label="Forecast")
    axes[0].fill_between(fc.index, fc["return_lower"], fc["return_upper"],
                         color="tab:red", alpha=0.2, label="95% interval")
    axes[0].set_title(f"Daily log returns: last {history} days and "
                      f"{len(fc)}-day forecast")
    axes[1].plot(prices.index[-history:], prices.iloc[-history:],
                 color="tab:blue", linewidth=1, label="Actual")
    axes[1].plot(fc.index, fc["price"], color="tab:red", label="Forecast")
    axes[1].fill_between(fc.index, fc["price_lower"], fc["price_upper"],
                         color="tab:red", alpha=0.2, label="95% interval")
    axes[1].set_title("Price (USD): recent history and forecast path")
    for ax in axes:
        ax.legend(loc="upper left")
        ax.grid(True, alpha=0.3)
    save_figure(fig, "arima_forecast.png")


# ------------------------- Part 3.1: Feature engineering ----------------
def build_features(prices: pd.DataFrame,
                   returns: pd.Series) -> tuple[pd.DataFrame, pd.Series]:
    """Leak-free features X_t (known at day t) and target y_t = r_t+1."""
    close, volume = prices["Adj Close"], prices["Volume"]
    X = pd.DataFrame(index=returns.index)
    for lag in RETURN_LAGS:
        X[f"ret_lag{lag}"] = returns.shift(lag)
    X["vol_7"] = returns.rolling(7).std()
    X["vol_30"] = returns.rolling(30).std()
    X["dlog_volume"] = np.log(volume).diff()
    X["volume_ratio"] = volume / volume.rolling(20).mean()
    X["ma_ratio"] = close.rolling(7).mean() / close.rolling(30).mean() - 1
    X["gk_vol_7"] = np.sqrt(garman_klass(prices).rolling(7).mean())
    X["rsi_14"] = rsi(close)
    y = returns.shift(-1).rename("target")
    data = pd.concat([X, y], axis=1).dropna()
    return data.drop(columns="target"), data["target"]


# ------------------------- Part 3.2-3.3: Split and scale ----------------
def split_and_scale(X: pd.DataFrame, y: pd.Series,
                    train_fraction: float = TRAIN_FRACTION) -> tuple:
    """Chronological split; StandardScaler fitted on the train set only."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=train_fraction, shuffle=False)
    scaler = StandardScaler().fit(X_train)
    X_train = pd.DataFrame(scaler.transform(X_train),
                           index=X_train.index, columns=X.columns)
    X_test = pd.DataFrame(scaler.transform(X_test),
                          index=X_test.index, columns=X.columns)
    return X_train, X_test, y_train, y_test


# ------------------------- Part 3.4: Model and evaluation ---------------
def evaluate(y_true: pd.Series, y_pred: np.ndarray, name: str) -> pd.Series:
    """MSE, RMSE, MAE, R^2 and directional hit rate for one prediction."""
    mse = mean_squared_error(y_true, y_pred)
    hit_rate = np.mean(np.sign(y_pred) == np.sign(y_true))
    if np.all(y_pred == 0):
        hit_rate = np.nan      # a zero forecast has no direction
    return pd.Series({"MSE": mse,
                      "RMSE": np.sqrt(mse),
                      "MAE": mean_absolute_error(y_true, y_pred),
                      "R2": r2_score(y_true, y_pred),
                      "Hit rate": hit_rate}, name=name)


# ------------------------- Bonus A: Advanced features -------------------
def garman_klass(prices: pd.DataFrame) -> pd.Series:
    """Daily Garman-Klass variance estimate from open, high, low, close."""
    log_hl = np.log(prices["High"] / prices["Low"])
    log_co = np.log(prices["Close"] / prices["Open"])
    return 0.5 * log_hl ** 2 - (2 * np.log(2) - 1) * log_co ** 2


def rsi(close: pd.Series, window: int = RSI_WINDOW) -> pd.Series:
    """Relative Strength Index, 0-100, with Wilder's exponential smoothing."""
    change = close.diff()
    gain = change.clip(lower=0).ewm(alpha=1 / window, adjust=False).mean()
    loss = (-change).clip(lower=0).ewm(alpha=1 / window, adjust=False).mean()
    return 100 - 100 / (1 + gain / loss)


# ------------------------- Bonus B: Random forest -----------------------
def tune_random_forest(X_train: pd.DataFrame,
                       y_train: pd.Series) -> GridSearchCV:
    """Random forest tuned by grid search with 5-fold TimeSeriesSplit."""
    forest = RandomForestRegressor(n_estimators=200, random_state=SEED,
                                   n_jobs=-1)
    search = GridSearchCV(forest, RF_GRID, cv=TimeSeriesSplit(n_splits=5),
                          scoring="neg_mean_squared_error")
    return search.fit(X_train, y_train)


def plot_importances(model, columns: pd.Index) -> pd.Series:
    """Horizontal bar chart of MDI feature importances; returns them."""
    importances = pd.Series(model.feature_importances_,
                            index=columns).sort_values()
    fig, ax = plt.subplots(figsize=(8, 4.5))
    importances.plot.barh(ax=ax, color="tab:blue")
    ax.set_title("Random forest feature importances (MDI)")
    ax.set_xlabel("Importance")
    save_figure(fig, "feature_importances.png")
    return importances


# ------------------------- Bonus C: Direction classification ------------
def classify_direction(X_train: pd.DataFrame, X_test: pd.DataFrame,
                       y_train: pd.Series, y_test: pd.Series) -> pd.Series:
    """Logistic regression on up (1) / down (0) days; test-set metrics."""
    up_train = (y_train > 0).astype(int)
    up_test = (y_test > 0).astype(int)
    clf = LogisticRegression(random_state=SEED).fit(X_train, up_train)
    pred = clf.predict(X_test)
    prob_up = clf.predict_proba(X_test)[:, 1]
    tn, fp, fn, tp = confusion_matrix(up_test, pred).ravel()
    return pd.Series({"Accuracy": accuracy_score(up_test, pred),
                      "Precision": precision_score(up_test, pred),
                      "Recall": recall_score(up_test, pred),
                      "F1": f1_score(up_test, pred),
                      "ROC-AUC": roc_auc_score(up_test, prob_up),
                      "TP": tp, "FP": fp, "TN": tn, "FN": fn,
                      "Share of up days": up_test.mean()})


# ------------------------------- Reporting -------------------------------
def print_header(title: str) -> None:
    """Print a section banner."""
    print("\n" + "=" * REPORT_WIDTH)
    print(title.upper().center(REPORT_WIDTH))
    print("=" * REPORT_WIDTH)


def print_table(title: str, table, decimals: int = 4) -> None:
    """Print a titled, rounded table (Series or DataFrame) between rules."""
    if isinstance(table, pd.Series):
        table = table.to_frame("Value")
    print("\n" + title)
    print("-" * REPORT_WIDTH)
    print(table.round(decimals).to_string())
    print("-" * REPORT_WIDTH)


# ------------------------------- Main ----------------------------------
def main() -> None:
    """Run the Project 2 pipeline."""
    print_header("Data and exploratory analysis")

    # 1.1 ingestion and integrity
    raw = load_data(DATA_PATH)
    print(f"Raw data: {raw.shape[0]} rows x {raw.shape[1]} columns, "
          f"{raw.index[0].date()} to {raw.index[-1].date()}")
    print_table("Integrity checks (raw file)", pd.Series(check_integrity(raw)))
    prices = clean_prices(raw)
    print(f"Clean data: {prices.shape[0]} rows (unchanged)")

    # 1.2 log returns, with the additivity check sum(r_t) = ln(P_T / P_0)
    returns = log_returns(prices["Adj Close"])
    total = np.log(prices["Adj Close"].iloc[-1] / prices["Adj Close"].iloc[0])
    print(f"Log returns: {len(returns)} observations | sum of returns = "
          f"{returns.sum():.4f} | ln(P_last / P_first) = {total:.4f}")

    # 1.3 summary statistics
    print_table("Summary statistics of daily log returns",
                summary_stats(returns))

    # 1.4 stationarity: price should be I(1), returns I(0)
    adf_table = pd.concat([adf_test(prices["Adj Close"], "Price"),
                           adf_test(returns, "Log return")], axis=1)
    print_table("Augmented Dickey-Fuller tests", adf_table)

    # 1.5 figures
    plot_prices(prices["Adj Close"])
    plot_returns(returns)
    plot_distribution(returns)

    print_header("ARIMA modelling")

    # 2.2 order identification: correlograms and AIC/BIC grid
    returns = returns.asfreq("D")
    plot_correlograms(returns)
    grid = arima_grid_search(returns)
    print_table("AIC by (p, q)",
                grid.pivot(index="p", columns="q", values="AIC"), 1)
    print_table("BIC by (p, q)",
                grid.pivot(index="p", columns="q", values="BIC"), 1)
    best = grid.loc[grid["AIC"].idxmin()]
    order = (int(best["p"]), 0, int(best["q"]))
    print("Selected by AIC:", order)

    # 2.3 fit the selected model and check the residuals
    result = fit_arima(returns, order)
    print()
    print(result.summary())
    print_table("Ljung-Box test on residuals",
                residual_diagnostics(result, order))

    # 2.4 7-day forecast of returns, converted to prices
    fc = forecast_prices(result, prices["Adj Close"].iloc[-1])
    print_table(f"{FORECAST_DAYS}-day forecast of returns and prices", fc)
    plot_forecast(returns, prices["Adj Close"], fc)

    print_header("Predictive modelling")

    # 3.1-3.3 features, chronological 80/20 split, scaling
    X, y = build_features(prices, returns)
    X_train, X_test, y_train, y_test = split_and_scale(X, y)
    print(f"Features: {X.shape[1]} columns, {len(X)} rows")
    print(f"Train: {X_train.index[0].date()} to {X_train.index[-1].date()}"
          f" ({len(X_train)} rows)")
    print(f"Test:  {X_test.index[0].date()} to {X_test.index[-1].date()}"
          f" ({len(X_test)} rows)")

    # 3.4 linear regression vs naive baselines on the test set
    model = LinearRegression().fit(X_train, y_train)
    y_pred = model.predict(X_test)
    zero = np.zeros(len(y_test))
    train_mean = np.full(len(y_test), y_train.mean())
    scores = pd.concat([evaluate(y_test, y_pred, "Linear regression"),
                        evaluate(y_test, zero, "Zero baseline"),
                        evaluate(y_test, train_mean, "Train-mean baseline")],
                       axis=1)
    print_table("Test-set performance", scores, 5)
    print_table("Coefficients (standardized features)",
                pd.Series(model.coef_, index=X.columns), 5)

    print_header("Advanced features and models")

    # Bonus B: random forest with TimeSeriesSplit tuning
    search = tune_random_forest(X_train, y_train)
    forest_pred = search.best_estimator_.predict(X_test)
    print("Random forest best parameters:", search.best_params_)
    print_table("Random forest, test set",
                evaluate(y_test, forest_pred, "Random forest"), 5)
    print_table("Feature importances (MDI)",
                plot_importances(search.best_estimator_, X.columns))

    # Bonus C: directional classification
    print_table("Logistic regression (up/down), test set",
                classify_direction(X_train, X_test, y_train, y_test))

    print(f"\nFigures saved to {FIGURE_DIR}")


if __name__ == "__main__":
    main()
