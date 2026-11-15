"""
Option Pricing with Monte Carlo Simulation
Author: Owen Purnomo | Date: 11/01/2026

1. Are the Monte Carlo prices close to the BSM prices?

Yes the Monte Carlo prices are actually very close to the BSM prices, but they are not exactly the same due 
to the randomness in the Monte Carlo simulation. With 100,000 simulations and a seed of 42, the BSM 
call is 10.4506 and the MC call is 10.4205. BSM put is 5.5735 and the MC put is 5.6122
Here both BSM prices also fall inside the 95% confidence intervals of the MC estimates.

2. Why are they not exactly the same?

BSM computes the exact expected discounted payoff while Monte Carlo replaces that expectation 
with the average of 100,000 random payoffs, so it has sampling error.

3. What happens when you increase the number of simulations?

The estimate gets closer to the BSM prices. The improvement is slow, 
because error falls with the squareroot of the path count, so 100 times 
more paths only buys about 10 times more accuracy.

4. Which method is faster for this simple European option?

BSM is much faster for this European option. It is arithmetic plus two normal
CDF calls, while Monte Carlo has to simulate 100,000 random numbers,
exponentiate them, and average, which takes time and gets slower with every path added.

5. Why might Monte Carlo simulation still be useful for more complex options?

BSM only works because a simple European option has a formula.  A lot
of more complex options do not, like ones that pay off on the average price over time.
But Monte Carlo still works the same way: simulate prices, calculate the payoff, take the average, 
and discount it. In this script only the payoff function would need to change.

"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats

# -------------------- Settings --------------------------------

DEFAULT_S = 100  # Initial stock price
DEFAULT_K = 100  # Strike price
DEFAULT_T = 1.0    # Time to maturity in years
DEFAULT_R = 0.05   # Risk-free interest rate
DEFAULT_SIGMA = 0.2  # Annualized Volatility
DEFAULT_NUM_SIMULATIONS = 100_000
DEFAULT_SEED = 42
CONVERGENCE_SIZES = (1_000, 5_000, 10_000, 50_000, 100_000, 500_000)
FIGURE_PATH = (Path(__file__).resolve().parent.parent
               / "Visualization" / "Project 3" / "terminal_prices.png")


# -------------------- Black Scholes ------------------------------
def normal_cdf(x):
    """Standard normal CDF."""
    return stats.norm.cdf(x)


def d1_d2(S, K, T, r, sigma):
    """Return the BSM terms d1 and d2, which the call and put share."""
    sigma_sqrt_t = sigma * np.sqrt(T)
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / sigma_sqrt_t
    d2 = d1 - sigma_sqrt_t
    return d1, d2


def black_scholes_call(S, K, T, r, sigma):
    """BSM price for a European call option."""
    d1, d2 = d1_d2(S, K, T, r, sigma)
    return S * normal_cdf(d1) - K * np.exp(-r * T) * normal_cdf(d2)


def black_scholes_put(S, K, T, r, sigma):
    """BSM price of a European put option."""
    d1, d2 = d1_d2(S, K, T, r, sigma)
    return K * np.exp(-r * T) * normal_cdf(-d2) - S * normal_cdf(-d1)


def put_call_parity(S, K, T, r, sigma):
    """Return (C - P, S - K e^(-rT), difference) to verify parity."""
    left = (black_scholes_call(S, K, T, r, sigma)
            - black_scholes_put(S, K, T, r, sigma))
    right = S - K * np.exp(-r * T)
    return left, right, left - right


# ----------------------------- Monte Carlo -----------------------------
def simulate_terminal_prices(S, T, r, sigma, num_simulations, seed=None):
    """Simulate S_T under risk-neutral GBM as one array, with no loops."""
    rng = np.random.default_rng(seed)           # Random number generator
    z = rng.standard_normal(num_simulations)    # Vectorization
    drift = (r - 0.5 * sigma ** 2) * T
    diffusion = sigma * np.sqrt(T) * z
    return S * np.exp(drift + diffusion)


def call_payoff(terminal_prices, K):
    """Call payoff for every simulated path."""
    return np.maximum(terminal_prices - K, 0.0)


def put_payoff(terminal_prices, K):
    """Put payoff for every simulated path."""
    return np.maximum(K - terminal_prices, 0.0)


def discounted_mean(payoffs, r, T):
    """Present value of the average payoff: e^(-rT) * mean(payoffs)."""
    return np.exp(-r * T) * payoffs.mean()


def confidence_interval(payoffs, r, T, z_score=1.96):
    """95% confidence interval (low, high) for a Monte Carlo price."""
    price = discounted_mean(payoffs, r, T)
    std_error = np.exp(-r * T) * payoffs.std(ddof=1) / np.sqrt(len(payoffs))
    return price - z_score * std_error, price + z_score * std_error


def monte_carlo_prices(S, K, T, r, sigma, num_simulations, seed=None):
    """Return (call, put, terminal_prices) from one Monte Carlo draw."""
    terminal_prices = simulate_terminal_prices(
        S, T, r, sigma, num_simulations, seed)
    call = discounted_mean(call_payoff(terminal_prices, K), r, T)
    put = discounted_mean(put_payoff(terminal_prices, K), r, T)
    return call, put, terminal_prices


# -------------------------- Input Validation --------------------------
def validate_inputs(S, K, T, sigma, num_simulations):
    """Validate inputs, raise ValueError and message if invalid."""
    checks = (
        (S > 0, "Current stock price S must be greater than 0."),
        (K > 0, "Strike price K must be greater than 0."),
        (T > 0, "Time to maturity T must be greater than 0."),
        (sigma > 0, "Volatility sigma must be greater than 0."),
        (num_simulations > 0,
         "Number of simulations must be a positive integer."),
    )

    for is_valid, message in checks:
        if not is_valid:
            raise ValueError(message)


# ----------------------------- Report --------------------------------
def print_section(title, rows):
    """Print a titled block of 'Label: value' lines."""
    print(title)
    print("-" * len(title))
    for label, value in rows:
        print(f"{label}: {value}")
    print()


def print_report(results):
    """Print the pricing report in the format given in the assignment."""
    title = "Option Pricing Report"
    print(title)
    print("=" * len(title))
    print()
    print_section("Input Parameters", [
        ("Current Stock Price", f"{results['S']:.2f}"),
        ("Strike Price", f"{results['K']:.2f}"),
        ("Time to Maturity", f"{results['T']:.2f} years"),
        ("Risk-Free Rate", f"{results['r']:.2%}"),
        ("Volatility", f"{results['sigma']:.2%}"),
        ("Number of Simulations", f"{results['num_simulations']}"),
    ])
    print_section("Black-Scholes-Merton Prices", [
        ("Call Option Price", f"{results['bsm_call']:.4f}"),
        ("Put Option Price", f"{results['bsm_put']:.4f}"),
    ])
    left, right, gap = results["parity"]
    print_section("Put-Call Parity Check", [
        ("C - P", f"{left:.4f}"),
        ("S - K e^(-rT)", f"{right:.4f}"),
        ("Difference", f"{gap:.2e}"),
    ])
    print_section("Monte Carlo Prices", [
        ("Call Option Price", f"{results['mc_call']:.4f}"),
        ("Put Option Price", f"{results['mc_put']:.4f}"),
    ])
    call_low, call_high = results["call_ci"]
    put_low, put_high = results["put_ci"]
    print_section("95% Confidence Intervals (Monte Carlo)", [
        ("Call Option Price", f"[{call_low:.4f}, {call_high:.4f}]"),
        ("Put Option Price", f"[{put_low:.4f}, {put_high:.4f}]"),
    ])
    print_section("Comparison (Monte Carlo minus BSM)", [
        ("Call Price Difference",
         f"{results['mc_call'] - results['bsm_call']:.4f}"),
        ("Put Price Difference",
         f"{results['mc_put'] - results['bsm_put']:.4f}"),
    ])


# ----------------------------- Analysis ------------------------
def print_convergence_table(S, K, T, r, sigma, sizes, seed=None):
    """Print Monte Carlo prices and their errors versus BSM by path count."""
    bsm_call = black_scholes_call(S, K, T, r, sigma)
    bsm_put = black_scholes_put(S, K, T, r, sigma)
    header = (f"{'Paths':>10} {'MC Call':>10} {'MC Put':>10} "
              f"{'Call Error':>10} {'Put Error':>10}")
    print("Convergence Analysis")
    print("-" * len(header))
    print(header)
    for n in sizes:
        mc_call, mc_put, _ = monte_carlo_prices(S, K, T, r, sigma, n, seed)
        print(f"{n:>10,} {mc_call:>10.4f} {mc_put:>10.4f} "
              f"{mc_call - bsm_call:>10.4f} {mc_put - bsm_put:>10.4f}")
    print()


def plot_terminal_prices(terminal_prices, K, path=FIGURE_PATH):
    """Histogram of simulated S_T with the strike marked; saves a PNG."""
    plt.figure(figsize=(10, 6))
    plt.hist(terminal_prices, bins=50, alpha=0.7, edgecolor="black")
    plt.axvline(K, color="red", linestyle="--", label="Strike Price")
    plt.title("Distribution of Simulated Terminal Stock Prices")
    plt.xlabel("Simulated Terminal Stock Price")
    plt.ylabel("Frequency")
    plt.legend()
    plt.grid(True)
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=150)
    plt.show()


# ----------------------------- Command line --------------------------
def parse_args():
    """Read optional command-line inputs; each falls back to a default."""
    parser = argparse.ArgumentParser(description="BSM vs Monte Carlo pricing")
    parser.add_argument("--S", type=float, default=DEFAULT_S)
    parser.add_argument("--K", type=float, default=DEFAULT_K)
    parser.add_argument("--T", type=float, default=DEFAULT_T)
    parser.add_argument("--r", type=float, default=DEFAULT_R)
    parser.add_argument("--sigma", type=float, default=DEFAULT_SIGMA)
    parser.add_argument("--num_simulations", type=int,
                        default=DEFAULT_NUM_SIMULATIONS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return parser.parse_args()


# ----------------------------- Main ----------------------------------
def main():
    """Validate inputs, price the options, and print the report."""
    args = parse_args()
    try:
        validate_inputs(args.S, args.K, args.T, args.sigma,
                        args.num_simulations)
    except ValueError as error:
        sys.exit(f"Error: {error}")

    inputs = (args.S, args.K, args.T, args.r, args.sigma)
    mc_call, mc_put, terminal_prices = monte_carlo_prices(
        *inputs, args.num_simulations, args.seed)
    results = {
        "S": args.S, "K": args.K, "T": args.T,
        "r": args.r, "sigma": args.sigma,
        "num_simulations": args.num_simulations,
        "bsm_call": black_scholes_call(*inputs),
        "bsm_put": black_scholes_put(*inputs),
        "mc_call": mc_call,
        "mc_put": mc_put,
    }
    call_payoffs = call_payoff(terminal_prices, args.K)
    put_payoffs = put_payoff(terminal_prices, args.K)
    results["call_ci"] = confidence_interval(call_payoffs, args.r, args.T)
    results["put_ci"] = confidence_interval(put_payoffs, args.r, args.T)
    results["parity"] = put_call_parity(*inputs)
    print_report(results)
    print_convergence_table(*inputs, CONVERGENCE_SIZES, args.seed)
    plot_terminal_prices(terminal_prices, args.K)


if __name__ == "__main__":
    main()
