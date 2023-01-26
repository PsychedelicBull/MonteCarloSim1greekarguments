import numpy as np
import yfinance as yf


def monte_carlo_american_option_greeks(S0, K, r, T, sigma, n_trials, n_steps, type='call'):
    dt = T / n_steps
    S = np.zeros((n_steps + 1, n_trials))
    S[0] = S0
    for i in range(1, n_steps + 1):
        z = np.random.normal(size=n_trials)
        S[i] = S[i - 1] * np.exp((r - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * z)
    if type == 'call':
        payoff = np.maximum(S[-1] - K, 0)
    else:
        payoff = np.maximum(K - S[-1], 0)
    discounted_payoff = payoff * np.exp(-r * T)
    option_price = discounted_payoff.mean()

    # Greek inputs calculation
    dS = S[-1] - S0
    dS_mean = dS.mean()
    dS_std = dS.std()
    dS_mean_normalized = dS_mean / S0
    dt = T / n_steps
    Delta = dS_mean_normalized / sigma * np.sqrt(dt)
    Gamma = (dS_mean_normalized / (sigma * np.sqrt(dt)) - 1 / 2) * (S0 / dS_std)
    Theta = -S0 * dS_mean_normalized * 1 / dt
    Vega = S0 * dS_std * np.sqrt(dt)
    Rho = -T * payoff.mean()

    return option_price, Delta, Gamma, Theta, Vega, Rho


ticker = "aapl"
data = yf.download(ticker, start="2021-01-01", end="2021-01-31")

# Get the close price and calculate the volatility
S0 = data["Close"][-1]
sigma = data["Close"].pct_change().std() * np.sqrt(252)

# Define the option parameters
K = 100
r = 0.05
T = 1
n_trials = 10000
n_steps = 252

call_price, call_Delta, call_Gamma, call_Theta, call_Vega, call_Rho = monte_carlo_american_option_greeks(S0, K, r, T,
                                                                                                         sigma,
                                                                                                         n_trials,
                                                                                                         n_steps,
                                                                                                         type='call')
put_price, put_Delta, put_Gamma, put_Theta, put_Vega, put_Rho = monte_carlo_american_option_greeks
