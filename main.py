import numpy as np

# Input Option Greek values
delta = 0.5
gamma = 0.1
theta = -0.02
vega = 0.03
rho = 0.01

# Other input parameters
spot_price = 100
strike_price = 110
volatility = 0.2
interest_rate = 0.05
time_to_expiration = 1

# Number of simulations
num_simulations = 10000

# Generate random numbers for the simulations
np.random.seed(0)
rand1 = np.random.normal(0, 1, num_simulations)
rand2 = np.random.normal(0, 1, num_simulations)

# Simulate the price of the underlying asset at expiration
price_at_expiration = spot_price * np.exp((interest_rate - (volatility ** 2) / 2) * time_to_expiration + volatility * np.sqrt(time_to_expiration) * rand1)

# Calculate the payoff at expiration
payoff_at_expiration = np.maximum(price_at_expiration - strike_price, 0)

# Simulate the option price at time 0
price_at_time_0 = payoff_at_expiration * np.exp(-interest_rate * time_to_expiration)

# Add the Greek value perturbations
price_at_time_0 += delta * (price_at_expiration - spot_price) + gamma * (price_at_expiration - spot_price) ** 2 + theta * time_to_expiration + vega * rand2 + rho * interest_rate

# Calculate the average option price
option_price = np.mean(price_at_time_0)

print("Option price: ", option_price)



