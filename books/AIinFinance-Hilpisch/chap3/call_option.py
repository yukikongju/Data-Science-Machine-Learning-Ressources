#  _u: stock/option goes up ; _d: stock/option goes down
#  Strike Price: K=14.5; 
#  Bond: B0=10, B1_u = 11, B1_d = 11
#  Stock: S0=10, S1_u = 20, S1_d = 5
#  x: num of stocks, y: num of bonds => portfolio replication
#  https://www.investopedia.com/articles/optioninvestor/07/options_beat_market.asp

import numpy as np

# initialize market payoff matrix
S0, B0, K = 10, 10, 14.5
S1 = np.array([20, 5])
B1 = np.array([11, 11])
M0 = np.array((S0, B0))
M1 = np.array((S1, B1)).T
print(M1)

# get option payoff
C1 = np.maximum(S1 - K, 0)
print(C1)

# portfolio replication: Solve 20x + 11y = 5.5; 5x + 11y = 0
phi = np.linalg.solve(M1, C1)
print(phi)

# verify that portfolio is arbitrage-free
print(np.allclose(C1, np.dot(M1, phi)))

# get price of the replication-portfolio (equal to option arbitrage-free price)
C0 = np.dot(M0, phi)
print(C0)


