import numpy as np

np.random.seed(123)

tails = [0]

for x in range(10):
    coin = np.random.randint(0, 2)
    tails.append(tails[x] + coin)
print(tails)