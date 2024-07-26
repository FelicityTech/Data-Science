import numpy as np
import matplotlib.pyplot as plot

np.random.seed(123)
final_tails = []
for x in range(100):
    tails = [0]
    for x in range(10):
        coin = np.random.randint(0, 2)
        tails.append(tails[x] + coin)

    final_tails.append(tails[-1])

    