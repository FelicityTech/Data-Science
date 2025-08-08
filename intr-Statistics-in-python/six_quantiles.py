# Calculate the quintiles of co2_emission
print(np.quantile(food_consumption['co2_emission'], [0,0.2, 0.4, 0.6, 0.8, 1]))