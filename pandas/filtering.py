# Import cars data
import pandas as pd
cars = pd.read_csv('cars.csv', index_col = 0)

# Iterate over rows of cars
for label, rows in cars.iterrows():
    print(label)
    print(rows)