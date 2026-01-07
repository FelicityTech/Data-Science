continent_summary = unemployment.groupby("continent").agg(
    # Create the mean_rate_2021 column
    mean_rate_2021=('2021', 'mean'),
    # Create the std_rate_2021 column
    std_rate_2021=('2021', 'std')
)
print(continent_summary)