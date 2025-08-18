# Print mean and standard deviation grouped by continent
print(unemployment[["continent", "2019", "2020"]].groupby('continent').agg(
    mean_rating=("2019", "mean"),
    std_rating=("2020", "std")
))