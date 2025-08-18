# Print mean and standard deviation grouped by continent
print(unemployment[["continent", "2019", "2020"]].groupby('continent').agg(
["mean", "std"]
))