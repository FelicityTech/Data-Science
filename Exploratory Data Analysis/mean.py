# Print the mean and standard deviation of rates for 2019 and 2020 
print(unemployment[["2019", "2020"]].agg(["mean", "std"]))