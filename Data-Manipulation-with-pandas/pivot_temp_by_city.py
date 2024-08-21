# Add a year column to temperatures
temperatures["year"] = temperatures["date"].dt.year

# Pivot avg_temp_c by country and city vs year
temp_by_country_city_vs_year = temperatures.pivot_table(values="avg_temp_c",index=['country', 'city'], columns="year", aggfunc="mean")

# See the result
print(temp_by_country_city_vs_year)