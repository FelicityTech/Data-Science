-- Select all columns from cities
SELECT *
FROM cities;

SELECT * 
FROM cities
-- Inner join to countries
INNER JOIN countries
-- Match on country codes
ON cities.country_code = countries.code

-- Select name fields (with alias) and region 
SELECT cities.name AS city, countries.name AS country, region
FROM cities 
INNER JOIN countries 
ON cities.country_code = countries.code;