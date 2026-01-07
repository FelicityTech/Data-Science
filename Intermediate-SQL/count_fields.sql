-- Count the number of records in the people table
SELECT COUNT(*) AS count_records
fROM people;


-- Count the number of birthdates in the people table
SELECT COUNT(birthdate) AS count_birthdate
FROM people;

-- Count the records for languages and countries represented in the films table
SELECT COUNT(language) AS count_languages, COUNT(country) AS count_countries
FROM films;

-- Return the unique countries from the films table
SELECT DISTINCT country 
FROM films;

-- Count the distinct countries from the films table
SELECT COUNT(DISTINCT country) AS count_distinct_countries
FROM films;