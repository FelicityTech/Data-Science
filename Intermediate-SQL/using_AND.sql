-- Select the title and release_year for all German-language films released before 2000
SELECT title, release_year
FROM films
WHERE release_year < 2000
AND language = 'German';

-- Update the query to see all German-language films released after 2000
SELECT title, release_year
FROM films
WHERE release_year > 2000
	AND language = 'German';

-- Select all records for German-language films released after 2000 and before 2010
SELECT *
FROM films
WHERE release_year > 2000
    AND release_year < 2010
    AND language = 'German';