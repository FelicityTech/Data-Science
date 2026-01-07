-- Select the names that start with B
SELECT name
FROM people
WHERE name LIKE 'B%';

SELECT name
FROM people
-- Select the names that have r as the second letter
WHERE name LIKE '_r%'

SELECT name
FROM people
-- Select names that don't start with A
WHERE name NOT LIKE 'A%' 