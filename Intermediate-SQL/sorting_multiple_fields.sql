-- Select the release year, duration, and title sorted by release year and duration
SELECT release_year, duration, title
FROM films
ORDER BY release_year, duration;

-- Select the certification, release year, and title sorted by certification and release year
SELECT certification, release_year, title 
FROM films
ORDER BY certification, release_year DESC;