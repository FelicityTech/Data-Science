-- Round the average number of facebook_likes to one decimal place
SELECT ROUND(avg(facebook_likeS), 1) AS avg_facebook_likes
FROM reviews;


-- Calculate the average budget rounded to the thousands
SELECT ROUND(AVG(budget), -3) AS avg_budget_thousands
FROM films;