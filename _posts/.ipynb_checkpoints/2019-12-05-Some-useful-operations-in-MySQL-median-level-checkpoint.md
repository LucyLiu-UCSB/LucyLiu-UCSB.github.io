---
title: Some useful operations in MySQL--Median level
date: 2019-12-05 16：12
categories: [Technical Tools]
tags: [SQL]
---

The second post of SQL includes median level applications. 

## Running total calculation -- window function

Aggregations/rankings on a subset of rows relative to the current row being transformed by `SELECT`.

```sql
function(...) OVER(
    PARTITION BY...
    ORDER BY...
    ROWS BETWEEN ... AND ...
)
```

`Activity` table:

| player_id | device_id | event_date | games_played |
|----------:|----------:|-----------:|--------------|
| 1         | 2         | 2016-03-01 | 5            |
| 1         | 2         | 2016-05-02 | 6            |
| 1         | 3         | 2017-06-25 | 1            |
| 3         | 1         | 2016-03-02 | 0            |
| 3         | 4         | 2018-07-03 | 5            |

**Q:**  Reports for each player and date, how many games played so far by the player. That is, the total number of games played by the player until that date. 

```sql
### MS SQL
SELECT player_id, event_date, SUM(games_played) OVER (PARTITION BY player_id ORDER BY event_date) AS games_played_so_far
FROM Activity
ORDER BY player_id
```
The `ORDER BY` clause defines by what ordering the cumulation should happen.

Other useful windown functions are

```sql
ROW_NUMBER() OVER()
RANK() OVER()
DENSE_RANK() OVER()
```

```sql
### MySQL
SELECT a1.player_id, a1.event_date, SUM(a2.games_played) as games_played_so_far
FROM Activity a1, Activity a2
WHERE a1.player_id = a2.player_id AND a1.event_date >= a2.event_date
GROUP BY a1.player_id, a1.event_date
```

## conditions in SQL

`Transactions` table:

| id   | country | state    | amount | trans_date |
|-----:|--------:|---------:|-------:|-----------:|
| 121  | US      | approved | 1000   | 2018-12-18 |
| 122  | US      | declined | 2000   | 2018-12-19 |
| 123  | US      | approved | 2000   | 2019-01-01 |
| 124  | DE      | approved | 2000   | 2019-01-07 |

**Q:** find for each month and country, the number of transactions and their total amount, the number of approved transactions and their total amount.

```sql
##
SELECT LEFT(trans_date, 7) AS month
##
SELECT DATE_FORMAT(trans_date, '%Y-%m') AS month, country, 
       COUNT(*) AS trans_count,
       SUM(IF(state = 'approved', 1, 0)) AS approved_count, 
       SUM(amount) AS trans_total_amount, 
       SUM(CASE WHEN state = 'approved' THEN amount ELSE 0 END) AS approved_total_amount
FROM Transactions
GROUP BY month, country
                           
```

## function in SQL
Write a SQL query to get the nth highest salary from the `Employee` table.

```sql
CREATE FUNCTION getNthHighestSalary(N INT) RETURNS INT
BEGIN
DECLARE M INT;
SET M = N - 1;
  RETURN (
    
      SELECT DISTINCT Salary
      FROM Employee
      ORDER BY Salary DESC
      LIMIT 1 OFFSET M
      
  );END
```
