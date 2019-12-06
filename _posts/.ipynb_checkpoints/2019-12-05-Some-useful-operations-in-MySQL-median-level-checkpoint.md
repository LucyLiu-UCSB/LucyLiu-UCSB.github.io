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

```sql
### MySQL

```