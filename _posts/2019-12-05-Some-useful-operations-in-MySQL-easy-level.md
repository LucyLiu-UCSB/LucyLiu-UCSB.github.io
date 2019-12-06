---
title: Some useful operations in MySQL--Easy level
date: 2019-12-05 09：08
categories: [Technical Tools]
tags: [SQL]
---

I am practicing some SQL question to prepare for 2020 summer intership interviews. I learned some useful operations beyound the basic `SELECT`, `FROM`, `WHERE`, `GROUP BY` and `HAVING`. 

## Some useful DATE operations

The date record has data type as `DATE` in MySQL. It has some special operations.

**EX1:**


| Id(INT) | RecordDate(DATE) | Temperature(INT) |
|---------|:----------------:|-----------------:|
|       1 |       2015-01-01 |               10 |
|       2 |       2015-01-02 |               25 |
|       3 |       2015-01-03 |               20 |
|       4 |       2015-01-04 |               30 |

**Q:** Given a `Weather` table, find all dates' Ids with higher temperature compared to its previous (yesterday's) dates.

``` sql
SELECT w2.Id
FROM Weather w1, Weather w2
WHERE DATEDIFF(w2.RecordDate, w1.RecordDate) = 1 and w2.Temperature > w1.Temperature

### another option ### 
WHERE w1.RecordDate = DATE_SUB(w2.RecordDate, INTERVAL 1 DAY) and  w2.Temperature > w1.Temperature
```
```sql
WHERE RecordDate BETWEEN '2015-01-01' and '2015-01-10'

```
## `DISTINCT` result

About `DISTINCT` in MySQL, it filters out the replicated selected rows, not just one variale.

Table: `friend_request`

| sender_id | send_to_id |request_date|
|-----------|:----------:|-----------:|
| 1         | 2          | 2016_06-01 |
| 1         | 3          | 2016_06-01 |
| 1         | 4          | 2016_06-01 |
| 2         | 3          | 2016_06-02 |
| 3         | 4          | 2016-06-09 |

Table: `request_accepted`

| requester_id | accepter_id |accept_date |
|-------------:|------------:|-----------:|
| 1            | 2           | 2016_06-03 |
| 1            | 3           | 2016-06-08 |
| 2            | 3           | 2016-06-08 |
| 3            | 4           | 2016-06-09 |
| 3            | 4           | 2016-06-10 |


``` sql
SELECT DISTINCT requester_id, accepter_id
FROM request_accepted
```
The result is

    [[1, 2], [1, 3], [2, 3], [3, 4]]

**Q:** Write a query to find the overall acceptance rate of requests rounded to 2 decimals, which is the number of acceptance divide the number of requests.

The highlight of this question is to use `IFNULL` and `ROUND`.

```sql
SELECT ROUND(IFNULL(COUNT(DISTINCT requester_id, accepter_id)/COUNT(DISTINCT sender_id, send_to_id), 0), 2) AS accept_rate
FROM friend_request, request_accepted
```

## `DELETE` deplicate rows

Table `Person`

| Id | Email            |
|---:|-----------------:|
| 1  | john@example.com |
| 2  | bob@example.com  |
| 3  | john@example.com |

```sql
DELETE FROM Person
WHERE Id NOT IN (
    SELECT id from (
        SELECT MIN(Id) as id
        FROM Person
        GROUP BY Email
    ) AS Minp
)
```

## `LIMIT` and `OFFSET`

Write a SQL query to get the second highest salary from the `Employee` table.


| Id | Salary |
|---:|-------:|
| 1  | 100    |
| 2  | 200    |
| 3  | 300    |

```sql
SELECT (
    SELECT DISTINCT Salary
    FROM Employee
    ORDER BY Salary DESC
    LIMIT 1 OFFSET 1
) AS SecondHighestSalary

```