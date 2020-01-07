---
title: Sort multiple variables
date: 2019-12-31 14:41
categories: [Technical Tools, Python programming]
tags: [Python]
seo:
  date_modified: 2019-12-31 15:40:53 -0800
---

Usually, we are proficient at sorting the data frame/table by one variable. But there are cases that we need a second variable to break the ties. In this post, I will summarize how to do this in Python, R, and SQL. 

The question considered here is a table of temperature per day. We are asked to sort the temperature in descending order and while there is a lie, sort the data in ascending order.

| record_id| temperature|date       |
|---------:|-----------:|:----------|
|         1|          32|1987-01-01 |
|         2|          33|1987-01-02 |
|         3|          33|1987-01-03 |
|         4|          29|1987-01-04 |
|         5|          32|1987-01-05 |
|         6|          40|1987-01-06 |

## Python tuple

```python
import pandas as pd

rawdata = pd.read_csv('/Users/lucyliu/Desktop/smallData.csv')
rawdata['date'] = pd.to_datetime(rawdata.date)
dataTuple = list(rawdata.itertuples(index = False, name = None))

sorted(dataTuple, key=lambda x: (-x[1], x[2]))
```
    [(6, 40, Timestamp('1987-01-06 00:00:00')),
     (2, 33, Timestamp('1987-01-02 00:00:00')),
     (3, 33, Timestamp('1987-01-03 00:00:00')),
     (1, 32, Timestamp('1987-01-01 00:00:00')),
     (5, 32, Timestamp('1987-01-05 00:00:00')),
     (4, 29, Timestamp('1987-01-04 00:00:00'))]
    
In the `sorted` function, we used anonymous `lambda` function to select the variables, temperature and date. Also, there is a trick that adding `-` is equivalent to sorting in descending order.

## Python Pandas

`Pandas` is a standard package to do data wrangling. The `sort_values` method in pandas is efficient and easy to use.

```python
from tabulate import tabulate

sortedData = rawdata.sort_values(by=['temperature', 'date'], ascending = [False, True])
print(tabulate(sortedData, tablefmt="pipe", headers="keys", showindex = False))
```

|   record_id |   temperature | date                |
|------------:|--------------:|:--------------------|
|           6 |            40 | 1987-01-06 00:00:00 |
|           2 |            33 | 1987-01-02 00:00:00 |
|           3 |            33 | 1987-01-03 00:00:00 |
|           1 |            32 | 1987-01-01 00:00:00 |
|           5 |            32 | 1987-01-05 00:00:00 |
|           4 |            29 | 1987-01-04 00:00:00 |

## SQL

In SQL, we usually sort the table at the end of the command using `ORDER BY`.

```sql
ORDER BY temperature DESC, date ASC
```

## R programming

In R, the library `dplyr` has similar functions as in Pandas.

```r
library(dplyr)
smalldata %>% 
        arrange(desc(temperature), date) %>%
        kable(format = 'markdown')
```

| record_id| temperature|date       |
|---------:|-----------:|:----------|
|         6|          40|1987-01-06 |
|         2|          33|1987-01-02 |
|         3|          33|1987-01-03 |
|         1|          32|1987-01-01 |
|         5|          32|1987-01-05 |
|         4|          29|1987-01-04 |