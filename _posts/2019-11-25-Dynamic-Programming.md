---
title: Dynamic Programming
date: 2019-11-25 23:37
categories: [Computer Science, Algorithm]
tags: [Dynamic Programming]
---

I am following the free open courses, *Algorithms: Design and Analysis*, on [Stanford Lagnuita](https://lagunita.stanford.edu/courses) to learn Basic CS algorithms. While I am capable of understanding the Part1 fast and smoothly, I paused and repeated the video frequently when studying the dynamic programming in Part2. One could claim that there is nothing mysterious because the main tool used in the theoretical analysis is merely **Induction**, which we learned in high school. However, to find the suitable subproblem leading to fast running time, it is nontrivial. I feel that mathematicians and  computer scienties are clever in different way. The mathematicans can tackle questions systematiclly  with much longer logical chains, like proving one theorem using hundreds pages. On the contrary, these efficient algorithms do not have a deep logical argument but are really tricky.

The four examples discussed in the Professor Tim Roughgarden's lecture video are here.
- Weighted independent sets/ Professional house robber question
- Knapsack problem
- Needleman-Wunsch sequence alignment
- Optimal binary search trees


##  Weighted independent sets

This is an easy question on LeetCode:

You are a professional robber planning to rob houses along a street. Each house has a certain amount of money stashed, the only constraint stopping you from robbing each of them is that adjacent houses have security system connected and it will automatically contact the police if two adjacent houses were broken into on the same night.

Given a list of non-negative integers representing the amount of money of each house, determine the maximum amount of money you can rob tonight without alerting the police.

**a) Problem:**
- Input:  a sequence of vertices, with positive weight, \\(w_i, i = 1,\ldots, n\\). 
- Output: an independent set,  avoiding consecutive pairs of vertices, whose sum of vertex weights is the largest.

**b) Subproblem formulation:**

Let \\(G_i\\) = first \\(i\\)  vertices of \\(G = \{1, \ldots, n\}\\) and \\(A[i] = \\) value of max-weight independent set of \\(G_i\\).

- Initialization: \\(A[0] = 0, A[1] = w_1\\)
- Main loop: 
> For \\(i = 2, 3, \ldots, n:\\)
     \\[A[i] = \max \\{ A[i-1], A[i-2] + w_i\\}\\]
- Return: \\(A[-1]\\), the last element of A

**c) Python code:**
```python
def weightedIndependentSet(arr):
    if len(arr) == 0: return 0
    elif len(arr) <= 2: return max(arr)
    else:
        A = [arr[0], max(arr[0:2])]
        for i in range(2, len(arr)):
            A.append(max(A[i-1], A[i-2] + arr[i]))
    return A[-1]
```


## Knapsack problem


**a) Problem:**
- Input: \\(n\\) items, each coming with a value \\(v_i > 0\\) and a size \\(w_i \in Z_+\\). A capacity, \\(W \in Z_+\\).
- Output: the maximum value of selected items, with total size at most \\(W\\).

The first thought would be, well, I just sort the unit value of each item, and select items with the largest  value/size_unit to fill in the knapsack. However, since both the size of each item and capacity are integers, this native method is intractable.

**b) Subproblem formulation:**

In order to find the subproblem and the recurrence, we analyze the original question backwards.

Let \\(V_{i, x}\\) be the value of the best solution on that: \\
1) using only the first \\(i\\) items;\\
2) has total size \\(\leq x\\).

If we add \\(i\\)th item into the list, based on the optimal solution with only \\(i-1\\) items, we march on in two ways:
\\[V_{i, x} = \left\{ \begin{array}{lr}
V_{i-1, x} & \text{case 1: ith item exculded} \\\
V_{i-1, x - w_i} + v_i & \text{case 2: ith item inculded}
\end{array}\right. \\]

Therefore, the possible prefixes are item \\(\{1, 2, \ldots, i\}, i\leq n\\). In addition, to form the recursion, value of \\(V_{i-1, x - w_i}\\) is needed. So we also solve the question with all possible capacities \\(x\in \{0, 1, 2, \ldots, W\}\\).

- Initialization: \\(A[0, x] = 0\\) for \\(x = \{1, \ldots, W\}\\).
- Main loop: 
> For \\(i = 1, 2, \ldots, n\\):
>> For \\(x = 0, 1, \ldots, W\\):
>> \\[A[i, x] = \max\\{A[i-1, x], A[i-1, x-w_i] + v_i\\}\\]
- Return \\(A[n, W]\\)

**c) Python code:**