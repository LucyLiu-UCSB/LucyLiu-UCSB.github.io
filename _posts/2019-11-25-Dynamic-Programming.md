---
title: Dynamic Programming
date: 2019-11-25 23:37
categories: [Computer Science, Algorithm]
tags: [Dynamic Programming]
seo:
  date_modified: 2019-11-26 14:40:37 -0800
---

I am following the free open courses, *Algorithms: Design and Analysis*, on [Stanford Lagnuita](https://lagunita.stanford.edu/courses) to learn Basic CS algorithms. While I am capable of understanding the Part1 fast and smoothly, I paused and repeated the video frequently when studying the dynamic programming in Part2. One could claim that there is nothing mysterious because the main tool used in the theoretical analysis is mere **Induction**, which we learned in high school. However, to find suitable subproblem leading to fast running time, it is nontrivial. I feel that mathematicians and computer scientists are clever in different ways. The mathematicians can tackle questions systematically with much longer logical chains, like proving one theorem using hundreds of pages. On the contrary, these efficient algorithms do not have a deep logical argument but are tricky.

The four examples discussed in Professor Tim Roughgarden's lecture video are here.
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

In order to find the subproblem and the recurrence, we analyze the original question backward.

Let \\(V_{i, x}\\) be the value of the best solution on that: \\
1) using only the first \\(i\\) items;\\
2) has total size \\(\leq x\\).

If we add \\(i\\)th item into the list, based on the optimal solution with only \\(i-1\\) items, we march on in two ways:
\\[V_{i, x} = \left\\{ \begin{array}{lr}
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

**c) An example:**

<img src="/assets/img/sample/knapsack.jpg" alt="knapsackEx" width="700" class="center"/>

**d) Python code:**
```python
import numpy as np
def knapsack1(capacity, weights, values):
    
    if len(values) == 0 or capacity <= 0: return 0
    
    else:
        arr = np.zeros(shape = (len(values) + 1, capacity + 1))
        for i in range(1, len(values) + 1):
            for x in range(1, capacity + 1):
                case2 = arr[i-1, x - weights[i-1]] + values[i-1] if x - weights[i-1] >= 0 else 0
                arr[i, x] = max(arr[i-1, x], case2)
    return arr[-1, -1]  
```


## Needleman-Wunsch sequence alignment

Give two strings \\(X = x_1, \ldots, x_m\\) and \\(Y = y_1, \ldots, y_n\\), find the best alignment in terms of some metric. For instance, we can define score function as \\(\alpha(\text{gap}, x_i/y_j) = \alpha_g\\), \\(\alpha(x_i, y_j\|x_i = x_i) = \alpha_m\\) and \\(\alpha(x_i, y_j\|x_i \neq x_i) = \alpha_{no}\\), and we prefer higher score.

**a) Problem:**
- Input: two sequences and the score function
- Output: the maximum score or the best alignment

**b) Subproblem formulation:**

There are three relevant possibilities for the contents of the final position of an optimal alignment, which corresponds to three different ways of proceeding. \\\ 
Let \\(X' = X-x_m\\) and \\(Y' = Y-y_n\\). Define \\(S_{i,j} = \\) score of optimal alignment of \\(X_i = \{1, \ldots, i\}\\) and \\(Y_j = \{1, \ldots, j\}\\).
- case1: \\(x_m, y_n\\) matched/mismatched ->  alignment of \\(X'\\) and \\(Y'\\) is optimal.
- case2: \\(x_m\\) matched with a gap -> alignment of \\(X'\\) and \\(Y\\) is optimal.
- case3: \\(y_n\\) matched with a gap -> alignment of \\(X \\) and \\(Y'\\) is optimal.

This is a 2D recurrence. 
- Initialization: \\(A[i, 0] = A[0,i ] = i\cdot \alpha_g\\) for \\(i\geq 0\\).
- Main loop: 
> For \\(i = 1, 2, \ldots, m\\):
>> For \\(j = 1, 2, \ldots, n\\):
>> \\[A[i, j] = \max\left\\{ \begin{array}{lr}
A[i-1, j-1] + \alpha_m/\alpha_{no} & \text{case1} \\\
A[i-1, j] +\alpha_g &  \text{case2}\\\
A[i, j-1] +\alpha_g &  \text{case3}
\end{array}\right. \\]
- Return \\(A[m, n]\\)

**c) An example:**
<img src="/assets/img/sample/sequenceAlign.jpg" alt="sequenceAlign" width="900" class="center"/>
**d) Python code:**
```python
import numpy as np
def NeedlemanWunsch(seq1, seq2, alpha_g, alpha_m, alpha_no):
    
    if len(seq1) == 0 and len(seq2) == 0: return 0
    
    else:
        arr = np.zeros(shape = (len(seq1) + 1, len(seq2) + 1))
        arr[:, 0] = np.arange(len(seq1) + 1)*alpha_g
        arr[0, :] = np.arange(len(seq2) + 1)*alpha_g
        for i in range(1, len(seq1) + 1):
            for j in range(1, len(seq2) + 1):
                case1 = alpha_m if seq1[i-1] == seq2[j-1] else alpha_no
                arr[i, j] = max(arr[i-1, j-1] + case1,
                               arr[i-1, j] + alpha_g,
                               arr[i, j-1] + alpha_g)
    return arr
```

## Optimal binary search trees

Given set of probabilities over the keys, find the search tree that miminizes the average search time.


**a) Problem:**
- Input: frequenties \\(p_1, \ldots, p_n\\) for items \\(1, \ldots, n\\) (assume items in sorted order, \\(1<2<, \ldots ,< n\\)).
- Output: Compute a valid search tree that minimizes the weighted (average) search time, \\(C(T)\\).
\\[C(T) = \sum_{\text{items } i} p_i \times \{\text{depth of i in } T + 1 \}\\]

**b) Subproblem formulation:**

For \\(1\leq i\leq j \leq n\\), let \\(C_{i,j} = \\) weighted search cost of an optimal BST for the item \\(\{i, i+1, \ldots, j-1, j\}\\). We solve smallest subproblems with fewest number \\(j - i + 1\\) of itmes first.

- Initialization: \\(A[i, 0] = A[0,i ] = i\cdot \alpha_g\\) for \\(i\geq 0\\).
- Main loop: 
> For \\(s = 0, 1, \ldots, n-1\\): *Note: s represents (j-i)*
>> For \\(j = 1, 1, \ldots, n\\): *Note: i + s = j* 
>> \\[A[i, i + s] = \min_{r = i}^{i + s} \left\\{ \sum_{k = i}^{i + s} + A[i, r-1] + A[r+1, i+s]\right\\{ \\]
- Return \\(A[m, n]\\)

**c) R code:**

```r
optimalBST <- function(wvec) {
    
        n = length(wvec)
        res = matrix(nrow = n, ncol = n)
    
        for (s in 0:n-1) {
                for (i in 1:n) {
                        sss = NULL
                        for (r in i:min(s+i, n)) {
                                fir = sum(wvec[i:min(s+i, n)])
                                sec = ifelse(i <= r-1, res[i, r-1], 0)
                                thi = ifelse(r + 1 <= min(i+s, n), res[r+1, min(i+s, n)], 0)
                                sss = c(sss, fir + sec + thi)
                        }
                        if (i+s <= 7) { res[i, i +s] = min(sss)}
                }
        }
        return(res[1, n])        
}
```