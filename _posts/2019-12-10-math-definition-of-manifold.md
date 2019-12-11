---
title: Mathematical definition of Manifold
date: 2019-12-10 17:07
categories: [Manifold Learning, Mathematical essentials]
tags: [Manifold, Embedding]
---

The machine learning community has borrowed lots of terminologies from mathematics. To gain a deeper understanding of the feature learning/dimension reduction approaches, I make some notes of the rigorous definition in math. 
## Some terminologies in topology

An simple topology:
Let Z be the set\{1, 2, 3\}, and declare the open subsets to be \{1\}, \{1, 2\}, \{1, 2, 3\} and \\(\varnothing\\).

- For every neighborhood of 2 and 3, it includes 2. Therefore, \{2, 2, 2,...\} converges to both 2 and 3.
- The distinct points 1 and 2 do not have disjoint neighborhoods. 

1. **homeomorphism**: If \\(X\\) and \\(Y\\) are topological spaces, a homromorphism from \\(X\\) to \\(Y\\) is a bijective map \\(\varphi: X\rightarrow Y\\) such that  \\(\varphi\\) and \\(\varphi^{-1}\\) are continuous.
2. **Hausdorff Space**: If given any pair of distinct points \\(p_1, p_2\in X\\), there exist neighborhoods \\(U_1\\) of \\(p_1\\) and \\(U_2\\) of \\(p_2\\) with \\(U_1\cap U_2 = \varnothing\\).\\\
The problem with the above simple example is that there are too few open subsets, so neighborhoods do not have the similar and desired meaning as in metric space. The Hausdorff space requires that ''points can be separated by open sets''. As a result, if a sequence \\(\{p_i\}\\) converges to a limit \\(p\\), the limit is unique.   
3. **Countability Properties**: The Hausdorff properties ensures that a topological space has *enough* open subsets but we also want to restrict attention to spaces that do not have *too many* open subset. The right balance is struck by requiring the existence of a basis that is countable.

- **First countable**: \\(X\\) is first countable if there exists a countable neighborhood basis at each point. Since the sets of balls \\(B_r(p)\\) with rational \\(r\\) is a neighborhood basis at \\(p\\), every metric space is first countable.
- **Second countable**: A topological space is said to be second countable if it admits a countable basis for its topology. EG: taking the collection \\(\mathcal{B}\\) of subsets of Eculidean space to be 
\\[\mathcal{B} = \{B_r(x): r\text{ is rational and } x \text{ has rational coordinates}\}\\] the basis,  Eculidean space is second countable.

Here are some relationship between four different countability properties. First, every second countable space \\(X\\) is first countable, separable (has a countable dense subset), and Lindel\\(\ddot{o}\\)f (Every open cover of \\(X\\) has a countable subcover). Second,  every metric space is first countable, and  second countability, separability, and the Lindel\\(\ddot{o}\\)f property are all equivalent for metric spaces.


## Topological Manifolds

Suppose \\(M\\) is a topological space. We say that \\(M\\) is a **topological manifold of dimension n** or a **topological n-manifold** if it has the following properties:
- \\(M\\) is a Hausdorff space.
- \\(M\\) is second-countable.
- \\(M\\) is locally Euclidean of dimension n: each point of \\(M\\) has a neighborhood that is homeomorphic to an open subset of \\(\mathbb{R}^n\\).