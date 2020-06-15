---
title: Wasserstein Barycenter
date: 2020-04-30 11:00
categories: [Technical Tools, Statistics]
tags: [Optimal Transport, Computer Vision]
---

## Wasserstein Barycenter problem

The Wasserstein Barycenter problem focuses on solving a weighted mean of a collection probability distributions such that the weighted Wasserstein distance is minimized between the mean and the probability distribution in the collection. Following the sinkhorn algorithm, we consider the entropy regularized Wasserstein distance. The regularized optimization can be much faster than the orignal Wasserstein barycenter question.

## Mathematical formulation

For simplicity, we consider the situation that the ground metric is \\(L_2\\) norm in Eculidean space, i.e. the distance matrix $$\mathbf{C}_{ij} = \|\delta_i - \delta_j\|^2$$. Thus the regularized Wasserstein distance is 
\\[
W_{\epsilon,2}^2 = \min_{\mathbf{P}\in \mathbf{U}(\mathbf{a}, \mathbf{b})} \langle \mathbf{C}, \mathbf{P}\rangle - \varepsilon H(\mathbf{P}), \text{ where } - H(\mathbf{P}) =  \sum\_{i, j}\mathbf{P}\_{i,j}\log \mathbf{P}\_{i,j} - \mathbf{P}\_{i,j}.
\\]
Given a weigh vector \\( (\lambda_1, \ldots, \lambda_n) \\), the Wassertein Barycenter \\( \mathbf{a}\\) is defined through
\\[
\min_{\mathbf{P}, \mathbf{a}} \sum_{s} \lambda_s \left(\langle \mathbf{C}_s, \mathbf{P}_s \rangle - \varepsilon H(\mathbf{P}_s)\right) \text{ s.t. } \mathbf{P}_s\mathbb{1} = \mathbf{a}, \mathbf{P}^T_s\mathbb{1} = \mathbf{b}_s.
\\]
Note the \\(\mathbf{P}\\) is a collection of joint distribution \\(\mathbf{P}_1, \ldots,\mathbf{P}_S\\).

## Iterative projection algorithm

One can generalize the sinkhorn algorithm to the barycenter problem. We derive the algorithm by the dual formulation since the above convex optimization problem embraces strong duality, namely the dual problem has the same optimum with the primal problem.

Define the Language function as
\\[
\Lambda(\mathbf{P}, \mathbf{a}, \mathbf{f}, \mathbf{g}) = \sum_s \lambda_s \left(\langle \mathbf{C}_s, \mathbf{P}_s \rangle + \varepsilon \sum\_{i, j} \mathbf{P}^s\_{i, j} (\log \mathbf{P}\_{i,j} - 1) + \langle\mathbf{f}, \mathbf{a} - \mathbf{P}_s \mathbb{1}\rangle + \langle \mathbf{g}, \mathbf{b}_s - \mathbf{P}^T_s\mathbf{1}\rangle\right)
\\]

Since strong duality holds, we have
\\[\begin{align}
g(\mathbf{f}, \mathbf{g}) &= \inf_{\mathbf{P}, \mathbf{a}} \Lambda(\mathbf{P}, \mathbf{a}, \mathbf{f}, \mathbf{g}) \leq p* \\\\\
d* &= \max\_{\mathbf{f},\mathbf{g}} g(\mathbf{f}, \mathbf{g}) = p*.
\end{align}
\\]
The $$\max\min$$ question is 
\\[  \max\_{\mathbf{f},\mathbf{g}}\sum_s \lambda_s \left(\langle \mathbf{g}, \mathbf{b}\_s \rangle +\min\_{\mathbf{P}\_s} \langle \mathbf{C}\_s, \mathbf{P}\_s \rangle + \varepsilon \sum\_{i, j} \mathbf{P}^s\_{i, j} (\log \mathbf{P}^s\_{i,j} - 1) - \langle\mathbf{f}\_s,  \mathbf{P}\_s \mathbb{1}\rangle - \langle \mathbf{g},  \mathbf{P}^T_s\mathbb{1}\rangle\right) +\min_{\mathbf{a}}\langle \sum_s \lambda_s\mathbf{f}_s, \mathbf{a} \rangle.
\\]

First, the explicit minimization on \\(\mathbf{a}\\) gives the constraint \\(\sum_s\lambda_s\mathbf{f}_s = 0\\), otherwise, the function has maximum \\(-\infty\\) and thus \\(\mathbf{f}\\) is not the maximizer.

Define
\\[
\Lambda_s (\mathbf{P}\_s) = \langle \mathbf{C}\_s, \mathbf{P}\_s \rangle + \varepsilon \sum\_{i, j} \mathbf{P}^s\_{i, j} (\log \mathbf{P}^s\_{i,j} - 1) - \langle\mathbf{f}\_s,  \mathbf{P}\_s \mathbb{1}\rangle - \langle \mathbf{g},  \mathbf{P}^T_s\mathbb{1}\rangle.
\\]
The first order condition
\\[
\frac{\partial \Lambda_s (\mathbf{P}_s)}{\partial \mathbf{P}^s\_{i, j} } = \mathbf{C}^s\_{i, j} + \varepsilon \log\mathbf{P}^s\_{i,j} -\mathbf{f}_i - \mathbf{g}_j = 0
\\]
gives us 
\\[
\mathbf{P}^\*_s = \text{diag}(e^{\mathbf{f}_s/\varepsilon}) * e^{\frac{-\mathbf{C}_s}{\varepsilon}} * \text{diag}(e^{\mathbf{g}_s/\varepsilon}) = \text{diag}(\mathbf{u}_s) * \mathbf{K}_s * \text{diag}(\mathbf{v}_s).
\\]
\\[
\Lambda_s (\mathbf{P}^{\ast}\_s) =\varepsilon \langle \mathbf{P}^{\ast}\_s, -\log \mathbf{P}^{\ast}\_s \rangle + \varepsilon \langle \mathbf{P}^{\ast}\_s, \log \mathbf{P}^{\ast}\_s \rangle - \varepsilon \sum\_{i,j} \mathbf{P}^{s\ast}\_{i,j} = - \varepsilon\mathbf{P}^{s\ast}\_{i,j} .
\\]
Thus, the optimization question is transformed into 
\\[\max\_{\mathbf{f},\mathbf{g}}\sum_s \lambda_s \left(\langle \mathbf{g}\_s, \mathbf{b}\_s \rangle - \varepsilon \sum\_{i,j} e^{\frac{\mathbf{f}^s_i}{\varepsilon}} \mathbf{K}^s\_{i,j}
e^{\frac{\mathbf{g}^s_j}{\varepsilon}}\right), \text{ s.t. }\sum_s\lambda_s\mathbf{f}_s = 0.
\\]
With fixed \\(\mathbf{f}_i\\), the first order condition w.r.t \\(\mathbf{g}^s_j\\) results in
\\[
\mathbf{b}^s_j - \sum_i e^{\frac{\mathbf{f}^s_i}{\varepsilon}} \mathbf{K}^s\_{i,j}
e^{\frac{\mathbf{g}^s_j}{\varepsilon}} = 0 \rightarrow \text{diag}(\mathbf{v}_s) * \mathbf{K}^T_s * \text{diag}(\mathbf{u}_s) = \mathbf{b}_s.
\\]
With fixed \\(\mathbf{g}_s\\), we construct a Language function to solve \\(\mathbf{f}_s\\).
\\[
L(\mathbf{f}_s, l) = \sum_s \lambda_s \varepsilon \sum\_{i,j} e^{\frac{\mathbf{f}^s_i}{\varepsilon}} \mathbf{K}^s\_{i,j}
e^{\frac{\mathbf{g}^s_j}{\varepsilon}} - l\sum_s\lambda_s\mathbf{f}_s.
\\]
The first order condition gives us
\\[
\frac{\partial L}{\partial \mathbf{f}^s_i} = \lambda_s\sum\_{i,j} e^{\frac{\mathbf{f}^s_i}{\varepsilon}} \mathbf{K}^s\_{i,j}
e^{\frac{\mathbf{g}^s_j}{\varepsilon}} - l_i\lambda_s\mathbf{f}^s_i = 0 \rightarrow \mathbf{u}^s = e^{\frac{\mathbf{f}_s}{\varepsilon}} = \frac{l}{\mathbf{K}_s \mathbf{v}_s}.
\\]
By the constraint \\(\sum_s\lambda_s\mathbf{f}_s = 0\\) and \\(\sum_s \lambda_s = 1\\),
\\[
\sum_s \lambda_s\log \frac{l}{\mathbf{K}_s \mathbf{v}_s} = \log \frac{l}{\Pi_s(\mathbf{K}_s \mathbf{v}_s)^{\lambda_s}} = 0 \rightarrow \mathbf{a} = \Pi_s(\mathbf{K}_s \mathbf{v}_s)^{\lambda_s}.
\\]
---------------

**function wasserstein-barycenter**

**Input:** \\(\mathbf{C}_s, \mathbf{b}_s, \varepsilon\\)

- Initialization: \\(\mathbf{u}_s = \mathbf{v}_s = \mathbb{1}, \mathbf{K}_s = e^{\frac{-\mathbf{C}_s}{\varepsilon}}\\).
- Main loop: 
> While $$L$$ changes do:
> for $$s \in (1, \ldots, S)$$\\[\begin{align}
\mathbf{v}^{(i+1)}_s &= \frac{\mathbf{b}_s}{\mathbf{K}^T_s*\mathbf{u}^{(i)}_s} \\\\\
\mathbf{a}^{(i+1)} &= \Pi_s(\mathbf{K}_s \mathbf{v}^{(i+1)}_s)^{\lambda_s} \\\\\
\mathbf{u}^{(i+1)}_s &= \frac{\mathbf{a}^{(i+1)}}{\mathbf{K}_s * \mathbf{v}^{(i+1)}_s}
\end{align}\\]

**Return:** $$\mathbf{a} = \mathbf{P}_s\mathbb{1}, \mathbf{P}_s = \mathbf{u}_s*\mathbf{K}_s*\mathbf{v}_s$$, \\(L = \sum_s\text{trace}(\mathbf{C}^T_s\mathbf{P}_s)\\)

---------------

## References
1. [Solomon, J., De Goes, F., Peyré, G., Cuturi, M., Butscher, A., Nguyen, A. & Guibas, L. (2015).
    Convolutional wasserstein distances: Efficient optimal transportation on geometric domains](https://dl.acm.org/doi/10.1145/2766963)
2. [Duality(Optimization)](https://en.wikipedia.org/wiki/Duality_(optimization))
3. [Gabriel Peyré and Marco Cuturi, Computational Optimal Transport (2019)](https://optimaltransport.github.io/)