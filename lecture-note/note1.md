## Useful Identities for Computing Gradients

### 1

$$
\frac{\partial}{\partial \boldsymbol{X}} \boldsymbol{f}(\boldsymbol{X})^T = \left( \frac{\partial \boldsymbol{f} (\boldsymbol{X})}{\partial \boldsymbol{X}} \right)^T
$$

Proof:
$$
\begin{align*}
\frac{\partial}{\partial \boldsymbol{X}} \boldsymbol{f}(\boldsymbol{X})^T 
&= 
\frac{\partial}{\partial \boldsymbol{X}} 
\begin{bmatrix}
f_1(\boldsymbol{X}) \;
f_2(\boldsymbol{X}) \; 
\cdots
f_M(\boldsymbol{X}) 
\end{bmatrix} \\
&= 
\begin{bmatrix}
\frac{\partial}{\partial \boldsymbol{X}} f_1(\boldsymbol{X}) \;
\frac{\partial}{\partial \boldsymbol{X}} f_2(\boldsymbol{X}) \;
\cdots \;
\frac{\partial}{\partial \boldsymbol{X}} f_M(\boldsymbol{X}) \;
\end{bmatrix} \\
&= 
\left( \frac{\partial \boldsymbol{f} (\boldsymbol{X})}{\partial \boldsymbol{X}} \right)^T
\end{align*}
$$

### 2

$$
\frac{\partial}{\partial \boldsymbol{X}} \boldsymbol{f}(\boldsymbol{X})^{-1} = -\boldsymbol{f}(\boldsymbol{X})^{-1}\frac{\partial \boldsymbol{f} (\boldsymbol{X})} {\partial \boldsymbol{X}} \boldsymbol{f}(\boldsymbol{X})^{-1}
$$

$$
\begin{align*}
\frac{\partial \boldsymbol{f}(\boldsymbol{X})^{-1} \boldsymbol{f}(\boldsymbol{X})}{\partial \boldsymbol{X}} 
&= 
\boldsymbol{f}(\boldsymbol{X})^{-1} \frac{\partial \boldsymbol{f}(\boldsymbol{X})}{\partial \boldsymbol{X}} + \frac{\partial \boldsymbol{f}(\boldsymbol{X})^{-1} }{\partial \boldsymbol{X}} \boldsymbol{f}(\boldsymbol{X})
\\ &= \bold{0}
\end{align*}
$$



### 3

$$
\frac{\partial \boldsymbol{a}^T \boldsymbol{X}^{-1} \boldsymbol{b}}{\partial \boldsymbol{X}} = - (\boldsymbol{X}^{-1})^T \boldsymbol{a} \boldsymbol{b}^T (\boldsymbol{X}^{-1})^T
$$

Proof:
$$
\begin{align*}
\frac{\partial \boldsymbol{a}^T \boldsymbol{X}^{-1} \boldsymbol{b}}{\partial \boldsymbol{X}}
&= 
\frac{\partial \; \mathrm{tr}(\boldsymbol{a}^T \boldsymbol{X}^{-1} \boldsymbol{b})}{\partial \boldsymbol{X}}
\\ &= 
\frac{\partial \; \mathrm{tr}(\boldsymbol{X}^{-1} \boldsymbol{b} \boldsymbol{a}^T)}{\partial \boldsymbol{X}}
\end{align*}
$$

### 4

$$
\frac{\partial \boldsymbol{x}^\text{T} \boldsymbol{a}}{\partial \boldsymbol x} 
= \frac{\partial \boldsymbol{a}^\text{T} \boldsymbol{x}}{\partial \boldsymbol x} 
= \boldsymbol a^\text T
$$



### 5

$$
\frac{\partial \boldsymbol{a}^\text T \boldsymbol{X} \boldsymbol{b}}{\partial \boldsymbol{X}} = \boldsymbol{a} \boldsymbol{b}^\text T
$$

Proof:
$$
\begin{align*}
\frac{\partial \boldsymbol{a}^\text T \boldsymbol{X} \boldsymbol{b}}{\partial \boldsymbol{X}} 
&=
\frac{\partial \boldsymbol{a}^\text T }{\partial \boldsymbol{X}} \boldsymbol{X} \boldsymbol{b} +
\boldsymbol{a}^\text T\frac{\partial  \boldsymbol{X} }{\partial \boldsymbol{X}} \boldsymbol{b} + 
\boldsymbol{a}^\text T \boldsymbol{X}\frac{\partial  \boldsymbol{b}}{\partial \boldsymbol{X}} 
\\ &=
\boldsymbol{a} \boldsymbol{b}^\text T
\end{align*}
$$

### 6

$$
\frac{\partial \boldsymbol x^ \text T \boldsymbol A \boldsymbol x}{\partial \boldsymbol x} = \boldsymbol x^\text T (\boldsymbol A + \boldsymbol A^ \text T)
$$



### 7

$$
\frac{\partial}{\partial s} \left( (\mathbf{x} - A\mathbf{s})^\text{T} W (\mathbf{x} - A\mathbf{s}) \right) = -2(\mathbf{x} - A\mathbf{s})^\text{T} WA
$$

where $W$ is symmetric.

Proof:
$$
(\mathbf{x} - A\mathbf{s})^\text{T} W (\mathbf{x} - A\mathbf{s}) = \mathbf{x}^\text{T}W\mathbf{x} - \mathbf{x}^\text{T}WA\mathbf{s} - (A\mathbf{s})^\text{T}W\mathbf{x} + (A\mathbf{s})^\text{T}WA\mathbf{s}
$$










