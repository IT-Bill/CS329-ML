#### 1.1

$$
\begin{align*}
\pi_\text{ML} &= \frac 1 N \sum\limits_{n=1}^N t_n =\frac{N_1}{N}=\frac{N_1}{N_1+N_2}\\
\mu_{1\text{ML}} &= \frac 1 {N_1}\sum\limits_{n=1}^Nt_n x_n\\
\mu_{2\text{ML}} &= \frac 1 {N_2}\sum\limits_{n=1}^N(1-t_n) x_n\\
\Sigma_{i\text{ML}} &=\frac 1 {N_i}\sum\limits_{x_n\in\mathcal C_i} (x_n-\mu_i)(x_n-\mu_i)^\text T
\end{align*}
$$



#### 1.2

$$
\begin{align*}
\pi_\text{MAP} &= \frac{N_1+a_{0}}{N+a_0+b_0}=\frac{N_1+a_0}{N_1+N_2+a_0+b_0}\\
{\Sigma_{i\text{MAP}}}^{-1} &={\Sigma_{i\text{ML}}}^{-1}+{\Sigma_{i0}}^{-1}\\
{\Sigma_{i\text{MAP}}}^{-1}\mu_{i\text{MAP}} &= {\Sigma_{i\text{ML}}}^{-1}\mu_{i\text{ML}}+{\Sigma_{i0}}^{-1}m_{i0}\\
\end{align*}
$$

Hence,
$$
\begin{align*}
{\Sigma_{i\text{MAP}}} &= ({\Sigma_{i\text{ML}}}^{-1}+{\Sigma_{i0}}^{-1})^{-1}\\
\mu_{i\text{MAP}} &= ({\Sigma_{i\text{ML}}}^{-1}+{\Sigma_{i0}}^{-1})^{-1}({\Sigma_{i\text{ML}}}^{-1}\mu_{i\text{ML}}+{\Sigma_{i0}}^{-1}m_{i0})

\end{align*}
$$


#### 1.3

$$
\begin{align*}
p_\text{ML}(\mathcal C_1\vert x) &= \frac{\pi_\text{ML}\mathcal N(x\vert\mu_\text{1ML},\Sigma_\text{1ML})}{\pi_\text{ML}\mathcal N(x\vert\mu_\text{1ML},\Sigma_\text{1ML})+(1-\pi_\text{ML})\mathcal N(x\vert\mu_\text{2ML},\Sigma_\text{2ML})}\\
p_\text{MAP}(\mathcal C_1\vert x) &= \frac{\pi_\text{MAP}\mathcal N(x\vert\mu_\text{1MAP},\Sigma_\text{1MAP})}{\pi_\text{MAP}\mathcal N(x\vert\mu_\text{1MAP},\Sigma_\text{1MAP})+(1-\pi_\text{MAP})\mathcal N(x\vert\mu_\text{2MAP},\Sigma_\text{2MAP})}
\end{align*}
$$

---

#### 2.1

Use $w^\text{new}=w^\text{old}-H^{-1}\nabla E(w)$ to obtain $w_\text{ML}$,

where
$$
\begin{align*}
E(w) &= -\sum\limits_{n=1}^N  [t_n\ln y_n + (1-t_n)\ln(1-y_n)]\\
\nabla E(w) &= \sum\limits_{n=1}^N (y_n-t_n)\phi_n\\
H &= \nabla^2 E(w) = \sum\limits_{n=1}^N y_n(1-y_n)\phi_n\phi_n^\text T
\end{align*}
$$
Hence,
$$
q(w) = \mathcal N(w\vert w_\text{ML}, H^{-1})
$$

#### 2.2

Use $w^\text{new}=w^\text{old}-H^{-1}\nabla E(w)$ to obtain $w_\text{MAP}$,

where
$$
\begin{align*}
E(w) &= \frac{1}{2} (w-m_0)^\text T\Sigma_0^{-1}(w-m_0)-\sum\limits_{n=1}^N  [t_n\ln y_n + (1-t_n)\ln(1-y_n)]\\
\nabla E(w) &=\Sigma_0^{-1}(w-m_0)+ \sum\limits_{n=1}^N (y_n-t_n)\phi_n\\
H &= \nabla^2 E(w) = \Sigma_0^{-1}+\sum\limits_{n=1}^N y_n(1-y_n)\phi_n\phi_n^\text T
\end{align*}
$$
Hence,
$$
q(w) = \mathcal N(w\vert w_\text{MAP}, H^{-1})
$$

#### 2.3

For ML:
$$
p_{\text{ML}}(t|y(w,x)) = y^{t} \cdot (1-y)^{(1-t)}
$$
where
$$
y = \sigma(w_{\text{ML}}^T \phi(x))
$$
For MAP:
$$
p_{\text{MAP}}(t|y(w,x)) = y^{t} \cdot (1-y)^{(1-t)}
$$
where
$$
y = \sigma(w_{\text{MAP}}^T \phi(x))
$$