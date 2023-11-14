## CS329 Machine Learning(H) Quiz 2

### Before Quiz

$$
\begin{align*}
-(a_0-1)\ln\pi &- (b_0-1)\ln(1-\pi) + &\frac 1 2(\mu_1-m_{10})^\text T \Sigma_{10}^{-1}(\mu_1-m_{10}) + &\frac 1 2(\mu_2-m_{20})^\text T &\Sigma_{20}^{-1}(\mu_2-m_{20})\\
&\text{prior}  &\text{likelihood 1} \quad\quad\quad&\text{likelihood 2}\\
\end{align*}
$$

---

### Question 1

#### Question 1.1

$p(x)=\pi\mathcal N(x\vert\mu_1,\Sigma_1)+(1-\pi)\mathcal N(x\vert\mu_2,\Sigma_2)$

Given $\mathbf x=[x_1,\dots,x_N]$, $\mathbf t=[t_1,\dots,t_N]$

What's the ML estimation of $\mu_1,\Sigma_1,\mu_2,\Sigma_2,\pi$?

#### Solution 1.1

$$
\begin{align*}
\pi_\text{ML} &= \frac 1 N \sum\limits_{n=1}^N t_n =\frac{N_1}{N}=\frac{N_1}{N_1+N_2}\\
\mu_{1\text{ML}} &= \frac 1 {N_1}\sum\limits_{n=1}^Nt_n x_n\\
\mu_{2\text{ML}} &= \frac 1 {N_2}\sum\limits_{n=1}^N(1-t_n) x_n\\
\Sigma_{1\text{ML}} &=\frac 1 {N_1}\sum\limits_{x_n\in\mathcal C_1} (x_n-\mu_1)(x_n-\mu_1)^\text T\\
\Sigma_{2\text{ML}} &=\frac 1 {N_2}\sum\limits_{x_n\in\mathcal C_2} (x_n-\mu_2)(x_n-\mu_2)^\text T
\end{align*}
$$



#### Question 1.2

$p(\pi)\sim beta(a_0,b_0), p(\mu_i) = \mathcal N(m_{i0},\Sigma_{i0}), i=1,2$

Given $\mathbf x=[x_1,\dots,x_N]$, $\mathbf t=[t_1,\dots,t_N]$

What's the MAP estimation of $\mu_1,\Sigma_1,\mu_2,\Sigma_2,\pi$?

#### Solution 1.2

$$
\begin{align*}
\pi_\text{MAP} &= \frac{N_1+a_{0}}{N+a_0+b_0}=\frac{N_1+a_0}{N_1+N_2+a_0+b_0}\\
{\Sigma_{1\text{MAP}}}^{-1} &={\Sigma_{1\text{ML}}}^{-1}+{\Sigma_{10}}^{-1}\\
{\Sigma_{2\text{MAP}}}^{-1} &={\Sigma_{2\text{ML}}}^{-1}+{\Sigma_{20}}^{-1}\\
{\Sigma_{1\text{MAP}}}^{-1}\mu_{1\text{MAP}} &= {\Sigma_{1\text{ML}}}^{-1}\mu_{1\text{ML}}+{\Sigma_{10}}^{-1}m_{10}\\
{\Sigma_{2\text{MAP}}}^{-1}\mu_{2\text{MAP}} &= {\Sigma_{2\text{ML}}}^{-1}\mu_{2\text{ML}}+{\Sigma_{20}}^{-1}m_{20}\\
\end{align*}
$$

#### Question 1.3

What's $p(\mathcal C_1\vert x)$ for ML and MAP models respectively?

#### Solution 1.3

$$
\begin{align*}
p_\text{ML}(\mathcal C_1\vert x)&=\frac{p_\text{ML}(x,\mathcal C_1)}{p_\text{ML}(x)}=\frac{\pi_\text{ML}\mathcal N(x\vert\mu_\text{1ML},\Sigma_\text{1ML})}{\pi_\text{ML}\mathcal N(x\vert\mu_\text{1ML},\Sigma_\text{1ML})+(1-\pi_\text{ML})\mathcal N(x\vert\mu_\text{2ML},\Sigma_\text{2ML})}\\
p_\text{MAP}(\mathcal C_1\vert x)&=\frac{p_\text{MAP}(x,\mathcal C_1)}{p_\text{MAP}(x)}=\frac{\pi_\text{MAP}\mathcal N(x\vert\mu_\text{1MAP},\Sigma_\text{1MAP})}{\pi_\text{MAP}\mathcal N(x\vert\mu_\text{1MAP},\Sigma_\text{1MAP})+(1-\pi_\text{MAP})\mathcal N(x\vert\mu_\text{2MAP},\Sigma_\text{2MAP})}
\end{align*}
$$

---

### Question 2

#### Question 2.1

$y=\sigma(w^\text T\phi)$

Given $\mathbf x=[x_1,\dots,x_N]$, $\mathbf t=[t_1,\dots,t_N]$

What's the ML estimation of the mean and variance of $w$?

#### Solution 2.1

By Gauss-Newton iteration: $w^\text{new}=w^\text{old}-H^{-1}\nabla E(w)$, we obtain $w_\text{ML}$,

where
$$
\begin{align*}
E(w) &= -\sum\limits_{n=1}^N  [t_n\ln y_n + (1-t_n)\ln(1-y_n)]\\
\nabla E(w) &= \sum\limits_{n=1}^N (y_n-t_n)\phi_n\\
H &= \nabla^2 E(w) = \sum\limits_{n=1}^N y_n(1-y_n)\phi_n\phi_n^\text T
\end{align*}
$$
Hence
$$
q(w) = \mathcal N(w\vert w_\text{ML}, H^{-1})
$$

#### Question 2.2

$y=\sigma(w^\text T\phi)$, $p(w)\sim\mathcal N(m_0,\Sigma_0)$

Given $\mathbf x=[x_1,\dots,x_N]$, $\mathbf t=[t_1,\dots,t_N]$

What's the MAP estimation of the mean and variance of $w$?

#### Solution 2.2

By Gauss-Newton iteration: $w^\text{new}=w^\text{old}-H^{-1}\nabla E(w)$, we obtain $w_\text{ML}$,

where
$$
\begin{align*}
E(w) &= \frac{1}{2} (w-m_0)^\text T\Sigma_0^{-1}(w-m_0)-\sum\limits_{n=1}^N  [t_n\ln y_n + (1-t_n)\ln(1-y_n)]\\
\nabla E(w) &=\Sigma_0^{-1}(w-m_0)+ \sum\limits_{n=1}^N (y_n-t_n)\phi_n\\
H &= \nabla^2 E(w) = \Sigma_0^{-1}+\sum\limits_{n=1}^N y_n(1-y_n)\phi_n\phi_n^\text T
\end{align*}
$$
Hence 
$$
q(w) = \mathcal N(w\vert w_\text{MAP}, H^{-1})
$$

#### Question 2.3

$p(t|y(w,x))$ for ML and MAP estimation, respectively?

#### Solution 2.3

