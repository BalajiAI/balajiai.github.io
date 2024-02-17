---
title: High Variance in Policy gradients
subtitle: In this article, we'll briefly discuss about the problem of high variance in Policy gradients and techniques for variance reduction.
layout: default
date: 2024-02-17
keywords: Reinforcement Learning, Policy gradients, Variance
permalink: /high_variance_in_policy_gradients
includelink: true
---

We've talked about Policy gradients in our previous blog. Though vanilla policy gradient is theoretically simple and mathematically proven, it doesn't seem to work well in practice. The main reason is because of high variance which policy gradient exhibit. High variance comes from the rewards obtained in a trajectory. Due to inherent randomness present in both environment and policy action selection, the rewards obtained by a fixed policy can vary greatly.

Because of the high variance, we need more trajectories for an accurate estimation of the policy gradient.

$$
\begin{gathered}
  \nabla_{\theta} J(\theta) =  \sum_{i=0}^{N} \sum_{t=0}^{T} \nabla_{\theta} \log \pi_{\theta}(a_{t}|s_{t}) R_{t} \\
  \text{where $N$ = No.of.trajectories}
\end{gathered}
$$

---
<div class='figure'>
    <img src="/assets/variance_visualization.png"
         style="width: 60%; display: block; margin: 0 auto;"/>
    <div class='caption'>
        <span class='caption-label'>Figure 1.</span> Consider two Random variables $X$ and $Y$ with same expectation value. 
        But $Var(Y) > Var(X)$. So samples drawn from $Y$ will be much far away from the expectation value than the ones drawn from $X$. 
        In that sense, the difference b/w the sample mean of $Y$ and expectation value will be large. 
        In order to drown out the variance for accurate estimation of sample mean, we've to use large number of samples. 
        Note: Number of samples and variance affects the calculation of sample mean.
    </div>         
</div>
---

But collecting large number of trajectories just for a single policy update is infeasible. So researchers have proposed techniques for reducing the variance in policy gradients in the past decades. One common theme across these variance reduction techniques is, they always try to keep the gradient estimate unbiased (equal to its original value).

It's because, we can solve the variance problem by using large number of samples to estimate the gradient. But with bias, though we've infinite number of samples, our policy might converge to a local optimum or not converge at all. That's why RL researchers are so cautious about it.


## Baselines
---
One of the well known techniques to reduce variance is to introduce baseline into our policy gradient expression. The idea of baseline comes from the control variates method, which is a popular variance reduction technique used in Monte Carlo methods. So First, we'll talk about what control variates are and then we can derive baseline from it.

### Control Variates
Let's consider a random variable $X$ which has an expectation value of $\mathop{\mathrm{\mathbb{E}}}[X]$ and a high variance of $Var[X]$. Since $X$ has a large variance, we require large number of samples to accurately compute its mean. So what control variate method proposes to solve this problem is, to construct a new random variable $X^\*$ which has the same expectation value of $X$ ($\mathop{\mathrm{\mathbb{E}}}[X^\*] = \mathop{\mathrm{\mathbb{E}}}[X]$), but lesser variance ($Var[X^\*] < Var[X]$). So instead of computing sample mean of $X$ (requires large no.of.samples), it's just enough to compute the sample mean of $X^\*$ (requires less no.of.samples). $X^\*$ is constructed by the following means,
$$\begin{gather*}
 X^* = X - c (Y - \mu_Y)
\end{gather*}$$
Where $c$ is a constant and $Y$ is an another random variable which correlates with $X$. We'll prove that indeed $\mathop{\mathrm{\mathbb{E}}}[X^\*] = \mathop{\mathrm{\mathbb{E}}}[X]$ and $Var[X^\*] \leq Var[X]$ in the following.

$$\begin{align*}
 \mathop{\mathrm{\mathbb{E}}}[X^*] &= \mathop{\mathrm{\mathbb{E}}}[X - c (Y - \mu_Y)] \\
 &=\mathop{\mathrm{\mathbb{E}}}[X] - c \mathop{\mathrm{\mathbb{E}}}[(Y - \mu_Y)] \\
 &= \mathop{\mathrm{\mathbb{E}}}[X]
\end{align*}$$

$$\begin{align*}
 Var[X^*] &= \mathop{\mathrm{\mathbb{E}}}[X^{*2}] - \mathop{\mathrm{\mathbb{E}}}[X^*]^2 \\
 &= \mathop{\mathrm{\mathbb{E}}}[(X - c (Y - \mu_Y))^2] - \mathop{\mathrm{\mathbb{E}}}[X]^2 \\
 &= \mathop{\mathrm{\mathbb{E}}}[X^2] + \mathop{\mathrm{\mathbb{E}}}[c^2 (Y - \mu_Y)^2] - \mathop{\mathrm{\mathbb{E}}}[2X.c(Y - \mu_Y)] - \mathop{\mathrm{\mathbb{E}}}[X]^2 \\ 
 &= Var[X] + c^2 \mathop{\mathrm{\mathbb{E}}}[ (Y - \mu_Y)^2] - 2c\mathop{\mathrm{\mathbb{E}}}[X.(Y - \mu_Y)] \\  
 &= Var[X] + c^2 Var[Y] - 2cCovar[X,Y] \\  
\end{align*}$$

For the proof to continue, we've to find the value for $c$. Since we want $Var[X^\*]$ to be minimum, we take the derivative with respect to $c$ and set to zero, to arrive at the following:

$$\begin{gather*}
    c = \frac{Covar[X,Y]}{Var[Y]}
\end{gather*}$$

After substituting the value of $c$ in $Var[X^\*]$ we'll get,

$$\begin{align*}
    Var[X^*] &= Var[X] - \frac{Covar[X,Y]^2}{Var[Y]} \\
     &= Var[X] \biggl(1 - \frac{Covar[X,Y]^2}{Var[X]Var[Y]} \biggl) \\
    &= Var[X] (1 - \rho_{X,Y}^2)
\end{align*}$$

Where $\rho_{X,Y}$ is the correlation coefficient of $X$ and $Y$. Since $\rho_{X,Y}$ always takes a value in between 0 and 1 ($0 \leq \rho_{X,Y} \leq 1 $), we can show that, 

$$\begin{gather*}
 Var[X^*] \leq Var[X]   
\end{gather*}$$

There is a special case for control variate equation, where the constant c takes the value of 1 and $\mu_Y$'s value equals to 0,

$$\begin{gather*}
 X^* = X - Y
\end{gather*}$$

Our Baseline technique makes use of the above special case of control variate equation to construct a new policy gradient expression which has lesser variance,

$$\begin{align*}
  \nabla_{\theta} J(\theta) &= \mathop{\mathrm{\mathbb{E}}}_{\tau \sim \pi_{\theta}} \biggr[ \sum_{t=0}^{T} \nabla_{\theta} \log \pi_{\theta}(a_{t}|s_{t}) R_{t} - \sum_{t=0}^{T} \nabla_{\theta} \log \pi_{\theta}(a_{t}|s_{t}) b \biggr] \\
  &= \mathop{\mathrm{\mathbb{E}}}_{\tau \sim \pi_{\theta}} \biggr[ \sum_{t=0}^{T} \nabla_{\theta} \log \pi_{\theta}(a_{t}|s_{t}) (R_{t} - b) \biggr]
\end{align*}$$

Now, we're going to prove that the expectation of the newly introduced term into our policy gradient expression is zero. To simplify the proof, we're going to express the policy gradient expression interms of trajectories,

$$\begin{gather*}
  \nabla_{\theta} J(\theta)
  = \mathop{\mathrm{\mathbb{E}}}_{\tau \sim \pi_{\theta}} \bigr[ \nabla_{\theta} \log \pi_{\theta}(\tau) (R(\tau) - b) \bigr]
\end{gather*}$$

$$\begin{align*}
    \mathop{\mathrm{\mathbb{E}}}_{\tau \sim \pi_{\theta}} \bigr[ \nabla_{\theta} \log \pi_{\theta}(\tau) b \bigr] 	&= \int \pi_\theta(\tau) \nabla_\theta\log \pi_\theta(\tau)bd\tau \\
	&= \int {\pi_\theta(\tau)} \frac{\nabla_\theta\pi_\theta(\tau)}{\pi_\theta(\tau)}bd\tau \\
	&= b\nabla_\theta \int \pi_\theta(\tau)d\tau \\
	&= b\nabla_\theta .1 \hspace{2em}\because \int f(x)dx=1\\
	&= b \times 0 \\
	&= 0 \label{BL_bias_2}
\end{align*}$$

We've already proven that $Var[X^\*] \leq Var[X]$. The same result applies here, since we derived baseline from control variate equation. So, there is no need to reprove the Variance reduction proof again.

### Baseline expression
We've incorporated baseline into our policy gradient expression and proved that it doesn't add any bias to the policy gradient. What's left is the expression for baseline which achieves minimal variance. To find out the expression, we'll take the derivative of $Var[\nabla_{\theta} J(\theta)]$ with respect to $b$ and set to zero, and solve for $b$.

$$\begin{gather*}
    Var[X] = \mathop{\mathrm{\mathbb{E}}}[X^2] - \mathop{\mathrm{\mathbb{E}}}[X]^2 \\
    Var[\nabla_{\theta} J(\theta)] = \mathop{\mathrm{\mathbb{E}}}_{\tau \sim \pi_\theta}[(\nabla_{\theta} \log \pi_{\theta}(\tau) (R(\tau) - b))^2] - \mathop{\mathrm{\mathbb{E}}}_{\tau \sim \pi_\theta}[\nabla_{\theta} \log \pi_{\theta}(\tau) (R(\tau) - b)]^2 
\end{gather*}$$

To simplify the notation, we can write:

$$\begin{gather*}
    f(\tau) = \nabla_{\theta} \log \pi_{\theta}(\tau) \\
    Var[\nabla_{\theta} J(\theta)] = \mathop{\mathrm{\mathbb{E}}}_{\tau \sim \pi_\theta}[(f(\tau) (R(\tau) - b))^2] - \mathop{\mathrm{\mathbb{E}}}_{\tau \sim \pi_\theta}[f(\tau) (R(\tau) - b)]^2    
\end{gather*}$$

To further simplify our proof, we can safely ignore the second term in the above variance equation and can take the derivative wrt $b$,

$$\begin{align*}
    \frac{d}{db} Var[\nabla_{\theta} J(\theta)] &= \frac{d}{db} \mathop{\mathrm{\mathbb{E}}}_{\tau \sim \pi_\theta}[(f(\tau) (R(\tau) - b))^2] \\
&= \frac{d}{db} \mathop{\mathrm{\mathbb{E}}}_{\tau \sim \pi_\theta}[f(\tau)^2 (R(\tau)^2 + b^2 - 2R(\tau)b)] \\
&= \frac{d}{db} \Bigl( \mathop{\mathrm{\mathbb{E}}}_{\tau \sim \pi_\theta}[f(\tau)^2 R(\tau)^2] + b^2\mathop{\mathrm{\mathbb{E}}}_{\tau \sim \pi_\theta}[f(\tau)^2] - 2b\mathop{\mathrm{\mathbb{E}}}_{\tau \sim \pi_\theta}[f(\tau)^2R(\tau)] \Bigl) \\
&= 2b\mathop{\mathrm{\mathbb{E}}}_{\tau \sim \pi_\theta}[f(\tau)^2] - 2\mathop{\mathrm{\mathbb{E}}}_{\tau \sim \pi_\theta}[f(\tau)^2R(\tau)]
\end{align*}$$

Set the above result to zero and solve for $b$,

$$\begin{gather*}
    2b\mathop{\mathrm{\mathbb{E}}}_{\tau \sim \pi_\theta}[f(\tau)^2] - 2\mathop{\mathrm{\mathbb{E}}}_{\tau \sim \pi_\theta}[f(\tau)^2R(\tau)] = 0 \\
    b = \frac{\mathop{\mathrm{\mathbb{E}}}_{\tau \sim \pi_\theta}[f(\tau)^2R(\tau)]}{\mathop{\mathrm{\mathbb{E}}}_{\tau \sim \pi_\theta}[f(\tau)^2]} \\
    b = \frac{\mathop{\mathrm{\mathbb{E}}}_{\tau \sim \pi_\theta}[(\nabla_{\theta} \log \pi_{\theta}(\tau))^2R(\tau)]}{\mathop{\mathrm{\mathbb{E}}}_{\tau \sim \pi_\theta}[(\nabla_{\theta} \log \pi_{\theta}(\tau))^2]}    
\end{gather*}$$

So the optimal baseline expression is expected return over trajectories weighted by gradient magnitudes. But often in practice, people just use the average return as a baseline,

$$\begin{gather*}
    b = \mathop{\mathrm{\mathbb{E}}}_{\tau \sim \pi_\theta} [R(\tau)]
\end{gather*}$$

Another commonly used baseline expression is the state-value $v[s]$. Whereas Average return is a constant value which don't vary, state-value varies depending upon the state the agent is in. The following is the generic equation for state-dependent baselines like state-value, 

$$\begin{gather*}
  \nabla_{\theta} J(\theta)
  = \mathop{\mathrm{\mathbb{E}}}_{\tau \sim \pi_{\theta}} \biggr[ \sum_{t=0}^{T} \nabla_{\theta} \log \pi_{\theta}(a_{t}|s_{t}) (R_{t} - b(s_t)) \biggr]
\end{gather*}$$

### Intuition behind Baseline
In vanilla policy gradient, the probability of a trajectory $\pi_{\theta}(\tau)$ is incremented on par with the return $R(\tau)$ during the policy update. In policy gradient with baseline, the probability is incremented only if $R(\tau) - b$ is positive and decremented if it is negative.
In other words, a trajectory's probability is pushed up only when that particular trajectory's return is greater than the average return ($$R(\tau) > \mathop{\mathrm{\mathbb{E}}}_{\tau \sim \pi_\theta} [R(\tau)]$$).

### Neuroscience behind Baseline
Let's consider that you're in a room, where you've an access to a button, which when pressed will give you a 5 dollar bill. At the first press, you'll get an increase in dopamine level (it also means that you'll experience pleasure). During the 50th press, you'll still get an increase in dopamine level, but not the same increase that you've got in the first press. Even worse, during the 200th press, you won't get any increase in dopamine level at all, so you won't experience any joy or pleasure. It's because at each press, your baseline gets updated slowly and in the 200th press your baseline value gets updated completely such that it entirely cancels out the reward that you'd received.

In RL terms, let's say the reward that one gets in each press is equivalent to the amount that one receives. So for a 5 dollar bill, the reward is 5 points.
In our setup (environment), there is only a single timestep in each episode, so there is no need for the concept of Return.
The most important thing here is how the agent perceives the reward.
The agent doesn't perceive the reward $r$ as it is. Instead it perceives the reward as the relative difference to the baseline ($r - b$).
And our baseline $b$ is initially set to zero.
So at the first press, the reward which the agent perceives is $5-0 = 5$. In the 50th press, the baseline gets updated to $2.5$, so the perceived reward is $5-2.5 = 2.5$.
In the 200th press, the baseline gets updated to $5$, so the perceived reward becomes $5-5 = 0$.

But you can still feel pleasure, only if you get a reward which is greater than the baseline value. For example, let's say you'd received a 20 dollar bill instead of a 5 dollar bill, then the perceived reward is $20 - 5 = 15$.
This is the sole reason behind why drug addicts tend to increase their drug dosage in order for them to experience the pleasure.

## Actor Critic
---
After Baselines, the next class of methods which are widely used for variance reduction is Actor Critic methods. As the name suggests, these class of methods has two components, Actor (aka policy) and Critic (aka value function), which are parameterized and learnable. Here the Critic assists in optimizing the Actor. In exact words, instead of using the empirical returns computed from the collected trajectories, the policy uses approximate returns predicted by the value function for its optimization. This reduces the variance and also introduces (some!) bias into our gradient estimates.

$$\begin{gather*}
  \nabla_{\theta} J(\theta) = \mathop{\mathrm{\mathbb{E}}}_{\tau \sim \pi_{\theta}} \biggr[ \sum_{t=0}^{T} \nabla_{\theta} \log \underbrace{\pi_{\theta}(a_{t}|s_{t})}_{Actor} \underbrace{Q(s_{t}, a_{t})}_{Critic} \biggr]
\end{gather*}$$

Apart from variance reduction, another nice thing about actor-critic methods is we don't have to wait for an episode to get completed before updating the policy. Although it's rarely done in practice.

### Advantage Actor Critic (A2C)

The best-performing variant in the actor-critic class of methods is Advantage Actor Critic (A2C). In A2C algorithm, Advantage function $A(s_{t}, a_{t})$ is the critic. The equation that we got is equivalent to the policy gradient with baseline equation, since $A(s_{t}, a_{t}) \approx R_{t} - b(s_{t})$.

$$\begin{gather*}
  \nabla_{\theta} J(\theta) = \mathop{\mathrm{\mathbb{E}}}_{\tau \sim \pi_{\theta}} \biggr[ \sum_{t=0}^{T} \nabla_{\theta} \log \pi_{\theta}(a_{t}|s_{t}) A(s_{t}, a_{t})\biggr]
\end{gather*}$$

The next question which arises in our mind is, how to estimate the advantage of taking a specific action $a_{t}$ in a specific state $s_{t}$. One can estimate the advantage function $A(s_t,a_t)$ by the following ways,


1) $Q(s_{t}, a_{t}) - V(s_{t})$ : Advantage function decomposition

2) $r_{t} + \gamma r_{t+1} + \gamma^2 r_{t+1}+...+ \gamma^n r_{t+n} + \gamma^{n+1} V(s_{t+n+1}) - V(s_t)$ : TD(n) Error

3) $\sum_{l=0}^{\infty} (\gamma \lambda)^{l} \delta_{t+l}$, where $\delta_{t} = r_t + \gamma V(s_{t+1}) - V(s_t)$ : Generalized Advantage Estimation (GAE)

(1) method requires $Q$ & $V$ function to estimate the advantage of $a_t$ given $s_t$ whereas (2) method requires only the $V$-function. So in the second case, we only have to learn the $V$-function. But we've to choose the value of $n$ in the TD(n) carefully. Because, if we set the value too small, then the gradient exhibits low variance & high bias. Or if we set the value too large, then the gradient shows high variance & low bias. So we've to choose the value to something in between which exhibits a decent amount of bias & variance.

Generalized Advantage Estimation (GAE) improves over the TD(n) error method by overcoming the problem of having to explicitly choose the value for $n$. GAE estimates the advantage by taking an exponential weighted-average of individual advantages ($\hat{A}_t^{(1)}$, $\hat{A}_t^{(2)}$, ..., $\hat{A}_t^{(n)}$) calculated using TD errors (TD(1), TD(2), ..., TD(n)).

$$\begin{align*}
\hat{A}_t^{(1)} & := \delta_t = r_t + \gamma V(s_{t+1}) - V(s_t) \\
\hat{A}_t^{(2)} & := \delta_t + \gamma \delta_{t+1} = r_t + \gamma r_{t+1} + \gamma^2 V(s_{t+2}) - V(s_t)  \\
...\\
\hat{A}_t^{(n)} & := \sum_{l=0}^{n} \gamma ^{l} \delta_{t+l} = r_{t} + \gamma r_{t+1} + \gamma^2 r_{t+1}+...+ \gamma^{n-1} r_{t+n-1} + \gamma^{n} V(s_{t+n}) - V(s_t)
\end{align*}$$

If $n \rightarrow \infty$, then the advantage estimated using GAE becomes,

$$\begin{align*}
\hat{A}_t^{\text{GAE}(\gamma, \lambda)} & = (1-\lambda) (\hat{A}_t^{(1)} + \lambda \hat{A}_t^{(2)} + \lambda^2 \hat{A}_t^{(3)} + ...) \\  
& = (1-\lambda) (\delta_t + \lambda (\delta_t + \gamma \delta_{t+1}) + \lambda^2 (\delta_t + \gamma \delta_{t+1} + \gamma^2 \delta_{t+2} ) + ...) \\  
& = (1-\lambda) ( \delta_t (1+\lambda+\lambda^2+...) + \gamma \delta_{t+1} (\lambda+\lambda^2+\lambda^3+...) + \gamma^2 \delta_{t+2} (\lambda^2+\lambda^3+\lambda^3+...) + ...) \\
& = (1-\lambda) \biggl( \delta_t \biggl(\frac{1}{1-\lambda}\biggl) + \gamma \delta_{t+1} \biggl(\frac{\lambda}{1-\lambda}\biggl) + \gamma^2 \delta_{t+2} \biggl(\frac{\lambda^2}{1-\lambda}\biggl) +  ...) \\
& = \sum_{l=0}^{\infty} (\gamma \lambda)^{l} \delta_{t+l}
\end{align*}$$

There are two special cases for the above formula which are obtained by setting the $\lambda = 0$ and $\lambda = 1$,

$$\begin{align*}
\text{If }  \lambda=0,  \; \hat{A}_t^{\text{GAE}(\gamma,0)} & := \delta_t = r_t + \gamma V(s_{t+1}) - V(s_t) \\
\text{If }  \lambda=1,  \; \hat{A}_t^{\text{GAE}(\gamma,1)} & :=  \sum_{l=0}^{\infty} \gamma^{l} \delta_{t+l} = \sum_{l=0}^{\infty} \gamma^{l} r_{t+l} - V(s_t)
\end{align*}$$

Similar to the second method where $n$ determines the bias-variance tradeoff, In GAE, $\lambda$ determines it. Setting the value of $\lambda$ to a small value (say 0) in GAE makes the gradient estimates to have low variance & high bias. If we set the value too large (say 1), then the gradient exhibits (or should I say, advantage estimates) high variance & low bias.

## Conclusion
---
So, in this blog, we've seen how variance arises in Policy gradient methods and also discussed about the techniques for variance reduction such as Baselines & Actor Critic methods. While Baseline method tries to reduce the variance by following a common approach in Statistics called control variates, in the meanwhile, Actor critic methods takes a radical new approach to the variance reduction by making use of predictions from the value function. Although one can always make use of multiple trajectories to approximate the policy gradient inorder to factor out the variance :)

You can find the associated code for this blog [here](https://github.com/BalajiAI/High-Variance-in-Policy-gradients).

## References
---
[1] [HuggingFace DeepRL course-Variance Problem](https://huggingface.co/learn/deep-rl-course/unit6/variance-problem)

[2] [The Problem with Policy Gradient](https://mcneela.github.io/machine_learning/2019/06/03/The-Problem-With-Policy-Gradient.html)

[3] [Understanding Deep Learning - DeepRL chapter](https://udlbook.github.io/udlbook/)

[4] [Fundamentals of Policy Gradients](https://danieltakeshi.github.io/2017/03/28/going-deeper-into-reinforcement-learning-fundamentals-of-policy-gradients/)

[5] [Actor critic methods](https://mpatacchiola.github.io/blog/2017/02/11/dissecting-reinforcement-learning-4.html)

[6] [Asynchronous Methods for Deep Reinforcement Learning](https://arxiv.org/abs/1602.01783)

[7] [High-Dimensional Continuous Control Using Generalized Advantage Estimation](https://arxiv.org/abs/1506.02438)