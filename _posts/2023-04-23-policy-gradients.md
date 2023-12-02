---
title: Policy gradients demystified
subtitle: This article explains about Policy gradient methods and REINFORCE algorithm (Monte Carlo Policy Gradient) which is the simplest policy gradient method.
layout: default
date: 2023-04-23
keywords: Reinforcement Learning, Policy optimization
permalink: /policy_gradients
includelink: true
---

In the Model-free RL setting, the agent needs to learn either a policy
(Policy optimization), value function (Q-Learning) or both to attain its
goal. While Q-Learning is so famous (popularized by DQN), Policy
optimization is just starting to get notified through the work of
Reinforcement Learning from Human Feedback (RLHF).

## Policy Optimization
---
Policy Optimization techniques allows the agent to directly optimize
(thus improving) the policy, whereas Q-Learning first optimizes the
value function and then extracts the policy from it. So, Q-Learning
indirectly optimizes the policy. There are two ways by which one can
optimize the policy. First, Derivative free Optimization, in which one
modifies the policy's parameters multiple times, then measures the
performance of those modified policies and chooses a better performing
policy. Second, Policy gradient methods, in which there is no need to
modify the policy's parameters in order to find the direction of policy
update. Instead, it estimates the gradient for updating the policy
(gradient provides info about the direction). Since Policy gradient
methods are proven successful and useful, I'll talk about it below.

# Policy gradient methods
---
Before diving into Policy gradient methods, let's reflect on what we are
optimizing for (aka objective function) in the RL setting. The goal of
the agent in an environment is to maximize the expected sum of
discounted rewards (or expected return).

$$\begin{gathered}
  J(\theta) = \mathop{\mathrm{\mathbb{E}}}_{\tau \sim \pi_{\theta}}[R(\tau)] \\
  \text{where $R(\tau) = \sum_{t=0}^{T} \gamma^{t} r_{t}$ }
\end{gathered}$$

Policy gradient methods directly optimizes the policy ($\pi_{\theta}$)
by using the gradient of the policy's performance with respect to
policy's parameters ($\theta$). The policy's performance is measured by
the objective function on the trajectories collected by the current set
of policy's parameters.

$$\theta = \theta + \alpha \nabla_{\theta} J(\theta)$$

Now, you might be thinking on your mind that it's very simple. But it's
not. The main problem is how to compute the gradient? The Policy
gradient Theorem which can help us answer that question.

## Policy Gradient Theorem
---
The Policy Gradient Theorem simplifies the gradient expression
$\nabla_{\theta} J(\theta)$.

$$\begin{aligned}
\nabla_{\theta} J(\theta) & = \nabla_{\theta} \mathop{\mathrm{\mathbb{E}}}_{\tau \sim \pi_{\theta}}[R(\tau)] \\
 & = \nabla_{\theta} \sum_{\tau} p(\tau | \theta) R(\tau) &&\text{[Definition of Expected value]} \\
 & = \sum_{\tau} \nabla_{\theta} p(\tau | \theta) R(\tau) \\
 & = \sum_{\tau} \frac{p(\tau | \theta)}{p(\tau | \theta)} \nabla_{\theta} p(\tau | \theta) R(\tau) &&\text{[Multiply and divide the eqn by $p(\tau | \theta)$]} \\
 & = \sum_{\tau} p(\tau | \theta) \frac{\nabla_{\theta} p(\tau | \theta)}{p(\tau | \theta)} R(\tau) \\
 & = \sum_{\tau} p(\tau | \theta) \nabla_{\theta} \log p(\tau | \theta) R(\tau) &&\text{[$\nabla_{\theta} \log x = \frac{1}{x} \nabla_{\theta} x$]} \\
 & = \mathop{\mathrm{\mathbb{E}}}_{\tau \sim \pi_{\theta}}[\nabla_{\theta} \log p(\tau | \theta) R(\tau)] 
\end{aligned}$$

Now, we've arrived at the simplified version of
$\nabla_{\theta} J(\theta)$. Hey wait, how to calculate the probability
of a trajectory given policy's parameters $p(\tau | \theta)$.

## Probability of a Trajectory
---
Trajectory contains all the interactions between an agent and the
environment in a single episode. Though we'd fix a policy, and let it
interact with the environment for some episodes, the trajectories
collected would be different due to stochasticity of the policy and the
randomness in the environment dynamics. That's the exact reason we care
about the expected return calculated over the trajectories collected by
the policy.

Let's consider the following trajectory of an episode of $T$ time steps
from an agent's interactions,
$$\tau = (s_{0}, a_{0}, r_{0}, s_{1}, a_{1}, r_{1}, ... ,s_{T-1}, a_{T-1}, r_{T-1}, s_{T})$$

So, a trajectory is nothing but a combined occurrences of multiple
states and actions (rewards aren't included since they're usually tied
with the occurrence of states). So the probability of a trajectory can
be calculated as the joint probability of occurrence of multiple states
and actions.

$$p(\tau|\theta) = p(s_{0}) \prod_{t=0}^{T} \pi_{\theta}(a_{t}|s_{t}) \cdot p(s_{t+1}|s_{t},a_{t})$$

## Grad-Log-Probability of a Trajectory
---
First we need to calculate the log-probability of a trajectory
$\log p(\tau|\theta)$,

$$\begin{aligned}
\log p(\tau|\theta) & = \log \biggl( p(s_{0}) \prod_{t=0}^{T} \pi_{\theta}(a_{t}|s_{t}) \cdot p(s_{t+1}|s_{t},a_{t}) \biggl) \\
& = \log p(s_{0}) + \log \prod_{t=0}^{T} \pi_{\theta}(a_{t}|s_{t}) \cdot p(s_{t+1}|s_{t},a_{t}) &&\text{[Log of a Product = Sum of Logs]} \\
& = \log p(s_{0}) + \sum_{t=0}^{T}  \log [ \pi_{\theta}(a_{t}|s_{t}) \cdot p(s_{t+1}|s_{t},a_{t}) ] \\
& = \log p(s_{0}) + \sum_{t=0}^{T}  \log \pi_{\theta}(a_{t}|s_{t}) + \log p(s_{t+1}|s_{t},a_{t}) \\
\end{aligned}$$

Now we can calculate the gradient of the log-probability of a trajectory 
$\nabla_{\theta} \log p(\tau | \theta)$,

$$\nabla_{\theta} \log p(\tau|\theta) = \nabla_{\theta} \log p(s_{0}) + \sum_{t=0}^{T} \biggl( \nabla_{\theta} \log \pi_{\theta}(a_{t}|s_{t}) + \nabla_{\theta} \log p(s_{t+1}|s_{t},a_{t}) \biggl)$$

Neither the initial state distribution $s_{0}$ nor the state transition
function $p(s_{t+1}|s_{t},a_{t})$ depends upon the policy's parameters
$\theta$, so their gradients are zero. So the above equation becomes,

$$\nabla_{\theta} \log p(\tau|\theta) =  \sum_{t=0}^{T}\biggl(\nabla_{\theta} \log \pi_{\theta}(a_{t}|s_{t})\biggl)$$

## Putting it all together
---
We can plug $\nabla_{\theta} \log p(\tau|\theta)$ into our simplified
policy gradient expression to get the following,

$$\nabla_{\theta} J(\theta) = \mathop{\mathrm{\mathbb{E}}}_{\tau \sim \pi_{\theta}} \biggr[ \sum_{t=0}^{T} \nabla_{\theta} \log \pi_{\theta}(a_{t}|s_{t}) R(\tau) \biggr]$$

So the gradient is just probability of action times the return.

*Note*: Computing the expectation over many trajectories sampled from a
policy is costly. Instead one can estimate the gradient by using few
trajectories or even a single trajectory. This is why these algorithms
are called Monte Carlo policy gradients.

# REINFORCE algorithm
---
REINFORCE is the simplest Policy gradient method. It is also known as
Monte Carlo policy gradient. It makes use of the above simplified policy
gradient expression to compute the gradient.

There are different versions of this algorithm, which differs on
computing the gradient. In this section, I'll describe the REINFORCE
algorithm based on the *Reward to go policy gradient* version and
provide you with a pseudo code which you can base your implementation
on.

First let's take a look at our simplified PG expression,

$$\nabla_{\theta} J(\theta) = \mathop{\mathrm{\mathbb{E}}}_{\tau \sim \pi_{\theta}} \biggr[ \sum_{t=0}^{T} \nabla_{\theta} \log \pi_{\theta}(a_{t}|s_{t}) R(\tau) \biggr]$$

In the above equation, the log-probability of an action at each
time-step is scaled by the Return. This doesn't make any sense. Why?
Because, if we'd make use of the above expression, the agents won't
learn which actions in the trajectory have contributed more to the
return (since it treats all the actions in a trajectory the same), so it
won't learn what are the good actions. This blindly increases the
probabilities of all the actions in a trajectory if their return is
high. But we want to increase the probabilities of the actions which
contributed more to the return, since those are the actions that we want
out agents to choose in the next time, if it saw that particular state
again.

Instead we can use the following expression which allows the agent to
reinforce actions based on it's contribution to the Return.

$$\begin{gathered}
  \nabla_{\theta} J(\theta) = \mathop{\mathrm{\mathbb{E}}}_{\tau \sim \pi_{\theta}} \biggr[ \sum_{t=0}^{T} \nabla_{\theta} \log \pi_{\theta}(a_{t}|s_{t}) R_{t} \biggr] \\ \\
  \text{where $R_{t}$ is Return at time-step $t$}
\end{gathered}$$

## Pseudo code for REINFORCE
---
<div class='figure'>
    <img src="/assets/reinforce_pseudocode.png"
         style="width: 100%; display: block; margin: 0 auto;"/>
</div>

I would highly encourage you to implement this algorithm for yourself to
reinforce your knowledge.

*Note*: [Here](https://github.com/BalajiAI/Minimal-DeepRL/blob/main/src/reinforce.py) is my PyTorch implementation of REINFORCE algorithm for your reference.

## References
---
[1] [HuggingFace DeepRL course-Policy Gradient with PyTorch](https://huggingface.co/learn/deep-rl-course/unit4/introduction?fw=pt)

[2] [Spinning up in DeepRL-Intro to Policy Optimization](https://spinningup.openai.com/en/latest/spinningup/rl_intro3.html)

[3] [Lilianweng's blog-Policy gradient algorithms](https://lilianweng.github.io/posts/2018-04-08-policy-gradient/)

[4] [Going deeper into RL fundaments of policy gradients](https://danieltakeshi.github.io/2017/03/28/going-deeper-into-reinforcement-learning-fundamentals-of-policy-gradients/)

[5] [Gibberblot notes-Policy gradients](https://gibberblot.github.io/rl-notes/single-agent/policy-gradients.html)

[6] [John Lambert's blog-Understanding Policy gradients](https://johnwlambert.github.io/policy-gradients/)

[7] [Reinforcement Learning: An Introduction-Ch7:Policy gradient methods](http://incompleteideas.net/book/the-book-2nd.html)
