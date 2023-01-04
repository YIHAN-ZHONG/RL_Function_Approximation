# RL_Function_Approximation
We want to extend the methods we have seen for discrete state spaces to 
to problems with arbitrarily large state spaces. In the case of a video game for example, the number of possible images is much larer than the number of atoms in the universe. We can't expect to find an optimal policy or the optimal value function. 
Our goal instead is to find a good approximate solution using limited computational resources. 

The machine learning methods used to mimic the values functions and its predictions are supervised learning methods. When
the outputs are numbers, the process is often called function approximation.  

In this notebook, we are going to implement four approximation algorithm. All of the predictions methods are
described as updates to an estimated value function that shift its value at particular towrds an update target.

- Monte carlo update for prediction is: $St\rightarrow Gt$, where $Gt$ is an unbiased estimate of
$v_\pi(St)$ 

- The linear semi-gradient TD(0) update for prediction is: $St\rightarrow R_{t+1}+\gamma\hat{v}(S_{t+1},wt)$

- The n-step-TD state value approximation is: $St\rightarrow G_{t:t+n}$

