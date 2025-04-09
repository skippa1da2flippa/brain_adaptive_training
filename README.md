##  Intro

**Brain Adaptive** Linear class represent a dynamic way of including weight creation and 
weight removal throughout the training of a network. The assumption behind this technique is 
to emulate the rewiring process of the brain when a new set of information is learnt. 
The logic behind when two neurons should link or split apart depends on the neurons output itself, 
if two neurons share similar activation, post non linearity, they should connect, if not already, 
otherwise the edge should be removed. The threshold $\epsilon$ representing the maximum distance
within which two neurons should connect is directly learnt from the model by actively including 
$\epsilon$  within the forward pass of this layer. 

$$
\begin{align}
\text{Let }& W \in \mathbb{R}^{\text{out_features} \times \text{in_features}}, \text{ } M \in \{0, 1\}^{\text{out_features}
\times \text{in_features}}, \text{ } \\
&X \in \mathbb{R}^{\text{batch_size} \times * \times \text{in_features}}, \text{ }
b \in \mathbb{R}^{\text{out_features}}, \\
&\phi: \mathbb{R}^{*} \rightarrow \mathbb{R}^{*}, \text{ }\epsilon \in \mathbb{R}
\end{align}
$$

Where the symbol $*$ represents all the $k$ "hidden" dimensions $\forall k \geq 0$ which might lay between the first and the 
last ones.

## Mask random initialization

The tensor $M$ is randomly initialized with either zero constraining each row to have at least a cell 
set to one. In other words each neuron in the newly create layer $h + 1$ has at least one connection 
with at least one neuron in the previous layer $h$.

## Forward pass 

The neurons dynamic adaptation is performed via element-wise multiplication between the learnable tensor
$W$ and the mask $M$. Due to the initial absence of isolated neurons, the bias tensor $b$ is
left out the mask product. 

$$
\text{BA}_{\text{linear}}(X) = X\left[\frac{1}{\sqrt{\epsilon}}(M \odot W)\right]^{T} + b
$$

The bias $b$ is kept out of the transformation process to still send a value, this aligns with the 
assumption that a bias parameter should always send information to the neurons which then result in a 
possible activation.

## Mask update

The distance between a pair of any two neurons $n_i, n_j$ is computed as absolute difference between 
the two neurons, even though distance itself might be misleading for some task in which the distance 
itself might not be indicative for detecting the similarity between two neurons. As usual the answer
lays in the empirical path conditioned by the chosen datasets.

$$
\begin{align}
\tilde{x} &= \text{AVG}\{X\}\big{|}_{[1, \dots, 1 + k]} \in \mathbb{R}^{\text{in_features}}\\
\tilde{m} &= \text{AVG}\{\phi[\text{BA}_{\text{linear}}(X)]\}\big{|}_{[1, \dots, 1 + k]} \in \mathbb{R}^{\text{out_features}} \\
\end{align}
$$

The neurons output in the $h$-th layer $X$ is averaged through all the dimensions beside the last one, resulting in a 
one dimensional tensor, same process is applied onto the neurons output of the $h+1$-th layer. The application
of the non-linearity $\phi$ is only needed if the **BA** layer is not set as first layer of the network.

$$
\begin{align}
\tilde{D} &= \begin{pmatrix}
|\tilde{x}_{1} - \tilde{m}_{1}| & \dots & |\tilde{x}_{1} - \tilde{m}_{\text{out_features}}| \\
\vdots & \ddots & \dots \\
|\tilde{x}_{\text{in_features}} - \tilde{m}_{1}| & \dots & |\tilde{x}_{\text{in_features}} - \tilde{m}_{\text{out_features}}|
\end{pmatrix} \in \mathbb{R}^{\text{in_features} \times \text{out_features}} \\ \\
m_{ij} &= \begin{cases}
1 & \text{If } \tilde{d}_{ij} \leq \epsilon \\
0 & \text{otherwise}
\end{cases}
\end{align}
$$



## Complete framework

On initialization the layer is randomly initialized as explained in `Mask Random Initialization`, then at each 
forward pass, if the gradient computation is enabled, before returning the forward pass value the update 
routine is run.

1. $\text{Initialize parameters: } W, M, b, \epsilon$
2. $\text{For any forward pass with input: } X$ 
   1. $out \leftarrow X\left[\frac{1}{\sqrt{\epsilon}}(M \odot W)\right]^{T} + b$
   2. $\text{If the gradient computation is allowed: }$
      1. $\text{Update mask}$
   3. $\text{Return } out$




