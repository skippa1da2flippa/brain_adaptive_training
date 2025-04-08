**Brain Adaptive** Linear class represent a dynamic way of including weight creation and 
weight removal throughout the training of a network. The assumption behind this technique is 
to emulate the rewiring process of the brain when a new set of information is learnt. 
The logic behind when two neurons should link or split apart depends on the neurons output itself, 
if two neurons share similar activation, post non-linearity, they should connect, if not already, 
otherwise the edge should be removed. The threshold $\epsilon$ representing the maximum distance
within which two neurons should connect is directly learnt from the model by actively including 
$\epsilon$  within the forward pass of this layer. 

$$
\begin{align}
    \text{Let } W \in \mathbb{R}^{\text{out_features} \times \text{in_features}}, \text{ } M \in \{0, 1\}^{\text{out_features} \times \text{in_features}}
\end{align}
$$
