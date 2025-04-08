import torch.nn as nn
from torch import Tensor
from typing import Callable

"""
    Brain Adaptive Linear class represent a dynamic way of including weight creation and 
    weight removal throughout the training of a network. The assumption behind this technique is 
    to emulate the rewiring process of the brain when a new set of information is learnt. 
    The logic behind when two neurons should link or split apart depends on the neurons output itself, 
    if two neurons share similar activation, post non linearity, they should connect, if not already, 
    otherwise the edge should be removed. The threshold \epsilon representing the maximum distance
    within which two neurons should connect is directly learnt from the model by actively including 
    \epsilon within the forward pass of this layer. 
    BA_Linear has:
        a) two learnable parameters: 
            - Matrix W of size in_features x out_features
            - Real value \epsilon 
        b) one non learnable parameter: 
            - 0-1 Matrix M of size in_features x out_features
    
    The forward pass first zero's the non-existing connection for the weight matrix W by performing
    element-wise multiplication with the mask M. The result \hat{W} and the input matrix X are multiplied 
    and summed with, if existing, the bias. The result is then weighted with the threshold \epsilon, 
    and returned. The received matrix X, which represents the output (post non-linearity) of the 
    previous layer neurons, is used to compare the distance of each neuron, hence assuming that the number 
    of related neurons in the intput matrix X is exactly in_features and the number of neurons in the 
    new layer is out_features. The resulting comparison matrix N of size in_features x out_features
    where at position i,j represent the distance between the neuron i and the neuron j after the activation. 
    The logic behind the forward pass multiplication with the parameter \epsilon is that the stricter the neurons
    threshold the more relevant the already existing connection should be.
"""

class BA_Linear(nn.Module):
    def __init__(
        self, in_features: int, out_features: int,
        next_act: Callable[[Tensor], Tensor],
        start_active_neuron_ratio: float = 0.5,
        bias: bool = True,
        device: str = "cuda"
    ) -> None:

        super().__init__()


