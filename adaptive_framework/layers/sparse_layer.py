import torch.nn as nn
import torch as to
from typing import Callable
from adaptive_framework.utility.sparse_mask_init import generate_sparse_mask_torch

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

class BA_Linear(nn.Linear):
    def __init__(
        self, in_features: int, out_features: int,
        next_act: Callable[[to.Tensor], to.Tensor],
        eps: float = 0.5,
        start_active_neuron_ratio: float = 0.5,
        bias: bool = True,
        device: str = "cuda"
    ) -> None:
        super().__init__(
            in_features=in_features,
            out_features=out_features,
            bias=bias, device=device
        )


        self.adaptive_mask_prod: to.Tensor = nn.Parameter(
            data=generate_sparse_mask_torch(
                in_features=out_features,
                out_features=in_features,
                density=start_active_neuron_ratio,
                device=device
            ),
            requires_grad=False
        )

        self.eps: nn.Parameter = nn.Parameter(data=to.tensor(eps), requires_grad=True)
        self.next_act: Callable[[to.Tensor], to.Tensor] = next_act

    def forward(self, x: to.Tensor) -> to.Tensor:
        new_weights: to.Tensor = self.adaptive_mask_prod * self.weight
        new_out: to.Tensor = to.sqrt(1 / self.eps) * (x @ new_weights.T + self.bias)
        self.update_masks(neurons_in=x, neurons_out=new_out)

        return new_out

    def update_masks(self, neurons_in: to.Tensor, neurons_out: to.Tensor) -> None:
        if to.is_grad_enabled():
            # TODO this thing with the assumption of having a 4 dimensional tensor otherwise just three
            batch_averaged_in: to.Tensor = to.mean(input=neurons_in, dim=[0, 1])  # vector of size in_features
            batch_averaged_out: to.Tensor = self.next_act(to.mean(input=neurons_out, dim=[0, 1])) # vector of size out_features

            raw_distance_score: to.Tensor = batch_averaged_in.reshape(-1, 1) - batch_averaged_out.reshape(1, -1)
            neurons_distance_mat: to.Tensor = to.abs(raw_distance_score) # matrix of size in_features x out_features

            batch_activation_mask: to.Tensor = neurons_distance_mat <= self.eps
            self.adaptive_mask_prod[batch_activation_mask] = 1
            self.adaptive_mask_prod[~batch_activation_mask] = 0


