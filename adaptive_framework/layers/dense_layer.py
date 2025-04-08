from typing import Callable
import torch.nn as nn
import torch as to

class DenseLayer(nn.Linear):
    def __init__(
            self,
            in_features: int,
            out_features: int,
            eps: float = 0.5,
            bias: bool = True,
            device: str = "cuda",
            next_act: Callable[[to.Tensor], to.Tensor] = None,
    ) -> None:
        super().__init__(
            in_features=in_features,
            out_features=out_features,
            bias=bias, device=device
        )

        # TODO no inplace operation are allowed for this tensor thus it's better to initialize everything
        #  with Kaiming or something similar, and then use the prod_mask to zero everthing you need also for this mask
        self.adaptive_mask_sum: to.Tensor = nn.Parameter(
            data=to.zeros(in_features, out_features, device=device),
            requires_grad=True
        )
        self.adaptive_mask_prod: to.Tensor = nn.Parameter(
            data=to.ones(in_features, out_features, device=device),
            requires_grad=False
        )

        self.eps: nn.Parameter = nn.Parameter(data=to.tensor(eps), requires_grad=True)

        if not bias:
            self.bias = to.zeros(self.out_features, device=device, requires_grad=False)

        self.next_act: Callable[[to.Tensor], to.Tensor] = next_act
        self.old_active_neuron: to.Tensor = to.zeros(in_features, out_features, device=device, requires_grad=False)

    def forward(self, x: to.Tensor) -> to.Tensor:
        new_weights: to.Tensor = self.adaptive_mask_prod.T * (self.weight + to.sqrt(1 / self.eps) * self.adaptive_mask_sum.T)
        new_out: to.Tensor = x @ new_weights.T + self.bias
        self.update_masks(neurons_in=x, neurons_out=new_out)

        return new_out

    def update_masks(self, neurons_in: to.Tensor, neurons_out: to.Tensor) -> None:
        if to.is_grad_enabled():
            # TODO this thing with the assumption of having a 4 dimensional tensor otherwise just three
            batch_averaged_in: to.Tensor = to.mean(input=neurons_in, dim=[0, 1])  # vector of size in_features
            batch_averaged_out: to.Tensor = self.next_act(to.mean(input=neurons_out, dim=[0, 1])) # vector of size out_features

            raw_distance_score: to.Tensor = batch_averaged_in.reshape(-1, 1) - batch_averaged_out.reshape(1, -1)
            neurons_distance_mat: to.Tensor = to.abs(raw_distance_score) # matrix of size in_features x out_features

            self.old_active_neuron = self.adaptive_mask_prod.clone()
            batch_activation_mask: to.Tensor = neurons_distance_mat <= self.eps
            self.adaptive_mask_prod[batch_activation_mask] = 1
            self.adaptive_mask_prod[~batch_activation_mask] = 0