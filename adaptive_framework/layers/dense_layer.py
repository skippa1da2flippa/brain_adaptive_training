import torch.nn as nn
import torch as to

class DenseLayer(nn.Linear):
    def __init__(
            self,
            in_features: int,
            out_features: int,
            eps: float = 1e-5,
            bias: bool = True,
            device: str = "cuda"
    ) -> None:
        super().__init__(
            in_features=in_features,
            out_features=out_features,
            bias=bias, device=device
        )

        self.adaptive_mask_sum: to.Tensor = nn.Parameter(
            data=to.zeros(out_features, in_features, device=device),
            requires_grad=True
        )
        self.adaptive_mask_prod: to.Tensor = nn.Parameter(
            data=to.ones(out_features, in_features, device=device),
            requires_grad=False
        )

        self.eps: nn.Parameter = nn.Parameter(data=to.tensor(eps), requires_grad=True)

        if not bias:
            self.bias = to.zeros(self.out_features, device=device, requires_grad=False)

    def forward(self, x: to.Tensor) -> to.Tensor:
        new_weights: to.Tensor = self.weight * self.adaptive_mask_prod + self.adaptive_mask_sum
        new_out: to.Tensor = x @ new_weights.T + self.bias
        self.update_masks(neurons_out=new_out)

        return new_out

    def update_masks(self, neurons_out: to.Tensor) -> None:
        if to.is_grad_enabled():
            batch_averaged: to.Tensor = to.mean(input=neurons_out, dim=0)
            batch_mask: to
