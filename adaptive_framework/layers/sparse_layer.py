import torch.nn as nn
import torch as to
from typing import Callable
from adaptive_framework.utility.sparse_mask_init import generate_sparse_mask_torch


class BA_Linear(nn.Linear):
    """
        Implementation of a dynamic fully connected layer updating its topology throughout
        the training. The connections are generated (unmasked) or pruned (masked) according to
        the similarity between the neurons, such value is measured with a learnable parameter.

        Parameters
        ----------
        in_features : int
            Expected size of the last dimension input tensor.
        out_features : int
            Desired size of the last dimension output tensor.
        eps : float, optional (default=0.5)
            A learnable scalar parameter representing the maximal distance two neurons can have in order to
            be incident (connected).
        start_active_neuron_ratio : float, optional (default=0.5)
            Proportion of initially active (non-zero) connections in the sparse layer.
        bias : bool, optional (default=True)
            If set to False, the layer will not learn an additive bias.
        next_act : Callable[[torch.Tensor], torch.Tensor], optional
            A function representing the activation to be applied after the linear transformation. Defaults to identity.
        device : str, optional (default='cuda')
            The device on which the parameters and computations will be allocated.

        Attributes
        ----------
        adaptive_mask_prod : torch.nn.Parameter
            A fixed sparse mask (not trainable) used to control which weights are active.
        next_act : Callable
            The activation function to apply after the linear transformation in order to
            check the distance between two neurons activations.
        eps : torch.nn.Parameter
            A learnable scalar parameter initialized with `eps`.

        Notes
        -----
        The sparse mask is generated once during initialization using `generate_sparse_mask_torch`
        and update deterministically throughout the training given the parameter `eps`.
    """
    def __init__(
        self, in_features: int,
        out_features: int,
        eps: float = 0.5,
        start_active_neuron_ratio: float = 0.5,
        bias: bool = True,
        next_act: Callable[[to.Tensor], to.Tensor] = None,
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

        next_act = next_act if next_act is not None else lambda x: x
        self.next_act: Callable[[to.Tensor], to.Tensor] = next_act

        self.eps: nn.Parameter = nn.Parameter(data=to.tensor(eps), requires_grad=True)

    def forward(self, x: to.Tensor) -> to.Tensor:
        new_weights: to.Tensor = to.sqrt(1 / self.eps) * self.adaptive_mask_prod * self.weight
        new_out: to.Tensor =  x @ new_weights.T + self.bias
        self.update_masks(neurons_in=x, neurons_out=new_out)

        return new_out

    def update_masks(self, neurons_in: to.Tensor, neurons_out: to.Tensor) -> None:
        if to.is_grad_enabled():
            batch_averaged_in: to.Tensor = to.mean(
                input=neurons_in,
                dim=[x for x in range(neurons_in.dim() - 1)]
            )  # vector of size in_features

            batch_averaged_out: to.Tensor = to.mean(
                input=self.next_act(neurons_out),
                dim=[x for x in range(neurons_out.dim() - 1)]
            ) # vector of size out_features

            # TODO add a choosable distance metric like cosine sim also
            raw_distance_score: to.Tensor = batch_averaged_in.reshape(-1, 1) - batch_averaged_out.reshape(1, -1)
            neurons_distance_mat: to.Tensor = to.abs(raw_distance_score) # matrix of size in_features x out_features

            batch_activation_mask: to.Tensor = neurons_distance_mat <= self.eps
            self.adaptive_mask_prod[batch_activation_mask] = 1
            self.adaptive_mask_prod[~batch_activation_mask] = 0


