
import torch
import torch.optim

class CautiousSGD(torch.optim.Optimizer):
    """Custom SGD with momentum optimizer with https://arxiv.org/pdf/2411.16085"""

    def __init__(
        self,
        params,
        lr: float,
        momentum: float,
        weight_decay: float = 0.0,
    ):
        assert lr >= 0
        assert momentum >= 0 and momentum <= 1
        assert weight_decay >= 0
        defaults = dict(
            lr=lr,
            momentum=momentum,
            dampening=0,
            weight_decay=weight_decay,
            nesterov=False,
        )
        super().__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            weight_decay = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad.data

                if weight_decay != 0:
                    grad = grad.add(p.data, alpha=weight_decay)

                param_state = self.state[p]
                if "momentum_buffer" not in param_state:
                    param_state["momentum_buffer"] = torch.zeros_like(p.data)

                buf = param_state["momentum_buffer"]
                buf.mul_(momentum).add_(grad)

                mask = (buf * grad > 0).to(grad.dtype)
                epsilon = max(1.0 / torch.numel(mask), 0.01)
                mask_scaled = mask / (mask.mean() + epsilon)
                p.data.add_(buf * mask_scaled, alpha=-lr)

        return loss
