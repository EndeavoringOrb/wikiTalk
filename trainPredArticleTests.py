import torch
from torch.autograd.functional import jacobian


def compute_jacobian_params2(output, input, model_zeros, device):
    grad_output = torch.eye(output.size(1), device=device).unsqueeze(dim=1)
    jacobian = torch.autograd.grad(
        outputs=output,
        inputs=input,
        grad_outputs=grad_output,
        retain_graph=True,
        create_graph=True,
        is_grads_batched=True,
        allow_unused=True,
    )
    jacs = []
    for j, jac in enumerate(jacobian):
        if jac is not None:
            jacs.append(jac.view(output.size(1), -1))
        else:
            jacs.append(model_zeros[j].repeat(output.size(1), 1))
    return torch.cat(jacs, dim=-1).detach()


def compute_jacobian2(output, input, device):
    grad_output = torch.eye(output.size(1), device=device).unsqueeze(dim=1)
    jacobian = torch.autograd.grad(
        outputs=output,
        inputs=input,
        grad_outputs=grad_output,
        retain_graph=True,
        create_graph=True,
        is_grads_batched=True,
        allow_unused=True,
    )
    return jacobian[0]