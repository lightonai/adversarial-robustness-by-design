import torch
import torch.nn as nn

try:
    from lightonml.projections.sklearn import OPUMap
    from lightonopu import OPU
    print('lightonml and lightonopu ara available')
except:
    pass


class _RealOpu_(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, opu, backward_weight):
        ctx.save_for_backward(input, backward_weight)
        # output = opu.transform(input)
        output = torch.empty((input.shape[0], opu.transform(input[0]).shape[0]))
        for i in range(input.shape[0]):
            output[i] = opu.transform(input[i])

        return output

    @staticmethod
    def backward(ctx, grad_output):

        input, backward_weight = ctx.saved_tensors
        grad_input = grad_forward_weight = grad_backward_weight = None

        if ctx.needs_input_grad[0]:
            grad_input = grad_output.mm(backward_weight)
        if ctx.needs_input_grad[1]:
            grad_forward_weight = grad_output.t().mm(input)

        return grad_input, grad_forward_weight, grad_backward_weight


realopu = _RealOpu_.apply


class RealOpu(nn.Module):
    def __init__(self, in_features, out_features, device):
        super(RealOpu, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.device = device
        self.opu = OPU(n_components=self.out_features)  # OPUMap(n_components=self.out_features)
        self.opu.fit1d(n_features=in_features, online=True)
        self.backward_weight = nn.Parameter(
            torch.randn((self.in_features, self.out_features)) / self.out_features ** 0.5, requires_grad=False)

    def forward(self, x):
        x = ((x + 1) / 2).int()
        x = realopu(x, self.opu, self.backward_weight) / 255.
        x = x - x.mean()
        x = x.to(self.device)
        return x


class _SimulatedOPU_(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, forward_weight, backward_weight):
        ctx.save_for_backward(input, forward_weight, backward_weight)
        output = (input @ forward_weight.t()) ** 2

        return output

    @staticmethod
    def backward(ctx, grad_output):

        input, forward_weight, backward_weight = ctx.saved_tensors
        grad_input = grad_forward_weight = grad_backward_weight = None

        if ctx.needs_input_grad[0]:
            grad_input = grad_output.mm(backward_weight)
        if ctx.needs_input_grad[1]:
            grad_forward_weight = grad_output.t().mm(input)

        return grad_input, grad_forward_weight, grad_backward_weight


simulatedopu = _SimulatedOPU_.apply


class SimulatedOpu(nn.Module):
    def __init__(self, in_features, out_features):
        super(SimulatedOpu, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.forward_weight = nn.Parameter(
            torch.randn((self.in_features, self.out_features)) / self.out_features ** 0.5, requires_grad=False)
        self.backward_weight = nn.Parameter(
            torch.randn((self.in_features, self.out_features)) / self.out_features ** 0.5, requires_grad=False)

    def forward(self, x):
        return simulatedopu(x, self.forward_weight.t(), self.backward_weight.t())


class SignBackthrough(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        output = torch.sign(input)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        # input, = ctx.saved_tensors
        grad_input = None
        if ctx.needs_input_grad[0]:
            # if identity grad_input = grad_output
            # if Tanh(beta * x): beta * (1 - tanh(beta * x) ** 2)
            grad_input = grad_output
        return grad_input


sign_backthrough = SignBackthrough.apply


class Sign(nn.Module):
    def __init__(self, sign_, sign_back=False):
        super(Sign, self).__init__()
        self.sign_ = sign_
        self.sign_back = sign_back

    def forward(self, x):
        if self.sign_:
            if self.sign_back:
                return sign_backthrough(x - x.mean())
            else:
                return torch.sign(x - x.mean())
        else:
            return x


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x
