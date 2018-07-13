import torch

class MyReLU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return input.clamp(min=0)

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input < 0] = 0
        return grad_input

def sigmoid(x):
    """
    A numerically stable version of the logistic sigmoid function.
    """
    pos_mask = (x >= 0)
    neg_mask = (x < 0)
    z = torch.zeros_like(x)
    z[pos_mask] = torch.exp(-x[pos_mask])
    z[neg_mask] = torch.exp(x[neg_mask])
    top = torch.ones_like(x)
    top[neg_mask] = z[neg_mask]
    return top / (1 + z)

class MySigmoid(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        return sigmoid(input)

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        return (1 - sigmoid(grad_input)) * sigmoid(grad_input)

class MyNewActivation(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        sigmoid_output = sigmoid(input)
        ctx.save_for_backward(sigmoid_output, input)
        return sigmoid_output * input

    @staticmethod
    def backward(ctx, grad_output):
        sigmoid_output, input, = ctx.saved_tensors
        grad_input = grad_output.clone()

        grad_input_sigmoid = input * grad_input
        sigmoid_grad = (1 - sigmoid(grad_input_sigmoid)) * sigmoid(grad_input_sigmoid)

        input_grad = sigmoid_output * grad_input

        return sigmoid_grad + input_grad


dtype = torch.float
device = torch.device("cpu")

N, D_in, H, D_out = 64, 1000, 100, 10

x = torch.randn(N, D_in, device=device, dtype=dtype)
y = torch.randn(N, D_out, device=device, dtype=dtype)

w1 = torch.randn(D_in, H, device=device, dtype=dtype, requires_grad=True)
w2 = torch.randn(H, D_out, device=device, dtype=dtype, requires_grad=True)

learning_rate = 1e-6

for t in range(500):
    relu = MyReLU.apply
    y_pred = relu(x.mm(w1)).mm(w2)

    #my_sigmoid = MySigmoid.apply
    #y_pred = my_sigmoid(x.mm(w1)).mm(w2)

    #my_new_activation = MyNewActivation.apply
    #y_pred = my_new_activation(x.mm(w1)).mm(w2)

    loss = (y_pred - y).pow(2).sum()
    print(t, loss.item())

    loss.backward()

    with torch.no_grad():
        w1 -= learning_rate * w1.grad
        w2 -= learning_rate * w2.grad

        w1.grad.zero_()
        w2.grad.zero_()

