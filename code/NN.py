import torch 
class TwoLayerNet(torch.nn.Module):
    def __init__(self, N_sym, H1, H2, N_element, bias = True, scaling = None):
        """
        In the constructor we instantiate two nn.Linear modules and assign them as
        member variables.
        """
        super(TwoLayerNet, self).__init__()
        self.linear1 = torch.nn.Linear(N_sym, H1, bias = bias)
        self.Tanh1 = torch.nn.Tanh()
        self.linear2 = torch.nn.Linear(H1, H2, bias = bias)
        self.Tanh2 = torch.nn.Tanh()
        self.linear3 = torch.nn.Linear(H2, N_element)
        self.scaling = scaling


    def forward(self, x):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.
        """
        h1_tanh = self.Tanh1(self.linear1(x))
        h2_tanh = self.Tanh2(self.linear2(h1_tanh))
        y_pred = self.linear3(h2_tanh)
        return y_pred


class Linear(torch.nn.Module):
    def __init__(self, N_sym, N_element, bias = True, scaling = None):
        """
        In the constructor we instantiate two nn.Linear modules and assign them as
        member variables.
        """
        super(Linear, self).__init__()
        self.linear1 = torch.nn.Linear(N_sym, N_element, bias = bias)
        self.scaling = scaling


    def forward(self, x):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.
        """
        y_pred = self.linear1(x)
        return y_pred



