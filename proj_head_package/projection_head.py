import torch.nn as nn

class ProjectionHead(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dims=[512], activation=nn.ReLU()):
        super(ProjectionHead, self).__init__()
        layers = []

        # Input to first hidden layer
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(activation)
            prev_dim = hidden_dim

        # Add final layer for the output projection
        layers.append(nn.Linear(prev_dim, output_dim))

        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)