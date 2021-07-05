import torch
import torch.nn as nn
import torch.nn.functional as F


class QNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed, fc_units=[64, 64]):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc_units (list of int): List of number of nodes in hidden layers
        """
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc_layers = nn.ModuleList()  # pytorch correctly registers layers in this ModuleList
        for i in range(len(fc_units)):
            if i == 0:
                source_size = state_size
            else:
                source_size = fc_units[i - 1]
            self.fc_layers.append(nn.Linear(source_size, fc_units[i]))

        # last layer
        self.fc_layers.append(nn.Linear(fc_units[-1], action_size))

    def forward(self, state):
        """Build a network that maps state -> action values."""
        x = F.relu(self.fc_layers[0](state))
        for layer in self.fc_layers[1:-1]:
            x = F.relu(layer(x))
        return self.fc_layers[-1](x)