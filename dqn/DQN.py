import torch
import torch.nn as nn
import torch.nn.functional as F

class DQN(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=256, seed=0):
        super(DQN, self).__init__()

        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, action_size)

    def forward(self, state):
        x = self.fc1(state)
        x = F.relu(x)
        x = self.fc2(x)
        return x
    
    def select_action(state):
        

if __name__ == "__main__":
    state_size = 12
    action_size = 3
    net = DQN(state_size, action_size)
    state = torch.randn(1, state_size)
    output = net(state)
    print(output)
