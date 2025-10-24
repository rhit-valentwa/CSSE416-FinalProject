class DQN(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=256, seed=0):
        super(DQN, self).__init__()

        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, action_size)

    def forward(self, state):
        state = F.relu(self.fc1(state))
        return self.fc2(state)

if __name__ == "__main__":
    state_size = 12
    action_size = 3
    net = DQN(state_size, action_size)
    state = torch.randn(1, state_size)
    output = net(state)
    print(output)
