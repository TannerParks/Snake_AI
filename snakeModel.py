import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os


class Linear_QNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.linear1(x)) # Activation function
        x = self.linear2(x)
        return x

    def save(self, file_name="model.pth"):
        # model_path = "..model"
        model_path = "model"
        if not os.path.exists(model_path):  # If folder doesn't exist
            os.makedirs(model_path) # Make it

        file_name = os.path.join(model_path, file_name) # Make path for file
        torch.save(self.state_dict(), file_name)    # Save model to file by saving state dictionary


class QTrainer:
    def __init__(self, model, learning_rate, gamma):
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()  # Loss function (Mean Squared Error here)

    def train_step(self, state, action, reward, next_state, running):
        state = torch.tensor(state, dtype=torch.float)
        next_state = torch.tensor(next_state, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float)
        # (n, x)

        if len(state.shape) == 1:   # Add dimension to tensor
            # (1, x)
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            running = (running, )   # Convert to tuple

        # 1: predicted Q values with current state
        pred = self.model(state)    # Will look something like [1, 0, 0]

        target = pred.clone()
        for i in range(len(running)):
            Q_new = reward[i]
            if running[i]:
            #if not running[i]:
                Q_new = reward[i] + self.gamma * torch.max(self.model(next_state[i]))

            target[i][torch.argmax(action[i]).item()] = Q_new

        # 2: Q_new = r + y * max(next_predicted Q value) -> only do this if not done
        self.optimizer.zero_grad()
        loss = self.criterion(target, pred)
        loss.backward()

        self.optimizer.step()




