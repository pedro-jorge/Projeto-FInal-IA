import torch 
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F 
import os
from typing import *


class QNetwork(nn.Module):
    def __init__(self, activation: Callable = nn.ReLU):
        super(QNetwork, self).__init__()
        self.activation_name = activation.__name__ 
        
        self.net = nn.Sequential(
            nn.Linear(11, 100),
            activation(),
            nn.Linear(100, 100),
            activation(),
            nn.Linear(100, 100),
            activation(),
            nn.Linear(100, 3),
            nn.Softmax()
        )
        
    
    def forward(self, x):
        return self.net(x)
    


class QTrainer:
    def __init__(self, model, lr, gamma, optimizer, criterion):
        self.lr = lr 
        self.gamma = gamma 
        self.model = model 
        self.optimizer = optimizer 
        self.criterion = criterion
    
    
    def train(self, state, action, reward, next_state, done):
        state = torch.tensor(state, dtype=torch.float)
        next_state = torch.tensor(next_state, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float)
        
        if len(state.shape) == 1:
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            done = (done,)
        
        pred = self.model(state)
        target = pred.clone() 
        
        for idx in range(len(done)):
            Q_new = reward[idx]
            if not done[idx]:
                Q_new = reward[idx] + self.gamma * torch.max(self.model(next_state[idx]))
            target[idx][torch.argmax(action[idx]).item()] = Q_new
        
        self.optimizer.zero_grad()
        loss = self.criterion(target, pred)
        loss.backward()
        self.optimizer.step()