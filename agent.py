import numpy as np
import torch 
import random
import torch.optim as optim 
import torch.nn as nn
from main import * 
from nn import * 

from collections import deque


class Agent:
    def __init__(self):
        self.n_games = 0
        self.epsilon = 1/75
        self.memory = deque(maxlen=1500)
        self.batch_size = 800
        
        self.model = QNetwork()
        self.lr = 0.00005
        self.gamma = 0.9
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()
        
        self.trainer = QTrainer(self.model, self.lr, self.gamma, self.optimizer, self.criterion)
        
    
    def get_state(self, game):
        head = game.snake.body[0]
        
        left = [head[0]-BLOCK_SIZE, head[1]]
        right = [head[0]+BLOCK_SIZE, head[1]]
        up = [head[0], head[1]-BLOCK_SIZE]
        down = [head[0], head[1]+BLOCK_SIZE]
        
        dir_left = game.direction == LEFT 
        dir_right = game.direction == RIGHT 
        dir_up = game.direction == UP 
        dir_down = game.direction == DOWN
        
        
        state = (
            (dir_right and game.check_game_over(right)) or
            (dir_left and game.check_game_over(left)) or 
            (dir_up and game.check_game_over(up)) or 
            (dir_down and game.check_game_over(down)),
            
            (dir_right and game.check_game_over(down)) or 
            (dir_left and game.check_game_over(up)) or 
            (dir_up and game.check_game_over(right)) or 
            (dir_down and game.check_game_over(left)),
            
            (dir_right and game.check_game_over(up)) or 
            (dir_left and game.check_game_over(down)) or 
            (dir_up and game.check_game_over(left)) or
            (dir_down and game.check_game_over(right)),
            
            dir_left,
            dir_right,
            dir_up,
            dir_down,
            
            game.food.x < head[0],
            game.food.x > head[0],
            game.food.y < head[1],
            game.food.y > head[1]
        )
        
        return np.array(state, dtype=int)
    
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    
    
    def train_long_memory(self):
        if len(self.memory) > self.batch_size:
            mini_sample = random.sample(self.memory, self.batch_size)
        else:
            mini_sample = self.memory
    
    
    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train(state, action, reward, next_state, done)
    
    
    def get_action(self, state):
        self.epsilon = 80 - self.n_games 
        final_move = [0, 0, 0]
        
        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 2)
            final_move[move] = 1 
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1 
        
        return final_move

    

agent = Agent()
game = Game()

while True:
    state_old = agent.get_state(game)
    final_move = agent.get_action(state_old)
    reward, done, score = game.run_agent(final_move)
    state_new = agent.get_state(game)
    
    agent.train_short_memory(state_old, final_move, reward, state_new, done)
    agent.remember(state_old, final_move, reward, state_new, done)