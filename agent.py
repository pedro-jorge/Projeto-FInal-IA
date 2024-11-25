# Arquivo que contém o agente que controla o jogo

import numpy as np
import random
import torch 
import torch.optim as optim 
import torch.nn as nn
import matplotlib.pyplot as plt
from main import * 
from nn import * 

from collections import deque


class Agent:
    def __init__(self):
        # número de rodadas que o agente jogou
        self.n_games = 1
        #self.q_matrix = deque(maxlen=100_000)
        #self.batch_size = 1000
        
        self.max_score = 0
        
        self.model = QNetwork()
        # learning rate
        self.lr = 0.001
        # fator de desconto
        self.gamma = 0.9
        
        # otimizador
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        # função de custo/perda
        self.criterion = nn.MSELoss()
        
    
    
    def get_state(self, game: Game) -> np.ndarray:
        """Verifica qual é o estado atual do jogo.

        Args:
            game (Game): objeto que tem as informações do jogo em si

        Returns:
            np.ndarray: array que encapsula o estado do jogo. Esse array é a entrada para a rede neural
        """
        # pegando a informação da posição da cabeça da cobrinha
        head = game.snake.body[0]
        
        # pegando quais são as posições adjacentes à cabeça
        left = [head[0]-BLOCK_SIZE, head[1]]
        right = [head[0]+BLOCK_SIZE, head[1]]
        up = [head[0], head[1]-BLOCK_SIZE]
        down = [head[0], head[1]+BLOCK_SIZE]
        
        # variáveis booleanas que capturam qual é a atual 
        # direção que a cobrinha está andando
        dir_left = game.direction == LEFT 
        dir_right = game.direction == RIGHT 
        dir_up = game.direction == UP 
        dir_down = game.direction == DOWN
        
        # criação do array que encapsula o estado
        state = [
            # 1: booleano que diz se a cobrinha
            # irá esbarrar na borda da tela no próximo passo 
            # da direção em que ela segue
            (dir_right and game.check_game_over(right)) or
            (dir_left and game.check_game_over(left)) or 
            (dir_up and game.check_game_over(up)) or 
            (dir_down and game.check_game_over(down)),
            
            # 2: booleano que diz se a cobrinha
            # irá esbarrar no próprio corpo se virar a cabeça 
            # para a direita (tendo a cabeça como referencial)
            (dir_right and game.check_game_over(down)) or 
            (dir_left and game.check_game_over(up)) or 
            (dir_up and game.check_game_over(right)) or 
            (dir_down and game.check_game_over(left)),
            
            # 3: booleano que diz se a cobrinha 
            # irá esbarrar no próprio corpo se virar a cabeça
            # para a esquerda (tendo a cabeça como referencial)
            (dir_right and game.check_game_over(up)) or 
            (dir_left and game.check_game_over(down)) or 
            (dir_up and game.check_game_over(left)) or
            (dir_down and game.check_game_over(right)),
            
            # booleanos que dizem se a cobrinha está
            # seguindo na direção left, right, up, down
            dir_left,
            dir_right,
            dir_up,
            dir_down,
            
            # booleanos que encapsulam se 
            # a comida está antes ou depois da posição da cobrinha
            # em ambos os eixos
            game.food.x < head[0],
            game.food.x > head[0],
            game.food.y < head[1],
            game.food.y > head[1]
        ]
        
        return np.array(state, dtype=float)
    
    
    def get_action(self, state: np.ndarray) -> List:
        """O agente decide uma ação. Se o número de rodadas for baixo (estiver no início do treinamento), há alta probabilidade da escolha ser aleatória. Conforme o jogo avança, a probabilidade do agente tomar uma decisão "inteligente" com base na rede neural aumenta.

        Args:
            state (np.ndarray): vetor que representa o estado do jogo

        Returns:
            List: lista que representa o movimento escolhido [continuar na mesma direção, virar à direita (com relação à cabeça), virar à esquerda (com relação à cabeça)]
        """
        # lista com os possíveis movimentos
        final_move = [0, 0, 0]
        
        # com probabilidade 1/número de rodadas, o agente escolhe uma ação aleatória
        if np.random.rand() < 1/self.n_games and self.max_score < 50:
            move = np.random.randint(0, 2)
            final_move[move] = 1
        
        # caso o agente decida não se mover aleatoriamente, ele consulta a rede neural
        else:
            state = torch.tensor(state, dtype=torch.float)
            # valores para os possiveis movimentos
            prediction = self.model(state)
            # pega o maior valor dentre os três
            move = torch.argmax(prediction).item()
            final_move[move] = 1
        
        return final_move
    
    
    def train(self, _state: np.ndarray, _action: np.ndarray, _reward: int, _next_state: np.ndarray, game_over: bool):
        """Executa uma iteração de treinamento da rede neural do agente.

        Args:
            _state (np.ndarray): estado atual do jogo
            _action (np.ndarray): ação tomada pelo agente
            _reward (int): recompensa adquirida
            _next_state (np.ndarray): estado seguinte ao estado anterior com a ação tomada
            game_over (bool): se foi game over
        """
        # converte os parâmetros para tensor 
        # para serem utilizados pelo pytorch
        state = torch.tensor(_state, dtype=torch.float)
        next_state = torch.tensor(_next_state, dtype=torch.float)
        action = torch.tensor(_action, dtype=torch.long)
        reward = torch.tensor(_reward, dtype=torch.float)
        
        # adicionando uma dimensão extra se necessário
        if len(state.shape) == 1:
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            game_over = (game_over,)
        
        # capturando a saída da rede (a ação a ser tomada)
        pred = self.model(state)
        target = pred.clone() 

        # para cada estado no batch
        for idx in range(len(game_over)):
            # o novo Q-value é a recompensa já existente
            Q_new = reward[idx]
            # se não foi game over, atualizamos o Q-value com base na equação de Bellman
            if not game_over[idx]:
                Q_new = reward[idx] + self.gamma * torch.max(self.model(next_state[idx]))
            
            # atualizamos o target com a nova recompensa
            target[idx][torch.argmax(action[idx]).item()] = Q_new
    
        # medimos a diferença entre o target, com a lr recompensa calculada 
        # e a recompensa obtida pela ação tomada
        # e fazemos a backpropagation
        self.optimizer.zero_grad()
        loss = self.criterion(target, pred)
        loss.backward()
        self.optimizer.step()
        
        
agent = Agent()
game = Game()
scores = []
mean_scores = []

while True:
    state = agent.get_state(game)
    move = agent.get_action(state)
    reward, is_game_over, score = game.run_agent(move)
    state_new = agent.get_state(game)
    
    agent.train(state, move, reward, state_new, is_game_over)
    
    if is_game_over:
        agent.n_games += 1
        if score > agent.max_score:
            agent.max_score = score
            #agent.model.save()
            
        scores.append(score)
        mean_scores.append(sum(scores)/agent.n_games)
        
    if agent.n_games % 200 == 0:
        plt.xlabel(f"Número de rodadas")
        plt.ylabel(f"Pontos")
        
        plt.plot(list(range(1, agent.n_games)), scores, label="Pontuação")
        plt.plot(list(range(1, agent.n_games)), mean_scores, label="Pontuação Média")
        plt.legend()
        
        plt.savefig(f"11_relu256_3.png")
        
        print(f"Melhor pontuação: {max(scores)}")
        print(f"Pontuação média: {max(mean_scores)}")
        print(f"Função de ativação: {agent.model.activation_name}")
        break
    