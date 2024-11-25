# Arquivo que contém a estrutura da rede neural para o Deep Q-Learning

import torch 
import torch.nn as nn
import torch.optim as optim

from typing import *


class QNetwork(nn.Module):
    """Classe que declara a rede neural do Deep Q-Learning. Possui 11 neurônios de entrada e 3 de saída:
    
    Os neurônios de entrada são todos booleanos:
    [
        1: O próximo bloco na direção atual dará game over?
        2: As vizinhanças da cabeça à direita darão game over?
        3: As vizinhanças da cabeça à esquerda darão game over?
        4: A direção é à esquerda?
        5: A direção é à esquerda?
        6: A direção é para cima?
        7: A direção é para baixo?
        8: A coordenada x da comida é menor do que a coordenada x da cabeça?
        9: A coordenada x da comida é maior do que a coordenada x da cabeça?
        10: A coordenada y da comida é menor do que a coordenada y da cabeça?
        11: A coordenada y é maior do que a coordenada y da cabeça?
    ]
    
    Os neurônios de saída dizem a probabilidade da cobrinha continuar na mesma direção, virar à esquerda ou à direita (em relação à cabeça):
    [
        1: continuar na mesma direção
        2: virar à direita
        3: virar à esquerda
    ]
    
    As camadas escondidas são arbitrárias e podem ser modificadas a cada experimento.
    """
    def __init__(self, activation: Callable = nn.ReLU):
        """Inicializa a representação da rede neural.

        Args:
            activation (Callable, optional): qual função de ativação usar entre cada camada escondida. Defaults to nn.ReLU.
        """
        super(QNetwork, self).__init__()
        self.activation_name = activation.__name__ 
        
        self.net = nn.Sequential(
            # a primeira camada precisa começar com 11 neurônios
            nn.Linear(11, 256),
            activation(),
            # nn.Linear(100, 100),
            # activation(),
            # a última camada precisa terminar com 3 neurônios
            nn.Linear(256, 3),
            #nn.Softmax()
        )
        
    
    def forward(self, x):
        return self.net(x)