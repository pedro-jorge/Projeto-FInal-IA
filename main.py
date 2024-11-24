import pygame
import sys
import numpy as np
from random import randint 

pygame.init()

# possíveis inputs do jogo: 
# teclas (na setinha do teclado) direita, esquerda, cima e baixo
RIGHT = pygame.K_RIGHT
LEFT = pygame.K_LEFT 
UP = pygame.K_UP
DOWN = pygame.K_DOWN

# tamanho de cada bloco da cobrinha
# e da comida
BLOCK_SIZE = 20 

# cores que serão usadas nos blocos
# branco para a cobrinha
WHITE1 = (255, 255, 255)
WHITE2 = (200, 200, 200)
# vermelho para a maçã
RED = (200, 0, 0)
# preto para o fundo da tela
BLACK = (0, 0, 0)

# classe que define e gerencia a cobrinha
class Snake:
    def __init__(self, x: int, y: int):
        """Instancia uma cobrinha de tamanho 1 (um bloco)

        Args:
            x (int): posição no eixo x
            y (int): posição no eixo y
        """
        # quando a cobrinha se move, ela dá um passo do mesmo
        # tamanho que o bloco
        self.step = BLOCK_SIZE
        self.x = x
        self.y = y
        # cria o corpo da cobrinha com tamanho 1
        self.body = [[x, y]]
    
    
    def move(self, direction) -> None:
        """Faz a cobrinha se mexer de acordo com a direção dada"""
        
        if direction == RIGHT:
            self.x += self.step
        
        elif direction == LEFT:
            self.x -= self.step 
        
        elif direction == UP:
            self.y -= self.step 
            
        elif direction == DOWN:
            self.y += self.step
        
        self.body.insert(0, [self.x, self.y])


# classe que define e gerencia a comida
class Food:
    def __init__(self, max_x: int, max_y: int):
        """Instancia um objeto comida.

        Args:
            max_x (int): a posição máxima onde a comida pode surgir no eixo x
            max_y (int): a posição máxima onde a comida pode surgir no eixo y
        """
        
        self.x = None 
        self.y = None 
        self.max_x = max_x
        self.max_y = max_y
        
        
    def update(self):
        """Gera uma posição aleatória para a comida. A cada rodada em que a cobrinha consegue comer, esse método é chamado."""
        # gerando as posições levando em consideração o tamanho da tela
        self.x = (randint(BLOCK_SIZE, self.max_x-BLOCK_SIZE)//BLOCK_SIZE)*BLOCK_SIZE
        self.y = (randint(BLOCK_SIZE, self.max_y-BLOCK_SIZE)//BLOCK_SIZE)*BLOCK_SIZE
         
    
# classe que roda o jogo
class Game:
    def __init__(self):
        # largura da tela
        self.width = 640
        # altura da tela
        self.height = 480
        # tupla com largura x altura
        self.screen_size = (self.width, self.height)
        # velocidade visual do jogo
        self.fps = 120
        
        # criando a tela efetivamente
        self.screen = pygame.display.set_mode(self.screen_size)
        self.clock = pygame.time.Clock()
        
        # criando a fonte para mostrar os textos
        pygame.display.set_caption("Snake")
        self.font = pygame.font.SysFont("Arial", 25)
        
        # cria a cobrinha inicial no meio da tela
        self.snake = Snake(self.width/2, self.height/2)
        # cria o objeto comida
        self.food = Food(self.width, self.height)
        
        # determina a direção inicial para a cobrinha (à direita)
        self.direction = RIGHT
        # gera a comida em uma posição aleatória
        self.food.update()
        
        # cria os valores de pontuação da rodada atual
        self.score = 0
        # cria os valores de melhor pontuação dentre todas as rodadas
        self.max_score = 0
        
        self.round = 0
    
    
    def check_game_over(self, pos = None) -> bool:
        """
        Verifica se a cobrinha esbarrou em si mesma ou nas bordas da tela.

        Returns:
            bool: se é ou não game over
        """
        
        if pos == None:
            pos = (self.snake.x, self.snake.y)
        
        # checa se a cobrinha esbarrou na borda esquerda ou direita
        if pos[0]> self.width-BLOCK_SIZE or pos[0] < 0:
            return True
    
        # checa se a cobrinha esbarrou na borda de cima ou de baixo
        if pos[1] >self.height-BLOCK_SIZE or pos[1] < 0:
            return True 
        
        # checa se a cobrinha esbarrou no próprio corpo
        if pos in self.snake.body[1:]:
            return True 
        
        return False

    
    def update_ui(self):
        """Atualiza as informações visualmente."""
        # colore a tela de preto
        self.screen.fill(BLACK)
        
        # desenha todos os blocos do corpo da cobrinha
        for point in self.snake.body:
            pygame.draw.rect(self.screen, WHITE1, pygame.Rect(point[0], point[1], BLOCK_SIZE, BLOCK_SIZE))
            pygame.draw.rect(self.screen, WHITE2, pygame.Rect(point[0]+4, point[1]+4, 12, 12))
        
        # desenha a comida
        pygame.draw.rect(self.screen, RED, pygame.Rect(self.food.x, self.food.y, BLOCK_SIZE, BLOCK_SIZE))

        # mostra os textos de pontuação
        text = f"Pontuação: {self.score} --- Pontuação máxima: {self.max_score} -- Round: {self.round}"
        scores = self.font.render(text, True, WHITE1)
        self.screen.blit(scores, [0, 0])
        
        pygame.display.flip()

        
    def run(self):
        """Roda o jogo efetivamente."""
        while True:
            for event in pygame.event.get():
                # verifica se o usuário está clicando no X para fechar
                if event.type == pygame.QUIT:
                    sys.exit()
                # captura qual tecla das setinhas foi apertada
                elif event.type == pygame.KEYDOWN:
                    self.direction = event.key 
            
            # move a cobrinha de acordo com a direção 
            self.snake.move(self.direction)
            
            # verifica se é a cobrinha esbarrou em si mesma ou nas bordas
            if self.check_game_over():
                # reinicia a cobrinha no meio da tela e com tamanho 1
                self.snake = Snake(self.width/2, self.height/2)
                # reinicia a pontuação
                self.score = 0
                continue
            
            # verifica se a cobrinha comeu a comida
            if self.snake.x == self.food.x and self.snake.y == self.food.y:
                # gera uma nova comida em um lugar aleatório
                self.food.update()
                # incrementa a pontuação
                self.score+=1
                # verifica se é a pontuação atingida é a maior até agora
                if self.score > self.max_score:
                    self.max_score = self.score 
            else:
                self.snake.body.pop()
            
            # atualiza as informações na tela
            self.update_ui()
            self.clock.tick(self.fps)

    
    def run_agent(self, key):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                sys.exit()
        
        directions = [RIGHT, DOWN, LEFT, UP]
        idx = directions.index(self.direction)
        
        if np.array_equal(key, [1, 0, 0]):
            key = directions[idx]
        elif np.array_equal(key, [0, 1, 0]):
            key = directions[(idx+1)%4]
        else:
            key = directions[(idx-1)%4]
        
        self.direction = key 
        self.snake.move(self.direction)
        reward = -1
        game_over = False
        
        # if self.score > 50:
        #     self.fps = 15
        
        if self.check_game_over():
            self.snake = Snake(self.width/2, self.height/2)
            self.score = 0
            self.round += 1
            reward -= 10
            game_over = True
            
            return reward, game_over, self.score
        
        if self.snake.x == self.food.x and self.snake.y == self.food.y:
            self.food.update()
            self.score += 1
            
            if self.score > self.max_score:
                self.max_score = self.score 
            
            reward = 15
        
        else:
            self.snake.body.pop()
        
        self.update_ui()
        self.clock.tick(self.fps)
        
        return reward, game_over, self.score
        
        

if __name__ == "__main__":
    game = Game()
    game.run()