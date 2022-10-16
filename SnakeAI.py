import math

import pygame
import random
from collections import namedtuple
from math import dist
import numpy as np

white = (255, 255, 255)
black = (0, 0, 0)
red = (255, 0, 0)
green = (0, 255, 0)

block = 20  # Size of a block
speed = 40

width = 600
height = 600

#Points = namedtuple("Point", "x, y, parent") # Hash lookup for snake coordinates


class Point:
    """Using this instead of the named tuple to improve the AI in future iterations."""
    __slots__ = ('x', 'y', 'depth', 'children')

    def __init__(self, x, y, depth):
        self.x = x
        self.y = y
        self.depth = depth
        self.children = []

        match self.depth != 0:  # Gets all children down to the depth given
            case True:
                self.find_children()

    def __repr__(self):
        return f'Point(x = {self.x}, y = {self.y}, depth = {self.depth})'

    def find_children(self):
        """Finds all the children of the parent points to a certain depth."""
        pt_u = Point(self.x, self.y - block, self.depth - 1)
        pt_d = Point(self.x, self.y + block, self.depth - 1)
        pt_l = Point(self.x - block, self.y, self.depth - 1)
        pt_r = Point(self.x + block, self.y, self.depth - 1)

        self.children = [pt_u, pt_d, pt_l, pt_r]  # child_up, child_down, child_left, child_right

class Fruit:
    def __init__(self, window, snake_x, snake_y):
        self.window = window
        self.snake_x = snake_x
        self.snake_y = snake_y
        self.x = None
        self.y = None
        self.move()

    def spawn(self):
        """Draws the fruit on the board."""
        pygame.draw.rect(self.window, red, [self.x, self.y, block, block])
        # pygame.display.update()

    def move(self):
        """Moves fruit to a random location that isn't already occupied by the snake."""
        while True:
            x = random.randrange(0, width - block, block)
            y = random.randrange(0, height - block, block)

            if (x, y) not in zip(self.snake_x, self.snake_y):  # Checks if position is occupied by the snake already
                self.x = x
                self.y = y
                break


class Snake:
    def __init__(self, window, length):
        self.window = window
        self.length = length
        self.x = [300] * length
        self.y = [300] * length
        self.direction = "right"  # Default starting direction

    def move_U(self):
        """Move up. Action prevented if you're moving down."""
        match self.direction:
            case "down":
                pass
            case _:
                self.direction = "up"

    def move_D(self):
        """Move down. Action prevented if you're moving up."""
        match self.direction:
            case "up":
                pass
            case _:
                self.direction = "down"

    def move_L(self):
        """Move left. Action prevented if you're moving right."""
        match self.direction:
            case "right":
                pass
            case _:
                self.direction = "left"

    def move_R(self):
        """Move right. Action prevented if you're moving left."""
        match self.direction:
            case "left":
                pass
            case _:
                self.direction = "right"

    def grow(self):
        """Increases the length of the snake by one."""
        self.length += 1
        self.x.append(0)  # Appends new set of (x, y) coords to corresponding lists
        self.y.append(0)

    def draw(self):
        """Draws the snake."""
        self.window.fill(black)
        for i in range(self.length):
            pygame.draw.rect(self.window, black, [self.x[i], self.y[i], block, block])  # Outlines snake segments
            pygame.draw.rect(self.window, green, [self.x[i] + 2, self.y[i] + 2, 16, 16])

    def slither(self, action):
        """Action tells the snake which direction to go dependent on the game state."""
        directions = ["right", "down", "left", "up"]  # Clockwise so we can make turns easier below
        idx = directions.index(self.direction)

        # Change the direction based on the action
        match action:
            case [1, 0, 0]:  # Continue in same direction
                self.direction = directions[idx]
                #print("straight")
            case [0, 1, 0]:  # Make a right turn
                self.direction = directions[(idx + 1) % 4]
                #print("right")
            case [0, 0, 1]:  # Make a left turn
                self.direction = directions[(idx - 1) % 4]
                #print("left")

        for i in range(self.length - 1, 0, -1):  # shift blocks when direction is changed
            self.x[i] = self.x[i - 1]
            self.y[i] = self.y[i - 1]

        match self.direction:
            case "up":
                self.y[0] -= block
            case "down":
                self.y[0] += block
            case "left":
                self.x[0] -= block
            case "right":
                self.x[0] += block

        self.draw()


class Game:
    def __init__(self):
        self.running = True
        self.force_quit = False
        pygame.init()
        self.window = pygame.display.set_mode((width, height))
        pygame.display.set_caption("Snake")
        self.clock = pygame.time.Clock()
        self.score = 0

        # For the AI
        self.fruit = None
        self.snake = None
        self.frame_iteration = None
        self.reward = 0
        self.dist_to_fruit = math.inf
        self.reset()

    def reset(self):
        self.snake = Snake(self.window, 1)
        self.fruit = Fruit(self.window, self.snake.x, self.snake.y)
        self.frame_iteration = 0
        self.score = 0
        self.running = True

    def distance(self):
        fruit = (self.fruit.x, self.fruit.y)
        snake = (self.snake.x[0], self.snake.y[0])
        dis = round(math.dist(snake, fruit))
        return dis

    def nearest_wall(self): # TODO
        dis = 0
        return dis

    #def draw_grid(self):
    #    for x in range(0, width, block):
    #        for y in range(0, height, block):
    #            rect = pygame.Rect(x, y, block, block)
    #            pygame.draw.rect(self.window, white, rect, 1)

    def display_score(self):
        """Displays the number of fruits eaten."""
        font = pygame.font.SysFont("Verdana", 25)
        score = font.render(f"SCORE: {self.score}", True, (200, 200, 200))
        self.window.blit(score, (0, 0))

    def collision(self, pt=None):
        """Checks if two objects collided."""
        if pt is None:  # points to snake
            pt = Point(self.snake.x[0], self.snake.y[0], 3)
        snake = list(zip(self.snake.x, self.snake.y))  # Makes a temporary coordinate list for the snake
        head = (pt.x, pt.y)
        print(head)

        if (pt.x > width - block) or (pt.x < 0) or (pt.y < 0) or (pt.y > height - block):
            return 1
        if head in snake[1:]:
            return 1
        return False

    def play(self, action):
        """Starts running the base of the game and allows for key presses."""
        self.frame_iteration += 1

        for event in pygame.event.get():
            match event.type:
                case pygame.QUIT:
                    print("Quitting")
                    self.running = False

        # HUMAN---------------------------------------------------------------HUMAN
                case pygame.KEYDOWN:
                    match event.key:
                        case pygame.K_UP:
                            self.snake.move_U()
                        case pygame.K_DOWN:
                            self.snake.move_D()
                        case pygame.K_LEFT:
                            self.snake.move_L()
                        case pygame.K_RIGHT:
                            self.snake.move_R()
                        case pygame.K_ESCAPE:
                            # self.running = False
                            self.force_quit = True
        # HUMAN---------------------------------------------------------------HUMAN

        # print(f"FRAME: {self.frame_iteration}\t LENGTH: {self.snake.length}")

        self.snake.slither(action)
        self.fruit.spawn()
        self.display_score()
        pygame.display.update()
        #self.draw_grid()
        #print(self.dist_to_fruit)

        self.reward = 0


        #self.nearest_wall()



        # Check if distance to fruit has decreased
        dis = self.distance()
        match dis <= self.dist_to_fruit:
            #case True:
            #    self.reward = 1
            case False:
                self.reward = -1
        self.dist_to_fruit = dis
        #print(self.dist_to_fruit, self.reward)

        # Check if the snake hit anything
        match self.collision():
            case 1:  # 1 - Game over
                print("Hit wall or snake")
                self.reward = -50
                self.running = False

        # Check if snake ate a fruit
        match self.snake.x[0] == self.fruit.x and self.snake.y[0] == self.fruit.y:
            case True:
                self.reward = 50
                self.dist_to_fruit = math.inf
                self.score += 1
                self.snake.grow()  # Add a block to the snake
                self.fruit.move()

        # End game after a certain amount of time
        match self.frame_iteration > 100 * self.snake.length:  # Game ends after 100 * snake_length frames
            case True:
                print("TOOK TOO LONG")
                self.running = False

        self.clock.tick(speed)
        #print(self.reward)

        return self.reward, self.running, self.score


if __name__ == "__main__":
    game = Game()
    while game.running and not game.force_quit:
        game.play(None)

    pygame.quit()
