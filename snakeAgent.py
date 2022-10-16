import torch
import random
import numpy as np
from collections import deque
from SnakeAI import Game, Point, block
from snakeModel import Linear_QNet, QTrainer
from snakeHelper import plot

BATCH_SIZE = 1000   # How many samples we pull from memory


class Agent:
    def __init__(self):
        self.num_games = 0
        self.epsilon = 0 # Randomness
        self.gamma = 0.9  # Discount rate (value needs to be smaller than 1)
        self.memory = deque(maxlen=100000)  # Deque automatically removes values when it exceeds the max length
        self.model = Linear_QNet(11, 256, 3)    # 11 inputs (state list), 256 hidden (value can be played with), 3 outputs (action)
        self.trainer = QTrainer(self.model, learning_rate=0.001, gamma=self.gamma)

    def get_state(self, game):
        snake = game.snake
        #print(snake.x[0], snake.y[0])
        # Check one block ahead in every direction of the head
        pt_u = Point(snake.x[0], snake.y[0] - block, 3)
        pt_d = Point(snake.x[0], snake.y[0] + block, 3)
        pt_l = Point(snake.x[0] - block, snake.y[0], 3)
        pt_r = Point(snake.x[0] + block, snake.y[0], 3)



        #exit()

        #print(f"FRUIT = {Chess.fruit.x, Chess.fruit.y}")

        state = [
            # Danger straight
            (game.snake.direction == "right" and game.collision(pt_r)) or
            (game.snake.direction == "left" and game.collision(pt_l)) or
            (game.snake.direction == "up" and game.collision(pt_u)) or
            (game.snake.direction == "down" and game.collision(pt_d)),

            # Danger right
            (game.snake.direction == "up" and game.collision(pt_r)) or
            (game.snake.direction == "down" and game.collision(pt_l)) or
            (game.snake.direction == "left" and game.collision(pt_u)) or
            (game.snake.direction == "right" and game.collision(pt_d)),

            # Danger left
            (game.snake.direction == "down" and game.collision(pt_r)) or
            (game.snake.direction == "up" and game.collision(pt_l)) or
            (game.snake.direction == "right" and game.collision(pt_u)) or
            (game.snake.direction == "left" and game.collision(pt_d)),

            # Direction
            game.snake.direction == "left",
            game.snake.direction == "right",
            game.snake.direction == "up",
            game.snake.direction == "down",

            # Fruit direction
            game.fruit.x < game.snake.x[0], # Left
            game.fruit.x > game.snake.x[0], # Right
            game.fruit.y < game.snake.y[0], # Up
            game.fruit.y > game.snake.y[0],  # Down

            # Nearest wall (TODO)
            #game.nearest_wall()
        ]

        #print(np.array(state, dtype=int))
        return np.array(state, dtype=int)   # Array that turns any booleans into integers

    def get_action(self, state):
        """Randomly changes direction to explore environment but the better the model gets the less random the changes."""
        self.epsilon = 80 - self.num_games
        action = [0, 0, 0]
        if random.randint(0, 200) < self.epsilon:   # Random move
            move = random.randint(0, 2)
            action[move] = 1
        else: # As epsilon decreases, moves become less random
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0) # Looks something like [5.0, 2.7, 0.1]
            move = torch.argmax(prediction).item()  # [5.0, 2.7, 0.1] max is 5.0 at index 0
            action[move] = 1

        return action

    def remember(self, state, action, reward, next_state, running):
        self.memory.append((state, action, reward, next_state, running))    # Stored in memory as single tuple

    def train_long(self):
        if len(self.memory) > BATCH_SIZE:
            sample = random.sample(self.memory, BATCH_SIZE) # Gets 1000 random states from memory
        else:
            sample = self.memory

        states, actions, rewards, next_states, runnings = zip(*sample)
        self.trainer.train_step(states, actions, rewards, next_states, runnings)

    def train_short(self, state, action, reward, next_state, running):
        """Saves all needed information at each step of the snake."""
        self.trainer.train_step(state, action, reward, next_state, running)


def train():
    scores = [] # For plotting
    mean_scores = []    # For plotting
    total_score = 0
    record = 0
    agent = Agent()
    game = Game()

    while True: # Training loop
        prev_state = agent.get_state(game)  # Get previous state
        action = agent.get_action(prev_state)   # Get action

        # Perform move and get next state
        reward, running, score = game.play(action)
        next_state = agent.get_state(game)

        # Train short memory
        agent.train_short(prev_state, action, reward, next_state, running)

        # Remember
        agent.remember(prev_state, action, reward, next_state, running)

        #print(agent.memory)

        if not running:
            # Train long memory
            game.reset()
            agent.num_games += 1
            agent.train_long()

            if score > record:
                record = score  # Update the highest score
                agent.model.save()  # Save model if new high score

            print(f"\nGame {agent.num_games}\tScore {score}\tRecord {record}")

            # Update score stats and plot
            scores.append(score)
            total_score += score
            mean_score = total_score / agent.num_games
            mean_scores.append(mean_score)
            plot(scores, mean_scores)

        if game.force_quit:
            break


if __name__ == "__main__":
    train()
