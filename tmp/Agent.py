import numpy as np
import torch as T
from DNN import DNN
from Memory import Memory
rnd = np.random


class Agent(object):
    def __init__(self, GAMMA, EPSILON, LR, NUM_ACTIONS, INPUT_SHAPE, MEMORY_SIZE, BATCH_SIZE, EPSILON_MIN=0.01,
                 EPSILON_DEC=5e-7, REPLACE_COUNTER=1000, NAME="", CHECKPOINT_DIR='models/'):
        self.GAMMA = GAMMA
        self.EPSILON = EPSILON
        self.LR = LR
        self.NUM_ACTIONS = NUM_ACTIONS
        self.INPUT_SHAPE = INPUT_SHAPE
        self.BATCH_SIZE = BATCH_SIZE
        self.EPSILON_MIN = EPSILON_MIN
        self.EPSILON_DEC = EPSILON_DEC
        self.REPLACE_COUNTER = REPLACE_COUNTER
        self.CHECKPOINT_DIR = CHECKPOINT_DIR
        self.ACTION_SPACE = [i for i in range(self.NUM_ACTIONS)]
        self.learning_counter = 0
        self.memory = Memory(MEMORY_SIZE, INPUT_SHAPE, NUM_ACTIONS)
        self.q_eval = DNN(LR, NUM_ACTIONS, INPUT_SHAPE, NAME+"_q_eval", self.CHECKPOINT_DIR)
        self.q_next = DNN(LR, NUM_ACTIONS, INPUT_SHAPE, NAME+"_q_next", self.CHECKPOINT_DIR)

    def store_transition(self, state, action, reward, resulted_state, done):
        self.memory.store_transition(state, action, reward, resulted_state, done)

    def sample_memory(self):
        states, actions, rewards, resulted_states, dones = self.memory.sample_buffer(self.BATCH_SIZE)
        states = T.tensor(states)
        rewards = T.tensor(rewards)
        actions = T.tensor(actions)
        resulted_states = T.tensor(resulted_states)
        dones = T.tensor(dones)

        return states, actions, rewards, resulted_states, dones

    def choose_action(self, state, SEED, train_mode=True):
        rnd.seed(SEED)
        if train_mode:
            random_number = rnd.random()
            # print(random_number)
            if random_number > self.EPSILON:
                state = T.tensor(state, dtype=T.float)
                expected_values = self.q_eval.forward(state)
                action = T.argmax(expected_values).item()
                print("Q:", action + 4)
            else:
                action = rnd.choice(self.ACTION_SPACE)
                print("R:", action + 4)
        else:
            state = T.tensor(state, dtype=T.float)  # state = T.tensor([state], dtype=T.float)
            expected_values = self.q_eval.forward(state)
            action = T.argmax(expected_values).item()

        return action

    def replace_target_network(self):
        if self.REPLACE_COUNTER is not None and self.learning_counter % self.REPLACE_COUNTER == 0:
            self.q_next.load_state_dict(self.q_eval.state_dict())

    def decrement_epsilon(self):
        if self.EPSILON > self.EPSILON_MIN:
            self.EPSILON = self.EPSILON - self.EPSILON_DEC
        else:
            self.EPSILON = self.EPSILON_MIN

    def learn(self):
        if self.memory.counter < self.BATCH_SIZE:
            return

        self.q_eval.optimizer.zero_grad()
        self.replace_target_network()
        states, actions, rewards, resulted_states, dones = self.sample_memory()
        indexes = np.arange(self.BATCH_SIZE)

        q_pred = self.q_eval.forward(states)[indexes, actions]  # dims: batch_size * n_actions
        q_next = self.q_next.forward(resulted_states)
        q_eval = self.q_eval.forward(resulted_states)

        max_actions = T.argmax(q_eval, dim=1)
        q_next[dones] = 0.0

        target = rewards + self.GAMMA * q_next[indexes, max_actions]

        loss = self.q_eval.criterion(target, q_pred)
        loss.backward()

        self.q_eval.optimizer.step()

        self.learning_counter += 1
        self.decrement_epsilon()

    def save_models(self):
        self.q_eval.save_checkpoint()
        self.q_next.save_checkpoint()

    def load_models(self):
        self.q_eval.load_checkpoint()
        self.q_next.load_checkpoint()

    def extract_model(self):
        self.q_eval.load_checkpoint()

        return self.q_eval
