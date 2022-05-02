from tmp.Environment import Environment
from tmp.Agent import Agent
import numpy as np
import random
import sys
from Functions import plot_learning_curve, calculate_input_shape
import torch as T


class Priority_Assignment(object):
    def __init__(self, NUM_NODES, NUM_REQUESTS, NUM_SERVICES, NUM_PRIORITY_LEVELS):
        self.SWITCH = "pri_asg"
        self.NUM_NODES = NUM_NODES
        self.NUM_REQUESTS = NUM_REQUESTS
        self.NUM_SERVICES = NUM_SERVICES
        self.NUM_PRIORITY_LEVELS = NUM_PRIORITY_LEVELS
        self.NUM_GAMES = 500
        self.EPSILON = 1
        self.srv_plc_env_obj = Environment(NUM_NODES, NUM_PRIORITY_LEVELS, NUM_REQUESTS, NUM_SERVICES, "srv_plc")
        self.env_obj = Environment(NUM_NODES, NUM_PRIORITY_LEVELS, NUM_REQUESTS, NUM_SERVICES, self.SWITCH)
        self.NUM_ACTIONS = NUM_PRIORITY_LEVELS
        self.INPUT_SHAPE = calculate_input_shape(NUM_NODES, NUM_REQUESTS, NUM_PRIORITY_LEVELS, self.SWITCH)
        self.FILE_NAME = self.SWITCH + "_" + str(NUM_NODES) + "_" + str(NUM_PRIORITY_LEVELS) + "_" + str(
            NUM_REQUESTS) + "_" + str(NUM_SERVICES) + "_" + str(self.NUM_GAMES)
        self.agent = Agent(GAMMA=0.99, EPSILON=self.EPSILON, LR=0.0001, NUM_ACTIONS=self.NUM_ACTIONS,
                           INPUT_SHAPE=self.INPUT_SHAPE, MEMORY_SIZE=50000, BATCH_SIZE=32,
                           EPSILON_MIN=0.1, EPSILON_DEC=5e-4, REPLACE_COUNTER=10000, NAME=self.FILE_NAME,
                           CHECKPOINT_DIR='models/')

    def get_allocations(self, srv_plc_q):
        try:
            assigned_nodes = []
            state = self.srv_plc_env_obj.get_state("srv_plc")

            for i in range(self.NUM_REQUESTS):
                raw_action = T.argmax(srv_plc_q.forward(T.tensor(state, dtype=T.float))).item()
                action = {"req_id": i, "node_id": raw_action + len(self.srv_plc_env_obj.net_obj.get_first_tier_nodes())}
                assigned_nodes.append(raw_action + len(self.srv_plc_env_obj.net_obj.get_first_tier_nodes()))
                # print(action)

                result = self.srv_plc_env_obj.model_obj.solve(action)

                if result["done"]:
                    print("there is no feasible solution for the input scenario!")
                    break
                else:
                    self.srv_plc_env_obj.update_state(action, result)
                    state = self.srv_plc_env_obj.get_state("srv_plc")

            # print(f'\nAssigned Nodes: {assigned_nodes}')
            return assigned_nodes

        except:
            print("there is no trained model for the input scenario!")
            return -1

    def train(self, srv_plc_q):
        best_score = -np.inf
        n_steps = 0
        scores, eps_history, steps_array = [], [], []

        for i in range(self.NUM_GAMES):
            done = False
            score = 0
            SEED = int(random.randrange(sys.maxsize) / (10 ** 15))

            self.env_obj.reset(SEED)
            self.srv_plc_env_obj.reset(SEED)
            assigned_nodes = self.get_allocations(srv_plc_q)

            state = self.env_obj.get_state(self.SWITCH, assigned_nodes)
            # parse_state(state, self.NUM_NODES, self.NUM_REQUESTS, self.env_obj, self.SWITCH)

            while not done:
                selected_request = self.env_obj.req_obj.REQUESTS.min()
                raw_action = self.agent.choose_action(state)
                action = {"req_id": selected_request, "pri_id": raw_action + 1}
                print(action)

                resulted_state, reward, done, info = self.env_obj.step(action, "pri_asg", assigned_nodes)
                # parse_state(resulted_state, self.NUM_NODES, self.NUM_REQUESTS, self.env_obj, self.SWITCH)

                score += reward

                self.agent.store_transition(state, raw_action, reward, resulted_state, int(done))
                self.agent.learn()

                state = resulted_state
                n_steps += 1

            scores.append(score)
            steps_array.append(n_steps)

            avg_score = np.mean(scores[-100:])
            print('episode:', i, 'score:', score, 'best_score: %.1f, eps: %.4f'
                  % (best_score, self.agent.EPSILON), 'steps', n_steps)

            if avg_score > best_score:
                self.agent.save_models()
                best_score = avg_score

            eps_history.append(self.agent.EPSILON)

        scores = np.array(scores)
        x = [i + 1 for i in range(len(scores))]
        plot_learning_curve(steps_array, scores / (10000 * self.NUM_REQUESTS), eps_history,
                            filename=self.FILE_NAME + '.png')