from tmp.Environment import Environment
from tmp.Agent import Agent
import numpy as np
import random
import sys
from Functions import plot_learning_curve, calculate_input_shape


class Service_Placement(object):
    def __init__(self, NUM_NODES, NUM_REQUESTS, NUM_SERVICES, NUM_PRIORITY_LEVELS):
        self.SWITCH = "srv_plc"
        self.NUM_NODES = NUM_NODES
        self.NUM_REQUESTS = NUM_REQUESTS
        self.NUM_SERVICES = NUM_SERVICES
        self.NUM_PRIORITY_LEVELS = NUM_PRIORITY_LEVELS
        self.NUM_GAMES = 500
        self.EPSILON = 1
        self.env_obj = Environment(NUM_NODES, NUM_PRIORITY_LEVELS, NUM_REQUESTS, NUM_SERVICES, self.SWITCH)
        self.NUM_ACTIONS = (NUM_NODES - len(self.env_obj.net_obj.get_first_tier_nodes()))  # 1,100,1,1
        # self.NUM_ACTIONS = NUM_NODES  # 1,1,100,1
        self.INPUT_SHAPE = calculate_input_shape(NUM_NODES, NUM_REQUESTS, NUM_PRIORITY_LEVELS, self.SWITCH)
        self.FILE_NAME = self.SWITCH + "_" + str(NUM_NODES) + "_" + str(NUM_PRIORITY_LEVELS) + "_" + str(
            NUM_REQUESTS) + "_" + str(NUM_SERVICES) + "_" + str(self.NUM_GAMES)
        self.agent = Agent(GAMMA=0.99, EPSILON=self.EPSILON, LR=0.0001, NUM_ACTIONS=self.NUM_ACTIONS,
                           INPUT_SHAPE=self.INPUT_SHAPE, MEMORY_SIZE=3000, BATCH_SIZE=32, EPSILON_MIN=0.1,
                           EPSILON_DEC=36e-5, REPLACE_COUNTER=10000, NAME=self.FILE_NAME, CHECKPOINT_DIR='models/')

    def train(self):
        f = open("txts/" + self.FILE_NAME + "_" + str(int(random.randrange(sys.maxsize) / (10 ** 15))) + '.txt', "a")

        best_score = -np.inf
        n_steps = 0
        scores, eps_history, steps_array = [], [], []

        print("episode score best_score eps steps *infeasible accuracy")
        f.write("\n")

        for i in range(self.NUM_GAMES):
            done = False
            score = 0
            sum_accuracy = 0
            # SEED = int(random.randrange(sys.maxsize) / (10 ** 15))
            SEED_MAIN = 3555

            self.env_obj.reset(SEED_MAIN)
            state = self.env_obj.get_state(self.SWITCH)
            #parse_state(state, self.NUM_NODES, self.NUM_REQUESTS, self.env_obj)

            flag = False
            while not done:
                SEED = int(random.randrange(sys.maxsize) / (10 ** 15))

                selected_request = self.env_obj.req_obj.REQUESTS.min()
                raw_action = self.agent.choose_action(state, SEED)
                action = {"req_id": selected_request,
                          "node_id": raw_action + len(self.env_obj.net_obj.get_first_tier_nodes())}  # 1,100,1,1
                # action = {"req_id": selected_request, "node_id": raw_action}  # 1,1,100,1
                # print(action)

                resulted_state, reward, done, info, accuracy = self.env_obj.step(action, "srv_plc")
                # parse_state(resulted_state, self.NUM_NODES, self.NUM_REQUESTS, self.env_obj)

                if "infeasible" in info:
                    flag = True

                score += reward
                sum_accuracy += accuracy

                self.agent.store_transition(state, raw_action, reward, resulted_state, int(done))
                self.agent.learn()

                state = resulted_state

                # print(n_steps)
                n_steps += 1

            scores.append(score)
            steps_array.append(n_steps)

            avg_score = np.mean(scores[-100:])
            print('episode:', i, 'score:', score, 'best_score: %.1f eps: %.4f'
                  % (best_score, self.agent.EPSILON), 'steps', n_steps, "Yes" if flag else "No",
                  round(sum_accuracy/self.NUM_REQUESTS, 5))
            f.write(str(i) + " " + str(score) + " " + str(best_score) + " " +
                    str(round(self.agent.EPSILON, 5)) + " " + str(n_steps) + " " +
                    str(round(sum_accuracy / self.NUM_REQUESTS, 5)))
            """
            print(i, round(score, 5), round(best_score, 5), round(self.agent.EPSILON, 5), n_steps,
                  "Yes" if flag else "No", round(sum_accuracy/self.NUM_REQUESTS, 5))
            f.write(str(i) + " " + str(round(score, 5)) + " " + str(round(best_score, 5)) + " " +
                    str(round(self.agent.EPSILON, 5)) + " " + str(n_steps) + " " +
                    str(round(sum_accuracy/self.NUM_REQUESTS, 5)))
            """
            f.write("\n")

            if avg_score > best_score:
                self.agent.save_models()
                best_score = avg_score

            eps_history.append(self.agent.EPSILON)

        scores = np.array(scores)
        x = [i + 1 for i in range(len(scores))]
        plot_learning_curve(steps_array, scores / (10000 * self.NUM_REQUESTS), eps_history,
                            filename="figs/" + self.FILE_NAME + "_" + str(SEED_MAIN) + '.png')
        # plot_learning_curve(steps_array, scores / self.NUM_REQUESTS, eps_history, filename="figs/" + self.FILE_NAME +
        #                     "_" + str(SEED_MAIN) + '.png')

        f.close()

    def test(self, TEST_SIZE=100):
        try:
            self.agent.load_models()

            best_score = -np.inf
            n_steps = 0
            scores, eps_history, steps_array = [], [], []

            for i in range(TEST_SIZE):
                done = False
                score = 0
                SEED = int(random.randrange(sys.maxsize) / (10 ** 15))

                self.env_obj.reset(SEED)
                state = self.env_obj.get_state(self.SWITCH)

                while not done:
                    selected_request = self.env_obj.req_obj.REQUESTS.min()
                    raw_action = self.agent.choose_action(state, False)
                    action = {"req_id": selected_request,
                              "node_id": raw_action + len(self.env_obj.net_obj.get_first_tier_nodes())}

                    resulted_state, reward, done, info = self.env_obj.step(action, "srv_plc")

                    score += reward

                    state = resulted_state
                    n_steps += 1

                scores.append(score)
                steps_array.append(n_steps)

                avg_score = np.mean(scores[-100:])
                print('episode:', i, 'score:', score, 'avg_score: %.1f' % (avg_score), 'steps', n_steps)

                if avg_score > best_score:
                    best_score = avg_score

                eps_history.append(self.agent.EPSILON)

            scores = np.array(scores)
            x = [i + 1 for i in range(len(scores))]
            plot_learning_curve(steps_array, scores / (10000 * self.NUM_REQUESTS), eps_history,
                                filename=self.FILE_NAME + '_test.png')

        except:
            print("there is no trained model for the input scenario!")

    def get_allocations(self, env_obj):
        try:
            assigned_nodes = []
            self.agent.load_models()

            # env_obj.reset(SEED)
            state = self.env_obj.get_state(self.SWITCH)

            for i in range(self.NUM_REQUESTS):
                raw_action = self.agent.choose_action(state, False)
                action = {"req_id": i, "node_id": raw_action + len(self.env_obj.net_obj.get_first_tier_nodes())}
                assigned_nodes.append(raw_action + len(self.env_obj.net_obj.get_first_tier_nodes()))
                # print(action)

                result = self.env_obj.model_obj.solve(action)

                if result["done"]:
                    print("there is no feasible solution for the input scenario!")
                    break
                else:
                    self.env_obj.update_state(action, result)
                    state = self.env_obj.get_state(self.SWITCH)

            # print(f'\nAssigned Nodes: {assigned_nodes}')
            return assigned_nodes

        except:
            print("there is no trained model for the input scenario!")
            return -1