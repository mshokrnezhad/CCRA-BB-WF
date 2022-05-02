from Network import Network
from Request import Request
from Service import Service
from Functions import specify_requests_entry_nodes, assign_requests_to_services
from CPLEX import CPLEX
import numpy as np
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))

class Environment:
    def __init__(self, NUM_NODES=9, NUM_PRIORITY_LEVELS=1, NUM_REQUESTS=2, NUM_SERVICES=3, SWITCH="none"):
        self.NUM_NODES = NUM_NODES
        self.NUM_REQUESTS = NUM_REQUESTS
        self.NUM_SERVICES = NUM_SERVICES
        self.NUM_PRIORITY_LEVELS = NUM_PRIORITY_LEVELS
        self.SWITCH = SWITCH
        self.net_obj = Network(NUM_NODES, NUM_PRIORITY_LEVELS, self.SWITCH)
        self.REQUESTS_ENTRY_NODES = specify_requests_entry_nodes(self.net_obj.FIRST_TIER_NODES, np.arange(NUM_REQUESTS))
        self.req_obj = Request(NUM_REQUESTS, self.net_obj.NODES, self.REQUESTS_ENTRY_NODES, self.SWITCH)
        self.srv_obj = Service(NUM_SERVICES)
        self.REQUESTED_SERVICES = assign_requests_to_services(np.arange(NUM_SERVICES), np.arange(NUM_REQUESTS))
        self.model_obj = CPLEX(self.net_obj, self.req_obj, self.srv_obj, self.REQUESTED_SERVICES,
                               self.REQUESTS_ENTRY_NODES)

    def get_state(self, switch, assigned_nodes=[]):
        net_state = self.net_obj.get_state(switch)
        req_state = self.req_obj.get_state(switch, assigned_nodes)
        env_state = np.concatenate((req_state, net_state))

        # normalized_env_state = scaler.fit_transform(env_state.reshape(-1, 1))

        # return normalized_env_state.reshape(1, -1)[0]
        return env_state

    def step(self, action, switch, assigned_nodes=[]):
        result = self.model_obj.solve(action, switch, assigned_nodes)
        optimum_result = self.model_obj.solve({}, switch, assigned_nodes)
        accuracy = 0

        print("*:   ", optimum_result["g"][action["req_id"]])

        if result["done"]:
            reward = 0
        else:
            accuracy = result["OF"] / optimum_result["OF"]
            reward = ((1 - ((result["OF"] - optimum_result["OF"]) / result["OF"]))**2) * 10000
            """
            if accuracy <= 1.1:
                reward = 1
            else:
                # reward = 1/(result["OF"] - optimum_result["OF"])
                # reward = round(reward, 5)
                reward = 0
            """

            self.update_state(action, result)

        resulted_state = self.get_state(switch, assigned_nodes)

        return resulted_state, int(reward), result["done"] or len(self.req_obj.REQUESTS) == 0, result["info"], accuracy

    def update_state(self, action, result):
        self.net_obj.update_state(action, result, self.req_obj)
        self.req_obj.update_state(action)

    def reset(self, SEED):
        self.REQUESTS_ENTRY_NODES = specify_requests_entry_nodes(self.net_obj.FIRST_TIER_NODES,
                                                                 np.arange(self.NUM_REQUESTS), SEED)
        self.req_obj = Request(self.NUM_REQUESTS, self.net_obj.NODES, self.REQUESTS_ENTRY_NODES, self.SWITCH, SEED)

        self.net_obj = Network(self.NUM_NODES, self.NUM_PRIORITY_LEVELS, self.SWITCH)
        self.srv_obj = Service(self.NUM_SERVICES)
        self.REQUESTED_SERVICES = assign_requests_to_services(np.arange(self.NUM_SERVICES),
                                                              np.arange(self.NUM_REQUESTS))
        self.model_obj = CPLEX(self.net_obj, self.req_obj, self.srv_obj, self.REQUESTED_SERVICES,
                               self.REQUESTS_ENTRY_NODES)
