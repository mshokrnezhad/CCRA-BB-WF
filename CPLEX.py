# v3 for GlobeCom2022
from docplex.mp.model import Model
import numpy as np


class CPLEX:

    def __init__(self, net_obj, req_obj, srv_obj, REQUESTED_SERVICES, REQUESTS_ENTRY_NODES):
        self.net_obj = net_obj
        self.req_obj = req_obj
        self.srv_obj = srv_obj
        self.REQUESTED_SERVICES = REQUESTED_SERVICES
        self.REQUESTS_ENTRY_NODES = REQUESTS_ENTRY_NODES
        self.M = 10 ** 6
        self.Z = [(s, v) for s in srv_obj.SERVICES for v in net_obj.NODES]
        self.G = [(r, v) for r in req_obj.REQUESTS for v in net_obj.NODES]
        self.P = [(r, p, k) for r in req_obj.REQUESTS for p in net_obj.PATHS for k in net_obj.PRIORITIES]
        self.Rho = [(r, k) for r in req_obj.REQUESTS for k in net_obj.PRIORITIES]
        self.LINK_DELAYS = net_obj.LINK_DELAYS_DICT
        self.EPSILON = 0.001
        self.PATH_PER_NODE = (net_obj.NUM_NODES - 1) * net_obj.NUM_PATHS_UB

    def initialize_model(self):
        mdl = Model('CCRA')
        z = mdl.binary_var_dict(self.Z, name='z')
        g = mdl.binary_var_dict(self.G, name='g')
        req_path = mdl.binary_var_dict(self.P, name='req_path')
        rpl_path = mdl.binary_var_dict(self.P, name='rpl_path')
        rho = mdl.binary_var_dict(self.Rho, name='rho')
        d = mdl.continuous_var_dict([r for r in self.req_obj.REQUESTS], name='d')
        req_paths_cost = mdl.continuous_var_dict([r for r in [0]], name='req_paths_cost')
        rpl_paths_cost = mdl.continuous_var_dict([r for r in [0]], name='rpl_paths_cost')
        nodes_cost = mdl.continuous_var_dict([r for r in self.req_obj.REQUESTS], name='nodes_cost')

        return mdl, z, g, req_path, rpl_path, rho, d, req_paths_cost, rpl_paths_cost

    def define_model(self, mdl, z, g, req_path, rpl_path, rho, d, req_paths_cost, rpl_paths_cost):

        # region OF
        # print("defining OF...")
        mdl.minimize(
            mdl.sum(
                mdl.sum(
                    g[r, v]
                    for r in self.req_obj.REQUESTS
                ) * self.net_obj.DC_COSTS[v]
                for v in self.net_obj.NODES
            )
            +
            mdl.sum(
                (
                    mdl.sum(
                        (req_path[r, p, k] * self.net_obj.LINKS_PATHS_MATRIX[l, p])
                        for p in self.net_obj.PATHS_PER_HEAD[self.REQUESTS_ENTRY_NODES[r]]
                        for k in self.net_obj.PRIORITIES
                    )
                    +
                    mdl.sum(
                        (rpl_path[r, p, k] * self.net_obj.LINKS_PATHS_MATRIX[l, p])
                        for p in self.net_obj.PATHS_PER_TAIL[self.REQUESTS_ENTRY_NODES[r]]
                        for k in self.net_obj.PRIORITIES
                    )
                ) * self.net_obj.LINK_COSTS[l]
                for l in self.net_obj.LINKS
                for r in self.req_obj.REQUESTS
            )
            # just to control z. it could be removed without changing the result
            # + mdl.sum(z[s, v] for s in self.srv_obj.SERVICES for v in self.net_obj.NODES)
            # minimize the sum of delays just to test
            # + mdl.sum(d[r] for r in self.req_obj.REQUESTS) * 1000
        )
        # endregion

        # region C1
        # print("defining C1...")
        mdl.add_constraints(
            mdl.sum(
                g[r, v]
                for v in self.net_obj.NODES
            ) == 1
            for r in self.req_obj.REQUESTS
        )
        # endregion

        """
        # region C2
        # print("defining C2...")
        mdl.add_constraints(
            g[r, v]
            <=
            z[s, v]
            for r in self.req_obj.REQUESTS
            for v in self.net_obj.NODES
            for s in self.srv_obj.SERVICES
            if s == self.REQUESTED_SERVICES[r]
        )
        # endregion
        """

        # region C3 & C4
        # print("defining C3 & C4...")
        mdl.add_constraints(
            mdl.sum(
                g[r, v] * self.req_obj.CAPACITY_REQUIREMENTS[r]
                for r in self.req_obj.REQUESTS
            )
            <=
            self.net_obj.DC_CAPACITIES[v]
            for v in self.net_obj.NODES
        )
        # endregion

        # region C5
        # print("defining C5...")
        mdl.add_constraints(
            mdl.sum(
                rho[r, k]
                for k in self.net_obj.PRIORITIES
            )
            == 1
            for r in self.req_obj.REQUESTS
        )
        # endregion

        # region C6
        # print("defining C6...")
        mdl.add_constraints(
            mdl.sum(
                req_path[r, p, k]
                for k in self.net_obj.PRIORITIES
                for p in self.net_obj.PATHS_PER_HEAD[self.REQUESTS_ENTRY_NODES[r]]
                if self.net_obj.PATHS_DETAILS[p][-1] == v
            )
            ==
            g[r, v]
            for r in self.req_obj.REQUESTS
            for v in self.net_obj.NODES
        )
        # endregion

        # region C7
        # print("defining C7...")
        mdl.add_constraints(
            mdl.sum(
                rpl_path[r, p, k]
                for k in self.net_obj.PRIORITIES
                for p in self.net_obj.PATHS_PER_TAIL[self.REQUESTS_ENTRY_NODES[r]]
                if self.net_obj.PATHS_DETAILS[p][0] == v
            )
            ==
            g[r, v]
            for r in self.req_obj.REQUESTS
            for v in self.net_obj.NODES
        )
        # endregion

        # region C8
        # print("defining C8...")
        mdl.add_constraints(
            mdl.sum(
                req_path[r, p, k]
                for p in self.net_obj.PATHS_PER_HEAD[self.REQUESTS_ENTRY_NODES[r]]
            )
            == rho[r, k]
            for r in self.req_obj.REQUESTS
            for k in self.net_obj.PRIORITIES
        )
        # endregion

        # region C9
        # print("defining C9...")
        mdl.add_constraints(
            mdl.sum(
                rpl_path[r, p, k]
                for p in self.net_obj.PATHS_PER_TAIL[self.REQUESTS_ENTRY_NODES[r]]
            )
            == rho[r, k]
            for r in self.req_obj.REQUESTS
            for k in self.net_obj.PRIORITIES
        )
        # endregion

        """
        # region C10
        # print("defining C10...")
        mdl.add_constraint(
            mdl.sum(
                (
                    (req_path[r, 32, 1] * self.net_obj.LINKS_PATHS_MATRIX[5, 32])
                    +
                    (rpl_path[r, 32, 1] * self.net_obj.LINKS_PATHS_MATRIX[5, 32])
                ) * self.req_obj.BW_REQUIREMENTS[r]
                for r in self.req_obj.REQUESTS
            )
            <=
            self.net_obj.LINK_BWS[5]
        )
        # endregion
        """

        # region C10
        # print("defining C10...")
        mdl.add_constraints(
            mdl.sum(
                (
                    mdl.sum(
                        (req_path[r, p, k] * self.net_obj.LINKS_PATHS_MATRIX[l, p])
                        for p in self.net_obj.PATHS_PER_HEAD[self.REQUESTS_ENTRY_NODES[r]]
                        for k in self.net_obj.PRIORITIES
                    )
                    +
                    mdl.sum(
                        (rpl_path[r, p, k] * self.net_obj.LINKS_PATHS_MATRIX[l, p])
                        for p in self.net_obj.PATHS_PER_TAIL[self.REQUESTS_ENTRY_NODES[r]]
                        for k in self.net_obj.PRIORITIES
                    )
                ) * self.req_obj.BW_REQUIREMENTS[r]
                for r in self.req_obj.REQUESTS
            )
            <=
            self.net_obj.LINK_BWS[l]
            for l in self.net_obj.LINKS
        )
        # endregion

        # region C11
        # print("defining C11...")
        mdl.add_constraints(
            mdl.sum(
                (
                    mdl.sum(
                        (req_path[r, p, k] * self.net_obj.LINKS_PATHS_MATRIX[l, p])
                        for p in self.net_obj.PATHS_PER_HEAD[self.REQUESTS_ENTRY_NODES[r]]
                    )
                    +
                    mdl.sum(
                        (rpl_path[r, p, k] * self.net_obj.LINKS_PATHS_MATRIX[l, p])
                        for p in self.net_obj.PATHS_PER_TAIL[self.REQUESTS_ENTRY_NODES[r]]
                    )
                ) * self.req_obj.BURST_SIZES[r]
                for r in self.req_obj.REQUESTS
            )
            <=
            self.net_obj.BURST_SIZE_LIMIT_PER_PRIORITY[k]
            for l in self.net_obj.LINKS
            for k in self.net_obj.PRIORITIES
        )
        # endregion

        # region C13'
        # print("defining C13'...")
        mdl.add_constraints(
            mdl.sum(
                (
                    mdl.sum(
                        (req_path[r, p, k] * self.net_obj.LINKS_PATHS_MATRIX[l, p])
                        for p in self.net_obj.PATHS_PER_HEAD[self.REQUESTS_ENTRY_NODES[r]]
                    )
                    +
                    mdl.sum(
                        (rpl_path[r, p, k] * self.net_obj.LINKS_PATHS_MATRIX[l, p])
                        for p in self.net_obj.PATHS_PER_TAIL[self.REQUESTS_ENTRY_NODES[r]]
                    )
                ) * self.req_obj.BW_REQUIREMENTS[r]
                for r in self.req_obj.REQUESTS
            )
            <=
            self.net_obj.LINK_BWS_LIMIT_PER_PRIORITY[l, k]
            for l in self.net_obj.LINKS
            for k in self.net_obj.PRIORITIES
        )
        # endregion

        # region C14'
        # print("defining C14'...")
        mdl.add_constraints(
            mdl.sum(
                ((req_path[r, p, k] * self.net_obj.LINKS_PATHS_MATRIX[l, p]) * self.net_obj.LINK_DELAYS[l][k])
                for p in self.net_obj.PATHS_PER_HEAD[self.REQUESTS_ENTRY_NODES[r]]
                for l in self.net_obj.LINKS
                for k in self.net_obj.PRIORITIES
            )
            +
            mdl.sum(
                ((rpl_path[r, p, k] * self.net_obj.LINKS_PATHS_MATRIX[l, p]) * self.net_obj.LINK_DELAYS[l][k])
                for p in self.net_obj.PATHS_PER_TAIL[self.REQUESTS_ENTRY_NODES[r]]
                for l in self.net_obj.LINKS
                for k in self.net_obj.PRIORITIES
            )
            +
            mdl.sum(
                (g[r, v] * self.net_obj.PACKET_SIZE / (self.net_obj.DC_CAPACITIES[v] + self.EPSILON))
                for v in self.net_obj.NODES
            )
            ==
            d[r]
            for r in self.req_obj.REQUESTS
        )
        # endregion

        # region C15
        # print("defining C15...")
        mdl.add_constraints(
            d[r]
            <=
            self.req_obj.DELAY_REQUIREMENTS[r]
            for r in self.req_obj.REQUESTS
        )
        # endregion

        """
        # region req_paths_cost
        # print("defining req_paths_cost...")
        mdl.add_constraint(
            mdl.sum(
                (req_path[2, p, k] * self.net_obj.LINKS_PATHS_MATRIX[l, p]) * self.net_obj.LINK_COSTS[l]
                for p in self.net_obj.PATHS_PER_HEAD[self.REQUESTS_ENTRY_NODES[2]]
                for k in self.net_obj.PRIORITIES
                for l in self.net_obj.LINKS
            )
            ==
            req_paths_cost[0]
        )
        # endregion

        # region rpl_paths_cost
        # print("defining rpl_paths_cost...")
        mdl.add_constraint(
            mdl.sum(
                (rpl_path[2, p, k] * self.net_obj.LINKS_PATHS_MATRIX[l, p]) * self.net_obj.LINK_COSTS[l]
                for p in self.net_obj.PATHS_PER_TAIL[self.REQUESTS_ENTRY_NODES[2]]
                for k in self.net_obj.PRIORITIES
                for l in self.net_obj.LINKS
            )
            ==
            rpl_paths_cost[0]
        )
        # endregion

        mdl.add_constraint(
            req_path[0, 4, 1]
            ==
            1
        )

        mdl.add_constraint(
            rpl_path[0, 30, 1]
            ==
            1
        )

        mdl.add_constraint(
            req_path[1, 14, 1]
            ==
            1
        )

        mdl.add_constraint(
            rpl_path[1, 32, 1]
            ==
            1
        )

        mdl.add_constraint(
            req_path[2, 14, 1]
            ==
            1
        )

        mdl.add_constraint(
            rpl_path[2, 32, 1]
            ==
            1
        )
        """

        return mdl

    def solve(self, action, switch="none", assigned_nodes=[], assigned_priorities=[], assigned_paths=[]):
        # print("initializing model...")
        mdl, z, g, req_path, rpl_path, rho, d, req_paths_cost, rpl_paths_cost = self.initialize_model()

        # print("defining model...")
        mdl = self.define_model(mdl, z, g, req_path, rpl_path, rho, d, req_paths_cost, rpl_paths_cost)

        msg = ""

        """
        if switch == "srv_plc" and len(action) != 0:
            mdl = self.add_service_placement_action_constraint(mdl, g, action)

        if switch == "pri_asg":
            if not assigned_nodes:
                msg = "the assigned nodes array is empty!"
            else:
                self.add_assigned_nodes_constraints(mdl, g, assigned_nodes)

            if len(action) != 0:
                mdl = self.add_priority_assignment_action_constraint(mdl, rho, action)
        """

        # print("running model.solve()...")
        # mdl.parameters.timelimit = 60
        # mdl.log_output = True
        solution = mdl.solve()
        # print(solution, "\n")
        # mdl.solve()

        parsed_solution = {"pairs": {}, "g": {}, "k": {}, "req_paths": {}, "rpl_paths": {}, "info": "", "OF": 0, "done": True}

        if msg != "":
            print(f"error: {msg}")
            return parsed_solution
        else:
            try:
                status = str(mdl.solve_details.status)

                g = np.array([[1 if g[r, v].solution_value > 0 else 0 for v in self.net_obj.NODES] for r in range(self.req_obj.NUM_REQUESTS)])
                g = np.array([g[r, :].argmax() for r in range(self.req_obj.NUM_REQUESTS)])

                k = np.array([[1 if rho[r, k].solution_value > 0 else 0 for r in range(self.req_obj.NUM_REQUESTS)] for k in self.net_obj.PRIORITIES])
                k = np.array([k[:, r].argmax() for r in range(self.req_obj.NUM_REQUESTS)])

                req_paths = np.array([[[1 if req_path[r, p, k].solution_value > 0 else 0 for k in self.net_obj.PRIORITIES] for p in self.net_obj.PATHS] for r in range(self.req_obj.NUM_REQUESTS)])
                req_paths = np.array([[req_paths[r, p, :].argmax() for p in self.net_obj.PATHS] for r in range(self.req_obj.NUM_REQUESTS)])
                req_paths = np.array([req_paths[r, :].argmax() for r in range(self.req_obj.NUM_REQUESTS)])
                req_paths = np.array([self.net_obj.PATHS_DETAILS[req_paths[r]] for r in range(self.req_obj.NUM_REQUESTS)], dtype=object)

                rpl_paths = np.array([[[1 if rpl_path[r, p, k].solution_value > 0 else 0 for k in self.net_obj.PRIORITIES] for p in self.net_obj.PATHS] for r in range(self.req_obj.NUM_REQUESTS)])
                rpl_paths = np.array([[rpl_paths[r, p, :].argmax() for p in self.net_obj.PATHS] for r in range(self.req_obj.NUM_REQUESTS)])
                rpl_paths = np.array([rpl_paths[r, :].argmax() for r in range(self.req_obj.NUM_REQUESTS)])
                rpl_paths = np.array([self.net_obj.PATHS_DETAILS[rpl_paths[r]] for r in range(self.req_obj.NUM_REQUESTS)], dtype=object)

                parsed_pairs = {}
                parsed_g = {}
                parsed_k = {}
                parsed_req_paths = {}
                parsed_rpl_paths = {}
                for r in self.req_obj.REQUESTS:
                    parsed_pairs[r] = (self.REQUESTS_ENTRY_NODES[r], g[r])
                    parsed_g[r] = g[r]
                    parsed_k[r] = k[r]
                    parsed_req_paths[r] = req_paths[r]
                    parsed_rpl_paths[r] = rpl_paths[r]

                parsed_solution["pairs"] = parsed_pairs
                parsed_solution["g"] = parsed_g
                parsed_solution["k"] = parsed_k
                parsed_solution["req_paths"] = parsed_req_paths
                parsed_solution["rpl_paths"] = parsed_rpl_paths
                parsed_solution["info"] = status
                parsed_solution["OF"] = solution.get_objective_value()
                parsed_solution["done"] = False

                return parsed_solution
            except:
                parsed_solution["info"] = str(mdl.solve_details.status)
                return parsed_solution

    """
    def add_service_placement_action_constraint(self, mdl, g, action):
        mdl.add_constraint(g[action["req_id"], action["node_id"]] == 1)

        return mdl

    def add_assigned_nodes_constraints(self, mdl, g, assigned_nodes):
        for i in self.req_obj.REQUESTS:
            # print(f"{i} - {assigned_nodes[i]}")
            mdl.add_constraint(g[i, assigned_nodes[i]] == 1)

        return mdl

    def add_priority_assignment_action_constraint(self, mdl, rho, action):
        mdl.add_constraint(rho[action["req_id"], action["pri_id"]] == 1)

        return mdl
    """
