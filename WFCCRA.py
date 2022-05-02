# v1 for GlobeCom2022
import numpy as np


class WFCCRA:

    def __init__(self, net_obj, req_obj, srv_obj, REQUESTED_SERVICES, REQUESTS_ENTRY_NODES):
        self.net_obj = net_obj
        self.req_obj = req_obj
        self.srv_obj = srv_obj
        self.REQUESTED_SERVICES = REQUESTED_SERVICES
        self.REQUESTS_ENTRY_NODES = REQUESTS_ENTRY_NODES
        self.EPSILON = 0.001

    def solve(self):
        z = np.zeros((self.srv_obj.NUM_SERVICES, self.net_obj.NUM_NODES))
        g = np.zeros((self.req_obj.NUM_REQUESTS, self.net_obj.NUM_NODES))
        rho = np.zeros((self.req_obj.NUM_REQUESTS, len(self.net_obj.PRIORITIES)))
        req_path = np.zeros((self.req_obj.NUM_REQUESTS, len(self.net_obj.PATHS), len(self.net_obj.PRIORITIES)))
        rpl_path = np.zeros((self.req_obj.NUM_REQUESTS, len(self.net_obj.PATHS), len(self.net_obj.PRIORITIES)))

        DC_CAPACITIES = self.net_obj.DC_CAPACITIES
        LINK_BWS = self.net_obj.LINK_BWS
        LINK_BURSTS = np.array([self.net_obj.BURST_SIZE_LIMIT_PER_PRIORITY for l in self.net_obj.LINKS])
        # sorted_requests = self.sort_requests()
        sorted_requests = np.argsort(self.req_obj.DELAY_REQUIREMENTS)
        resources = {}
        costs = {}
        cost_details = {}

        for r in sorted_requests:
            resources_per_req = []
            costs_per_req = []
            cost_details_per_req = []
            for v in self.net_obj.NODES:
                if self.req_obj.CAPACITY_REQUIREMENTS[r] <= DC_CAPACITIES[v]:
                    z[self.REQUESTED_SERVICES[r]][v] = 1
                    g[r][v] = 1
                if z[self.REQUESTED_SERVICES[r]][v] == 1 and g[r][v] == 1:
                    for k in self.net_obj.PRIORITIES:
                        rho[r][k] = 1
                        for p1 in np.intersect1d(self.net_obj.PATHS_PER_HEAD[self.REQUESTS_ENTRY_NODES[r]], self.net_obj.PATHS_PER_TAIL[v]):
                            flag1 = True
                            for l1 in np.where(self.net_obj.LINKS_PATHS_MATRIX[:, p1] == 1)[0]:
                                if self.req_obj.BW_REQUIREMENTS[r] > LINK_BWS[l1]:
                                    flag1 = False
                                if self.req_obj.BURST_SIZES[r] > LINK_BURSTS[l1, k]:
                                    flag1 = False
                            if flag1:
                                req_path[r][p1][k] = 1
                                for p2 in np.intersect1d(self.net_obj.PATHS_PER_HEAD[v], self.net_obj.PATHS_PER_TAIL[self.REQUESTS_ENTRY_NODES[r]]):
                                    flag2 = True
                                    for l2 in np.where(self.net_obj.LINKS_PATHS_MATRIX[:, p2] == 1)[0]:
                                        if self.req_obj.BW_REQUIREMENTS[r] > LINK_BWS[l2] or self.req_obj.BURST_SIZES[r] > LINK_BURSTS[l2][k]:
                                            flag2 = False
                                    if flag2:
                                        rpl_path[r][p2][k] = 1
                                        d = 0
                                        c = 0
                                        req_paths_cost = 0
                                        rpl_paths_cost = 0
                                        for l1 in np.where(self.net_obj.LINKS_PATHS_MATRIX[:, p1] == 1)[0]:
                                            d += self.net_obj.LINK_DELAYS[l1][k]
                                        for l2 in np.where(self.net_obj.LINKS_PATHS_MATRIX[:, p2] == 1)[0]:
                                            d += self.net_obj.LINK_DELAYS[l2][k]
                                        d += self.net_obj.PACKET_SIZE / (self.net_obj.DC_CAPACITIES[v] + self.EPSILON)
                                        if d <= self.req_obj.DELAY_REQUIREMENTS[r]:
                                            c += self.net_obj.DC_COSTS[v]
                                            for l1 in np.where(self.net_obj.LINKS_PATHS_MATRIX[:, p1] == 1)[0]:
                                                c += self.net_obj.LINK_COSTS[l1]
                                                req_paths_cost += self.net_obj.LINK_COSTS[l1]
                                            for l2 in np.where(self.net_obj.LINKS_PATHS_MATRIX[:, p2] == 1)[0]:
                                                c += self.net_obj.LINK_COSTS[l2]
                                                rpl_paths_cost += self.net_obj.LINK_COSTS[l1]
                                            resources_per_req.append([v, k, p1, p2])
                                            costs_per_req.append(c)
                                            cost_details_per_req.append([req_paths_cost, rpl_paths_cost])
            min_index = np.array(costs_per_req).argmin()
            resources[r] = resources_per_req[min_index]
            costs[r] = costs_per_req[min_index]

            DC_CAPACITIES[resources[r][0]] = DC_CAPACITIES[resources[r][0]] - self.req_obj.CAPACITY_REQUIREMENTS[r]
            for l1 in np.where(self.net_obj.LINKS_PATHS_MATRIX[:, resources[r][2]] == 1)[0]:
                LINK_BWS[l1] = LINK_BWS[l1] - self.req_obj.BW_REQUIREMENTS[r]
                LINK_BURSTS[l1, resources[r][1]] = LINK_BURSTS[l1, resources[r][1]] - self.req_obj.BURST_SIZES[r]
            for l2 in np.where(self.net_obj.LINKS_PATHS_MATRIX[:, resources[r][3]] == 1)[0]:
                LINK_BWS[l2] = LINK_BWS[l2] - self.req_obj.BW_REQUIREMENTS[r]
                LINK_BURSTS[l2, resources[r][1]] = LINK_BURSTS[l2, resources[r][1]] - self.req_obj.BURST_SIZES[r]
            cost_details[r] = cost_details_per_req[min_index]

        parsed_solution = {"pairs": {}, "g": {}, "k": {}, "req_paths": {}, "rpl_paths": {}, "info": "No more info on WF-CCRA", "OF": 0, "done": None}
        parsed_pairs = {}
        parsed_g = {}
        parsed_k = {}
        parsed_req_paths = {}
        parsed_rpl_paths = {}

        OF = sum(costs.values())
        req_paths_cost = 0
        rpl_paths_cost = 0
        for r in self.req_obj.REQUESTS:
            parsed_g[r] = resources[r][0]
            parsed_pairs[r] = (self.REQUESTS_ENTRY_NODES[r], parsed_g[r])
            parsed_k[r] = resources[r][1]
            parsed_req_paths[r] = self.net_obj.PATHS_DETAILS[resources[r][2]]
            parsed_rpl_paths[r] = self.net_obj.PATHS_DETAILS[resources[r][3]]

        req_paths_cost += cost_details[2][0]
        rpl_paths_cost += cost_details[2][1]
        # print(req_paths_cost)
        # print(rpl_paths_cost)

        parsed_solution["pairs"] = parsed_pairs
        parsed_solution["g"] = parsed_g
        parsed_solution["k"] = parsed_k
        parsed_solution["req_paths"] = parsed_req_paths
        parsed_solution["rpl_paths"] = parsed_rpl_paths
        parsed_solution["OF"] = OF

        return parsed_solution

    def sort_requests(self):
        # sort request indexes in terms of delay requirements
        sorted_requirements = np.sort(self.req_obj.DELAY_REQUIREMENTS)
        sorted_requests = []

        for requirement in sorted_requirements:
            sorted_requests.append(np.where(self.req_obj.DELAY_REQUIREMENTS == requirement)[0][0])

        return sorted_requests
