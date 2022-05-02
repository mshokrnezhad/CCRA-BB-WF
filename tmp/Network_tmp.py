import random
import math
import matplotlib.pyplot as plt
import numpy as np
rnd = np.random


class Network:
    def __init__(self, NUM_NODES, SEED=4, NUM_TIERS=3, TIER_HEIGHT=100, TIER_WIDTH=20,
                 DC_CAPACITY_UNIT=50, DC_COST_UNIT=50, LINK_BW_LB=100, LINK_BW_UB=150,
                 LINK_COST_LB=10, LINK_COST_UB=20, NUM_PRIORITY_LEVELS=1, BURST_SIZE_LIMIT=50,
                 PACKET_SIZE=10):
        self.NUM_NODES = NUM_NODES
        self.NUM_TIERS = NUM_TIERS
        self.TIER_HEIGHT = TIER_HEIGHT
        self.TIER_WIDTH = TIER_WIDTH
        rnd.seed(SEED)
        self.NODES = np.arange(NUM_NODES)
        self.X_LOCS = [rnd.randint(self.get_tier_num(i)*TIER_WIDTH,
                                   (self.get_tier_num(i)+1)*TIER_WIDTH) for i in self.NODES]
        self.Y_LOCS = rnd.rand(len(self.NODES))*TIER_HEIGHT
        self.DISTANCES = np.array([[np.hypot(self.X_LOCS[i]-self.X_LOCS[j], self.Y_LOCS[i]-self.Y_LOCS[j])
                                    for j in self.NODES] for i in self.NODES])
        self.DC_CAPACITIES = np.array([rnd.randint(self.get_tier_num(i)*DC_CAPACITY_UNIT,
                                                   (self.get_tier_num(i)+1)*DC_CAPACITY_UNIT)
                                       for i in self.NODES])
        self.DC_COSTS = np.array([rnd.randint((self.NUM_TIERS-self.get_tier_num(i)-1)*DC_COST_UNIT,
                                              (self.NUM_TIERS-self.get_tier_num(i))*DC_COST_UNIT)
                                  for i in self.NODES])
        self.LINKS = self.create_links()
        self.LINK_BW_LB = LINK_BW_LB
        self.LINK_BW_UB = LINK_BW_UB
        self.LINKS_BWS = self.get_links_bws()
        self.LINKS_COSTS = {(i, j): rnd.randint(
            LINK_COST_LB, LINK_COST_UB) for (i, j) in self.LINKS}
        self.PRIORITIES = [n for n in range(0, NUM_PRIORITY_LEVELS+1)]
        self.BURST_SIZE_LIMIT = BURST_SIZE_LIMIT
        self.PACKET_SIZE = PACKET_SIZE
        self.LINKS_BWS_CMLTV_L_PP = self.cumulative_link_bw_limit_per_priority()
        self.LINK_DELAYS = self.link_delays()


    def nodes(self):
        return self.NODES

    def first_tier_nodes(self):
        return [i for i in self.NODES if self.get_tier_num(i) == 0]

    def dc_capacities(self):
        return self.DC_CAPACITIES

    def dc_costs(self):
        return self.DC_COSTS

    def links(self):
        return self.LINKS

    def links_costs(self):
        return self.LINKS_COSTS

    def links_bws(self):
        return self.LINKS_BWS

    def distances(self):
        return self.DISTANCES

    def get_tier_num(self, i):
        tier_size = math.ceil(self.NUM_NODES / self.NUM_TIERS)

        for t in range(self.NUM_TIERS):
            if t * tier_size <= i <= (t + 1) * tier_size:
                tier_num = t

        return tier_num

    def is_j_neighbor_of_i(self, i, j):
        if abs(self.get_tier_num(i) - self.get_tier_num(j)) <= 1:
            close_neighbors = {k: self.DISTANCES[i, k] for k in self.NODES if self.get_tier_num(
                k) == self.get_tier_num(j) and k != i}
            if j == min(close_neighbors, key=close_neighbors.get):
                return True
            else:
                return False
        else:
            return False

    def create_links(self):
        temp_links = []
        for i in self.NODES:
            for j in self.NODES:
                if i != j and self.is_j_neighbor_of_i(i, j):
                    if (i, j) not in temp_links:
                        temp_links.append((i, j))
                    if (j, i) not in temp_links:
                        temp_links.append((j, i))
        return temp_links

    def make_links_undirected(self):
        temp_links = []
        for i, j in self.LINKS:
            if i < j:
                temp_links.append((i, j))
        return temp_links

    def get_services_of_each_DC(self, service_placement):
        service_placement_dict = {}
        for i in self.NODES:
            temp_string = ''
            for (s, j) in service_placement:
                if i == j:
                    temp_string += str(s) if temp_string == '' else '/' + str(s)
            service_placement_dict[i] = temp_string if temp_string != '' else '-'
        return service_placement_dict

    def get_active_DCs_Locations(self, service_placement_dict):
        active_DCs_X_LOCS = []
        active_DCs_Y_LOCS = []
        for i in self.NODES:
            if service_placement_dict[i] != '-':
                active_DCs_X_LOCS.append(self.X_LOCS[i])
                active_DCs_Y_LOCS.append(self.Y_LOCS[i])
        return active_DCs_X_LOCS, active_DCs_Y_LOCS

    def get_routers_locations(self, service_placement_dict):
        routers_X_LOCS = []
        routers_Y_LOCS = []
        for i in self.NODES:
            if service_placement_dict[i] == '-':
                routers_X_LOCS.append(self.X_LOCS[i])
                routers_Y_LOCS.append(self.Y_LOCS[i])
        return routers_X_LOCS, routers_Y_LOCS

    def plot(self, service_placement=[], srv_assignment=[]):
        service_placement_dict = self.get_services_of_each_DC(
            service_placement)
        active_DCs_X_LOCS, active_DCs_Y_LOCS = self.get_active_DCs_Locations(
            service_placement_dict)
        routers_X_LOCS, routers_Y_LOCS = self.get_routers_locations(
            service_placement_dict)

        plt.scatter(active_DCs_X_LOCS, active_DCs_Y_LOCS,
                    c='r', label='active DC')
        plt.scatter(routers_X_LOCS, routers_Y_LOCS,
                    c='b', label='router')

        for i in self.NODES:
            """ plt.annotate('$(%d, %d)$' %
                         (self.X_LOCS[i], self.Y_LOSC[i]),
                         (self.X_LOCS[i]+1, self.Y_LOCS[i]+4))
            plt.annotate('$CAP:%d$' %
                         (self.DC_CAPACITIES[i]),
                         (self.X_LOCS[i]+1, self.Y_LOCS[i]))
            plt.annotate('$COST:%d$' %
                         (self.DC_COSTS[i]),
                         (self.X_LOCS[i]+1, self.Y_LOCS[i]-4)) """
            plt.annotate('$%d$' % (i), (self.X_LOCS[i]+1, self.Y_LOCS[i]))
            plt.annotate(
                service_placement_dict[i], (self.X_LOCS[i]+1, self.Y_LOCS[i]-2))

        sample_node = random.choice(self.make_links_undirected())
        for i, j in self.make_links_undirected():
            plt.plot([self.X_LOCS[i], self.X_LOCS[j]], [
                     self.Y_LOCS[i], self.Y_LOCS[j]], c='g', label='link' if (i, j) == sample_node else None)
        """ plt.annotate('$%d-%d$' %
                         (self.LINKS_BWS[i, j], self.LINKS_COSTS[i, j]),
                         ((self.X_LOCS[i]+self.X_LOCS[j])/2, (self.Y_LOCS[i]+self.Y_LOCS[j])/2)) """

        plt.legend()
        plt.show()

    def priorities(self):
        return self.PRIORITIES

    def get_links_bws(self):
        links_bws = {}
        for i in self.NODES:
            for j in self.NODES:
                if i > j and (i, j) in self.LINKS:
                    rnd_bw = rnd.randint(self.LINK_BW_LB, self.LINK_BW_UB)
                    links_bws[(i, j)] = rnd_bw
                    links_bws[(j, i)] = rnd_bw
        return links_bws

    def burst_size_limit_per_priotity(self):
        return [(len(self.priorities())-i)*self.BURST_SIZE_LIMIT if i > 0 else 0
                for i in self.priorities()]

    def cumulative_burst_size_limit_per_priotity(self):
        return [np.array(self.burst_size_limit_per_priotity()[:i+1]).sum() for i in self.priorities()]

    def link_bw_limit_per_priority(self):
        links_bws = {}
        for i in self.NODES:
            for j in self.NODES:
                if i > j and (i, j) in self.LINKS:
                    for n in self.priorities():
                        if n > 0:
                            links_bws[(i, j), n] = (
                                (len(self.priorities())-n) /
                                np.array(self.priorities()).sum())*self.LINKS_BWS[i, j]
                            links_bws[(j, i), n] = (
                                (len(self.priorities())-n) /
                                np.array(self.priorities()).sum())*self.LINKS_BWS[i, j]
                        else:
                            links_bws[(i, j), n] = 0
                            links_bws[(j, i), n] = 0
        return links_bws

    def cumulative_link_bw_limit_per_priority(self):
        # print("...cumulative_link_bw_limit_per_priority...")
        cumulative_links_bws = {}
        for i in self.NODES:
            for j in self.NODES:
                if i > j and (i, j) in self.LINKS:
                    array = []
                    for n in self.priorities():
                        array.append(
                            self.link_bw_limit_per_priority()[(i, j), n])
                    for n in self.priorities():
                        cumulative_links_bws[(i, j), n] = np.array(array)[
                            :n].sum()
                        cumulative_links_bws[(j, i), n] = np.array(array)[
                            :n].sum()
        return cumulative_links_bws

    def link_delays(self):
        links_delays = {}
        for i in self.NODES:
            for j in self.NODES:
                if i > j and (i, j) in self.LINKS:
                    for n in self.priorities():
                        if n > 0:
                            delay = (self.cumulative_burst_size_limit_per_priotity()[n] + self.PACKET_SIZE) / (
                                self.LINKS_BWS[i, j]-self.LINKS_BWS_CMLTV_L_PP[(i, j), n])
                            + self.PACKET_SIZE/self.LINKS_BWS[i, j]
                            links_delays[(i, j), n] = round(delay, 3)
                            links_delays[(j, i), n] = round(delay, 3)
                        else:
                            links_delays[(i, j), n] = 1000
                            links_delays[(j, i), n] = 1000
        return links_delays

    """ def link_delays(self):
        links_delays = {}
        for i in self.NODES:
            for j in self.NODES:
                if i > j and (i, j) in self.LINKS:
                    for n in self.priorities():
                        if n > 0:
                            delay = 1
                            links_delays[(i, j), n] = round(delay, 3)
                            links_delays[(j, i), n] = round(delay, 3)
                        else:
                            links_delays[(i, j), n] = 1000
                            links_delays[(j, i), n] = 1000
        return links_delays """
