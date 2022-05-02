import math
import matplotlib.pyplot as plt
import numpy as np
rnd = np.random


class Request:
    def __init__(self, NUM_REQUESTS, SEED=0, CAPACITY_REQUIREMENT_LB=5, CAPACITY_REQUIREMENT_UB=10,
                 BW_REQUIREMENT_LB=5, BW_REQUIREMENT_UB=10, DLY_REQUIREMENT_LB=10, DLY_REQUIREMENT_UB=20,
                 BURST_SIZE_LB=5, BURST_SIZE_UB=6):
        self.NUM_REQUESTS = NUM_REQUESTS
        self.REQUESTS = np.arange(NUM_REQUESTS)
        self.CAPACITY_REQUIREMENTS = np.array(
            [rnd.randint(CAPACITY_REQUIREMENT_LB, CAPACITY_REQUIREMENT_UB) for i in self.REQUESTS])
        self.BW_REQUIREMENTS = np.array(
            [rnd.randint(BW_REQUIREMENT_LB, BW_REQUIREMENT_UB) for i in self.REQUESTS])
        self.DLY_REQUIREMENTS = np.array(
            [rnd.randint(DLY_REQUIREMENT_LB, DLY_REQUIREMENT_UB) for i in self.REQUESTS])
        self.BURST_SIZES = np.array(
            [rnd.randint(BURST_SIZE_LB, BURST_SIZE_UB) for i in self.REQUESTS])

    def requests(self):
        return self.REQUESTS

    def capacity_rquirements(self):
        return self.CAPACITY_REQUIREMENTS

    def bw_rquirements(self):
        return self.BW_REQUIREMENTS

    def delay_rquirements(self):
        return self.DLY_REQUIREMENTS

    def burst_sizes(self):
        return self.BURST_SIZES
