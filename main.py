from Network import Network
from Request import Request
from Service import Service
from Functions import specify_requests_entry_nodes, assign_requests_to_services, filter_paths_per_entry_nodes, create_system, create_feasible_seeds_set, read_seeds_set
from CPLEX import CPLEX
from WFCCRA import WFCCRA
import numpy as np
import random

NUM_NODES = 20
NUM_PRIORITY_LEVELS = 4
NUM_REQUESTS = 10
NUM_REQUESTS_LB = 10
NUM_REQUESTS_UB = 150
NUM_SERVICES = 5
SWITCH = "none"
SAMPLE_COUNT = 10

# create_feasible_seeds_set(NUM_NODES, NUM_PRIORITY_LEVELS, NUM_REQUESTS, NUM_SERVICES, 1000)
SEEDS = read_seeds_set()

f_BB = open("BB.txt", "w")
f_WF = open("WF.txt", "w")
f_index = open("index.txt", "w")

for NUM_REQUESTS in range(NUM_REQUESTS_LB, NUM_REQUESTS_UB+1):
    itr = 0
    while itr < SAMPLE_COUNT:
        f_index.write(str(NUM_REQUESTS) + "\n")
        SEED = np.random.choice(SEEDS)
        print("-------------------------------")
        print(NUM_REQUESTS, itr, SEED)

        try:
            net_obj, req_obj, srv_obj, REQUESTS_ENTRY_NODES, REQUESTED_SERVICES = create_system(NUM_NODES, NUM_PRIORITY_LEVELS, NUM_REQUESTS, NUM_SERVICES, SEED)

            print("creating WF_CCRA_obj...")
            WF_CCRA_obj = WFCCRA(net_obj, req_obj, srv_obj, REQUESTED_SERVICES, REQUESTS_ENTRY_NODES)
            print("running WF-CCRA...")
            result = WF_CCRA_obj.solve()
            print("***   OF:", result["OF"])
            f_WF.write(str(result["OF"]) + "\n")
            # print("pairs:", result["pairs"])
            # print("k:", result["k"])
            # print("req_paths:", result["req_paths"])
            # print("rpl_paths:", result["rpl_paths"])

            net_obj, req_obj, srv_obj, REQUESTS_ENTRY_NODES, REQUESTED_SERVICES = create_system(NUM_NODES, NUM_PRIORITY_LEVELS, NUM_REQUESTS, NUM_SERVICES, SEED)

            print("creating CPLEX_obj...")
            BB_CCRA_obj = CPLEX(net_obj, req_obj, srv_obj, REQUESTED_SERVICES, REQUESTS_ENTRY_NODES)
            print("running B&B-CCRA...")
            optimum_result = BB_CCRA_obj.solve({}, SWITCH)
            # Solver(net_obj, req_obj, srv_obj, REQUESTED_SERVICES, REQUESTS_ENTRY_NODES)
            print("***   OF:", optimum_result["OF"])
            f_BB.write(str(optimum_result["OF"]) + "\n")
            # print("pairs:", optimum_result["pairs"])
            # print("k:", optimum_result["k"])
            # print("req_paths:", optimum_result["req_paths"])
            # print("rpl_paths:", optimum_result["rpl_paths"])

            itr += 1
        except:
            """
            f_BB.write(str(0) + "\n")
            f_WF.write(str(0) + "\n")
            """
            print("the sample is infeasible :(")

f_index.close()
f_BB.close()
f_WF.close()

"""
print("\n\n\n\n\n\n\n\n\n\n\n\n")
print(net_obj.PATHS_DETAILS.index([1, 2, 4]))
print(np.where(net_obj.LINKS_PATHS_MATRIX[:, 17] == 1)[0])
print(net_obj.LINK_BWS)
print(req_obj.BW_REQUIREMENTS[2])

print(net_obj.LINK_COSTS_DICT[(9,6)])
print(net_obj.LINK_COSTS_DICT[(6,0)])

print(net_obj.PATHS_DETAILS.index([9, 6, 0]))
print(net_obj.PATHS_DETAILS[163])

print(net_obj.LINKS_LIST[16])
print(net_obj.LINKS_LIST[21])
print(net_obj.LINK_COSTS[16])
print(net_obj.LINK_COSTS[21])

p = 163

for l in net_obj.LINKS:
    if net_obj.LINKS_PATHS_MATRIX[l, p] == 1:
        print(l)
        print(net_obj.LINK_COSTS[l])
"""

"""
pairs_diff_count = 0
priority_diff_count = 0
req_paths_diff_count = 0
rpl_paths_diff_count = 0
for r in req_obj.REQUESTS:
    if optimum_result["req_paths"][r] != result["req_paths"][r]:
        req_paths_diff_count += 1
    if optimum_result["rpl_paths"][r] != result["rpl_paths"][r]:
        rpl_paths_diff_count += 1
    if optimum_result["pairs"][r] != result["pairs"][r]:
        pairs_diff_count += 1
    if optimum_result["k"][r] != result["k"][r]:
        priority_diff_count += 1
        
print("\npairs_diff_count", ":", pairs_diff_count)
print("req_paths_diff_count", ":", req_paths_diff_count)
print("rpl_paths_diff_count", ":", rpl_paths_diff_count)
print("priority_diff_count", ":", priority_diff_count)

srv_plc_obj = Service_Placement(NUM_NODES, NUM_REQUESTS, NUM_SERVICES, NUM_PRIORITY_LEVELS)
srv_plc_obj.train()
srv_plc_obj.test(100)

srv_plc_q = srv_plc_obj.agent.extract_model()

pri_asg_obj = Priority_Assignment(NUM_NODES, NUM_REQUESTS, NUM_SERVICES, NUM_PRIORITY_LEVELS)
pri_asg_obj.train(srv_plc_q)
"""
