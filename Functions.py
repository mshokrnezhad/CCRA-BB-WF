# from docplex.mp.model import Model
import numpy as np
import matplotlib.pyplot as plt
from Network import Network
from Request import Request
from Service import Service
from tqdm import tqdm

rnd = np.random


def specify_requests_entry_nodes(FIRST_TIER_NODES, REQUESTS, SEED=0):
    rnd.seed(SEED)
    return np.array([rnd.choice(FIRST_TIER_NODES) for i in REQUESTS])


def assign_requests_to_services(SERVICES, REQUESTS, SEED=0):
    rnd.seed(SEED)
    return np.array([rnd.choice(SERVICES) for i in REQUESTS])


def calculate_input_shape(NUM_NODES, NUM_REQUESTS, NUM_PRIORITY_LEVELS, switch):
    counter = 0

    if switch == "srv_plc":
        counter = (2 * NUM_REQUESTS) + (5 * NUM_NODES) + ((2 + NUM_PRIORITY_LEVELS) * (NUM_NODES ** 2))
    """
    if switch == "srv_plc":
        counter = (2 * NUM_REQUESTS) + (5 * NUM_NODES) + ((2) * (NUM_NODES ** 2))
    """
    if switch == "pri_asg":
        counter = (2 * NUM_REQUESTS) + (6 * NUM_NODES) + ((2 + NUM_PRIORITY_LEVELS) * (NUM_NODES ** 2))

    return counter


def parse_state(state, NUM_NODES, NUM_REQUESTS, env_obj, switch="srv_plc"):
    np.set_printoptions(suppress=True, linewidth=100)
    counter = 0

    print("\n^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")
    """
    print("ACTIVE REQUESTS:")
    print(state[counter:NUM_REQUESTS].astype(int))
    counter += NUM_REQUESTS

    print("\nPER NODE REQUEST CAPACITY REQUIREMENTS:")
    print(state[counter:counter + NUM_NODES].astype(int))
    counter += NUM_NODES

    print("\nPER NODE REQUEST BW REQUIREMENTS:")
    print(state[counter:counter + NUM_NODES].astype(int))
    counter += NUM_NODES

    print("\nREQUEST DELAY REQUIREMENTS:")
    print(state[counter:counter + NUM_REQUESTS].astype(int))
    counter += NUM_REQUESTS

    print("\nPER NODE REQUEST BURST SIZES:")
    print(state[counter:counter + NUM_NODES].astype(int))
    counter += NUM_NODES

    if switch == "pri_asg":
        print("\nPER ASSIGNED NODE REQUEST BW REQUIREMENTS:")
        print(state[counter:counter + NUM_NODES].astype(int))
        counter += NUM_NODES

    print("\nDC CAPACITIES:")
    print(state[counter:counter + NUM_NODES].astype(int))
    counter += NUM_NODES

    print("\nDC COSTS:")
    print(state[counter:counter + NUM_NODES].astype(int))
    counter += NUM_NODES

    print("\nLINK BWS MATRIX:")
    print(state[counter:counter + NUM_NODES ** 2].astype(int).reshape(NUM_NODES, NUM_NODES))
    counter += NUM_NODES ** 2

    print("\nLINK COSTS MATRIX:")
    print(state[counter:counter + NUM_NODES ** 2].astype(int).reshape(NUM_NODES, NUM_NODES))
    counter += NUM_NODES ** 2

    print("\nLINK DELAYS MATRIX:")
    link_delays_matrix = state[counter:counter + env_obj.net_obj.NUM_PRIORITY_LEVELS * (NUM_NODES ** 2)]. \
        reshape(env_obj.net_obj.NUM_PRIORITY_LEVELS, NUM_NODES, NUM_NODES)
    # since we removed null index 0, index 0 of link_delays_matrix is for priority 1 and so on.
    for n in range(0, env_obj.net_obj.NUM_PRIORITY_LEVELS):
        print(f"Priority: {n + 1}")
        print(link_delays_matrix[n])
    counter += env_obj.net_obj.NUM_PRIORITY_LEVELS * (NUM_NODES ** 2)
    """

    print("ACTIVE REQUESTS:")
    print(state[counter:NUM_REQUESTS])
    counter += NUM_REQUESTS

    print("\nPER NODE REQUEST CAPACITY REQUIREMENTS:")
    print(state[counter:counter + NUM_NODES])
    counter += NUM_NODES

    print("\nPER NODE REQUEST BW REQUIREMENTS:")
    print(state[counter:counter + NUM_NODES])
    counter += NUM_NODES

    print("\nREQUEST DELAY REQUIREMENTS:")
    print(state[counter:counter + NUM_REQUESTS])
    counter += NUM_REQUESTS

    print("\nPER NODE REQUEST BURST SIZES:")
    print(state[counter:counter + NUM_NODES])
    counter += NUM_NODES

    if switch == "pri_asg":
        print("\nPER ASSIGNED NODE REQUEST BW REQUIREMENTS:")
        print(state[counter:counter + NUM_NODES])
        counter += NUM_NODES

    print("\nDC CAPACITIES:")
    print(state[counter:counter + NUM_NODES])
    counter += NUM_NODES

    print("\nDC COSTS:")
    print(state[counter:counter + NUM_NODES])
    counter += NUM_NODES

    print("\nLINK BWS MATRIX:")
    print(state[counter:counter + NUM_NODES ** 2].reshape(NUM_NODES, NUM_NODES))
    counter += NUM_NODES ** 2

    print("\nLINK COSTS MATRIX:")
    print(state[counter:counter + NUM_NODES ** 2].reshape(NUM_NODES, NUM_NODES))
    counter += NUM_NODES ** 2

    print("\nLINK DELAYS MATRIX:")
    link_delays_matrix = state[counter:counter + env_obj.net_obj.NUM_PRIORITY_LEVELS * (NUM_NODES ** 2)]. \
        reshape(env_obj.net_obj.NUM_PRIORITY_LEVELS, NUM_NODES, NUM_NODES)
    # since we removed null index 0, index 0 of link_delays_matrix is for priority 1 and so on.
    for n in range(0, env_obj.net_obj.NUM_PRIORITY_LEVELS):
        print(f"Priority: {n + 1}")
        print(link_delays_matrix[n])
    counter += env_obj.net_obj.NUM_PRIORITY_LEVELS * (NUM_NODES ** 2)

    print("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n")


def plot_learning_curve(x, scores, epsilons, filename=""):
    fig = plt.figure()
    s_plt1 = fig.add_subplot(111, label="1")  # "234" means "2x3 grid, 4th subplot".
    s_plt2 = fig.add_subplot(111, label="2", frame_on=False)

    s_plt1.plot(x, epsilons, color="C0")
    s_plt1.set_xlabel("Training Steps", color="C0")
    s_plt1.set_ylabel("Epsilon", color="C0")
    s_plt1.tick_params(axis="x", color="C0")
    s_plt1.tick_params(axis="y", color="C0")

    n = len(scores)
    running_avg = np.empty(n)
    for i in range(n):
        running_avg[i] = np.mean(scores[max(0, i - 100):(i + 1)])

    s_plt2.scatter(x, running_avg, color="C1")
    s_plt2.axes.get_xaxis().set_visible(False)
    s_plt2.yaxis.tick_right()
    s_plt2.set_ylabel('Score', color="C1")
    s_plt2.yaxis.set_label_position('right')
    s_plt2.tick_params(axis='y', colors="C1")

    # plt.show()
    plt.savefig(filename)


def filter_paths_per_entry_nodes(paths_details, entry_nodes, requests):
    paths_per_entry_nodes = np.zeros((len(requests), len(paths_details)))

    for r in requests:
        for p in range(len(paths_details)):
            if entry_nodes[r] == paths_details[p][0]:
                paths_per_entry_nodes[r][p] = 1

    paths_per_entry_nodes = paths_per_entry_nodes.astype(int)
    return paths_per_entry_nodes


def create_system(NUM_NODES, NUM_PRIORITY_LEVELS, NUM_REQUESTS, NUM_SERVICES, SEED):
    # print("creating system...")
    # print("creating net_obj...")
    net_obj = Network(NUM_NODES, NUM_PRIORITY_LEVELS, SEED=SEED)
    REQUESTS_ENTRY_NODES = specify_requests_entry_nodes(net_obj.FIRST_TIER_NODES, np.arange(NUM_REQUESTS), SEED=SEED)
    # print("creating req_obj...")
    req_obj = Request(NUM_REQUESTS, net_obj.NODES, REQUESTS_ENTRY_NODES, SEED=SEED)
    # print("creating srv_obj...")
    srv_obj = Service(NUM_SERVICES, SEED=SEED)
    REQUESTED_SERVICES = assign_requests_to_services(np.arange(NUM_SERVICES), np.arange(NUM_REQUESTS), SEED=SEED)

    return net_obj, req_obj, srv_obj, REQUESTS_ENTRY_NODES, REQUESTED_SERVICES


def create_feasible_seeds_set(NUM_NODES, NUM_PRIORITY_LEVELS, NUM_REQUESTS, NUM_SERVICES, SET_LENGTH):

    f = open("SEEDS_SET.txt", "w")
    SEED_SET = []
    SEED = 0

    for i in tqdm(range(SET_LENGTH)):
        flag = False
        while not flag:
            SEED += 1
            try:
                net_obj, req_obj, srv_obj, REQUESTS_ENTRY_NODES, REQUESTED_SERVICES = create_system(NUM_NODES, NUM_PRIORITY_LEVELS, NUM_REQUESTS, NUM_SERVICES, SEED)
                SEED_SET.append(SEED)
                # print(SEED)
                f.write(str(SEED) + "\n")
                flag = True
            except:
                pass
    f.close()
    # print(SEED_SET)


def read_seeds_set(FILENAME="SEEDS_SET.txt"):
    f = open(FILENAME, "r")  # opens the file in read mode
    seeds = f.read().splitlines()  # puts the file into an array
    f.close()
    return [int(seed) for seed in seeds]
