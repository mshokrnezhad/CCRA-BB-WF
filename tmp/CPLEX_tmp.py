from docplex.mp.model import Model
from Network_tmp import Network
from Service_tmp import Service
from Request_tmp import Request
from Functions_tmp import specify_requests_entry_nodes
from Functions_tmp import assign_requests_to_services
import random
# from functions import subplot
# import numpy as np
# import matplotlib.pyplot as plt
# import cplex
# import sys
# old_stdout = sys.stdout
# log_file = open("log.txt","w")
# sys.stdout = log_file

def Solver_tmp(net_obj1, req_obj1, srv_obj1, requested_services1, requests_entry_nodes1):
    rnd = random.randint(1, 1000)
    print("Temp Model:")

    #print("Building network on " + str(rnd) + "...")
    net_obj = Network(9, 4)
    #print("Building services...")
    srv_obj = Service(3, 4)
    #print("Building requests...")
    req_obj = Request(4, 4)
    #print("Building inputs...")
    seed = 0
    requests_entry_nodes = specify_requests_entry_nodes(net_obj, req_obj, seed)
    seed = 1
    requested_services = assign_requests_to_services(srv_obj, req_obj, seed)
    print(net_obj.link_bw_limit_per_priority())
    M = 10 ** 6
    OF_WEIGHTS = [1, 0, 1, 100]
    Z = [(s, j) for s in srv_obj.services() for j in net_obj.nodes()]
    G = [(i, j) for i in req_obj.requests() for j in net_obj.nodes()]
    F = [(i, (j, m)) for i in req_obj.requests() for (j, m) in net_obj.links()]
    P = [(i, n) for i in req_obj.requests() for n in net_obj.priorities()]
    PP = [(i, (j, m), n) for i in req_obj.requests()
          for (j, m) in net_obj.links() for n in net_obj.priorities()]
    PL = [((j, m), n) for (j, m) in net_obj.links() for n in net_obj.priorities()]
    LINK_DELAYS = net_obj.LINK_DELAYS
    epsilon = 0.001

    #print("Solving the problem is started...")
    mdl = Model('RASP')
    z = mdl.binary_var_dict(Z, name='z')
    g = mdl.binary_var_dict(G, name='g')
    g_sup = mdl.binary_var_dict(G, name='g_sup')
    req_flw = mdl.continuous_var_dict(PP, lb=0, name='req_flw')
    res_flw = mdl.continuous_var_dict(PP, lb=0, name='res_flw')
    flw = mdl.continuous_var_dict([(j, m) for j in net_obj.nodes() for m in net_obj.nodes()
                                   if (j, m) in net_obj.links()], lb=0, name='flw')
    p = mdl.binary_var_dict(P, name='p')
    req_pp = mdl.binary_var_dict(PP, name='req_pp')
    res_pp = mdl.binary_var_dict(PP, name='res_pp')
    req_d = mdl.continuous_var_dict([i for i in req_obj.requests()], name='req_d')
    res_d = mdl.continuous_var_dict([i for i in req_obj.requests()], name='res_d')
    d = mdl.continuous_var_dict([i for i in req_obj.requests()], name='d')

    mdl.minimize(mdl.sum(z[s, j] for s, j in Z) * OF_WEIGHTS[0]
                 +
                 mdl.sum(mdl.sum(g[i, j] for i in req_obj.requests())
                         * net_obj.dc_costs()[j] for j in net_obj.nodes()) * OF_WEIGHTS[1]
                 +
                 mdl.sum(mdl.sum(flw[j, m]) * net_obj.links_costs()[j, m]
                         for (j, m) in net_obj.links()) * OF_WEIGHTS[2]
                 +
                 mdl.sum(d[i] for i in req_obj.requests()) * OF_WEIGHTS[3]
                 )

    mdl.add_constraints(
        mdl.sum(g[i, j] for j in net_obj.nodes()) == 1 for i in req_obj.requests())
    mdl.add_constraints(g[i, j] <= z[s, j] for i in req_obj.requests() for j in net_obj.nodes()
                        for s in srv_obj.services() if s == requested_services[i])
    mdl.add_constraints(mdl.sum(g[i, j] * req_obj.capacity_rquirements()[i]
                                for i in req_obj.requests()) <= net_obj.dc_capacities()[j] for j in net_obj.nodes())
    mdl.add_constraints(g_sup[i, j] == 1 - g[i, j]
                        for i in req_obj.requests() for j in net_obj.nodes())
    mdl.add_constraint(g[0, 7] == 1)
    mdl.add_constraint(g[1, 8] == 1)

    mdl.add_constraints(
        mdl.sum(p[i, n] for n in net_obj.priorities()) == 1 for i in req_obj.requests())

    mdl.add_constraints(mdl.sum(req_flw[i, (j, m), n] for n in net_obj.priorities()
                                for j in net_obj.nodes() for m in net_obj.nodes()
                                if j == requests_entry_nodes[i] and (j, m) in net_obj.links())
                        >= req_obj.bw_rquirements()[i] for i in req_obj.requests())
    mdl.add_constraints(mdl.sum(req_flw[i, (j, m), n] for n in net_obj.priorities()
                                for m in net_obj.nodes() for j in net_obj.nodes()
                                if j != requests_entry_nodes[i] and (j, m) in net_obj.links())
                        >= 0 for i in req_obj.requests())
    mdl.add_indicator_constraints(
        mdl.indicator_constraint(g[i, j], mdl.sum(req_flw[i, (m, j), n] for n in net_obj.priorities()
                                                  for m in net_obj.nodes() if (m, j) in net_obj.links())
                                 >= req_obj.bw_rquirements()[i]) for i in req_obj.requests() for j in net_obj.nodes())
    mdl.add_indicator_constraints(
        mdl.indicator_constraint(g_sup[i, j], mdl.sum(req_flw[i, (m, j), n] for n in net_obj.priorities()
                                                      for m in net_obj.nodes() if (m, j) in net_obj.links()) >= 0)
        for i in req_obj.requests() for j in net_obj.nodes())
    mdl.add_indicator_constraints(
        mdl.indicator_constraint(g_sup[i, j], mdl.sum(req_flw[i, (x, j), n] for n in net_obj.priorities()
                                                      for x in net_obj.nodes() if (x, j) in net_obj.links()) ==
                                 mdl.sum(req_flw[i, (j, m), n] for n in net_obj.priorities()
                                         for m in net_obj.nodes() if (j, m) in net_obj.links()))
        for i in req_obj.requests() for j in net_obj.nodes() if j != requests_entry_nodes[i])
    mdl.add_constraints(req_flw[i, (j, m), n] <= p[i, n] * M for i in req_obj.requests() for n in net_obj.priorities()
                        for j in net_obj.nodes() for m in net_obj.nodes() if (j, m) in net_obj.links())

    mdl.add_constraints(mdl.sum(res_flw[i, (m, j), n] for n in net_obj.priorities()
                                for m in net_obj.nodes() for j in net_obj.nodes()
                                if j == requests_entry_nodes[i] if (m, j) in net_obj.links())
                        >= req_obj.bw_rquirements()[i] for i in req_obj.requests())
    mdl.add_constraints(mdl.sum(res_flw[i, (m, j), n] for n in net_obj.priorities()
                                for m in net_obj.nodes() for j in net_obj.nodes()
                                if j != requests_entry_nodes[i] if (m, j) in net_obj.links())
                        >= 0 for i in req_obj.requests())
    mdl.add_indicator_constraints(
        mdl.indicator_constraint(g[i, j], mdl.sum(res_flw[i, (j, m), n] for n in net_obj.priorities()
                                                  for m in net_obj.nodes() if (j, m) in net_obj.links()) >=
                                 req_obj.bw_rquirements()[i])
        for i in req_obj.requests() for j in net_obj.nodes())
    mdl.add_indicator_constraints(
        mdl.indicator_constraint(g_sup[i, j], mdl.sum(res_flw[i, (j, m), n] for n in net_obj.priorities()
                                                      for m in net_obj.nodes() if (j, m) in net_obj.links()) >= 0)
        for i in req_obj.requests() for j in net_obj.nodes())
    mdl.add_indicator_constraints(
        mdl.indicator_constraint(g_sup[i, j], mdl.sum(res_flw[i, (j, x), n] for n in net_obj.priorities()
                                                      for x in net_obj.nodes() if (j, x) in net_obj.links()) ==
                                 mdl.sum(res_flw[i, (m, j), n] for n in net_obj.priorities() for m in net_obj.nodes()
                                         if (m, j) in net_obj.links()))
        for i in req_obj.requests() for j in net_obj.nodes() if j != requests_entry_nodes[i])
    mdl.add_constraints(res_flw[i, (j, m), n] <= p[i, n] * M for i in req_obj.requests() for n in net_obj.priorities()
                        for j in net_obj.nodes() for m in net_obj.nodes() if (j, m) in net_obj.links())

    mdl.add_constraints(
        flw[j, m] == mdl.sum(req_flw[i, (j, m), n] for n in net_obj.priorities() for i in req_obj.requests()) +
        mdl.sum(res_flw[i, (j, m), n] for n in net_obj.priorities()
                for i in req_obj.requests())
        for j in net_obj.nodes() for m in net_obj.nodes() if (j, m) in net_obj.links())
    mdl.add_constraints(flw[j, m] + flw[m, j] <= net_obj.links_bws()[j, m] for j in net_obj.nodes()
                        for m in net_obj.nodes() if j < m and (j, m) in net_obj.links())

    mdl.add_constraints(req_pp[i, (j, m), n] >= req_flw[i, (j, m), n] / net_obj.links_bws()[j, m]
                        for j in net_obj.nodes() for m in net_obj.nodes() if (j, m) in net_obj.links()
                        for i in req_obj.requests() for n in net_obj.priorities())
    mdl.add_constraints(req_pp[i, (j, m), n] <= req_flw[i, (j, m), n] for j in net_obj.nodes()
                        for m in net_obj.nodes() if (j, m) in net_obj.links() for i in req_obj.requests()
                        for n in net_obj.priorities())
    """ mdl.add_constraints(mdl.sum(req_pp[i, (j, m), n] for n in net_obj.priorities()) <= 1
                        for j in net_obj.nodes() for m in net_obj.nodes() if (j, m) in net_obj.links() for i in req_obj.requests()) """

    mdl.add_constraints(res_pp[i, (j, m), n] >= res_flw[i, (j, m), n] / net_obj.links_bws()[j, m]
                        for j in net_obj.nodes() for m in net_obj.nodes() if (j, m) in net_obj.links()
                        for i in req_obj.requests() for n in net_obj.priorities())
    mdl.add_constraints(res_pp[i, (j, m), n] <= res_flw[i, (j, m), n] for j in net_obj.nodes()
                        for m in net_obj.nodes() if (j, m) in net_obj.links() for i in req_obj.requests()
                        for n in net_obj.priorities())
    """ mdl.add_constraints(mdl.sum(res_pp[i, (j, m), n] for n in net_obj.priorities()) <= 1
                        for j in net_obj.nodes()
                        for m in net_obj.nodes() if (j, m) in net_obj.links() for i in req_obj.requests()) """

    mdl.add_constraints(req_d[i] == mdl.sum(mdl.sum(req_pp[i, (j, m), n] * LINK_DELAYS[(j, m), n]
                                                    for n in net_obj.priorities())
                                            for j in net_obj.nodes() for m in net_obj.nodes()
                                            if (j, m) in net_obj.links()) for i in req_obj.requests())
    mdl.add_constraints(res_d[i] == mdl.sum(mdl.sum(res_pp[i, (j, m), n] * LINK_DELAYS[(j, m), n]
                                                    for n in net_obj.priorities())
                                            for j in net_obj.nodes() for m in net_obj.nodes()
                                            if (j, m) in net_obj.links()) for i in req_obj.requests())
    mdl.add_constraints(d[i] == req_d[i] + res_d[i] + mdl.sum(g[i, j] * net_obj.PACKET_SIZE / (
            net_obj.DC_CAPACITIES[j] + epsilon) for j in net_obj.nodes()) for i in req_obj.requests())

    mdl.add_constraints(mdl.sum((req_pp[i, (j, m), n]) * req_obj.burst_sizes()[i] for i in req_obj.requests())
                        <= net_obj.burst_size_limit_per_priotity()[n] for n in net_obj.priorities()
                        for j in net_obj.nodes() for m in net_obj.nodes() if (j, m) in net_obj.links())
    mdl.add_constraints(mdl.sum((res_pp[i, (j, m), n]) * req_obj.burst_sizes()[i] for i in req_obj.requests())
                        <= net_obj.burst_size_limit_per_priotity()[n] for n in net_obj.priorities()
                        for j in net_obj.nodes() for m in net_obj.nodes() if (j, m) in net_obj.links())
    mdl.add_constraints(mdl.sum(req_flw[i, (j, m), n] + res_flw[i, (j, m), n] + req_flw[i, (m, j), n] +
                                res_flw[i, (m, j), n] for i in req_obj.requests()) <=
                        net_obj.link_bw_limit_per_priority()
                        [(j, m), n] for n in net_obj.priorities()
                        for j in net_obj.nodes() for m in net_obj.nodes() if (j, m) in net_obj.links())

    # mdl.parameters.timelimit = 60
    #mdl.log_output = True
    solution = mdl.solve()
    x = mdl.has_objective()
    y = mdl._objective_value()
    print(y)

    """service_placement = [(s, j) for (s, j) in Z if z[s, j].solution_value > 0.9]
    service_assignment = [(i, j) for (i, j) in G if g[i, j].solution_value > 0.9]
    net_obj.plot(service_placement) """

    # achieved_delays = [d[i].solution_value for i in req_obj.requests()]
    # fig = plt.figure(constrained_layout=True, figsize=(14, 7))
    # subfigs = fig.subfigures(nrows=2, ncols=1)
    # for row, subfig in enumerate(subfigs):
    #     """ subfig.suptitle('Request ID: %d, Request Entry Node: %d, Requested Service: %d, Service Place: %d, Required BW: %d, Required Capacity: %d, Required Minimum Delay: %d, \n Assigned Priority: %d, Achieved Delay: %f' % (
    #         row, requests_entry_nodes[row], requested_services[row], requested_service, req_obj.bw_rquirements()[row], req_obj.capacity_rquirements()[row], req_obj.delay_rquirements()[row], assigned_priority, achieved_delays[row]))
    #     """
    #     """ if row == 0:
    #         subfig.suptitle('Request Path')
    #     if row == 1:
    #         subfig.suptqitle('Reply Path') """
    #
    #     axs = subfig.subplots(nrows=1, ncols=3)
    #     for col, ax in enumerate(axs):
    #         requested_service = [
    #             j for (s, j) in G if s == col and g[s, j].solution_value > 0.9][0]
    #         assigned_priority = [
    #             n for (i, n) in P if i == col and p[i, n].solution_value > 0.9][0]
    #         if col == 0:
    #             service_color = 'violet'
    #             link_color = 'g'
    #         else:
    #             service_color = 'darkblue'
    #             link_color ='lightgreen'
    #         if row == 0:
    #             subplot(col, requests_entry_nodes[col], requested_services[col], net_obj.X_LOCS, net_obj.Y_LOCS,
    #                     g, G, net_obj.NODES, PP, req_flw, net_obj.LINKS, ax, service_color, link_color)
    #             #ax.set_title("Path from Entry Node to Service Place")
    #         else:
    #             subplot(col, requests_entry_nodes[col], requested_services[col], net_obj.X_LOCS, net_obj.Y_LOCS,
    #                     g, G, net_obj.NODES, PP, res_flw, net_obj.LINKS, ax, service_color, link_color)
    #             #ax.set_title("Path from Service Place to Entry Node")
    # plt.show()

    # sys.stdout = old_stdout
    # log_file.close()

    print("\n")
