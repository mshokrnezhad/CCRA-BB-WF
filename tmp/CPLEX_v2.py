from docplex.mp.model import Model
import numpy as np

def Solver(net_obj, req_obj, srv_obj, REQUESTED_SERVICES, REQUESTS_ENTRY_NODES):

    M = 10 ** 6
    Z = [(s, v) for s in srv_obj.SERVICES for v in net_obj.NODES]
    G = [(r, v) for r in req_obj.REQUESTS for v in net_obj.NODES]
    P = [(r, p, k) for r in req_obj.REQUESTS for p in net_obj.PATHS for k in net_obj.PRIORITIES]
    Rho = [(r, k) for r in req_obj.REQUESTS for k in net_obj.PRIORITIES]
    LINK_DELAYS = net_obj.LINK_DELAYS_DICT
    EPSILON = 0.001

    mdl = Model('CCRA')
    z = mdl.binary_var_dict(Z, name='z')
    g = mdl.binary_var_dict(G, name='g')
    req_path = mdl.binary_var_dict(P, name='req_path')
    rpl_path = mdl.binary_var_dict(P, name='rpl_path')
    rho = mdl.binary_var_dict(Rho, name='rho')
    d = mdl.continuous_var_dict([r for r in req_obj.REQUESTS], name='d')

    # region OF
    print("defining OF...")
    mdl.minimize(
        mdl.sum(
            mdl.sum(
                g[r, v]
                for r in req_obj.REQUESTS
            ) * net_obj.DC_COSTS[v]
            for v in net_obj.NODES
        )
        +
        mdl.sum(
            mdl.sum(
                ((req_path[r, p, k] + rpl_path[r, p, k]) * net_obj.LINKS_PATHS_MATRIX[l, p])
                for p in net_obj.PATHS
                for k in net_obj.PRIORITIES
            ) * net_obj.LINK_COSTS[l]
            for l in net_obj.LINKS
            for r in req_obj.REQUESTS
        )
        # just to control z. it could be removed without changing the result
        +
        mdl.sum(
            z[s, v]
            for s in srv_obj.SERVICES
            for v in net_obj.NODES
        )
        # minimize the sum of delays just to test
        # + mdl.sum(d[r] for r in req_obj.REQUESTS) * 1000
    )
    # endregion

    # region C1
    print("defining C1...")
    mdl.add_constraints(
        mdl.sum(
            g[r, v]
            for v in net_obj.NODES
        ) == 1
        for r in req_obj.REQUESTS
    )
    # endregion

    # region C2
    print("defining C2...")
    mdl.add_constraints(
        g[r, v]
        <=
        z[s, v]
        for r in req_obj.REQUESTS
        for v in net_obj.NODES
        for s in srv_obj.SERVICES
        if s == REQUESTED_SERVICES[r]
    )
    # endregion

    # region C3 & C4
    print("defining C3 & C4...")
    mdl.add_constraints(
        mdl.sum(
            g[r, v] * req_obj.CAPACITY_REQUIREMENTS[r]
            for r in req_obj.REQUESTS
        )
        <=
        net_obj.DC_CAPACITIES[v]
        for v in net_obj.NODES
    )
    # endregion

    # region C5
    print("defining C5...")
    mdl.add_constraints(
        mdl.sum(
            rho[r, k]
            for k in net_obj.PRIORITIES
        )
        == 1
        for r in req_obj.REQUESTS
    )
    # endregion

    # region C6
    print("defining C6...")
    mdl.add_constraints(
        mdl.sum(
            req_path[r, p, k]
            for k in net_obj.PRIORITIES
            for p in net_obj.PATHS
            if net_obj.PATHS_DETAILS[p][0] == REQUESTS_ENTRY_NODES[r] and net_obj.PATHS_DETAILS[p][-1] == v
        )
        ==
        g[r, v]
        for r in req_obj.REQUESTS
        for v in net_obj.NODES
    )
    # endregion

    # region C7
    print("defining C7...")
    mdl.add_constraints(
        mdl.sum(
            rpl_path[r, p, k]
            for k in net_obj.PRIORITIES
            for p in net_obj.PATHS
            if net_obj.PATHS_DETAILS[p][0] == v and net_obj.PATHS_DETAILS[p][-1] == REQUESTS_ENTRY_NODES[r]
        )
        ==
        g[r, v]
        for r in req_obj.REQUESTS
        for v in net_obj.NODES
    )
    # endregion

    # region C8
    print("defining C8...")
    mdl.add_constraints(
        mdl.sum(
            req_path[r, p, k]
            for p in net_obj.PATHS
        )
        == rho[r, k]
        for r in req_obj.REQUESTS
        for k in net_obj.PRIORITIES
    )
    # endregion

    # region C9
    print("defining C9...")
    mdl.add_constraints(
        mdl.sum(
            rpl_path[r, p, k]
            for p in net_obj.PATHS
        )
        == rho[r, k]
        for r in req_obj.REQUESTS
        for k in net_obj.PRIORITIES
    )
    # endregion

    # region C10
    print("defining C10...")
    mdl.add_constraints(
        mdl.sum(
            mdl.sum(
                ((req_path[r, p, k] + rpl_path[r, p, k]) * net_obj.LINKS_PATHS_MATRIX[l, p])
                for p in net_obj.PATHS
                for k in net_obj.PRIORITIES
            ) * req_obj.BW_REQUIREMENTS[r]
            for r in req_obj.REQUESTS
        )
        <=
        net_obj.LINK_BWS[l]
        for l in net_obj.LINKS
    )
    # endregion

    # region C11
    print("defining C11...")
    mdl.add_constraints(
        mdl.sum(
            mdl.sum(
                ((req_path[r, p, k] + rpl_path[r, p, k]) * net_obj.LINKS_PATHS_MATRIX[l, p])
                for p in net_obj.PATHS
            ) * req_obj.BURST_SIZES[r]
            for r in req_obj.REQUESTS
        )
        <=
        net_obj.BURST_SIZE_LIMIT_PER_PRIORITY[k]
        for l in net_obj.LINKS
        for k in net_obj.PRIORITIES
    )
    # endregion

    # region C13'
    print("defining C13'...")
    mdl.add_constraints(
        mdl.sum(
            mdl.sum(
                ((req_path[r, p, k] + rpl_path[r, p, k]) * net_obj.LINKS_PATHS_MATRIX[l, p])
                for p in net_obj.PATHS
            ) * req_obj.BW_REQUIREMENTS[r]
            for r in req_obj.REQUESTS
        )
        <=
        net_obj.LINK_BWS_LIMIT_PER_PRIORITY[l, k]
        for l in net_obj.LINKS
        for k in net_obj.PRIORITIES
    )
    # endregion

    # region C14'
    print("defining C14'...")
    mdl.add_constraints(
        mdl.sum(
            mdl.sum(
                (((req_path[r, p, k] + rpl_path[r, p, k]) * net_obj.LINKS_PATHS_MATRIX[l, p]) * net_obj.LINK_DELAYS[l][k])
                for p in net_obj.PATHS
                for l in net_obj.LINKS
                for k in net_obj.PRIORITIES
            )
            +
            mdl.sum(
                (g[r, v] * net_obj.PACKET_SIZE / (net_obj.DC_CAPACITIES[v] + EPSILON))
                for v in net_obj.NODES
            )
        )
        ==
        d[r]
        for r in req_obj.REQUESTS
    )
    # endregion

    # region C15
    print("defining C15...")
    mdl.add_constraints(
        d[r]
        <=
        req_obj.DELAY_REQUIREMENTS[r]
        for r in req_obj.REQUESTS
    )
    # endregion

    mdl.log_output = True
    solution = mdl.solve()

    return  solution