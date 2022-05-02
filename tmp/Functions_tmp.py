import numpy as np
rnd = np.random


def specify_requests_entry_nodes(net, req, seed):
    rnd.seed(seed)
    return np.array([rnd.choice(net.first_tier_nodes()) for i in req.requests()])


def assign_requests_to_services(srv, req, seed):
    rnd.seed(seed)
    return np.array([rnd.choice(srv.services()) for i in req.requests()])


def subplot(request, request_entry_node, requested_service, X_LOCS, Y_LOCS, g, G, NODES, PP, flw, LINKS,
            ax, service_color, link_color):
    service_node = [j for (s, j) in G if s ==
                    request and g[s, j].solution_value > 0.9][0]
    active_links = {(j, m): round(flw[(i, (j, m), n)].solution_value, 2) for (
        i, (j, m), n) in PP if flw[(i, (j, m), n)].solution_value > 0.1 and i == request}
    routers_X_LOCS = []
    routers_Y_LOCS = []
    for i in NODES:
        if i != request_entry_node and i != service_node:
            routers_X_LOCS.append(X_LOCS[i])
            routers_Y_LOCS.append(Y_LOCS[i])
    links = [(j, m) for (i, (j, m), n) in PP if i ==
             request and flw[(i, (j, m), n)].solution_value > 0.1]

    for i, j in LINKS:
        if (i, j) in links:
            ax.plot([X_LOCS[i], X_LOCS[j]], [
                    Y_LOCS[i], Y_LOCS[j]], c=link_color, zorder=10, linewidth=4)
        else:
            ax.plot([X_LOCS[i], X_LOCS[j]], [
                Y_LOCS[i], Y_LOCS[j]], c='lightblue', zorder=10, linestyle=':')
    ax.scatter(X_LOCS[request_entry_node],
               Y_LOCS[request_entry_node], s=100, c='r',  zorder=15)
    ax.scatter(X_LOCS[service_node], Y_LOCS[service_node],
               s=100, c=service_color, zorder=15)
    ax.scatter(routers_X_LOCS, routers_Y_LOCS, c='g', zorder=15)
    """ ax.annotate('$%d$' % (request_entry_node),
                (X_LOCS[request_entry_node]+1, Y_LOCS[request_entry_node]+1), zorder=20)
    ax.annotate('$%d$' % (service_node),
                (X_LOCS[service_node]+1, Y_LOCS[service_node]+1), zorder=20) """
    for i in NODES:
        ax.annotate(i, (X_LOCS[i]+2, Y_LOCS[i]-1), zorder=20)
    for (i, j) in active_links:
        ax.annotate(active_links[(i, j)], ((
            X_LOCS[i]+X_LOCS[j])/2, (Y_LOCS[i]+Y_LOCS[j])/2), zorder=20)
