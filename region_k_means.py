
# Author: You Wan <wanyou9@gmail.com>
# Data: 2021.10.13
# License: BSD 3 clause


from collections import defaultdict
import numpy
import networkx as nx
from .base import (
    w_to_g,
    move_ok,
    ok_moves,
    region_neighbors,
    _centroid,
    _closest,
    _seeds,
    _sse,
    is_neighbor,
)


def region_k_means(X, n_clusters, w):
    """Solve the region-K-means problem, the K-means with the constraint
    that each cluster forms a spatially connected component. An update version
    of pysal/spopt/region/region_k_means.py

    Parameters
    ----------

    X : {numpy.ndarray, list}
        The observations to cluster shaped ``(n_samples, n_features)``.

    n_clusters : int
        The number of clusters to form.

    w : libpysal.weights.W
        ...

    Returns
    -------

    label : numpy.ndarray
        Integer array with shape ``(n_samples,)``, where ``label[i]`` is the
        code or index of the centroid the ``i``th observation is closest to.

    centroid : numpy.ndarray
       Floating point array of centroids in the shape of ``(k, n_features)``
       found at the last iteration of ``region_k_means``.

    iters : int
        The number of iterations for the reassignment phase.

    """

    data = X
    areas = numpy.arange(w.n).astype(int)
    k = len(n_clusters)
    sse_list = []
    seeds = n_clusters  # _seeds(areas, k)

    # initial assignment phase
    label_initial = numpy.array([-1] * w.n).astype(int)
    for i, seed in enumerate(seeds):
        label_initial[seed] = i
    label = to_assign_initial(label_initial, areas, data, k, w)
    # reassignment phase
    changed = []
    g = w_to_g(w)
    iters = 1

    # want to loop this until candidates is empty
    regions = [areas[label == r].tolist() for r in range(k)]
    sse_list.append(_sse(regions, data))
    centroid = _centroid(regions, data)
    closest = numpy.array(_closest(data, centroid))
    candidates = areas[closest != label]
    candidates = ok_moves(candidates, regions, label, closest, g, w, areas)
    while candidates:
        # make moves
        for area in candidates:
            label[area] = closest[area]
        for ir in range(k):
            iregion = areas[label == ir].tolist()
            ir_graph = g.subgraph(nodes=iregion)
            largest_cc = max(nx.connected_components(ir_graph), key=len)
            cand2 = set(ir_graph).difference(largest_cc)
            if len(cand2)>0:
                # print("disconnected component")
                for area in cand2:
                    label[area] = -1
        to_assign = areas[label == -1]
        if to_assign.size > 0:
            label_loop = to_assign_initial(label, areas, data, k, w)
            label = label_loop
        regions = [areas[label == r].tolist() for r in range(k)]
        sse_list.append(_sse(regions, data))
        centroid = _centroid(regions, data)
        closest = numpy.array(_closest(data, centroid))
        candidates = areas[closest != label]
        candidates = ok_moves(candidates, regions, label, closest, g, w, areas)
        iters += 1

    return centroid, label, iters, sse_list

def to_assign_initial(labels, areas, data, k, w):
    to_assign = areas[labels == -1]
    a_list = w.to_adjlist(remove_symmetric=False)
    c = 0
    while to_assign.size > 0:
        assignments = defaultdict(list)
        for rid in range(k):
            region = areas[labels == rid]
            neighbors = region_neighbors(a_list, region)
            neighbors = [j for j in neighbors if j in to_assign]
            if neighbors:
                d_min = numpy.inf
                centroid = data[region].mean(axis=0)
                for neighbor in neighbors:
                    d = ((data[neighbor] - centroid) ** 2).sum()
                    if d < d_min:
                        idx = neighbor
                        d_min = d
                assignments[idx].append([rid, d_min])
        for key in assignments:
            assignment = assignments[key]
            if len(assignment) == 1:
                r, d = assignment[0]
                labels[key] = r
            else:
                d_min = numpy.inf
                for match in assignment:
                    r, d = match
                    if d < d_min:
                        idx = r
                        d_min = d
                labels[key] = idx

        to_assign = areas[labels == -1]
    return labels