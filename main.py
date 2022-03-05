#!/usr/bin/env python3

"""
Anchor point deployment and assignment simulator

Based on: https://github.com/pfondo/upf-allocation-simulator

@ author:
    Pablo Fondo-Ferreiro <pfondo@gti.uvigo.es>
    David Candal-Ventureira <dcandal@gti.uvigo.es>
"""

import argparse
import time
import networkx as nx
from random import random, seed, gauss, sample, randint, choice
import statistics
from math import log10
from numpy import array as np_array, mean as np_mean, percentile as np_percentile, vstack as np_vstack
from scipy.stats import gaussian_kde
from sys import stderr, stdout
import copy

from sklearn import cluster
from sklearn.cluster import KMeans

import community as community_louvain
import itertools
from networkx.algorithms.community.centrality import girvan_newman as girvan_newman

from scipy.stats import sem as st_sem, t as st_t

DEBUG = False
DEFAULT_SEED = 0
PL_THRESHOLD = 2
DISTANCE_THRESHOLD = 500

DEFAULT_MIN_UPF = 1
DEFAULT_MAX_UPF = 10

DEFAULT_ITERATION_DURATION = 5  # In seconds
DEFAULT_TIME_DEPLOYMENT = 1  # In seconds
DEFAULT_TIME_REMOVAL = 0.1  # In seconds
DEFAULT_COST_RELOCATION = 1  # Adimensional (NOTE: Could also be in bytes)

DEFAULT_ALPHA1 = 0.7
DEFAULT_ALPHA2 = 0.1
DEFAULT_ALPHA3 = 0.1
DEFAULT_ALPHA4 = 0.1


class UE:
    def __init__(self, id, x=0, y=0, bs=None):
        self._id = id
        self._x = float(x)
        self._y = float(y)
        self._bs = bs
        if self._bs is not None:
            self._bs.add_UE(self)

    def get_x(self):
        return self._x

    def get_y(self):
        return self._y

    def get_coords(self):
        return [self._x, self._y]

    def set_coords(self, x, y):
        self._x, self._y = (float(x), float(y))

    def get_bs(self):
        return self._bs

    def set_bs(self, bs):
        if self._bs is not None:
            self._bs.remove_UE(self)
        if bs is not None:
            bs.add_UE(self)
        self._bs = bs

    def get_id(self):
        return self._id

    def get_pl(self):
        if self._bs is None:
            return float("inf")
        else:
            distance = self._bs.get_distance_coords(self._x, self._y)
            return compute_path_loss(distance)

    def update_bs(self, bs, pl=float("inf")):
        if bs is None or pl + PL_THRESHOLD < self.get_pl():
            self.set_bs(bs)

    def update(self, x, y, bs, pl=float("inf")):
        self.set_coords(x, y)
        self.update_bs(bs, pl)

    def update_unconditional(self, x, y, bs):
        self.set_coords(x, y)
        self.set_bs(bs)


class BS:
    def __init__(self, id_, x, y, UPF=False):
        self._id = int(id_)
        self._x = float(x)
        self._y = float(y)
        self._UPF = UPF
        self.UEs = []

    def get_id(self):
        return self._id

    def get_x(self):
        return self._x

    def get_y(self):
        return self._y

    def get_coords(self):
        return [self._x, self._y]

    def add_UE(self, ue):
        self.UEs.append(ue)

    def remove_UE(self, ue):
        self.UEs.remove(ue)

    def get_UEs(self):
        return self.UEs

    def get_numUEs(self):
        return len(self.UEs)

    def clear_UEs(self):
        self.UEs = []

    def get_distance(self, bs2):
        return ((bs2.get_x()-self.get_x())**2 + (bs2.get_y()-self.get_y())**2)**0.5

    def get_distance_coords(self, x, y):
        return ((x-self.get_x())**2 + (y-self.get_y())**2)**0.5


''' Generates a set of connected components by conecting all the base stations
    that are positioned less than DISTANCE_THRESHOLD meters apart
'''


def generate_graph(bs_file):
    G = nx.Graph()
    BSs = {}
    highest_bs_id = -1

    with open(bs_file) as f:
        for _, line in enumerate(f):
            bs_data = line.strip().split()
            bs = BS(bs_data[0], bs_data[1], bs_data[2])
            BSs[bs.get_id()] = bs
            if bs.get_id() > highest_bs_id:
                highest_bs_id = bs.get_id()
            G.add_node(bs)
            for other_bs in G.nodes:
                if other_bs is not bs and bs.get_distance(other_bs) < DISTANCE_THRESHOLD:
                    G.add_edge(bs, other_bs)

    join_components(G)

    G_shortest_path_lengths = dict(nx.all_pairs_shortest_path_length(G))

    max_num_hops = nx.algorithms.distance_measures.diameter(G)

    return G, BSs, G_shortest_path_lengths, highest_bs_id, max_num_hops


''' Connects the connected components to compose a single giant component.
    This procedure is performed as follow: On each iteration, from the second
    biggest connected component, the node which is at the shortest distance
    from one of the nodes of the giant component is determined, and an edge
    between these two nodes.

    This way, on each iteration the connected components are joined to the giant
    component in order of size to achieve a single giant component.
'''


def join_components(G):
    while True:
        connected_components = list(nx.connected_components(G))
        if len(list(connected_components)) < 2:
            break
        connected_components = sorted(
            connected_components, key=len, reverse=True)

        bs1 = bs2 = distance = None
        for bs in connected_components[1]:
            for bs_giant_component in connected_components[0]:
                d = bs.get_distance(bs_giant_component)
                if distance is None or d < distance:
                    distance = d
                    bs1 = bs
                    bs2 = bs_giant_component
        G.add_edge(bs1, bs2)


''' Determines the nearest BS to the coordinates x and y
'''


def get_optimal_bs(BSs, x, y):
    distance = None
    bs = None
    for node in BSs.values():
        d = node.get_distance_coords(x, y)
        if distance is None or d < distance:
            distance = d
            bs = node
    return bs, distance


def compute_path_loss(distance):
    # p1 = 46.61
    # p2 = 3.63
    # std = 9.83
    # return p1 + (p2 * 10 * log10(distance)) + gauss(0, std)
    return 46.61 + (3.63 * 10 * log10(distance)) + gauss(0, 9.83)


''' Generates the list of UEs of the new iteration: updates data from previous
    iteration, removes UEs that do not appear in the new stage and adds those
    that did not appear in the last stage.
'''


def read_UE_data(ue_file, BSs, iteration_duration):
    UEs_last_iteration = {}

    first_timestamp_iteration = None

    UEs_new_iteration = {}

    with open(ue_file) as f:
        for line in f:
            # Read UEs from new iteration
            line = line.strip().split()
            timestamp = int(line[0])
            id_ = int(line[1].split("_")[0].split("#")[0])
            x = float(line[2])
            y = float(line[3])
            speed = float(line[4])  # Unused
            pl = None
            if len(line) > 5:
                bs = BSs[int(line[5])]
            else:
                bs, distance = get_optimal_bs(BSs, x, y)
                pl = compute_path_loss(distance)

            if first_timestamp_iteration == None:
                first_timestamp_iteration = timestamp

            if timestamp - first_timestamp_iteration > iteration_duration:
                # Iteration finished: Yield results
                for ue in [ue for id_, ue in UEs_last_iteration.items() if id_ not in UEs_new_iteration.keys()]:
                    ue.update_bs(None)

                UEs_last_iteration = UEs_new_iteration
                yield UEs_new_iteration

                # Resumed execution for next iteration: Initialize values for this iteration
                UEs_new_iteration = {}
                first_timestamp_iteration = timestamp

            # Update UE already present in previous iteration
            if id_ in UEs_last_iteration:
                ue = UEs_last_iteration[id_]
                if pl:
                    ue.update(x, y, bs, pl)
                else:
                    ue.update_unconditional(x, y, bs)
            # Only the last appearance of each UE in the iteration is considered
            elif id_ in UEs_new_iteration:
                ue = UEs_new_iteration[id_]
                if pl:
                    ue.update(x, y, bs, pl)
                else:
                    ue.update_unconditional(x, y, bs)
            # Se crea un nuevo UE
            else:
                ue = UE(id_, x, y, bs)
            UEs_new_iteration[id_] = ue

# Deprecated: Used to generate synthetic data


def generate_UE_data_random(BSs):
    UEs_last_iteration = {}

    for i in range(100):
        UEs_new_iteration = {}
        for j in range(10000):
            id_ = j  # int(j*random())
            x = (4503.09786887-28878.1970746)*random()+28878.1970746
            y = (3852.34416744-36166.012178)*random()+36166.012178
            bs, distance = get_optimal_bs(BSs, x, y)
            pl = compute_path_loss(distance)

            if id_ in UEs_last_iteration:
                ue = UEs_last_iteration[id_]
                ue.update(x, y, bs, pl)
            elif id_ in UEs_new_iteration:
                ue = UEs_new_iteration[id_]
                ue.update(x, y, bs, pl)
            else:
                ue = UE(x, y, bs)
            UEs_new_iteration[id_] = ue
        for ue in [ue for id_, ue in UEs_last_iteration.items() if id_ not in UEs_new_iteration.keys()]:
            ue.update_bs(None)

        UEs_last_iteration = UEs_new_iteration
        yield UEs_new_iteration


def UPF_allocation_random(G: nx.Graph, num_UPFs, BSs_with_UPF_ids_previous, G_shortest_path_lengths, highest_bs_id):
    BSs_with_UPF_ids = set(
        sample([x for x in range(G.number_of_nodes())], num_UPFs))
    return BSs_with_UPF_ids


def analyze_allocation(G: nx.Graph, BSs_with_UPF_ids):
    # Used for printing nodes and analyzing the allocations
    import matplotlib.pyplot as plt

    x = []
    y = []

    colors = []

    bs: BS
    for bs in G.nodes():
        if bs.get_numUEs() == 0:
            continue
        for _ in range(bs.get_numUEs()):
            x.append(bs.get_x() / 1000)
            y.append(bs.get_y() / 1000)
            colors.append(bs.get_numUEs())

    # plt.scatter(x, y, c=colors)

    xy = np_vstack([x, y])
    z = gaussian_kde(xy)(xy)

    # sc = plt.scatter(x, y, c=z, s=10)

    hb = plt.hexbin(x, y, gridsize=10, cmap="Blues")

    plt.xlabel("x coordinate (km)")
    plt.ylabel("y coordinate (km)")

    plt.colorbar(hb)

    x2 = []
    y2 = []
    for bs in G.nodes():
        if bs.get_id() in BSs_with_UPF_ids:
            x2.append(bs.get_x() / 1000)
            y2.append(bs.get_y() / 1000)

    plt.scatter(x2, y2, color=['r' for x in range(
        len(BSs_with_UPF_ids))], marker='x')

    plt.show()


def UPF_allocation_greedy_percentile(G: nx.Graph, num_UPFs, BSs_with_UPF_previous, G_shortest_path_lengths, highest_bs_id):
    BSs_with_UPF_ids = set()

    # Inf value initialization
    latencies_list = [
        G.number_of_nodes() + 1 for _ in range(highest_bs_id + 1)]
    num_ues_list = [0 for _ in range(highest_bs_id + 1)]
    tot_ues = 0
    for bs in G.nodes:
        num_ues_list[bs.get_id()] = bs.get_numUEs()
        tot_ues += bs.get_numUEs()

    done_BSs = [False for _ in range(highest_bs_id + 1)]

    for _ in range(num_UPFs):
        best_bs = None
        best_latency = None
        best_acc_latency = None
        for bs in G.nodes:
            if not done_BSs[bs.get_id()]:
                # Check latency if bs is selected
                acc_latency = 0
                latencies_current_list = list(latencies_list)
                for bs2 in G.nodes:
                    new_latency = min(
                        G_shortest_path_lengths[bs2][bs], latencies_current_list[bs2.get_id()])
                    latencies_current_list[bs2.get_id()] = new_latency
                    acc_latency += new_latency

                # Calculate 90th percentile of ue latency
                latency = None
                acc_num_ues = 0
                for lat, num_ues_bs in sorted(zip(latencies_current_list, num_ues_list), key=lambda x: x[0], reverse=True):
                    acc_num_ues += num_ues_bs
                    if acc_num_ues >= 0.1 * tot_ues:
                        latency = lat
                        break

                assert(latency != None)

                if best_bs == None or latency < best_latency or (latency == best_latency and acc_latency < best_acc_latency):
                    best_bs = bs
                    best_latency = latency
                    best_acc_latency = acc_latency

        BSs_with_UPF_ids.add(best_bs.get_id())
        done_BSs[best_bs.get_id()] = True
        for bs2 in G.nodes:
            new_latency = G_shortest_path_lengths[bs2][best_bs]
            if new_latency < latencies_list[bs2.get_id()]:
                latencies_list[bs2.get_id()] = new_latency

    return BSs_with_UPF_ids


# Greedy implementation iteratively picking the eNB which reduces the average latency the most
def UPF_allocation_greedy_average(G: nx.Graph, num_UPFs, BSs_with_UPF_ids_previous, G_shortest_path_lengths, highest_bs_id):
    BSs_with_UPF_ids = set()

    # Inf value initialization
    latencies_list = [
        G.number_of_nodes() + 1 for _ in range(highest_bs_id + 1)]
    num_ues_list = [0 for _ in range(highest_bs_id + 1)]
    tot_ues = 0
    for bs in G.nodes:
        num_ues_list[bs.get_id()] = bs.get_numUEs()
        tot_ues += bs.get_numUEs()

    done_BSs = [False for _ in range(highest_bs_id + 1)]

    for _ in range(num_UPFs):
        best_bs = None
        best_latency = None
        for bs in G.nodes:
            if not done_BSs[bs.get_id()]:
                # Check latency if bs is selected
                acc_latency = 0
                for bs2 in G.nodes:
                    new_latency = min(
                        G_shortest_path_lengths[bs2][bs], latencies_list[bs2.get_id()])
                    acc_latency += (new_latency * num_ues_list[bs2.get_id()])

                # Calculate average latency
                latency = acc_latency / tot_ues

                if best_bs == None or latency < best_latency:
                    best_bs = bs
                    best_latency = latency

        BSs_with_UPF_ids.add(best_bs.get_id())
        done_BSs[best_bs.get_id()] = True
        for bs2 in G.nodes:
            new_latency = G_shortest_path_lengths[bs2][best_bs]
            if new_latency < latencies_list[bs2.get_id()]:
                latencies_list[bs2.get_id()] = new_latency

    return BSs_with_UPF_ids

# Implementation of kmeans considering active BSs


def UPF_allocation_kmeans(G: nx.Graph, num_UPFs, BSs_with_UPF_previous, G_shortest_path_lengths, highest_bs_id):
    BSs_with_UPF_ids = set()

    # Calculate k-means clustering considering active BSs
    features = []
    for bs in G.nodes:
        if bs.get_numUEs() > 0:
            features.append([bs.get_x(), bs.get_y()])

    kmeans = KMeans(
        init="k-means++",  # "random" / "k-means++"
        n_clusters=min(num_UPFs, len(features)),
        n_init=2,
        max_iter=1000,
        random_state=0  # To allow for reproducibility
    )

    # In case scaling want to be applied
    # scaler = StandardScaler()
    # scaled_features = scaler.fit_transform(features)
    # kmeans.fit(scaled_features)

    kmeans.fit(features)

    # Pick UPFs closer to cluster centers
    done_BSs = [False for _ in range(highest_bs_id + 1)]

    # for center_x, center_y in scaler.inverse_transform(kmeans.cluster_centers_):
    for center_x, center_y in kmeans.cluster_centers_:
        best_UPF = None
        best_distance = None
        for bs in G.nodes:
            if not done_BSs[bs.get_id()]:
                distance = bs.get_distance_coords(center_x, center_y)
                if best_UPF == None or distance < best_distance:
                    best_UPF = bs.get_id()
                    best_distance = distance

        BSs_with_UPF_ids.add(best_UPF)
        done_BSs[best_UPF] = True

    # Add UPFs until reaching desired number of UPFs (for addressing corner cases with very few active BSs)
    if len(BSs_with_UPF_ids) < num_UPFs:
        BSs_with_UPF_ids.update(set(sample([x for x in range(G.number_of_nodes(
        )) if x not in BSs_with_UPF_ids], num_UPFs - len(BSs_with_UPF_ids))))

    assert(len(BSs_with_UPF_ids) == num_UPFs)

    return BSs_with_UPF_ids

# Static implementation of kmeans considering all BSs


def UPF_allocation_kmeans_static(G: nx.Graph, num_UPFs, BSs_with_UPF_previous, G_shortest_path_lengths, highest_bs_id):
    BSs_with_UPF_ids = set()

    # Pick UPFs closer to cluster centers
    done_BSs = [False for _ in range(highest_bs_id + 1)]

    # for center_x, center_y in scaler.inverse_transform(kmeans.cluster_centers_):
    for center_x, center_y in kmeans.cluster_centers_:
        best_UPF = None
        best_distance = None
        for bs in G.nodes:
            if not done_BSs[bs.get_id()]:
                distance = bs.get_distance_coords(center_x, center_y)
                if best_UPF == None or distance < best_distance:
                    best_UPF = bs.get_id()
                    best_distance = distance

        BSs_with_UPF_ids.add(best_UPF)
        done_BSs[best_UPF] = True

    # Add UPFs until reaching desired number of UPFs (for addressing corner cases with very few active BSs)
    if len(BSs_with_UPF_ids) < num_UPFs:
        BSs_with_UPF_ids.update(set(sample([x for x in range(G.number_of_nodes(
        )) if x not in BSs_with_UPF_ids], num_UPFs - len(BSs_with_UPF_ids))))

    # print(BSs_with_UPF_ids, num_UPFs, file = stderr)

    assert(len(BSs_with_UPF_ids) == num_UPFs)

    return BSs_with_UPF_ids

# Implementation of kmeans considering active eNBs, greedily selecting the best eNB in each cluster


def UPF_allocation_kmeans_greedy_average(G: nx.Graph, num_UPFs, BSs_with_UPF_ids_previous, G_shortest_path_lengths, highest_bs_id):
    BSs_with_UPF_ids = set()

    # Calculate k-means clustering considering active BSs
    features = []
    BSs = []
    for bs in G.nodes:
        if bs.get_numUEs() > 0:
            features.append([bs.get_x(), bs.get_y()])
            BSs.append(bs)

    kmeans = KMeans(
        init="k-means++",  # "random" / "k-means++"
        n_clusters=min(num_UPFs, len(features)),
        n_init=2,
        max_iter=1000,
        random_state=0  # To allow for reproducibility
    )

    # In case scaling wants to be enabled
    # scaler = StandardScaler()
    # scaled_features = scaler.fit_transform(features)
    # kmeans.fit(scaled_features)

    kmeans.fit(features)

    cluster_BS_ids_list = [[] for _ in range(num_UPFs)]
    for i in range(len(kmeans.labels_)):
        cluster_BS_ids_list[kmeans.labels_[i]].append(i)

    # Greedily pick the best UPF in each cluster
    for cluster in range(num_UPFs):
        # cluster_BS_ids = [x for x in range(len(kmeans.labels_)) if kmeans.labels_[x] == cluster]
        cluster_BS_ids = cluster_BS_ids_list[cluster]

        best_bs = None
        best_acc_latency = None
        for bs_index in cluster_BS_ids:
            bs = BSs[bs_index]
            # Check latency if bs is selected
            acc_latency = 0
            for bs2_index in cluster_BS_ids:
                bs2 = BSs[bs2_index]
                new_latency = G_shortest_path_lengths[bs2][bs]
                acc_latency += (new_latency * bs2.get_numUEs())

            # Calculate average latency
            if best_bs == None or acc_latency < best_acc_latency:
                best_bs = bs
                best_acc_latency = acc_latency

        if best_bs != None:
            BSs_with_UPF_ids.add(best_bs.get_id())

    # Add UPFs until reaching desired number of UPFs (for addressing corner cases with very few active base stations)
    if len(BSs_with_UPF_ids) < num_UPFs:
        BSs_with_UPF_ids.update(set(sample([x for x in range(G.number_of_nodes(
        )) if x not in BSs_with_UPF_ids], num_UPFs - len(BSs_with_UPF_ids))))

    assert(len(BSs_with_UPF_ids) == num_UPFs)

    return BSs_with_UPF_ids

# Implementation of clustering based on community detection (Louvain modularity maximization) considering active BSs, greedily selecting the best eNB in each cluster


def UPF_allocation_modularity_greedy_average(G: nx.Graph, num_UPFs, BSs_with_UPF_ids_previous, G_shortest_path_lengths, highest_bs_id):
    BSs_with_UPF_ids = set()

    # Louvain modularity maximization using active BSs
    G_active = nx.Graph(G)
    for bs in G.nodes:
        if bs.get_numUEs() == 0:
            G_active.remove_node(bs)

    partition = community_louvain.best_partition(G_active)

    cluster_BSs_list = [[] for _ in range(num_UPFs)]
    for bs, cluster in partition.items():
        # NOTE: Cluster joining can be improved
        cluster_BSs_list[cluster % num_UPFs].append(bs)

    # Greedily pick the best UPF in each cluster
    for cluster in range(num_UPFs):
        cluster_BS_ids = cluster_BSs_list[cluster]

        best_bs = None
        best_acc_latency = None
        for bs in cluster_BS_ids:
            # Check latency if bs is selected
            acc_latency = 0
            for bs2 in cluster_BS_ids:
                new_latency = G_shortest_path_lengths[bs2][bs]
                acc_latency += (new_latency * bs2.get_numUEs())

            # Calculate average latency
            if best_bs == None or acc_latency < best_acc_latency:
                best_bs = bs
                best_acc_latency = acc_latency

        if best_bs != None:
            BSs_with_UPF_ids.add(best_bs.get_id())

    # Add UPFs until reaching desired number of UPFs
    if len(BSs_with_UPF_ids) < num_UPFs:
        BSs_with_UPF_ids.update(set(sample([x for x in range(G.number_of_nodes(
        )) if x not in BSs_with_UPF_ids], num_UPFs - len(BSs_with_UPF_ids))))

    assert(len(BSs_with_UPF_ids) == num_UPFs)

    return BSs_with_UPF_ids


''' Determines in which BS a UPF is instantiated:
    Generic function which calls the specific allocation algorithm
'''


def UPF_allocation(algorithm, G,  BSs_with_UPF_ids_previous, G_shortest_path_lengths, highest_bs_id, generate_new_allocation=True):
    BSs_with_UPF_ids = set()

    if generate_new_allocation or BSs_with_UPF_ids_previous == None:
        BSs_with_UPF_ids = globals()["UPF_allocation_{}".format(algorithm)](
            G,  BSs_with_UPF_ids_previous, G_shortest_path_lengths, highest_bs_id)
    else:
        for bs_id in BSs_with_UPF_ids_previous:
            BSs_with_UPF_ids.add(bs_id)

    if DEBUG:
        print(BSs_with_UPF_ids, file=stderr)

    return BSs_with_UPF_ids


# Assignment methods

# Unused
def UPF_assignment_random(G: nx.Graph, num_UPFs, UE_to_UPF_assignment_previous, BSs_with_UPF_ids_previous, G_shortest_path_lengths, highest_bs_id):
    UE_to_UPF_assignment = {}
    BSs_with_UPF_ids = set()

    bs: BS
    for bs in G.nodes():
        ue: UE
        for ue in bs.get_UEs():
            # Determine which node will have a UPF to serve this UE
            upf_node = randint(0, G.number_of_nodes() - 1)
            UE_to_UPF_assignment[ue.get_id()] = upf_node
            BSs_with_UPF_ids.add(upf_node)

    return UE_to_UPF_assignment, BSs_with_UPF_ids

# Unused


def UPF_assignment_random_bs(G: nx.Graph, num_UPFs, UE_to_UPF_assignment_previous, BSs_with_UPF_ids_previous, G_shortest_path_lengths, highest_bs_id):
    UE_to_UPF_assignment = {}
    BSs_with_UPF_ids = set()

    bs: BS
    for bs in G.nodes():
        # Determine which node will have a UPF to serve the UEs in this BS
        upf_node = randint(0, G.number_of_nodes() - 1)
        for ue in bs.get_UEs():
            UE_to_UPF_assignment[ue.get_id()] = upf_node
        BSs_with_UPF_ids.add(upf_node)

    return UE_to_UPF_assignment, BSs_with_UPF_ids


def UPF_assignment_greedy_overhead(G: nx.Graph, BSs, num_UPFs, UE_to_UPF_assignment_previous, BSs_with_UPF_ids_previous, G_shortest_path_lengths, highest_bs_id, iteration_duration, time_deployment, time_removal, cost_relocation, alpha1, alpha2, alpha3, alpha4, num_UEs, max_num_hops):
    #UE_to_UPF_assignment = {}
    BS_to_UPF_assignment_previous = convert_ue_assignment_to_bs_assignment(
        BSs, UE_to_UPF_assignment_previous)

    BS_to_UPF_assignment = {}
    BSs_with_UPF_ids = set()

    bs: BS

    latencies_list = [
        G.number_of_nodes() + 1 for _ in range(highest_bs_id + 1)]
    num_ues_list = [0 for _ in range(highest_bs_id + 1)]
    tot_ues = 0
    for bs in G.nodes:
        num_ues_list[bs.get_id()] = bs.get_numUEs()
        tot_ues += bs.get_numUEs()

    done_BSs = [False for _ in range(highest_bs_id + 1)]

    for _ in range(num_UPFs):
        best_bs = None
        best_f_objective_function = None
        for bs in G.nodes:
            if not done_BSs[bs.get_id()]:
                # Check objective function if bs is selected

                # f2: Resource usage
                f2_num_UPFs = len(BSs_with_UPF_ids) + 1

                # f3: Deployment overhead
                f3_deployment_overhead = get_deployment_overhead(
                    BSs_with_UPF_ids_previous, BSs_with_UPF_ids | set([bs.get_id()]), time_deployment, time_removal, iteration_duration)

                # f1: Vehicle latency (90-th percentile)
                f_objective_function = get_objective_function(
                    G, 0, f2_num_UPFs, f3_deployment_overhead, 0, alpha1, alpha2, alpha3, alpha4, time_deployment, time_removal, cost_relocation, num_UEs, max_num_hops)
                for bs2 in G.nodes:
                    if bs2.get_numUEs() == 0:
                        continue
                    new_latency = latencies_list[bs2.get_id()]

                    # new_latency = min(
                    #     G_shortest_path_lengths[bs2][bs], latencies_list[bs2.get_id()])

                    f_objective_keep = get_objective_function(G, num_ues_list[bs2.get_id()] * latencies_list[bs2.get_id(
                    )] / tot_ues, 0, 0, 0, alpha1, alpha2, alpha3, alpha4, time_deployment, time_removal, cost_relocation, num_UEs, max_num_hops)

                    f4_control_plane_reassignment_overhead = 0
                    if (bs2.get_id() in BS_to_UPF_assignment_previous):
                        f4_control_plane_reassignment_overhead = num_ues_list[bs2.get_id(
                        )] * G_shortest_path_lengths[BSs[BS_to_UPF_assignment_previous[bs2.get_id()]]][bs]

                    f_objective_relocate = get_objective_function(G, num_ues_list[bs2.get_id()] * G_shortest_path_lengths[bs2][bs] / tot_ues, 0, 0, f4_control_plane_reassignment_overhead *
                                                                  cost_relocation, alpha1, alpha2, alpha3, alpha4, time_deployment, time_removal, cost_relocation, num_UEs, max_num_hops)

                    if f_objective_relocate < f_objective_keep:
                        f_objective_function += f_objective_relocate
                    else:
                        f_objective_function += f_objective_keep

                if best_bs == None or f_objective_function < best_f_objective_function:
                    best_bs = bs
                    best_f_objective_function = f_objective_function

        # NOTE: Experimental
        # if previous_f_objective_function != None and best_f_objective_function > previous_f_objective_function:
        #     # Do not add more UPFs if the best objective function increases
        #     break

        previous_f_objective_function = best_f_objective_function

        upf_node = best_bs.get_id()
        done_BSs[upf_node] = True
        BSs_with_UPF_ids.add(upf_node)
        # for ue in bs.get_UEs():
        #    UE_to_UPF_assignment[ue.get_id()] = upf_node

        for bs2 in G.nodes:
            if bs2.get_numUEs() == 0:
                continue
            new_latency = latencies_list[bs2.get_id()]

            f_objective_keep = get_objective_function(G, latencies_list[bs2.get_id(
            )], 0, 0, 0, alpha1, alpha2, alpha3, alpha4, time_deployment, time_removal, cost_relocation, num_UEs, max_num_hops)
            f_objective_relocate = get_objective_function(G, G_shortest_path_lengths[bs2][best_bs], 0, 0, G_shortest_path_lengths[bs2][
                best_bs] * cost_relocation, alpha1, alpha2, alpha3, alpha4, time_deployment, time_removal, cost_relocation, num_UEs, max_num_hops)

            # if G_shortest_path_lengths[bs2][bs] < latencies_list[bs2.get_id()]:
            if f_objective_relocate < f_objective_keep:
                new_latency = G_shortest_path_lengths[bs2][best_bs]
                latencies_list[bs2.get_id()] = new_latency
                # Update the assignment of all the UEs in bs2
                BS_to_UPF_assignment[bs2.get_id()] = upf_node

    # Convert back to original format BS_to_UPF_assignment -> UE_to_UPF_assignment
    UE_to_UPF_assignment = convert_bs_assignment_to_ue_assignment(
        BSs, BS_to_UPF_assignment)

    return UE_to_UPF_assignment, BSs_with_UPF_ids


# PABLO: Working on this!
def UPF_assignment_simulated_annealing_previous(G: nx.Graph, BSs, num_UPFs, UE_to_UPF_assignment_previous, BSs_with_UPF_ids_previous, G_shortest_path_lengths, highest_bs_id, iteration_duration, time_deployment, time_removal, cost_relocation, alpha1, alpha2, alpha3, alpha4, num_UEs, max_num_hops):
    BS_to_UPF_assignment_previous = convert_ue_assignment_to_bs_assignment(
        BSs, UE_to_UPF_assignment_previous)

    #print("\nRunning simulated annealing", file=stderr)

    BSs_with_UPF_ids = copy.deepcopy(BSs_with_UPF_ids_previous)
    BS_to_UPF_assignment = copy.deepcopy(BS_to_UPF_assignment_previous)

    # If there was not UPFs in the previous interval, assign them using the greedy_overhead algorithm
    if (BSs_with_UPF_ids == None or len(BSs_with_UPF_ids) == 0):
        print("\nRunning simulated annealing previous", file=stderr)
    # TODO: Do this only for the first iteration or never
    UE_to_UPF_assignment, BSs_with_UPF_ids = UPF_assignment_greedy_overhead(G, BSs, num_UPFs, UE_to_UPF_assignment_previous, BSs_with_UPF_ids_previous, G_shortest_path_lengths,
                                                                            highest_bs_id, iteration_duration, time_deployment, time_removal, cost_relocation, alpha1, alpha2, alpha3, alpha4, num_UEs, max_num_hops)
    BS_to_UPF_assignment = convert_ue_assignment_to_bs_assignment(
        BSs, UE_to_UPF_assignment)

    # Assign users which where not present in the previous interval to the closest active BS
    bs: BS
    for bs in G.nodes():
        if bs.get_numUEs() > 0 and bs.get_id() not in BS_to_UPF_assignment:
            _, upf_node = get_minimum_hops_from_BS_to_any_UPF(
                G, BSs, bs, BSs_with_UPF_ids, G_shortest_path_lengths)
            BS_to_UPF_assignment[bs.get_id()] = upf_node.get_id()

    #BSs_with_UPF_ids = set(BS_to_UPF_assignment.values())

    latencies_list = [0 for _ in range(highest_bs_id + 1)]
    num_ues_list = [0 for _ in range(highest_bs_id + 1)]
    tot_ues = 0
    tot_latencies = 0
    for bs in G.nodes:
        if bs.get_numUEs() > 0:
            latencies_list[bs.get_id()] = G_shortest_path_lengths[BSs[bs.get_id()]
                                                                  ][BSs[BS_to_UPF_assignment[bs.get_id()]]]
            num_ues_list[bs.get_id()] = bs.get_numUEs()
            tot_ues += bs.get_numUEs()
            tot_latencies += bs.get_numUEs() * latencies_list[bs.get_id()]

    # Counts the number of BSs assigned to a given UPF; Used for speedup calculations
    count_BSs_per_UPF = [0 for _ in range(highest_bs_id + 1)]
    for bs_id in BS_to_UPF_assignment.values():
        count_BSs_per_UPF[bs_id] += 1

    active_BSs = []
    bs: BS
    for bs in G.nodes():
        if bs.get_numUEs() > 0:
            active_BSs.append(bs.get_id())

    f1_latency = tot_latencies/tot_ues
    f2_num_upfs = len(BSs_with_UPF_ids)
    f3_deployment_overhead = get_deployment_overhead(
        BSs_with_UPF_ids_previous, BSs_with_UPF_ids, time_deployment, time_removal, iteration_duration)
    f4_cp_reassignment_overhead_keep = get_control_plane_reassignment_overhead_bs(
        BSs, BS_to_UPF_assignment_previous, BS_to_UPF_assignment, cost_relocation, G_shortest_path_lengths)
    f_objective_keep = get_objective_function(G, f1_latency, f2_num_upfs, f3_deployment_overhead, f4_cp_reassignment_overhead_keep,
                                              alpha1, alpha2, alpha3, alpha4, time_deployment, time_removal, cost_relocation,
                                              num_UEs, max_num_hops)

    number_of_iterations = 100000
    for iteration in range(number_of_iterations):
        # Generate new candidate solution by mutating current solution
        # Pick random active BS
        # Assign all the UEs in the BS to a random BS
        #ue_bs_id_change = active_BSs[randint(0, len(active_BSs) - 1)]
        ue_bs_id_change = choice(active_BSs)
        upf_node_id_change = randint(0, G.number_of_nodes() - 1)

        if upf_node_id_change == BS_to_UPF_assignment[ue_bs_id_change]:
            continue

        new_distance = G_shortest_path_lengths[BSs[ue_bs_id_change]
                                               ][BSs[upf_node_id_change]]
        f1_latency_new = f1_latency + \
            ((new_distance - latencies_list[ue_bs_id_change])
             * num_ues_list[ue_bs_id_change] / tot_ues)

        f2_num_upfs_new = f2_num_upfs
        BSs_with_UPF_added = set()
        BSs_with_UPF_removed = set()
        if upf_node_id_change not in BSs_with_UPF_ids:
            f2_num_upfs_new += 1
            BSs_with_UPF_added.add(upf_node_id_change)
        if count_BSs_per_UPF[BS_to_UPF_assignment[ue_bs_id_change]] <= 1:
            f2_num_upfs_new -= 1
            BSs_with_UPF_removed.add(BS_to_UPF_assignment[ue_bs_id_change])

        # Check that the resulting number of UPFs needed (f2_num_upfs_new) does not exceed
        # the maximum number of UPFs allowed (num_UPFs)
        if f2_num_upfs_new > num_UPFs:
            continue

        f3_deployment_overhead_new = f3_deployment_overhead + get_deployment_overhead(
            BSs_with_UPF_removed, BSs_with_UPF_added, time_deployment, time_removal, iteration_duration)

        f4_cp_reassignment_overhead_relocate = 0
        if ue_bs_id_change in BS_to_UPF_assignment_previous:
            f4_cp_reassignment_overhead_relocate = G_shortest_path_lengths[BSs[
                BS_to_UPF_assignment_previous[ue_bs_id_change]]][BSs[upf_node_id_change]] * cost_relocation
        f_objective_relocate = get_objective_function(G, f1_latency_new,
                                                      f2_num_upfs_new, f3_deployment_overhead_new, f4_cp_reassignment_overhead_relocate,
                                                      alpha1, alpha2, alpha3, alpha4, time_deployment, time_removal, cost_relocation,
                                                      num_UEs, max_num_hops)

        if f_objective_relocate <= f_objective_keep:
            #print("{} {}".format(ue_bs_id_change, upf_node_id_change), file=stderr)
            # if f_objective_function <= previous_f_objective_function:
            # Accept new candidate if it does not degrade (increase) the objective function
            count_BSs_per_UPF[BS_to_UPF_assignment[ue_bs_id_change]] -= 1
            count_BSs_per_UPF[upf_node_id_change] += 1

            BS_to_UPF_assignment[ue_bs_id_change] = upf_node_id_change
            BSs_with_UPF_ids.add(upf_node_id_change)
            BSs_with_UPF_ids -= BSs_with_UPF_removed
            latencies_list[ue_bs_id_change] = new_distance

            assert(f2_num_upfs_new == len(BSs_with_UPF_ids))

            f1_latency = f1_latency_new
            f2_num_upfs = f2_num_upfs_new
            f3_deployment_overhead = f3_deployment_overhead_new
            f4_cp_reassignment_overhead_keep = f4_cp_reassignment_overhead_relocate
            f_objective_keep = f_objective_relocate

    # Convert back to original format BS_to_UPF_assignment -> UE_to_UPF_assignment
    UE_to_UPF_assignment = convert_bs_assignment_to_ue_assignment(
        BSs, BS_to_UPF_assignment)

    return UE_to_UPF_assignment, BSs_with_UPF_ids


def old_UPF_assignment_simulated_annealing_previous(G: nx.Graph, BSs, num_UPFs, UE_to_UPF_assignment_previous, BSs_with_UPF_ids_previous, G_shortest_path_lengths, highest_bs_id, iteration_duration, time_deployment, time_removal, cost_relocation, alpha1, alpha2, alpha3, alpha4, num_UEs, max_num_hops):
    UE_to_UPF_assignment = copy.deepcopy(UE_to_UPF_assignment_previous)
    BSs_with_UPF_ids = copy.deepcopy(BSs_with_UPF_ids_previous)

    #print("\nRunning simulated annealing", file=stderr)

    # If there was not UPFs in the previous interval, assign them using the greedy_overhead algorithm
    if (BSs_with_UPF_ids == None or len(BSs_with_UPF_ids) == 0):
        print("\nRunning old simulated annealing previous", file=stderr)
    UE_to_UPF_assignment, BSs_with_UPF_ids = UPF_assignment_greedy_overhead(G, BSs, num_UPFs, UE_to_UPF_assignment_previous, BSs_with_UPF_ids_previous, G_shortest_path_lengths,
                                                                            highest_bs_id, iteration_duration, time_deployment, time_removal, cost_relocation, alpha1, alpha2, alpha3, alpha4, num_UEs, max_num_hops)

    # Assign users which where not present in the previous interval to the closest active BS

    bs: BS
    for bs in G.nodes():
        _, upf_node = get_minimum_hops_from_BS_to_any_UPF(
            G, BSs, bs, BSs_with_UPF_ids, G_shortest_path_lengths)
        ue: UE
        for ue in bs.get_UEs():
            if ue.get_id() not in UE_to_UPF_assignment:
                UE_to_UPF_assignment[ue.get_id()] = upf_node.get_id()

    # TODO: Use a different data structure rather than UE_to_UPF_assignment aggregating UEs in each BS (e.g., BS_to_UPF_assignment) for speeding up the process

    latencies_list = [
        G.number_of_nodes() + 1 for _ in range(highest_bs_id + 1)]
    num_ues_list = [0 for _ in range(highest_bs_id + 1)]
    tot_ues = 0
    for bs in G.nodes:
        num_ues_list[bs.get_id()] = bs.get_numUEs()
        tot_ues += bs.get_numUEs()

    # f1: Vehicle latency (90-th percentile)
    UE_hops_list = get_UE_hops_list_assignment(
        G, BSs, BSs_with_UPF_ids, UE_to_UPF_assignment, G_shortest_path_lengths)
    f1_num_hops_90th = np_percentile(UE_hops_list, 90)
    # TODO: Try average instead of percentile for speed-up!

    # f2: Resource usage
    f2_num_UPFs = len(BSs_with_UPF_ids)

    # f3: Deployment overhead
    f3_deployment_overhead = get_deployment_overhead(
        BSs_with_UPF_ids_previous, BSs_with_UPF_ids, time_deployment, time_removal, iteration_duration)

    # f4: Control-plane reassignment overhead
    f4_control_plane_reassignment_overhead = get_control_plane_reassignment_overhead(
        BSs, UE_to_UPF_assignment_previous, UE_to_UPF_assignment, cost_relocation, G_shortest_path_lengths)

    # f: Objective function
    previous_f_objective_function = get_objective_function(G, f1_num_hops_90th, f2_num_UPFs, f3_deployment_overhead, f4_control_plane_reassignment_overhead,
                                                           alpha1, alpha2, alpha3, alpha4, time_deployment, time_removal, cost_relocation, num_UEs, max_num_hops)

    active_BSs = []
    bs: BS
    for bs in G.nodes():
        if bs.get_numUEs() > 0:
            active_BSs.append(bs.get_id())

    number_of_iterations = 10
    for iteration in range(number_of_iterations):
        BSs_with_UPF_ids_tmp = copy.deepcopy(BSs_with_UPF_ids)
        UE_to_UPF_assignment_tmp = copy.deepcopy(UE_to_UPF_assignment)
        latencies_list_tmp = copy.deepcopy(latencies_list)

        # Generate new candidate solution by mutating current solution
        # Pick random active BS
        # Assign all the UEs in the BS to a random BS
        ue_bs_id_change = active_BSs[randint(0, len(active_BSs) - 1)]
        upf_node_id_change = randint(0, G.number_of_nodes() - 1)

        for ue in BSs[ue_bs_id_change].get_UEs():
            UE_to_UPF_assignment_tmp[ue.get_id()] = upf_node_id_change

        # Recalculate set of active BSs
        BSs_with_UPF_ids_tmp = set(UE_to_UPF_assignment_tmp.values())

        # f1: Vehicle latency (90-th percentile)

        UE_hops_list = get_UE_hops_list_assignment(
            G, BSs, BSs_with_UPF_ids, UE_to_UPF_assignment_tmp, G_shortest_path_lengths)
        f1_num_hops_90th = np_percentile(UE_hops_list, 90)
        # TODO: Try average instead of percentile for speed-up!

        # f2: Resource usage
        f2_num_UPFs = len(BSs_with_UPF_ids_tmp)

        # f3: Deployment overhead
        f3_deployment_overhead = get_deployment_overhead(
            BSs_with_UPF_ids_previous, BSs_with_UPF_ids_tmp, time_deployment, time_removal, iteration_duration)

        # f4: Control-plane reassignment overhead
        f4_control_plane_reassignment_overhead = get_control_plane_reassignment_overhead(
            BSs, UE_to_UPF_assignment_previous, UE_to_UPF_assignment_tmp, cost_relocation, G_shortest_path_lengths)

        # f: Objective function
        f_objective_function = get_objective_function(G, f1_num_hops_90th, f2_num_UPFs, f3_deployment_overhead, f4_control_plane_reassignment_overhead,
                                                      alpha1, alpha2, alpha3, alpha4, time_deployment, time_removal, cost_relocation, num_UEs, max_num_hops)

        if f_objective_function <= previous_f_objective_function:
            # Accept new candidate if it decreases the objective function
            UE_to_UPF_assignment = copy.deepcopy(UE_to_UPF_assignment_tmp)
            BSs_with_UPF_ids = copy.deepcopy(BSs_with_UPF_ids_tmp)
            latencies_list = copy.deepcopy(latencies_list_tmp)
            f_objective_function = previous_f_objective_function

    return UE_to_UPF_assignment, BSs_with_UPF_ids


def UPF_assignment_greedy_overhead_per_bs(G: nx.Graph, BSs, num_UPFs, UE_to_UPF_assignment_previous, BSs_with_UPF_ids_previous, G_shortest_path_lengths, highest_bs_id, iteration_duration, time_deployment, time_removal, cost_relocation, alpha1, alpha2, alpha3, alpha4, num_UEs, max_num_hops):
    #UE_to_UPF_assignment = {}
    BS_to_UPF_assignment_previous = convert_ue_assignment_to_bs_assignment(
        BSs, UE_to_UPF_assignment_previous)

    BS_to_UPF_assignment = {}
    BSs_with_UPF_ids = set()

    bs: BS

    latencies_list = [
        G.number_of_nodes() + 1 for _ in range(highest_bs_id + 1)]
    num_ues_list = [0 for _ in range(highest_bs_id + 1)]
    tot_ues = 0
    for bs in G.nodes:
        num_ues_list[bs.get_id()] = bs.get_numUEs()
        tot_ues += bs.get_numUEs()

    done_BSs = [False for _ in range(highest_bs_id + 1)]

    for _ in range(num_UPFs):
        best_bs = None
        best_f_objective_function = None
        best_BSs_with_UPF_ids = None
        #best_UE_to_UPF_assignment = None
        best_BS_to_UPF_assignment = None
        best_latencies_list = None
        for bs in G.nodes:
            if not done_BSs[bs.get_id()]:
                # Check objective function if bs is selected
                BSs_with_UPF_ids_tmp = copy.deepcopy(BSs_with_UPF_ids)
                #UE_to_UPF_assignment_tmp = copy.deepcopy(UE_to_UPF_assignment)
                BS_to_UPF_assignment_tmp = copy.deepcopy(BS_to_UPF_assignment)
                latencies_list_tmp = copy.deepcopy(latencies_list)

                BSs_with_UPF_ids_tmp.add(bs.get_id())

                # f1: Vehicle latency (90-th percentile)
                acc_latency = 0
                for bs2 in G.nodes:
                    if bs2.get_numUEs() == 0:
                        continue
                    new_latency = latencies_list_tmp[bs2.get_id()]

                    # new_latency = min(
                    #     G_shortest_path_lengths[bs2][bs], latencies_list[bs2.get_id()])

                    f_objective_keep = get_objective_function(G, latencies_list_tmp[bs2.get_id(
                    )], 0, 0, 0, alpha1, alpha2, alpha3, alpha4, time_deployment, time_removal, cost_relocation, num_UEs, max_num_hops)
                    f_objective_relocate = get_objective_function(G, G_shortest_path_lengths[bs2][bs], 0, 0, G_shortest_path_lengths[bs2][
                                                                  bs] * cost_relocation, alpha1, alpha2, alpha3, alpha4, time_deployment, time_removal, cost_relocation, num_UEs, max_num_hops)

                    # if G_shortest_path_lengths[bs2][bs] < latencies_list[bs2.get_id()]:
                    if f_objective_relocate < f_objective_keep:
                        # TODO: Update criteria (f_objective_function instead of latency)
                        new_latency = G_shortest_path_lengths[bs2][bs]
                        latencies_list_tmp[bs2.get_id()] = new_latency
                        # Update the assignment of all the UEs in bs2
                        BS_to_UPF_assignment_tmp[bs2.get_id()] = bs.get_id()
                        # for ue in bs2.get_UEs():
                        #     UE_to_UPF_assignment_tmp[ue.get_id()] = bs.get_id()

                    acc_latency += (new_latency * num_ues_list[bs2.get_id()])

                # f2: Resource usage
                f2_num_UPFs = len(BSs_with_UPF_ids_tmp)

                # f3: Deployment overhead
                f3_deployment_overhead = get_deployment_overhead(
                    BSs_with_UPF_ids_previous, BSs_with_UPF_ids_tmp, time_deployment, time_removal, iteration_duration)

                # f4: Control-plane reassignment overhead
                # f4_control_plane_reassignment_overhead = get_control_plane_reassignment_overhead(
                #     BSs, UE_to_UPF_assignment_previous, UE_to_UPF_assignment_tmp, cost_relocation, G_shortest_path_lengths)

                f4_control_plane_reassignment_overhead = get_control_plane_reassignment_overhead_bs(
                    BSs, BS_to_UPF_assignment_previous, BS_to_UPF_assignment_tmp, cost_relocation, G_shortest_path_lengths)

                # Calculate average latency
                f1_num_hops_average = acc_latency / tot_ues

                # f: Objective function
                f_objective_function = get_objective_function(G, f1_num_hops_average, f2_num_UPFs, f3_deployment_overhead, f4_control_plane_reassignment_overhead,
                                                              alpha1, alpha2, alpha3, alpha4, time_deployment, time_removal, cost_relocation, num_UEs, max_num_hops)

                if best_bs == None or f_objective_function < best_f_objective_function:
                    best_bs = bs
                    best_f_objective_function = f_objective_function
                    best_BSs_with_UPF_ids = copy.deepcopy(BSs_with_UPF_ids_tmp)
                    best_BS_to_UPF_assignment = copy.deepcopy(
                        BS_to_UPF_assignment_tmp)
                    best_latencies_list = copy.deepcopy(latencies_list_tmp)

        # NOTE: Experimental
        # if previous_f_objective_function != None and best_f_objective_function > previous_f_objective_function:
        #     # Do not add more UPFs if the best objective function increases
        #     break

        previous_f_objective_function = best_f_objective_function

        upf_node = best_bs.get_id()
        done_BSs[upf_node] = True
        # BSs_with_UPF_ids.add(upf_node)
        # for ue in bs.get_UEs():
        #    UE_to_UPF_assignment[ue.get_id()] = upf_node

        BSs_with_UPF_ids = copy.deepcopy(best_BSs_with_UPF_ids)
        BS_to_UPF_assignment = copy.deepcopy(best_BS_to_UPF_assignment)
        latencies_list = copy.deepcopy(best_latencies_list)

        # for bs2 in G.nodes:
        #     new_latency = G_shortest_path_lengths[bs2][best_bs]
        #     if new_latency < latencies_list[bs2.get_id()]:
        #         latencies_list[bs2.get_id()] = new_latency

    # Convert back to original format BS_to_UPF_assignment -> UE_to_UPF_assignment
    UE_to_UPF_assignment = convert_bs_assignment_to_ue_assignment(
        BSs, BS_to_UPF_assignment)

    return UE_to_UPF_assignment, BSs_with_UPF_ids


def UPF_assignment_greedy_overhead_old_per_ue(G: nx.Graph, BSs, num_UPFs, UE_to_UPF_assignment_previous, BSs_with_UPF_ids_previous, G_shortest_path_lengths, highest_bs_id, iteration_duration, time_deployment, time_removal, cost_relocation, alpha1, alpha2, alpha3, alpha4, num_UEs, max_num_hops):
    UE_to_UPF_assignment = {}
    BSs_with_UPF_ids = set()

    bs: BS

    latencies_list = [
        G.number_of_nodes() + 1 for _ in range(highest_bs_id + 1)]
    num_ues_list = [0 for _ in range(highest_bs_id + 1)]
    tot_ues = 0
    for bs in G.nodes:
        num_ues_list[bs.get_id()] = bs.get_numUEs()
        tot_ues += bs.get_numUEs()

    done_BSs = [False for _ in range(highest_bs_id + 1)]

    previous_f_objective_function = None

    for _ in range(num_UPFs):
        best_bs = None
        best_f_objective_function = None
        best_BSs_with_UPF_ids = None
        best_UE_to_UPF_assignment = None
        best_latencies_list = None
        for bs in G.nodes:
            if not done_BSs[bs.get_id()]:
                # Check objective function if bs is selected
                BSs_with_UPF_ids_tmp = copy.deepcopy(BSs_with_UPF_ids)
                UE_to_UPF_assignment_tmp = copy.deepcopy(UE_to_UPF_assignment)
                latencies_list_tmp = copy.deepcopy(latencies_list)

                BSs_with_UPF_ids_tmp.add(bs.get_id())

                # f1: Vehicle latency (90-th percentile)
                acc_latency = 0
                for bs2 in G.nodes:
                    new_latency = latencies_list_tmp[bs2.get_id()]

                    # new_latency = min(
                    #     G_shortest_path_lengths[bs2][bs], latencies_list[bs2.get_id()])

                    f_objective_keep = get_objective_function(G, latencies_list_tmp[bs2.get_id(
                    )], 0, 0, 0, alpha1, alpha2, alpha3, alpha4, time_deployment, time_removal, cost_relocation, num_UEs, max_num_hops)
                    f_objective_relocate = get_objective_function(G, G_shortest_path_lengths[bs2][bs], 0, 0, G_shortest_path_lengths[bs2][
                                                                  bs] * cost_relocation, alpha1, alpha2, alpha3, alpha4, time_deployment, time_removal, cost_relocation, num_UEs, max_num_hops)

                    # if G_shortest_path_lengths[bs2][bs] < latencies_list[bs2.get_id()]:
                    if f_objective_relocate < f_objective_keep:
                        # TODO: Update criteria (f_objective_function instead of latency)
                        new_latency = G_shortest_path_lengths[bs2][bs]
                        latencies_list_tmp[bs2.get_id()] = new_latency
                        # Update the assignment of all the UEs in bs2
                        for ue in bs2.get_UEs():
                            UE_to_UPF_assignment_tmp[ue.get_id()] = bs.get_id()

                    acc_latency += (new_latency * num_ues_list[bs2.get_id()])

                # f2: Resource usage
                f2_num_UPFs = len(BSs_with_UPF_ids_tmp)

                # f3: Deployment overhead
                f3_deployment_overhead = get_deployment_overhead(
                    BSs_with_UPF_ids_previous, BSs_with_UPF_ids_tmp, time_deployment, time_removal, iteration_duration)

                # f4: Control-plane reassignment overhead
                f4_control_plane_reassignment_overhead = get_control_plane_reassignment_overhead(
                    BSs, UE_to_UPF_assignment_previous, UE_to_UPF_assignment_tmp, cost_relocation, G_shortest_path_lengths)

                # Calculate average latency
                f1_num_hops_average = acc_latency / tot_ues

                # f: Objective function
                f_objective_function = get_objective_function(G, f1_num_hops_average, f2_num_UPFs, f3_deployment_overhead, f4_control_plane_reassignment_overhead,
                                                              alpha1, alpha2, alpha3, alpha4, time_deployment, time_removal, cost_relocation, num_UEs, max_num_hops)

                if best_bs == None or f_objective_function < best_f_objective_function:
                    best_bs = bs
                    best_f_objective_function = f_objective_function
                    best_BSs_with_UPF_ids = copy.deepcopy(BSs_with_UPF_ids_tmp)
                    best_UE_to_UPF_assignment = copy.deepcopy(
                        UE_to_UPF_assignment_tmp)
                    best_latencies_list = copy.deepcopy(latencies_list_tmp)

        # NOTE: Experimental
        # if previous_f_objective_function != None and best_f_objective_function > previous_f_objective_function:
        #     # Do not add more UPFs if the best objective function increases
        #     break

        previous_f_objective_function = best_f_objective_function

        upf_node = best_bs.get_id()
        done_BSs[upf_node] = True
        # BSs_with_UPF_ids.add(upf_node)
        # for ue in bs.get_UEs():
        #    UE_to_UPF_assignment[ue.get_id()] = upf_node

        BSs_with_UPF_ids = copy.deepcopy(best_BSs_with_UPF_ids)
        UE_to_UPF_assignment = copy.deepcopy(best_UE_to_UPF_assignment)
        latencies_list = copy.deepcopy(best_latencies_list)

        # for bs2 in G.nodes:
        #     new_latency = G_shortest_path_lengths[bs2][best_bs]
        #     if new_latency < latencies_list[bs2.get_id()]:
        #         latencies_list[bs2.get_id()] = new_latency

    return UE_to_UPF_assignment, BSs_with_UPF_ids

# Legacy assignment methods (i.e., they do not consider deployment and reassignment overheads)


def UPF_assignment_old(algorithm, G: nx.Graph, BSs, num_UPFs, UE_to_UPF_assignment_previous, BSs_with_UPF_ids_previous, G_shortest_path_lengths, highest_bs_id):
    UE_to_UPF_assignment = {}
    BSs_with_UPF_ids = globals()["UPF_allocation_{}".format(algorithm[4:])](
        G, num_UPFs, BSs_with_UPF_ids_previous, G_shortest_path_lengths, highest_bs_id)

    bs: BS
    for bs in G.nodes():
        _, upf_node = get_minimum_hops_from_BS_to_any_UPF(
            G, BSs, bs, BSs_with_UPF_ids, G_shortest_path_lengths)
        ue: UE
        for ue in bs.get_UEs():
            # Determine which node will have a UPF to serve this UE
            UE_to_UPF_assignment[ue.get_id()] = upf_node.get_id()

    return UE_to_UPF_assignment, BSs_with_UPF_ids

# Determines which UPF is used to serve each UE


# def UPF_assignment(algorithm, G, num_UPFs, UE_to_UPF_assignment_previous, BSs_with_UPF_ids_previous, G_shortest_path_lengths, highest_bs_id):
def UPF_assignment(algorithm, G: nx.Graph, BSs, num_UPFs, UE_to_UPF_assignment_previous, BSs_with_UPF_ids_previous, G_shortest_path_lengths, highest_bs_id, iteration_duration, time_deployment, time_removal, cost_relocation, alpha1, alpha2, alpha3, alpha4, num_UEs, max_num_hops):
    if "old_" in algorithm:
        UE_to_UPF_assignment, BSs_with_UPF_ids = UPF_assignment_old(algorithm, G, BSs, num_UPFs, UE_to_UPF_assignment_previous,
                                                                    BSs_with_UPF_ids_previous, G_shortest_path_lengths, highest_bs_id)
    else:
        UE_to_UPF_assignment, BSs_with_UPF_ids = globals()["UPF_assignment_{}".format(
            algorithm)](G, BSs, num_UPFs, UE_to_UPF_assignment_previous, BSs_with_UPF_ids_previous, G_shortest_path_lengths, highest_bs_id, iteration_duration, time_deployment, time_removal, cost_relocation, alpha1, alpha2, alpha3, alpha4, num_UEs, max_num_hops)

    if DEBUG:
        print(UE_to_UPF_assignment)
        print(BSs_with_UPF_ids)

    return UE_to_UPF_assignment, BSs_with_UPF_ids


# Metrics
def get_minimum_hops_from_BS_to_UPF(G, bs, upf_assigned, G_shortest_path_lengths):
    # hops = max(G_shortest_path_lengths[bs][other_bs] - 1, 0) #TODO: SHOULD WE INCLUDE THE -1 OR NOT?
    hops = max(G_shortest_path_lengths[bs][upf_assigned], 0)
    return hops


def get_minimum_hops_from_BS_to_any_UPF(G, BSs, bs, BSs_with_UPF_ids, G_shortest_path_lengths):
    hops = None
    bs_with_upf = None
    for other_bs_id in BSs_with_UPF_ids:
        other_bs = BSs[other_bs_id]
        try:
            # h = len(nx.shortest_path(G, source=bs,
            #                          target=other_bs)) - 1  # Dijkstra
            # Pre-computed Floyd-Wharsall
            # h = G_shortest_path_lengths[bs][other_bs] - 1 #TODO: SHOULD WE INCLUDE THE -1 OR NOT? -> NOT
            h = G_shortest_path_lengths[bs][other_bs]
        except:
            continue
        if hops is None or h < hops:
            hops = h
            bs_with_upf = other_bs

    if hops is None:
        # To handle the cases in which there is not an UPF deployed
        hops = G.number_of_nodes() / 2
        # raise Exception("No reachable UPF from BS {}".format(bs.get_id()))

    return hops, bs_with_upf


''' Returns a list with the number of hops to the nearest UPF for each UE
    NOTE: The indexes of the list do not correspond to the IDs of the UEs
'''


def get_UE_hops_list(G, BSs, BSs_with_UPF_ids, G_shortest_path_lengths):
    UE_hops_list = []
    for bs in G.nodes:
        num_UEs = bs.get_numUEs()
        if num_UEs < 1:
            continue
        hops, _ = get_minimum_hops_from_BS_to_any_UPF(
            G, BSs, bs, BSs_with_UPF_ids, G_shortest_path_lengths)
        UE_hops_list.extend([hops]*num_UEs)
    return UE_hops_list


def get_UE_hops_list_assignment(G, BSs, BSs_with_UPF_ids, UE_to_UPF_assignment, G_shortest_path_lengths):
    UE_hops_list = []

    bs: BS
    for bs in G.nodes:
        ue: UE
        for ue in bs.get_UEs():
            hops = None
            if ue.get_id() in UE_to_UPF_assignment:
                upf_assigned = BSs[UE_to_UPF_assignment[ue.get_id()]]
                hops = get_minimum_hops_from_BS_to_UPF(
                    G, bs, upf_assigned, G_shortest_path_lengths)
            else:
                hops, _ = get_minimum_hops_from_BS_to_any_UPF(
                    G, BSs, bs, BSs_with_UPF_ids, G_shortest_path_lengths)
            UE_hops_list.append(hops)
    return UE_hops_list


def get_deployment_overhead(BSs_with_UPF_previous, BSs_with_UPF, time_deployment, time_removal, slot_duration):
    intersection_size = len(BSs_with_UPF_previous & BSs_with_UPF)
    deployed_upfs = len(BSs_with_UPF) - intersection_size
    removed_upfs = len(BSs_with_UPF_previous) - intersection_size

    deployment_overhead = (deployed_upfs *
                           time_deployment + removed_upfs * time_removal)  # / slot_duration

    # print(BSs_with_UPF_previous, BSs_with_UPF)

    # print("+{} -{} -> {}".format(deployed_upfs, removed_upfs, deployment_overhead), file=stderr)

    return deployment_overhead


def check_assignment_ok(BSs, UE_to_UPF_assignment, BSs_with_UPF_ids):
    # Checks that all users have been assigned
    bs: BS
    for bs in BSs.values():
        ue: UE
        for ue in bs.get_UEs():
            if ue.get_id() not in UE_to_UPF_assignment:
                print("UE {} not in {}".format(
                    ue.get_id(), UE_to_UPF_assignment))
                return False
            if UE_to_UPF_assignment[ue.get_id()] not in BSs_with_UPF_ids:
                print("UPF {} not in {}".format(
                    UE_to_UPF_assignment[ue.get_id()], BSs_with_UPF_ids))
                return False
    return True


def get_control_plane_reassignment_overhead(BSs, UE_to_UPF_assignment_previous, UE_to_UPF_assignment, cost_relocation, G_shortest_path_lengths):
    control_plane_reassignment_overhead = 0
    for ue in UE_to_UPF_assignment:
        upf_old = None
        if ue in UE_to_UPF_assignment_previous:
            upf_old = UE_to_UPF_assignment_previous[ue]
        upf_new = UE_to_UPF_assignment[ue]
        if upf_old != upf_new and upf_old != None:
            control_plane_reassignment_overhead += cost_relocation * \
                G_shortest_path_lengths[BSs[upf_old]][BSs[upf_new]]

    return control_plane_reassignment_overhead


def get_control_plane_reassignment_overhead_bs(BSs, BS_to_UPF_assignment_previous, BS_to_UPF_assignment, cost_relocation, G_shortest_path_lengths):
    control_plane_reassignment_overhead = 0
    for bs_id in BS_to_UPF_assignment:
        upf_old = None
        if bs_id in BS_to_UPF_assignment_previous:
            upf_old = BS_to_UPF_assignment_previous[bs_id]
        upf_new = BS_to_UPF_assignment[bs_id]
        if upf_old != upf_new and upf_old != None:
            control_plane_reassignment_overhead += cost_relocation * \
                G_shortest_path_lengths[BSs[upf_old]
                                        ][BSs[upf_new]] * BSs[bs_id].get_numUEs()

    return control_plane_reassignment_overhead


def get_objective_function(G: nx.Graph, f1_num_hops, f2_num_UPFs, f3_deployment_overhead, f4_control_plane_reassignment_overhead,
                           alpha1, alpha2, alpha3, alpha4, time_deployment, time_removal, cost_relocation, num_UEs, max_num_hops):
    max_num_UPFs = G.number_of_nodes()
    max_deployment_overhead = max_num_UPFs * max(time_deployment, time_removal)
    max_control_plane_reassignment_overhead = (
        1 + num_UEs) * max_num_hops * cost_relocation
    return alpha1 * f1_num_hops / max_num_hops + \
        alpha2 * f2_num_UPFs / max_num_UPFs + \
        alpha3 * f3_deployment_overhead / max_deployment_overhead + \
        alpha4 * f4_control_plane_reassignment_overhead / \
        max_control_plane_reassignment_overhead

# Auxiliar functions for converting formats


def convert_bs_assignment_to_ue_assignment(BSs, BS_to_UPF_assignment):
    UE_to_UPF_assignment = {}
    for bs_id in BS_to_UPF_assignment:
        bs = BSs[bs_id]
        for ue in bs.get_UEs():
            UE_to_UPF_assignment[ue.get_id()] = BS_to_UPF_assignment[bs_id]

    return UE_to_UPF_assignment


def convert_ue_assignment_to_bs_assignment(BSs, UE_to_UPF_assignment):
    BS_to_UPF_assignment = {}
    for bs_id in BSs:
        bs = BSs[bs_id]
        for ue in bs.get_UEs():
            # Assumes all UEs are assigned to the same UPF
            if ue.get_id() in UE_to_UPF_assignment:
                BS_to_UPF_assignment[bs_id] = UE_to_UPF_assignment[ue.get_id()]
            break

    return BS_to_UPF_assignment


# Used for debugging


def print_statistics(UE_hops_list, file=stdout):
    if file != None:
        print("Minimum: {}".format(min(UE_hops_list)), file=file)
        print("Maximum: {}".format(max(UE_hops_list)), file=file)
        print("Mean: {}".format(statistics.mean(UE_hops_list)), file=file)
        if len(UE_hops_list) > 1:
            print("Variance: {}".format(
                statistics.variance(UE_hops_list)), file=file)
            print("Standard deviation: {}".format(
                statistics.stdev(UE_hops_list)), file=file)


def mean_confidence_interval(data, confidence=0.95):
    if (min(data) == max(data)):
        m = min(data)
        h = 0
    else:
        a = 1.0*np_array(data)
        n = len(a)
        m, se = np_mean(a), st_sem(a)
        h = se * st_t._ppf((1+confidence)/2., n-1)
    # return '{:.3f} {:.3f} {:.3f}'.format(m, max(m-h, 0), m+h)
    return (m, max(m-h, 0), m+h)


def main():
    seed(0)

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--algorithm", help="Specifies the UPF allocation algorithm [Supported: old_random/old_greedy_percentile/old_greedy_average/old_kmeans_greedy_average/old_modularity_greedy_average/greedy_overhead].", required=True)
    parser.add_argument(
        "--minUPFs", help="Specifies the minimum number of UPFs to be allocated [Default: {}].".format(DEFAULT_MIN_UPF), type=int, default=DEFAULT_MIN_UPF)
    parser.add_argument(
        "--maxUPFs", help="Specifies the maximum number of UPFs to be allocated [Default: {}].".format(DEFAULT_MAX_UPF), type=int, default=DEFAULT_MAX_UPF)
    parser.add_argument(
        "--bsFile", help="File containing the information about the base stations [Format: each line contains the id, x coordinate and y coordinate of a base station separated by spaces].", required=True)
    parser.add_argument(
        "--ueFile", help="File containing the information about the users throughout the simulation [Format: each line contains the timestamp, user id, x coordinate, y coordinate, speed and, optionally, the base station id to which the user is attached].", required=True)
    parser.add_argument(
        "--iterationDuration", help="Duration of each time-slot in seconds [Default: {}].".format(DEFAULT_ITERATION_DURATION), type=int, default=DEFAULT_ITERATION_DURATION)
    parser.add_argument(
        "--timeDeployment", help="Time required for deploying an anchor point in seconds [Default: {}].".format(DEFAULT_TIME_DEPLOYMENT), type=float, default=DEFAULT_TIME_DEPLOYMENT)
    parser.add_argument(
        "--timeRemoval", help="Time required for removing an anchor point in seconds [Default: {}].".format(DEFAULT_TIME_REMOVAL), type=float, default=DEFAULT_TIME_REMOVAL)
    parser.add_argument(
        "--costRelocation", help="Cost for relocating the communications of a vehicle [Default: {}].".format(DEFAULT_COST_RELOCATION), type=int, default=DEFAULT_COST_RELOCATION)
    parser.add_argument(
        "--alpha1", help="Weight for the first parameter of the objective function (latency) [Default: {}].".format(DEFAULT_ALPHA1), type=float, default=DEFAULT_ALPHA1)
    parser.add_argument(
        "--alpha2", help="Weight for the first parameter of the objective function (latency) [Default: {}].".format(DEFAULT_ALPHA2), type=float, default=DEFAULT_ALPHA2)
    parser.add_argument(
        "--alpha3", help="Weight for the first parameter of the objective function (latency) [Default: {}].".format(DEFAULT_ALPHA3), type=float, default=DEFAULT_ALPHA3)
    parser.add_argument(
        "--alpha4", help="Weight for the first parameter of the objective function (latency) [Default: {}].".format(DEFAULT_ALPHA4), type=float, default=DEFAULT_ALPHA4)
    args = parser.parse_args()
    algorithm = args.algorithm
    min_UPFs = args.minUPFs
    max_UPFs = args.maxUPFs
    ue_file = args.ueFile
    bs_file = args.bsFile
    iteration_duration = args.iterationDuration
    time_deployment = args.timeDeployment
    time_removal = args.timeRemoval
    cost_relocation = args.costRelocation
    alpha1 = args.alpha1
    alpha2 = args.alpha2
    alpha3 = args.alpha3
    alpha4 = args.alpha4

    # Generate graph
    G, BSs, G_shortest_path_lengths, highest_bs_id, max_num_hops = generate_graph(
        bs_file)

    for num_UPFs in range(min_UPFs, max_UPFs + 1):

        list_results_num_hops = []
        list_results_elapsed_time = []
        list_results_num_upfs = []
        list_results_deployment_overhead = []
        list_results_control_plane_reassignment_overhead = []
        list_results_objective_function = []

        global cluster_BSs_list
        if "kmeans" in algorithm:
            # Calculate k-means clustering considering all BSs
            features = []
            for bs in G.nodes:
                features.append([bs.get_x(), bs.get_y()])
            global kmeans
            kmeans = KMeans(
                init="k-means++",  # "random" / "k-means++"
                n_clusters=min(num_UPFs, len(features)),
                n_init=2,
                max_iter=1000,
                random_state=0  # To allow for reproducibility
            )

            kmeans.fit(features)

        for _ in range(1):
            # Clear UEs in BSs
            for bs in G.nodes():
                bs.clear_UEs()

            iteration = 0
            # First UPF allocation is random
            # BSs_with_UPF_ids = UPF_allocation(
            #     "random", G, None, G_shortest_path_lengths, highest_bs_id)

            UE_to_UPF_assignment_previous = {}
            BSs_with_UPF_ids_previous = set()

            UE_to_UPF_assignment, BSs_with_UPF_ids = UPF_assignment("old_random", G, BSs, num_UPFs, None, None, G_shortest_path_lengths,
                                                                    highest_bs_id, iteration_duration, time_deployment, time_removal, cost_relocation, alpha1, alpha2, alpha3, alpha4, 0, max_num_hops)

            upf_start_time = time.process_time()  # seconds
            print("Running {} for {} UPFs".format(
                algorithm, num_UPFs), file=stderr)
            for UEs in read_UE_data(ue_file, BSs, iteration_duration):
                iteration += 1

                # Calculate metrics
                # f1: Vehicle latency (90-th percentile)

                UE_hops_list = get_UE_hops_list_assignment(
                    G, BSs, BSs_with_UPF_ids, UE_to_UPF_assignment, G_shortest_path_lengths)

                f1_num_hops_90th = np_percentile(UE_hops_list, 90)
                list_results_num_hops.append(f1_num_hops_90th)
                # print(UE_hops_list, file=stderr)

                # f2: Resource usage
                f2_num_UPFs = len(BSs_with_UPF_ids)
                list_results_num_upfs.append(f2_num_UPFs)

                # f3: Deployment overhead
                f3_deployment_overhead = get_deployment_overhead(
                    BSs_with_UPF_ids_previous, BSs_with_UPF_ids, time_deployment, time_removal, iteration_duration)

                list_results_deployment_overhead.append(f3_deployment_overhead)

                # f4: Control-plane reassignment overhead
                f4_control_plane_reassignment_overhead = get_control_plane_reassignment_overhead(
                    BSs, UE_to_UPF_assignment_previous, UE_to_UPF_assignment, cost_relocation, G_shortest_path_lengths)
                list_results_control_plane_reassignment_overhead.append(
                    f4_control_plane_reassignment_overhead)

                # f: Objective function
                f_objective_function = get_objective_function(G, f1_num_hops_90th, f2_num_UPFs, f3_deployment_overhead, f4_control_plane_reassignment_overhead,
                                                              alpha1, alpha2, alpha3, alpha4, time_deployment, time_removal, cost_relocation, len(UEs), max_num_hops)

                list_results_objective_function.append(f_objective_function)

                if DEBUG:
                    print("UE hops to UPF: {}".format(
                        UE_hops_list), file=stderr)
                    print_statistics(UE_hops_list, file=stderr)
                    print("\n\n", file=stderr)

                start_time = time.process_time() * 1e3  # milliseconds

                # Calculate assignment for the next time slot

                BSs_with_UPF_ids_previous = BSs_with_UPF_ids
                UE_to_UPF_assignment_previous = UE_to_UPF_assignment

                UE_to_UPF_assignment, BSs_with_UPF_ids = UPF_assignment(algorithm, G, BSs, num_UPFs, UE_to_UPF_assignment_previous, BSs_with_UPF_ids_previous,
                                                                        G_shortest_path_lengths, highest_bs_id, iteration_duration, time_deployment, time_removal, cost_relocation, alpha1, alpha2, alpha3, alpha4, len(UEs), max_num_hops)

                assert check_assignment_ok(
                    BSs, UE_to_UPF_assignment, BSs_with_UPF_ids)

                # print(UE_to_UPF_assignment)

                end_time = time.process_time() * 1e3  # milliseconds
                elapsed_time = end_time - start_time

                list_results_elapsed_time.append(elapsed_time)

                print("\r  Iteration {}: {} UES, {} hops, {:.3f} ms".format(
                    iteration, len(UEs), int(f1_num_hops_90th), elapsed_time), end='', file=stderr)

                # if (iteration % 3000 == 0):
                #     print()
                #     analyze_allocation(G, BSs_with_UPF_ids)

        print("\r                                    \r  Number of iterations: {}".format(
            iteration), file=stderr)

        upf_end_time = time.process_time()  # seconds
        upf_elapsed_time = upf_end_time - upf_start_time

        mci_num_hops = mean_confidence_interval(list_results_num_hops, 0.95)
        mci_elapsed_time = mean_confidence_interval(
            list_results_elapsed_time, 0.95)
        mci_num_upfs = mean_confidence_interval(
            list_results_num_upfs, 0.95)
        mci_deployment_overhead = mean_confidence_interval(
            list_results_deployment_overhead, 0.95)
        mci_control_plane_reassignment_overhead = mean_confidence_interval(
            list_results_control_plane_reassignment_overhead, 0.95)
        mci_objective_function = mean_confidence_interval(
            list_results_objective_function, 0.95)

        print("  Elapsed time: {:.3f} seconds".format(
            upf_elapsed_time), file=stderr)

        print("  90th hops: {} {} seconds".format(
            mci_num_hops, mci_num_upfs), file=stderr)

        print("{} {} {:.3f} {:.3f} {:.3f} "
              "{:.3f} {:.3f} {:.3f} "
              "{:.3f} {:.3f} {:.3f} "
              "{:.3f} {:.3f} {:.3f} "
              "{:.3f} {:.3f} {:.3f} "
              "{:.3f} {:.3f} {:.3f}".format(algorithm, num_UPFs, mci_num_hops[0], mci_num_hops[1],
                                            mci_num_hops[2], mci_elapsed_time[0], mci_elapsed_time[1],
                                            mci_elapsed_time[2], mci_num_upfs[0], mci_num_upfs[1],
                                            mci_num_upfs[2], mci_deployment_overhead[0],
                                            mci_deployment_overhead[1], mci_deployment_overhead[2],
                                            mci_control_plane_reassignment_overhead[0],
                                            mci_control_plane_reassignment_overhead[1],
                                            mci_control_plane_reassignment_overhead[2],
                                            mci_objective_function[0], mci_objective_function[1],
                                            mci_objective_function[2]))


if __name__ == "__main__":
    main()

