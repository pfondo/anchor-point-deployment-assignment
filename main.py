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

DEFAULT_ALPHA1 = 0.5
DEFAULT_ALPHA2 = 0.25
DEFAULT_ALPHA3 = 0.25


class UE:
    def __init__(self, id, x=0, y=0, bs=None, pred_x=0, pred_y=0, pred_bs=None):
        self._id = id
        self._x = float(x)
        self._y = float(y)
        self._bs = bs
        self._pred_bs = pred_bs
        self._pred_x = pred_x
        self._pred_y = pred_y
        if self._bs is not None:
            self._bs.add_UE(self)
        if self._pred_bs is not None:
            self._pred_bs.add_pred_UE(self)

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

    def get_pred_coords(self):
        return [self._pred_x, self._pred_y]

    def set_pred_coords(self, pred_x, pred_y):
        self._pred_x, self._pred_y = (float(pred_x), float(pred_y))

    def get_pred_bs(self):
        return self._pred_bs

    def set_pred_bs(self, pred_bs):
        if self._pred_bs is not None:
            self._pred_bs.remove_pred_UE(self)
        if pred_bs is not None:
            pred_bs.add_pred_UE(self)
        self._pred_bs = pred_bs

    def get_id(self):
        return self._id

    def get_pl(self):
        if self._bs is None:
            return float("inf")
        else:
            distance = self._bs.get_distance_coords(self._x, self._y)
            return compute_path_loss(distance)

    def get_pred_pl(self):
        if self._bs is None:
            return float("inf")
        else:
            distance = self._bs.get_distance_coords(self._pred_x, self._pred_y)
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

    def update_pred_bs(self, pred_bs, pl=float("inf")):
        if pred_bs is None or pl + PL_THRESHOLD < self.get_pred_pl():
            self.set_pred_bs(pred_bs)

    def update_pred(self, pred_x, pred_y, pred_bs, pl=float("inf")):
        self.set_pred_coords(pred_x, pred_y)
        self.update_pred_bs(pred_bs, pl)

    def update_pred_unconditional(self, pred_x, pred_y, pred_bs):
        self.set_pred_coords(pred_x, pred_y)
        self.set_pred_bs(pred_bs)


class BS:
    def __init__(self, id_, x, y, UPF=False):
        self._id = int(id_)
        self._x = float(x)
        self._y = float(y)
        self._UPF = UPF
        self.UEs = []
        self.pred_UEs = []

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

    def add_pred_UE(self, ue):
        self.pred_UEs.append(ue)

    def remove_pred_UE(self, ue):
        self.pred_UEs.remove(ue)

    def get_pred_UEs(self):
        return self.pred_UEs

    def get_pred_numUEs(self):
        return len(self.pred_UEs)

    def clear_pred_UEs(self):
        self.pred_UEs = []

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
                # This line needs to be uncommented in order to have histeresis
                #pl = compute_path_loss(bs.get_distance_coords(x, y))
            else:
                bs, distance = get_optimal_bs(BSs, x, y)
                pl = compute_path_loss(distance)

            pred_x = None
            pred_y = None

            if len(line) > 7:
                pred_x = float(line[6])
                pred_y = float(line[7])
                # pred_x = float(line[2])
                # pred_y = float(line[3])

            pred_pl = None
            if len(line) > 8:
                pred_bs = BSs[int(line[8])]
                # This line needs to be uncommented in order to have histeresis
                #pred_pl = compute_path_loss(pred_bs.get_distance_coords(pred_x, pred_y))

            else:
                pred_bs, pred_distance = get_optimal_bs(BSs, pred_x, pred_y)
                pred_pl = compute_path_loss(pred_distance)

            if first_timestamp_iteration == None:
                first_timestamp_iteration = timestamp

            if timestamp - first_timestamp_iteration > iteration_duration:
                # Iteration finished: Yield results
                for ue in [ue for id_, ue in UEs_last_iteration.items() if id_ not in UEs_new_iteration.keys()]:
                    ue.update_bs(None)
                    ue.update_pred_bs(None)

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
                if pred_pl:
                    ue.update_pred(pred_x, pred_y, pred_bs, pred_pl)
                    #ue.update_pred_unconditional(pred_x, pred_y, pred_bs)
                else:
                    ue.update_pred_unconditional(pred_x, pred_y, pred_bs)
            # Only the last appearance of each UE in the iteration is considered
            elif id_ in UEs_new_iteration:
                ue = UEs_new_iteration[id_]
                if pl:
                    ue.update(x, y, bs, pl)
                else:
                    ue.update_unconditional(x, y, bs)
                if pred_pl:
                    ue.update_pred(pred_x, pred_y, pred_bs, pred_pl)
                    #ue.update_pred_unconditional(pred_x, pred_y, pred_bs)
                else:
                    ue.update_pred_unconditional(pred_x, pred_y, pred_bs)
            # Se crea un nuevo UE
            else:
                ue = UE(id_, x, y, bs, pred_x, pred_y, pred_bs)
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


def UPF_assignment_greedy_overhead(G: nx.Graph, BSs, num_UPFs, UE_to_UPF_assignment_previous, BSs_with_UPF_ids_previous, G_shortest_path_lengths, highest_bs_id, iteration_duration, time_deployment, time_removal, cost_relocation, alpha1, alpha2, alpha3, num_UEs, max_num_hops):
    BS_to_UPF_assignment = {}
    BSs_with_UPF_ids = set()

    bs: BS

    latencies_agg_dict = {}
    ues_agg_dict = {}
    tot_ues = 0
    for bs in G.nodes:
        tot_ues += bs.get_numUEs()

        ue: UE
        for ue in bs.get_UEs():
            upf_node_previous = -1
            if ue.get_id() in UE_to_UPF_assignment_previous:
                upf_node_previous = UE_to_UPF_assignment_previous[ue.get_id()]
            if bs.get_id() not in latencies_agg_dict:
                latencies_agg_dict[bs.get_id()] = {}
            if upf_node_previous not in latencies_agg_dict[bs.get_id()]:
                latencies_agg_dict[bs.get_id(
                )][upf_node_previous] = max_num_hops + 1
            if bs.get_id() not in ues_agg_dict:
                ues_agg_dict[bs.get_id()] = {}
            if upf_node_previous not in ues_agg_dict[bs.get_id()]:
                ues_agg_dict[bs.get_id()][upf_node_previous] = []

            ues_agg_dict[bs.get_id()][upf_node_previous].append(ue.get_id())

    done_BSs = [False for _ in range(highest_bs_id + 1)]

    for _ in range(num_UPFs):
        best_bs = None
        best_f_objective_function = None
        for bs in G.nodes:
            if not done_BSs[bs.get_id()]:
                # Check objective function if bs is selected

                # f2: Deployment overhead
                f2_deployment_overhead = get_deployment_overhead(
                    BSs_with_UPF_ids_previous, BSs_with_UPF_ids | set([bs.get_id()]), time_deployment, time_removal, iteration_duration)

                # f1 + f2: Vehicle latency (90-th percentile) and control-plane reassignment overhead
                f_objective_function = get_objective_function(
                    G, 0, f2_deployment_overhead, 0, alpha1, alpha2, alpha3, time_deployment, time_removal, cost_relocation, num_UEs, max_num_hops)
                for bs2 in G.nodes:
                    if bs2.get_numUEs() == 0:
                        continue

                    for upf_node_previous in ues_agg_dict[bs2.get_id()]:
                        num_UEs_agg = len(
                            ues_agg_dict[bs2.get_id()][upf_node_previous])
                        latency_agg = latencies_agg_dict[bs2.get_id(
                        )][upf_node_previous]
                        f3_control_plane_reassignment_overhead = 0
                        if (upf_node_previous != -1 and bs2.get_id() in BS_to_UPF_assignment and upf_node_previous in BS_to_UPF_assignment[bs2.get_id()]):
                            f3_control_plane_reassignment_overhead = num_UEs_agg * \
                                G_shortest_path_lengths[BSs[upf_node_previous]][BSs[BS_to_UPF_assignment[bs2.get_id(
                                )][upf_node_previous]]] * cost_relocation

                        f_objective_keep = get_objective_function(G, num_UEs_agg * latency_agg / tot_ues, 0, f3_control_plane_reassignment_overhead,
                                                                  alpha1, alpha2, alpha3, time_deployment, time_removal, cost_relocation, num_UEs, max_num_hops)

                        f3_control_plane_reassignment_overhead = 0
                        if (upf_node_previous != -1):
                            f3_control_plane_reassignment_overhead = num_UEs_agg * \
                                G_shortest_path_lengths[BSs[upf_node_previous]
                                                        ][bs] * cost_relocation

                        f_objective_relocate = get_objective_function(
                            G, num_UEs_agg * G_shortest_path_lengths[bs2][bs] / tot_ues, 0, f3_control_plane_reassignment_overhead, alpha1, alpha2, alpha3, time_deployment, time_removal, cost_relocation, num_UEs, max_num_hops)

                        if f_objective_relocate < f_objective_keep:
                            f_objective_function += f_objective_relocate
                        else:
                            f_objective_function += f_objective_keep

                if best_bs == None or f_objective_function < best_f_objective_function:
                    best_bs = bs
                    best_f_objective_function = f_objective_function

        previous_f_objective_function = best_f_objective_function

        upf_node = best_bs.get_id()
        done_BSs[upf_node] = True
        BSs_with_UPF_ids.add(upf_node)

        for bs2 in G.nodes:
            if bs2.get_numUEs() == 0:
                continue

            for upf_node_previous in ues_agg_dict[bs2.get_id()]:
                num_UEs_agg = len(
                    ues_agg_dict[bs2.get_id()][upf_node_previous])
                latency_agg = latencies_agg_dict[bs2.get_id(
                )][upf_node_previous]
                f3_control_plane_reassignment_overhead = 0
                if (upf_node_previous != -1 and bs2.get_id() in BS_to_UPF_assignment and upf_node_previous in BS_to_UPF_assignment[bs2.get_id()]):
                    f3_control_plane_reassignment_overhead = num_UEs_agg * \
                        G_shortest_path_lengths[BSs[upf_node_previous]][BSs[BS_to_UPF_assignment[bs2.get_id(
                        )][upf_node_previous]]] * cost_relocation

                f_objective_keep = get_objective_function(G, num_UEs_agg * latency_agg / tot_ues, 0, f3_control_plane_reassignment_overhead,
                                                            alpha1, alpha2, alpha3, time_deployment, time_removal, cost_relocation, num_UEs, max_num_hops)

                f3_control_plane_reassignment_overhead = 0
                if (upf_node_previous != -1):
                    f3_control_plane_reassignment_overhead = num_UEs_agg * \
                        G_shortest_path_lengths[BSs[upf_node_previous]
                                                ][best_bs] * cost_relocation

                f_objective_relocate = get_objective_function(
                    G, num_UEs_agg * G_shortest_path_lengths[bs2][best_bs] / tot_ues, 0, f3_control_plane_reassignment_overhead, alpha1, alpha2, alpha3, time_deployment, time_removal, cost_relocation, num_UEs, max_num_hops)

                if (bs2.get_id() not in BS_to_UPF_assignment or upf_node_previous not in BS_to_UPF_assignment[bs2.get_id()]) or f_objective_relocate < f_objective_keep:
                    new_latency = G_shortest_path_lengths[bs2][best_bs]
                    latencies_agg_dict[bs2.get_id()][upf_node_previous] = new_latency
                    # Update the assignment of all the UEs in bs2
                    if bs2.get_id() not in BS_to_UPF_assignment:
                        BS_to_UPF_assignment[bs2.get_id()] = {}
                    BS_to_UPF_assignment[bs2.get_id()][upf_node_previous] = upf_node

    # Convert back to original format BS_to_UPF_assignment -> UE_to_UPF_assignment
    UE_to_UPF_assignment = {}
    for bs2_id in BS_to_UPF_assignment:
        for upf_node_previous in BS_to_UPF_assignment[bs2_id]:
            for ue_id in ues_agg_dict[bs2_id][upf_node_previous]:
                UE_to_UPF_assignment[ue_id] = BS_to_UPF_assignment[bs2_id][upf_node_previous]

    return UE_to_UPF_assignment, BSs_with_UPF_ids

def UPF_assignment_prediction_greedy_overhead(G: nx.Graph, BSs, num_UPFs, UE_to_UPF_assignment_previous, BSs_with_UPF_ids_previous, G_shortest_path_lengths, highest_bs_id, iteration_duration, time_deployment, time_removal, cost_relocation, alpha1, alpha2, alpha3, num_UEs, max_num_hops):
    BS_to_UPF_assignment = {}
    BSs_with_UPF_ids = set()

    bs: BS

    latencies_agg_dict = {}
    ues_agg_dict = {}
    tot_ues = 0
    for bs in G.nodes:
        tot_ues += bs.get_pred_numUEs()

        ue: UE
        for ue in bs.get_pred_UEs():
            upf_node_previous = -1
            if ue.get_id() in UE_to_UPF_assignment_previous:
                upf_node_previous = UE_to_UPF_assignment_previous[ue.get_id()]
            if bs.get_id() not in latencies_agg_dict:
                latencies_agg_dict[bs.get_id()] = {}
            if upf_node_previous not in latencies_agg_dict[bs.get_id()]:
                latencies_agg_dict[bs.get_id(
                )][upf_node_previous] = max_num_hops + 1
            if bs.get_id() not in ues_agg_dict:
                ues_agg_dict[bs.get_id()] = {}
            if upf_node_previous not in ues_agg_dict[bs.get_id()]:
                ues_agg_dict[bs.get_id()][upf_node_previous] = []

            ues_agg_dict[bs.get_id()][upf_node_previous].append(ue.get_id())

    done_BSs = [False for _ in range(highest_bs_id + 1)]

    for _ in range(num_UPFs):
        best_bs = None
        best_f_objective_function = None
        for bs in G.nodes:
            if not done_BSs[bs.get_id()]:
                # Check objective function if bs is selected

                # f2: Deployment overhead
                f2_deployment_overhead = get_deployment_overhead(
                    BSs_with_UPF_ids_previous, BSs_with_UPF_ids | set([bs.get_id()]), time_deployment, time_removal, iteration_duration)

                # f1 + f2: Vehicle latency (90-th percentile) and control-plane reassignment overhead
                f_objective_function = get_objective_function(
                    G, 0, f2_deployment_overhead, 0, alpha1, alpha2, alpha3, time_deployment, time_removal, cost_relocation, num_UEs, max_num_hops)
                for bs2 in G.nodes:
                    if bs2.get_pred_numUEs() == 0:
                        continue
                    #new_latency = latencies_list[bs2.get_id()]

                    for upf_node_previous in ues_agg_dict[bs2.get_id()]:
                        num_UEs_agg = len(
                            ues_agg_dict[bs2.get_id()][upf_node_previous])
                        latency_agg = latencies_agg_dict[bs2.get_id(
                        )][upf_node_previous]
                        f3_control_plane_reassignment_overhead = 0
                        if (upf_node_previous != -1 and bs2.get_id() in BS_to_UPF_assignment and upf_node_previous in BS_to_UPF_assignment[bs2.get_id()]):
                            f3_control_plane_reassignment_overhead = num_UEs_agg * \
                                G_shortest_path_lengths[BSs[upf_node_previous]][BSs[BS_to_UPF_assignment[bs2.get_id(
                                )][upf_node_previous]]] * cost_relocation

                        f_objective_keep = get_objective_function(G, num_UEs_agg * latency_agg / tot_ues, 0, f3_control_plane_reassignment_overhead,
                                                                  alpha1, alpha2, alpha3, time_deployment, time_removal, cost_relocation, num_UEs, max_num_hops)

                        f3_control_plane_reassignment_overhead = 0
                        if (upf_node_previous != -1):
                            f3_control_plane_reassignment_overhead = num_UEs_agg * \
                                G_shortest_path_lengths[BSs[upf_node_previous]
                                                        ][bs] * cost_relocation

                        f_objective_relocate = get_objective_function(
                            G, num_UEs_agg * G_shortest_path_lengths[bs2][bs] / tot_ues, 0, f3_control_plane_reassignment_overhead, alpha1, alpha2, alpha3, time_deployment, time_removal, cost_relocation, num_UEs, max_num_hops)

                        if f_objective_relocate < f_objective_keep:
                            f_objective_function += f_objective_relocate
                        else:
                            f_objective_function += f_objective_keep

                if best_bs == None or f_objective_function < best_f_objective_function:
                    best_bs = bs
                    best_f_objective_function = f_objective_function

        previous_f_objective_function = best_f_objective_function

        upf_node = best_bs.get_id()
        done_BSs[upf_node] = True
        BSs_with_UPF_ids.add(upf_node)

        for bs2 in G.nodes:
            if bs2.get_pred_numUEs() == 0:
                continue

            for upf_node_previous in ues_agg_dict[bs2.get_id()]:
                num_UEs_agg = len(
                    ues_agg_dict[bs2.get_id()][upf_node_previous])
                latency_agg = latencies_agg_dict[bs2.get_id(
                )][upf_node_previous]
                f3_control_plane_reassignment_overhead = 0
                if (upf_node_previous != -1 and bs2.get_id() in BS_to_UPF_assignment and upf_node_previous in BS_to_UPF_assignment[bs2.get_id()]):
                    f3_control_plane_reassignment_overhead = num_UEs_agg * \
                        G_shortest_path_lengths[BSs[upf_node_previous]][BSs[BS_to_UPF_assignment[bs2.get_id(
                        )][upf_node_previous]]] * cost_relocation

                f_objective_keep = get_objective_function(G, num_UEs_agg * latency_agg / tot_ues, 0, f3_control_plane_reassignment_overhead,
                                                            alpha1, alpha2, alpha3, time_deployment, time_removal, cost_relocation, num_UEs, max_num_hops)

                f3_control_plane_reassignment_overhead = 0
                if (upf_node_previous != -1):
                    f3_control_plane_reassignment_overhead = num_UEs_agg * \
                        G_shortest_path_lengths[BSs[upf_node_previous]
                                                ][best_bs] * cost_relocation

                f_objective_relocate = get_objective_function(
                    G, num_UEs_agg * G_shortest_path_lengths[bs2][best_bs] / tot_ues, 0, f3_control_plane_reassignment_overhead, alpha1, alpha2, alpha3, time_deployment, time_removal, cost_relocation, num_UEs, max_num_hops)

                if (bs2.get_id() not in BS_to_UPF_assignment or upf_node_previous not in BS_to_UPF_assignment[bs2.get_id()]) or f_objective_relocate < f_objective_keep:
                    new_latency = G_shortest_path_lengths[bs2][best_bs]
                    latencies_agg_dict[bs2.get_id()][upf_node_previous] = new_latency
                    # Update the assignment of all the UEs in bs2
                    if bs2.get_id() not in BS_to_UPF_assignment:
                        BS_to_UPF_assignment[bs2.get_id()] = {}
                    BS_to_UPF_assignment[bs2.get_id()][upf_node_previous] = upf_node

    # Convert back to original format BS_to_UPF_assignment -> UE_to_UPF_assignment

    UE_to_UPF_assignment = {}
    for bs2_id in BS_to_UPF_assignment:
        for upf_node_previous in BS_to_UPF_assignment[bs2_id]:
            for ue_id in ues_agg_dict[bs2_id][upf_node_previous]:
                UE_to_UPF_assignment[ue_id] = BS_to_UPF_assignment[bs2_id][upf_node_previous]

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
def UPF_assignment(algorithm, G: nx.Graph, BSs, num_UPFs, UE_to_UPF_assignment_previous, BSs_with_UPF_ids_previous, G_shortest_path_lengths, highest_bs_id, iteration_duration, time_deployment, time_removal, cost_relocation, alpha1, alpha2, alpha3, num_UEs, max_num_hops):
    if "old_" in algorithm:
        UE_to_UPF_assignment, BSs_with_UPF_ids = UPF_assignment_old(algorithm, G, BSs, num_UPFs, UE_to_UPF_assignment_previous,
                                                                    BSs_with_UPF_ids_previous, G_shortest_path_lengths, highest_bs_id)
    else:
        UE_to_UPF_assignment, BSs_with_UPF_ids = globals()["UPF_assignment_{}".format(
            algorithm)](G, BSs, num_UPFs, UE_to_UPF_assignment_previous, BSs_with_UPF_ids_previous, G_shortest_path_lengths, highest_bs_id, iteration_duration, time_deployment, time_removal, cost_relocation, alpha1, alpha2, alpha3, num_UEs, max_num_hops)

    if DEBUG:
        print(UE_to_UPF_assignment)
        print(BSs_with_UPF_ids)

    return UE_to_UPF_assignment, BSs_with_UPF_ids


# Metrics
def get_minimum_hops_from_BS_to_UPF(G, bs, upf_assigned, G_shortest_path_lengths):
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
            #print("Changing UE {} from {} to {}".format(ue, upf_old, upf_new))

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


def get_objective_function(G: nx.Graph, f1_num_hops, f2_deployment_overhead, f3_control_plane_reassignment_overhead,
                           alpha1, alpha2, alpha3, time_deployment, time_removal, cost_relocation, num_UEs, max_num_hops):
    return alpha1 * get_f1_normalized(f1_num_hops, max_num_hops) + \
        alpha2 * get_f2_normalized(G, f2_deployment_overhead, time_deployment, time_removal) + \
        alpha3 * get_f3_normalized(f3_control_plane_reassignment_overhead,
                                   cost_relocation, num_UEs, max_num_hops)


def get_f1_normalized(f1_num_hops, max_num_hops):
    return f1_num_hops / max_num_hops


def get_f2_normalized(G: nx.Graph, f2_deployment_overhead, time_deployment, time_removal):
    max_num_UPFs = G.number_of_nodes()
    max_deployment_overhead = max_num_UPFs * max(time_deployment, time_removal)
    return f2_deployment_overhead / max_deployment_overhead


def get_f3_normalized(f3_control_plane_reassignment_overhead, cost_relocation, num_UEs, max_num_hops):
    # EXPERIMENTAL
    max_control_plane_reassignment_overhead = (
        1 + num_UEs) * max_num_hops * cost_relocation
    return f3_control_plane_reassignment_overhead / max_control_plane_reassignment_overhead


# Auxiliar functions for converting formats
def convert_bs_assignment_to_ue_assignment(BSs, BS_to_UPF_assignment):
    UE_to_UPF_assignment = {}
    for bs_id in BS_to_UPF_assignment:
        for upf_node in BS_to_UPF_assignment[bs_id]:
            for ue_id in BS_to_UPF_assignment[bs_id][upf_node]:
                UE_to_UPF_assignment[ue_id] = upf_node

    return UE_to_UPF_assignment


def convert_ue_assignment_to_bs_assignment(BSs, UE_to_UPF_assignment):
    BS_to_UPF_assignment = {}
    for bs_id in BSs:
        bs = BSs[bs_id]
        for ue in bs.get_UEs():
            if ue.get_id() in UE_to_UPF_assignment:
                upf_node = UE_to_UPF_assignment[ue.get_id()]
                if bs_id not in BS_to_UPF_assignment:
                    BS_to_UPF_assignment[bs_id] = {}
                if upf_node not in BS_to_UPF_assignment[bs_id]:
                    BS_to_UPF_assignment[bs_id][upf_node] = []
                BS_to_UPF_assignment[bs_id][upf_node].append(ue.get_id())

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
        "--algorithm", help="Specifies the UPF allocation algorithm [Supported: old_random/old_greedy_percentile/old_greedy_average/old_kmeans_greedy_average/old_modularity_greedy_average/greedy_overhead/prediction_greedy_overhead].", required=True)
    parser.add_argument(
        "--minUPFs", help="Specifies the minimum number of UPFs to be allocated [Default: {}].".format(DEFAULT_MIN_UPF), type=int, default=DEFAULT_MIN_UPF)
    parser.add_argument(
        "--maxUPFs", help="Specifies the maximum number of UPFs to be allocated [Default: {}].".format(DEFAULT_MAX_UPF), type=int, default=DEFAULT_MAX_UPF)
    parser.add_argument(
        "--bsFile", help="File containing the information about the base stations [Format: each line contains the id, x coordinate and y coordinate of a base station separated by spaces].", required=True)
    parser.add_argument(
        "--ueFile", help="File containing the information about the users throughout the simulation [Format: each line contains the timestamp, user id, x coordinate, y coordinate, speed and, optionally, base station id to which the user is attached, predicted x coordinate, predicted y coordinate and predicted base station id to which the user is attached].", required=True)
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
        list_results_f1_normalized = []
        list_results_f2_normalized = []
        list_results_f3_normalized = []

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
            # Clear UEs in BSs (and pred_UEs)
            for bs in G.nodes():
                bs.clear_UEs()
                bs.clear_pred_UEs()

            iteration = 0
            # First UPF allocation is random
            # BSs_with_UPF_ids = UPF_allocation(
            #     "random", G, None, G_shortest_path_lengths, highest_bs_id)

            UE_to_UPF_assignment_previous = {}
            BSs_with_UPF_ids_previous = set()

            UE_to_UPF_assignment, BSs_with_UPF_ids = UPF_assignment("old_random", G, BSs, num_UPFs, None, None, G_shortest_path_lengths,
                                                                    highest_bs_id, iteration_duration, time_deployment, time_removal, cost_relocation, alpha1, alpha2, alpha3, 0, max_num_hops)

            upf_start_time = time.process_time()  # seconds
            print("Running {} for {} UPFs".format(
                algorithm, num_UPFs), file=stderr)
            for UEs in read_UE_data(ue_file, BSs, iteration_duration):
                iteration += 1

                ## Experimental: @pfondo JUST MOVED
                if algorithm.startswith("prediction_"):
                    start_time = time.process_time() * 1e3  # milliseconds

                    # Calculate assignment for the next time slot

                    BSs_with_UPF_ids_previous = BSs_with_UPF_ids
                    UE_to_UPF_assignment_previous = UE_to_UPF_assignment

                    UE_to_UPF_assignment, BSs_with_UPF_ids = UPF_assignment(algorithm, G, BSs, num_UPFs, UE_to_UPF_assignment_previous, BSs_with_UPF_ids_previous,
                                                                            G_shortest_path_lengths, highest_bs_id, iteration_duration, time_deployment, time_removal, cost_relocation, alpha1, alpha2, alpha3, len(UEs), max_num_hops)

                    assert check_assignment_ok(
                        BSs, UE_to_UPF_assignment, BSs_with_UPF_ids)

                    # print(UE_to_UPF_assignment)

                    end_time = time.process_time() * 1e3  # milliseconds
                    elapsed_time = end_time - start_time

                    list_results_elapsed_time.append(elapsed_time)

                # Calculate metrics
                # f1: Vehicle latency (90-th percentile)

                UE_hops_list = get_UE_hops_list_assignment(
                    G, BSs, BSs_with_UPF_ids, UE_to_UPF_assignment, G_shortest_path_lengths)

                f1_num_hops_90th = np_percentile(UE_hops_list, 90)
                list_results_num_hops.append(f1_num_hops_90th)
                # print(UE_hops_list, file=stderr)

                f1_normalized = get_f1_normalized(
                    f1_num_hops_90th, max_num_hops)
                list_results_f1_normalized.append(f1_normalized)

                # Resource usage (f2 in previous formulation)
                actual_num_UPFs = len(BSs_with_UPF_ids)
                list_results_num_upfs.append(actual_num_UPFs)

                # f2: Deployment overhead
                f2_deployment_overhead = get_deployment_overhead(
                    BSs_with_UPF_ids_previous, BSs_with_UPF_ids, time_deployment, time_removal, iteration_duration)

                list_results_deployment_overhead.append(f2_deployment_overhead)

                f2_normalized = get_f2_normalized(
                    G, f2_deployment_overhead, time_deployment, time_removal)
                list_results_f2_normalized.append(f2_normalized)

                # f3: Control-plane reassignment overhead
                f3_control_plane_reassignment_overhead = get_control_plane_reassignment_overhead(
                    BSs, UE_to_UPF_assignment_previous, UE_to_UPF_assignment, cost_relocation, G_shortest_path_lengths)
                list_results_control_plane_reassignment_overhead.append(
                    f3_control_plane_reassignment_overhead)

                f3_normalized = get_f3_normalized(
                    f3_control_plane_reassignment_overhead, cost_relocation, len(UEs), max_num_hops)
                list_results_f3_normalized.append(f3_normalized)

                # f: Objective function
                f_objective_function = get_objective_function(G, f1_num_hops_90th, f2_deployment_overhead, f3_control_plane_reassignment_overhead,
                                                              alpha1, alpha2, alpha3, time_deployment, time_removal, cost_relocation, len(UEs), max_num_hops)

                list_results_objective_function.append(f_objective_function)

                if not algorithm.startswith("prediction_"):
                    start_time = time.process_time() * 1e3  # milliseconds

                    # Calculate assignment for the next time slot

                    BSs_with_UPF_ids_previous = BSs_with_UPF_ids
                    UE_to_UPF_assignment_previous = UE_to_UPF_assignment

                    UE_to_UPF_assignment, BSs_with_UPF_ids = UPF_assignment(algorithm, G, BSs, num_UPFs, UE_to_UPF_assignment_previous, BSs_with_UPF_ids_previous,
                                                                            G_shortest_path_lengths, highest_bs_id, iteration_duration, time_deployment, time_removal, cost_relocation, alpha1, alpha2, alpha3, len(UEs), max_num_hops)

                    assert check_assignment_ok(
                        BSs, UE_to_UPF_assignment, BSs_with_UPF_ids)

                    # print(UE_to_UPF_assignment)

                    end_time = time.process_time() * 1e3  # milliseconds
                    elapsed_time = end_time - start_time

                    list_results_elapsed_time.append(elapsed_time)

                if DEBUG:
                    print("UE hops to UPF: {}".format(
                        UE_hops_list), file=stderr)
                    print_statistics(UE_hops_list, file=stderr)
                    print("\n\n", file=stderr)

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
        mci_f1_normalized = mean_confidence_interval(
            list_results_f1_normalized, 0.95)
        mci_f2_normalized = mean_confidence_interval(
            list_results_f2_normalized, 0.95)
        mci_f3_normalized = mean_confidence_interval(
            list_results_f3_normalized, 0.95)

        print("  Elapsed time: {:.3f} seconds".format(
            upf_elapsed_time), file=stderr)

        print("  90th hops: {} {} seconds".format(
            mci_num_hops, mci_num_upfs), file=stderr)

        print("{} {} {:.3f} {:.3f} {:.3f} "
              "{:.3f} {:.3f} {:.3f} "
              "{:.3f} {:.3f} {:.3f} "
              "{:.3f} {:.3f} {:.3f} "
              "{:.3f} {:.3f} {:.3f} "
              "{:.3f} {:.3f} {:.3f} "
              "{:.6f} {:.6f} {:.6f} "
              "{:.6f} {:.6f} {:.6f} "
              "{:.6f} {:.6f} {:.6f}".format(algorithm, num_UPFs, mci_num_hops[0], mci_num_hops[1],
                                            mci_num_hops[2], mci_elapsed_time[0], mci_elapsed_time[1],
                                            mci_elapsed_time[2], mci_num_upfs[0], mci_num_upfs[1],
                                            mci_num_upfs[2], mci_deployment_overhead[0],
                                            mci_deployment_overhead[1], mci_deployment_overhead[2],
                                            mci_control_plane_reassignment_overhead[0],
                                            mci_control_plane_reassignment_overhead[1],
                                            mci_control_plane_reassignment_overhead[2],
                                            mci_objective_function[0], mci_objective_function[1],
                                            mci_objective_function[2], mci_f1_normalized[0],
                                            mci_f1_normalized[1], mci_f1_normalized[2],
                                            mci_f2_normalized[0], mci_f2_normalized[1],
                                            mci_f2_normalized[2], mci_f3_normalized[0],
                                            mci_f3_normalized[1], mci_f3_normalized[2]
                                            ))
        # print("DEBUG:", file = stderr)
        # print(list_results_num_hops, file = stderr)
        # print(list_results_deployment_overhead, file = stderr)
        # print(list_results_control_plane_reassignment_overhead, file = stderr)
        # print(list_results_f1_normalized, file = stderr)
        # print(list_results_f2_normalized, file = stderr)
        # print(list_results_f3_normalized, file = stderr)


if __name__ == "__main__":
    main()
