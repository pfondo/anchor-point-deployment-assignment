# anchor-point-deployment-assignment-simulator


## Description

Simulator for testing the deployment of anchor point network functions (e.g., User Plane Function (UPF) in 5G) and the assignment of the serving anchor point for each users in dynamic Multi-Access Edge Computing (MEC) scenarios.

The simulator allows the evaluation of algorithms for the dynamic deployment of anchor point network functions and the assignment of the serving anchor point for each user. The simulator calculates the latency between each UE and its serving anchor point, the execution time of the algorithm, the resource usage (number of UPF actually allocated), the deployment overhead, the control-plane reassignment overhead introduced in the network and the objective function defined as defined in http://www.investigo.biblioteca.uvigo.es/xmlui/handle/11093/3149.

## Execution

    usage: main.py [-h] --algorithm ALGORITHM [--minUPFs MINUPFS] [--maxUPFs MAXUPFS] --bsFile BSFILE --ueFile UEFILE [--iterationDuration ITERATIONDURATION] [--timeDeployment TIMEDEPLOYMENT] [--timeRemoval TIMEREMOVAL] [--costRelocation COSTRELOCATION] [--alpha1 ALPHA1] [--alpha2 ALPHA2] [--alpha3 ALPHA3] [--alpha4 ALPHA4]

    optional arguments:
    -h, --help            show this help message and exit
    --algorithm ALGORITHM
                            Specifies the UPF allocation algorithm [Supported: old_random/old_greedy_percentile/old_greedy_average/old_kmeans_greedy_average/old_modularity_greedy_average/greedy_overhead].
    --minUPFs MINUPFS     Specifies the minimum number of UPFs to be allocated [Default: 1].
    --maxUPFs MAXUPFS     Specifies the maximum number of UPFs to be allocated [Default: 10].
    --bsFile BSFILE       File containing the information about the base stations [Format: each line contains the id, x coordinate and y coordinate of a base station separated by spaces].
    --ueFile UEFILE       File containing the information about the users throughout the simulation [Format: each line contains the timestamp, user id, x coordinate, y coordinate, speed and, optionally, the base
                            station id to which the user is attached].
    --iterationDuration ITERATIONDURATION
                            Duration of each time-slot in seconds [Default: 5].
    --timeDeployment TIMEDEPLOYMENT
                            Time required for deploying an anchor point in seconds [Default: 1].
    --timeRemoval TIMEREMOVAL
                            Time required for removing an anchor point in seconds [Default: 0.1].
    --costRelocation COSTRELOCATION
                            Cost for relocating the communications of a vehicle [Default: 1].
    --alpha1 ALPHA1       Weight for the first parameter of the objective function (latency) [Default: 0.7].
    --alpha2 ALPHA2       Weight for the first parameter of the objective function (latency) [Default: 0.1].
    --alpha3 ALPHA3       Weight for the first parameter of the objective function (latency) [Default: 0.1].
    --alpha4 ALPHA4       Weight for the first parameter of the objective function (latency) [Default: 0.1].

## Results

The results are printed through the standard output stream with the following format:

    ALGORITHM NUM_UPFS LATENCY_AVG LATENCY_CI95_LOW LATENCY_CI95_HIGH EXECUTION_TIME_AVG EXECUTION_TIME_CI95_LOW EXECUTION_TIME_CI95_HIGH RESOURCE_USAGE_AVG RESOURCE_USAGE_CI95_LOW RESOURCE_USAGE_CI95_HIGH DEPLOYMENT_OVERHEAD_AVG DEPLOYMENT_OVERHEAD_CI95_LOW DEPLOYMENT_OVERHEAD_CI95_HIGH CONTROL_PLANE_REASSIGNMENT_OVERHEAD_AVG CONTROL_PLANE_REASSIGNMENT_OVERHEAD_CI95_LOW CONTROL_PLANE_REASSIGNMENT_OVERHEAD_CI95_HIGH OBJECTIVE_FUNCTION_AVG OBJECTIVE_FUNCTION_CI95_LOW OBJECTIVE_FUNCTION_CI95_HIGH

Status messages are printed through the standard error stream in order to provide information about the current status of the simulation.

## Adding an algorithm

An additional algorithm named "algX" can be added to the simulator by implementing a method with the following signature:

    def UPF_assignment_X(G: nx.Graph, BSs, num_UPFs, UE_to_UPF_assignment_previous, BSs_with_UPF_ids_previous, G_shortest_path_lengths, highest_bs_id, iteration_duration, time_deployment, time_removal, cost_relocation, alpha1, alpha2, alpha3, alpha4, num_UEs, max_num_hops)

The method must return two elements: The first one must be a dictionary being the key the ID of each UE and value the ID of the assigned MEC location (i.e., base station). The second one must be a set of no more than num_UPFs integers representing the IDs of the MEC location (i.e., base station) where UPFs are going to be deployed in the next interval.

## Copyright

This simulator has been elaborated based on https://github.com/pfondo/upf-allocation-simulator.

Copyright ⓒ 2021 Pablo Fondo Ferreiro <pfondo@gti.uvigo.es>, David Candal Ventureira <dcandal@gti.uvigo.es>

This simulator is licensed under the GNU General Public License, version 3 (GPL-3.0). For more information see LICENSE.txt
