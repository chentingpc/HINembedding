Introduction
------------------
Network embedding model extending LINE for heterogeneous networks.

__Warning__: Current code has not been well cleaned, and contains several hidden parameters (which may result in undesirable results).


Data format
-----------------
**network file (required)**

each line can be either:
```
src_node dst_node weight
```
or
```
src_node dst_node weight edge_type
```

_Note_:

1. separator allowed: tab or space. hence, no tab or space is allowed in ndoe name.
2. node name for different node type should be different.

**node_type file (optional?)**

each line should be
```
node node_type
```

currently only the first char of node_type is used for identification.

_Note_:

1. the node types associated with two ends of the edge are inferred from given data, and it is assumed that each edge type have at most two node types.
2. separator allowed: tab or space.

**path_conf_file (optional)**

each line includes:

```
edge_type edge_type_weight direction_conf proximity_conf sampling_ratio_power base_degree
```

Where

```edge_type_weight```: float in [0, inf). edge_type is weighted by random sampling, less weight means less chances being sampled. when tuning, let all edge type weights sum to some constant.

```direction_conf```: normal/reverse/bidirection. normal means only consider given direction; reverse means only consider reverse direction; bidirection means half-half for both given and reverse direction.

```proximity_conf```: single/context. single is use one vector for each node; context is to use additional context vector for target node.

```sampling_ratio_power```: float in (0, 1], used to scale degree.

```base_degree```: added to every node (of targeted type) when computing noise node sampler, note the number is in multiple of targeted min non-zero degree.

_Note_:

1. if this file is given, edge types not present in the file will not be used in network embedding.
2. if this file is not given, default option is: all edge types have weight sum of 10000, bidirection, single embedding vector, sampling ratio 0.75 with 1 base degree.
3. separator allowed: space.

Key parameters
---------------

**rho, sigma**: learning rates (network embedding, edge type bias in network embedding).

**lambda**: regularizations (network embedding).

Requirements
-----------------
The code can be run under Linux using Makefile for compilation.

Also, the GSL package is used and can be downloaded at http://www.gnu.org/software/gsl/
