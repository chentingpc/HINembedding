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

_Warning_:

1. When using (multiple) edge_type, adjacency matrices are normalized for each edge_type.
2. To change the weights for edge types, go to data_helper.h, and search "set your weights for edge types here".
3. Current second order similarity is used.

**node_type file**

each line should be
```
node node_type
```

currently only the first char of node_type is used for identification.

_Warning_:

1. each edge type can only contain at most two node types (one for source, one for destination).


Parameters
---------------

**rho, sigma**: learning rates (network embedding, edge type bias in network embedding).

**lambda**: regularizations (network embedding).


Requirements
-----------------
The code can be run under Linux using Makefile for compilation.

Also, the GSL package is used and can be downloaded at http://www.gnu.org/software/gsl/
