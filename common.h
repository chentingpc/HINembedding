#pragma once

//#define NOT_PA_TASK
#define PA_TASK
#define NEG_SAMPLING_POWER 0.75         // unigram downweighted

typedef float real;                     // Precision of float numbers
typedef unsigned int uint;
typedef long long int64;
typedef unsigned long long uint64;
const int hash_table_size = 5e7;        // better be at least several times larger than num_vertices
const int neg_table_size = 3e8;         // better be at least several times larger than num_edges
const double LOG_MIN = 1e-15;           // Smoother for log
#define SIGMOID_BOUND 10
const int sigmoid_table_size = 1000;
#define MAX_STRING 2000
#define MAX_LINE_LEN 65535

#define PATH_DIRECTION_NORMAL 0
#define PATH_DIRECTION_REVERSE 1
#define PATH_DIRECTION_BIDIRECTION 2
#define PATH_ORDER_SINGLE 0
#define PATH_ORDER_CONTEXT 1

struct Vertex {
  // content in this structure can be modified outside, which is not safe
  double degree;
  char *name;
  int type;
};


struct Graph {
  // content in this structure can be modified outside, which is not safe
  Vertex                  *vertex;
  int                     *vertex_type;
  double                  *vertex_degree_of_etype;

  int                     *edge_source_id;
  int                     *edge_target_id;
  double                  *edge_weight;
  int                     *edge_type;
  double                  *edge_type_w;

  bool                    *node_type_to_edge_type;

  int                     *num_vertices_p;
  int64                   *num_edges_p;
  int                     *num_node_type_p;
  int                     *num_edge_type_p;
};
