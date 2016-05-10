#pragma once
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <string.h>
#include <cassert>
#include <map>
#include <vector>
#include <utility>
#include "./common.h"
#include "./config.h"
#include "./utility.h"

using namespace std;

class DataHelper : public VertexHashTable {
  Graph                   graph;              // this compact struct is used for convinience
  Config                  *conf_p;
  bool                    *node_type_to_edge_type;
                                              // this is a matrix indexed by i * num_edge_type + j,
                                              // 1 if a node type involved with a edge type

  /* network vertex data struct */
  Vertex                  *vertex;
  int                     *vertex_type;       // init to 0
  double                  *vertex_degree_of_etype;
                                              // indexed by [vid * num_edge_type + e_type]
  int                     num_vertices;
  map<int, string>        node_type2name;     // type (int) => type (name, str)
  map<string, int>        node_name2type;     // type (name, str) => type (int)
  int                     num_node_type;      // init to 1

  /* network edge data struct */
  int                     *edge_source_id;    // edge info is stored separted for performance sake
  int                     *edge_target_id;
  double                  *edge_weight;
  int                     *edge_type;         // init to 0
  double                  *edge_type_w;       // edge weights sum for each meta-path
  int64                   num_edges;
  map<int, string>        edge_type2name;
  map<string, int>        edge_name2type;
  int                     num_edge_type;      // init to 1

  /* train/test data struct */
  vector<int>             train_group;
  vector<pair<int, int> > train_pairs;
  vector<real>            train_pairs_label;
  map<int, vector<int> >  train_src_features;
  map<int, vector<int> >  train_dst_features;
  vector<int>             test_group;         // each element specify the group size for ranking
  vector<pair<int, int> > test_pairs;         // test pairs, can be nodes or features
  vector<real>            test_pairs_label;   // labels for test pairs, if any
  vector<string>          test_pairs_type;    // task type of test pairs, if any
  vector<int>             test_pairs_etype;   // task type as edge type of test pairs
  map<int, vector<int> >  test_src_features;  // features (node indexes) for src nodes (LHS)
  map<int, vector<int> >  test_dst_features;  // features (node indexes) for dst nodes (RHS)
  map<string, pair<int, int> >
                          test_task_group_start_end;

  /* other auxilliary data */
  int                     max_num_vertices;

  /*
   * Read network from the training file
   */
  void load_network(string network_file, bool path_normalization = false) {
    FILE *fin;
    char name_v1[MAX_STRING], name_v2[MAX_STRING], type_name[MAX_STRING], line_buffer[MAX_LINE_LEN];
    vector<string> valid_paths;
    int vid, type, num_separator = 0;
    double weight;
    clock_t start, end;
    start = clock();

    // path selection
    bool do_path_selection = false;
    if (conf_p->path_file.size() > 0) {
      do_path_selection = true;
      int line_number = conf_p->path_line;  // select the path of this line, line number starting from 0
      printf("[WARNING!!!!!!!!!!!!!!!!!!!!!!!!!!!] do path selection using line %d of file %s!!!\n",
        line_number, conf_p->path_file.c_str());
      FILE *fp = fopen(conf_p->path_file.c_str(), "r");
      assert(fp != NULL);
      while (fgets(line_buffer, sizeof(line_buffer), fp)) {
        line_number--;
        if (line_number < 0) {
          int pos = 0;
          while (line_buffer[++pos] != '\n');
          assert(pos < MAX_LINE_LEN);  line_buffer[pos] = '\0';
          valid_paths = split(string(line_buffer), ' ');
          printf("valid paths: %s\n", line_buffer);
          break;
        }
      }
      fclose(fp);
    }

    // count number edges
    fin = fopen(network_file.c_str(), "rb");
    if (fin == NULL) {
      printf("ERROR: network file not found!\n");
      exit(1);
    }
    num_edges = 0;
    while (fgets(line_buffer, sizeof(line_buffer), fin)) num_edges++;
    fclose(fin);

    // init edges structure
    edge_source_id = new int[num_edges];
    edge_target_id = new int[num_edges];
    edge_weight = new double[num_edges];
    if (edge_source_id == NULL || edge_target_id == NULL || edge_weight == NULL) {
      printf("Error: memory allocation failed!\n");
      exit(1);
    }
    memset(edge_source_id, 0, sizeof(int) * num_edges);
    memset(edge_target_id, 0, sizeof(int) * num_edges);
    memset(edge_weight, 0, sizeof(double) * num_edges);
    edge_type = (int *)malloc(num_edges*sizeof(int));
    for (int64 i = 0; i < num_edges; i++) edge_type[i] = 0;
    vertex = (struct Vertex *)calloc(max_num_vertices, sizeof(struct Vertex));

    // load edge and vertex
    fin = fopen(network_file.c_str(), "rb");
    num_vertices = 0;
    for (int64 k = 0; k != num_edges; k++) {
      type_name[0] = '\0';
      fgets(line_buffer, sizeof(line_buffer), fin);
      if (num_separator == 0) {
        // read one line to find out the seporator, and be consistent
        for (size_t i = 0; i < MAX_LINE_LEN; i++) {
          if (line_buffer[i] == '\0') break;
          else if (line_buffer[i] == ' ' || line_buffer[i] == '\t') num_separator++;
        }
      }
      if (num_separator == 2) {
        sscanf(line_buffer, "%s %s %lf", name_v1, name_v2, &weight);
        // sscanf(line_buffer, "%s %s %s", name_v1, name_v2, type_name);
        weight = 1;
      }
      else if (num_separator == 3) {
        sscanf(line_buffer, "%s %s %lf %s", name_v1, name_v2, &weight, type_name);
      }
      else {
        printf("ERROR: seporator mis-match, check network file format..\n");
        exit(1);
      }

      /* edge type screening */
      bool go = false;
      if (do_path_selection) {
        for (vector<string>::const_iterator it = valid_paths.begin(); it != valid_paths.end();
            it ++) {
          if (strcmp(type_name, it->c_str()) == 0) {
            go = true;
            break;
          }
        }
      } else {
          go = true;  // debug
      }
      if (!go) {
        // still add the vertex
        vid = search_hash_table(name_v1, vertex);
        if (vid == -1) vid = add_vertex(name_v1, vertex, num_vertices, max_num_vertices);
        vid = search_hash_table(name_v2, vertex);
        if (vid == -1) vid = add_vertex(name_v2, vertex, num_vertices, max_num_vertices);
        k--;
        num_edges--;
        continue;
      }

      if (k % 10000 == 0) {
        printf("Reading edges: %.3lf%%%c", k / (double)(num_edges + 1) * 100, 13);
        fflush(stdout);
      }

      vid = search_hash_table(name_v1, vertex);
      if (vid == -1) vid = add_vertex(name_v1, vertex, num_vertices, max_num_vertices);
      vertex[vid].degree += weight;
      edge_source_id[k] = vid;

      vid = search_hash_table(name_v2, vertex);
      if (vid == -1) vid = add_vertex(name_v2, vertex, num_vertices, max_num_vertices);
      vertex[vid].degree += weight;
      edge_target_id[k] = vid;

      /* reverse edge
      // if (strcmp(type_name, "P1A") == 0) {
        int mid = edge_source_id[k];
        edge_source_id[k] = edge_target_id[k];
        edge_target_id[k] = mid;
        char midc = type_name[0];
        type_name[0] = type_name[2];
        type_name[2] = midc;
      // }
      */

      edge_weight[k] = weight;

      if (type_name[0] != '\0') {
        if (edge_name2type.find(type_name) == edge_name2type.end()) {
          type = num_edge_type++;
          edge_name2type[type_name] = type;
          edge_type2name[type] = type_name;
        } else {
          type = edge_name2type[type_name];
        }
        edge_type[k] = type;
      }
      // printf("%s %s %f %d\n", name_v1, name_v2, weight, type);
    }
    fclose(fin);
    printf("Number of (unique) edges: %lld          \n", num_edges);
    printf("Number of vertices: %d          \n", num_vertices);

    /* randomize network, by switching target nodes randomly, also adjust vertex weights */
    if (conf_p->net_randomize > 0) {
      // int valid_path_type = -1;
      // if (conf_p->net_randomize == 2) {
      //  assert(valid_paths.size() == 1);
      //  edge_name2type.at(valid_paths[0]);
      // }
      printf("[WARNING!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!] Network is randomized\n");
      for (int64 i = 1; i < num_edges; i++) {
        // if (valid_path_type != -1 && edge_type[i] != valid_path_type) continue;
        sample_target: int target_edge = rand() % (i+1);
        // if (valid_path_type != -1 && edge_type[target_edge] != valid_path_type) goto sample_target;
        int &src = edge_target_id[i]; const double &src_weight = edge_weight[i];
        int &dst = edge_target_id[target_edge]; const double &dst_weight = edge_weight[target_edge];
        vertex[src].degree += dst_weight - src_weight; vertex[dst].degree += src_weight - dst_weight;
        assert(vertex[src].degree > 0); assert(vertex[dst].degree > 0);
        int mid = src; src = dst; dst = mid;
      }
    }

    // edge weight computation
    edge_type_w = new double[num_edge_type];
    memset(edge_type_w, 0, sizeof(double) * num_edge_type);
    for (int64 i = 0; i < num_edges; i++) edge_type_w[edge_type[i]] += edge_weight[i];

    // normalize over row for each network
    const bool row_reweighting = false;  // debug
    if (row_reweighting) {
      double **_edge_type_degree = new double*[num_edge_type];
      for (int i = 0; i < num_edge_type; i++) {
        _edge_type_degree[i] = new double[num_vertices];
        memset(_edge_type_degree[i], 0, sizeof(double) * num_vertices);
      }
      for (int64 i = 0; i < num_edges; i++) {
        _edge_type_degree[edge_type[i]][edge_source_id[i]] += edge_weight[i];
      }
      for (int64 i = 0; i < num_edges; i++) {
        double deg = _edge_type_degree[edge_type[i]][edge_source_id[i]];
        edge_weight[i] = pow(edge_weight[i] / deg, NEG_SAMPLING_POWER);  // reweighting function
      }
      for (int i = 0; i < num_edge_type; i++) delete []_edge_type_degree[i];
    }

    // normalize over networks
    // only the relative values (instead of absolute values) of each meta-path matter
    double *weights = new double[num_edge_type]; 
    for (int i = 0; i < num_edge_type; i ++) weights[i] = 1;
    // set your weights for edge types here:
    // weights[0] = 0.1; weights[1] = 0.9;
    if (path_normalization) {
      double *_edge_type_w = new double[num_edge_type];
      memset(_edge_type_w, 0, sizeof(double) * num_edge_type);
      for (int64 i = 0; i < num_edges; i++) _edge_type_w[edge_type[i]] += edge_weight[i];
      // for (int64 i = 0; i < num_edges; i++) edge_weight[i] *= 1.000000 / _edge_type_w[edge_type[i]];
      for (int64 i = 0; i < num_edges; i++) edge_weight[i] *= weights[edge_type[i]] / _edge_type_w[edge_type[i]];
      delete [] _edge_type_w;
      printf("each meta-path edge weight is normalized (to one)\n");
    }

    /* processing node type, edge type */
    vertex_type = (int *)calloc(num_vertices, sizeof(int));
    for (int i = 0; i < num_vertices; i++) {
      vertex[i].type = 0;
      vertex_type[i] = 0;
    }
    num_node_type = 1;
    if (num_edge_type > 0) {
      printf("Number of edge type: %d          \n", num_edge_type);
      printf("[edge_name2type] edge type name => type index, edge_type_w\n");
      for (map<string, int>::const_iterator it = edge_name2type.begin();
          it != edge_name2type.end(); it++) {
        printf("\t%s => %d, %f\n", it->first.c_str(), it->second, edge_type_w[it->second]);
      }
    } else {
      num_edge_type = 1;
      printf("Number of edge type: %d          \n", num_edge_type);
      printf("[edge_name2type/edge_type2name not presented]\n");
    }

    node_type_to_edge_type = new bool[num_node_type * num_edge_type];
    for (int i = 0; i < num_node_type; i++) {
      int i_row_start = i * num_edge_type;
      for (int j = 0; j < num_edge_type; j++) {
        node_type_to_edge_type[i_row_start + j] = 1;
      }
    }

    // compute vertex_degree_of_etype
    vertex_degree_of_etype = new double[num_vertices * num_edge_type];
    memset(vertex_degree_of_etype, 0, sizeof(double) * num_vertices * num_edge_type);
    for (int64 i = 0; i < num_edges; i++) {
      int src = edge_source_id[i];
      int dst = edge_target_id[i];
      int e_type = edge_type[i];
      double w = edge_weight[i];
      vertex_degree_of_etype[src * num_edge_type + e_type] += w;
      vertex_degree_of_etype[dst * num_edge_type + e_type] += w;
    }

    graph.vertex = vertex;
    graph.vertex_type = vertex_type;
    graph.vertex_degree_of_etype = vertex_degree_of_etype;
    graph.edge_source_id = edge_source_id;
    graph.edge_target_id = edge_target_id;
    graph.edge_weight = edge_weight;
    graph.edge_type = edge_type;
    graph.edge_type_w = edge_type_w;
    graph.node_type_to_edge_type = node_type_to_edge_type;
    graph.num_vertices_p = &num_vertices;
    graph.num_edges_p = &num_edges;
    graph.num_node_type_p = &num_node_type;
    graph.num_edge_type_p = &num_edge_type;

    end = clock();
    printf("network loaded in %.2f (s)\n", (double)(end-start) / CLOCKS_PER_SEC);
  }

  /*
   * Read node to type_name file
   */
  void load_node_type(string node_type_file) {
    char line_buffer[MAX_LINE_LEN], node_name[MAX_STRING], type_name[MAX_STRING];
    int vid, type;
    int *num_node_in_type;
    FILE *fin = fopen(node_type_file.c_str(), "rb");
    if (fin == NULL) {
      printf("ERROR: node type file not exist..\n");
      exit(1);
    }
    num_node_type = 0;
    while (fgets(line_buffer, sizeof(line_buffer), fin)) {
      sscanf(line_buffer, "%s %s", node_name, type_name);
      type_name[1] = '\0'; // node type name only depends on the first char
      vid = search_hash_table(node_name, vertex);
      if (vid == -1) continue;
      if (node_name2type.find(type_name) == node_name2type.end()) {
        type = num_node_type++;
        node_name2type[type_name] = type;
        node_type2name[type] = type_name;
      } else {
        type = node_name2type[type_name];
      }
      vertex[vid].type = type;
      vertex_type[vid] = -1; // mark as loaded
    }
    for (int i = 0; i < num_vertices; i++) {
      // make sure every node has a type now
      assert(vertex_type[i] == -1);
      vertex_type[i] = vertex[i].type;
    }
    num_node_in_type = new int[num_node_type];
    memset(num_node_in_type, 0, sizeof(int) * num_node_type);
    for (int i = 0; i < num_vertices; i++)
      num_node_in_type[vertex_type[i]]++;
    printf("Number of node type: %d          \n", num_node_type);
    printf("[node_name2type] node type name => type index, num_node_in_type\n");
    for (map<string, int>::iterator it = node_name2type.begin(); it != node_name2type.end(); it ++)
      printf("\t%s => %d, %d\n", it->first.c_str(), it->second, num_node_in_type[it->second]);
    fclose(fin);

    _reload_node_type_to_edge_type();

    delete [] num_node_in_type;
  }

  /*
   * Re-construct the schema relation between node type and edge type
   */
  void _reload_node_type_to_edge_type(bool printing = true) {
    if (node_type_to_edge_type != NULL) {
      delete []node_type_to_edge_type;
    }
    node_type_to_edge_type = new bool[num_node_type * num_edge_type];
    memset(node_type_to_edge_type, 0, sizeof(bool) * num_node_type * num_edge_type);
    graph.node_type_to_edge_type = node_type_to_edge_type;

    // set to 1/true if any connectity from the node type to the edge type is observed in network
    for (int64 i = 0; i < num_edges; i++) {
      int src, dst, e_type, src_type, dst_type;
      src = edge_source_id[i];
      dst = edge_target_id[i];
      src_type = vertex_type[src];
      dst_type = vertex_type[dst];
      e_type = edge_type[i];
      node_type_to_edge_type[src_type * num_edge_type + e_type] = 1;
      node_type_to_edge_type[dst_type * num_edge_type + e_type] = 1;
    }

    // print the connectity schema
    if (printing) {
      printf("Node type to edge type schema in network: \n");
      for (int i = 0; i < num_node_type; i++) {
        int i_row_start = i * num_edge_type;
        for (int j = 0; j < num_edge_type; j++) {
          if (node_type_to_edge_type[i_row_start + j]) {
            string node_type_name("-");
            string edge_type_name("-");
            if (node_type2name.find(i) != node_type2name.end())
              node_type_name = node_type2name[i];
            if (edge_type2name.find(j) != edge_type2name.end())
              edge_type_name = edge_type2name[j];
            printf("\t%s ~~> %s\n",node_type_name.c_str(), edge_type_name.c_str());
          }
        }
      }
    }
  }

 public:
  explicit DataHelper(string network_file, string node_type_file = string(),
    bool path_normalization = false, Config *conf_p = NULL) :
      VertexHashTable(),
      conf_p(conf_p),
      node_type_to_edge_type(NULL),
      num_vertices(0),
      num_node_type(0),
      num_edges(0),
      num_edge_type(0),
      max_num_vertices(10000) {
    load_network(network_file, path_normalization);
    assert(hash_table_size > 10 * num_vertices);  // probably should set a bigger hash_table_size
    assert(neg_table_size > 2 * num_edges);       // probably should set a bigger neg_table_size

    if (node_type_file.size() > 0) {
      load_node_type(node_type_file);
    } else {
      printf("Number of node type: %d          \n", num_node_type);
      printf("[node_name2type/node_type2name not presented]\n");
    }
  }

  /*
   * loading test file of the form: "src dst label", where src/dst are nodes in network
   * TODO: support training file loading
   */
  void load_test(string test_file) {
    char name_v1[MAX_STRING], name_v2[MAX_STRING], line_buffer[MAX_LINE_LEN], type_name[MAX_STRING];
    int src, dst, num_separator = 0, skip = 0;
    int64 num_lines = 0;
    float label = 0;
    bool has_missing_test_etype = false;

    // saving real test into file
    bool save_real_test = false;
    FILE *fo = NULL;
    if (save_real_test) {
      printf("[INFO] saving test real to file..\n");
      string real_test_file("test.txt.regular_real");
      fo = fopen(real_test_file.c_str(), "wb");
      assert(fo != NULL);
    }

    FILE *fin = fopen(test_file.c_str(), "rb");
    if (fin == NULL) {
      printf("ERROR: test file not found!\n");
      exit(1);
    }
    while (fgets(line_buffer, sizeof(line_buffer), fin)) num_lines++;
    test_pairs.reserve(num_lines);
    test_pairs_label.reserve(num_lines);

    fin = fopen(test_file.c_str(), "rb");
    for (int64 i = 0; i != num_lines; i++) {
      type_name[0] = '\0';
      fgets(line_buffer, sizeof(line_buffer), fin);
      if (num_separator == 0) {
        // read one line to find out the seporator, and be consistent
        for (size_t i = 0; i < MAX_LINE_LEN; i++) {
          if (line_buffer[i] == '\0') break;
          else if (line_buffer[i] == ' ' || line_buffer[i] == '\t') num_separator++;
        }
      }
      if (num_separator == 2) {
        sscanf(line_buffer, "%s %s %f", name_v1, name_v2, &label);
      } else if (num_separator == 3) {
        sscanf(line_buffer, "%s %s %f %s", name_v1, name_v2, &label, type_name);
      } else {
        printf("ERROR: seporator mis-match, check test file format..\n");
        exit(1);
      }
      if (i % 10000 == 0) {
        printf("Reading test lines: %.3lf%%%c", i / (double)(num_lines + 1) * 100, 13);
        fflush(stdout);
      }

      src = search_hash_table(name_v1, vertex);
      if (src == -1) {skip++; continue;}  // debug, not big deal, just ignore unseen nodes
      // if (src == -1) printf("%s\n", name_v1);
      // assert(src != -1);
      dst = search_hash_table(name_v2, vertex);
      if (dst == -1) {skip++; continue;}
      // if (dst == -1) printf("%s\n", name_v2);
      // assert(dst != -1);

      test_pairs.push_back(make_pair(src, dst));
      test_pairs_label.push_back(label);
      if (type_name[0] != '\0') {
        test_pairs_type.push_back(string(type_name));
        int e_type = -1;
        if (edge_name2type.find(type_name) != edge_name2type.end()) {
          e_type = edge_name2type[type_name];
        }
        if (e_type == -1) {
          has_missing_test_etype = true;
          test_pairs_etype.push_back(-1);
        } else {
          test_pairs_etype.push_back(e_type);
        }
      }

      if (save_real_test) {
        fprintf(fo, "%s\t%s\t%f\t%s\n", name_v1, name_v2, label, type_name);
      }
    }

    if (has_missing_test_etype)
      printf("[WARNING!!!!!!!!!!] There are unkown edge type in test pairs.\n");

    printf("Number of test pairs: %ld          \n", test_pairs.size());
    if (skip > 0)
      printf("[WARNING!!!!!!!!] %d test points are skiped due to node is not in training.\n", skip);

    // save real test file
    if (save_real_test) {
      fclose(fo);
      exit(0);
    }

    assert(test_pairs.size() == test_pairs_label.size());
    fclose(fin);
  }

  /*
   * Loading paper-author train or test (w/ feature) file
   *
   * pa_file: paper-author train or test file
   * po_file: paper-feature train or test file
   *
   * require paper to be in int, and author and all features being in the network
   */
  void load_pa_trainortest(string pa_file, string po_file, bool is_training) {
    char name_v1[MAX_STRING], name_v2[MAX_STRING], line_buffer[MAX_LINE_LEN];
    int src, dst;
    int64 num_lines = 0;
    float label = 0;

    string                  trainortest_name;
    // vector<int>             *group, *target_group;
    vector<pair<int, int> > *pairs;
    vector<real>            *pairs_label;
    map<int, vector<int> >  *src_features, *dst_features;

    if (is_training) {
      trainortest_name = "train";
      pairs = &train_pairs;
      pairs_label = &train_pairs_label;
      src_features = &train_src_features;
      dst_features = &train_dst_features;
    } else {
      trainortest_name = "test";
      pairs = &test_pairs;
      pairs_label = &test_pairs_label;
      src_features = &test_src_features;
      dst_features = &test_dst_features;
    }

    FILE *fin = fopen(pa_file.c_str(), "rb");
    if (fin == NULL) {
      printf("ERROR: %s p2a file not found!\n", trainortest_name.c_str());
      exit(1);
    }
    while (fgets(line_buffer, sizeof(line_buffer), fin)) num_lines++;

    // load paper to authors candidate train or test pairs
    fin = fopen(pa_file.c_str(), "rb");
    pairs->reserve(num_lines);
    pairs_label->reserve(num_lines);
    for (int64 i = 0; i != num_lines; i++) {
      fscanf(fin, "%s %s %f", name_v1, name_v2, &label);
      if (i % 10000 == 0) {
        printf("Reading %s_p2a lines: %.3lf%%%c", trainortest_name.c_str(),
          i / (double)(num_lines + 1) * 100, 13);
        fflush(stdout);
      }
      src = atoi(name_v1);
      dst = search_hash_table(name_v2, vertex);
      assert(dst != -1);
      pairs->push_back(make_pair(src, dst));
      if (is_training) assert(label > 0);  // only positive pairs are given in training
      pairs_label->push_back(label);
    }
    assert(pairs->size() == pairs_label->size());
    fclose(fin);
    printf("Number of %s p2a pairs: %ld          \n", trainortest_name.c_str(), pairs->size());

    num_lines = 0;
    fin = fopen(po_file.c_str(), "rb");
    if (fin == NULL) {
      printf("ERROR: %s p2o file not found!\n", trainortest_name.c_str());
      exit(1);
    }
    while (fgets(line_buffer, sizeof(line_buffer), fin)) num_lines++;

    // load paper to features
    fin = fopen(po_file.c_str(), "rb");
    for (int64 i = 0; i != num_lines; i++) {
      fscanf(fin, "%s %s %f", name_v1, name_v2, &label);
      if (i % 10000 == 0) {
        printf("Reading %s_p2o lines: %.3lf%%%c", trainortest_name.c_str(),
          i / (double)(num_lines + 1) * 100, 13);
        fflush(stdout);
      }
      src = atoi(name_v1);
      dst = search_hash_table(name_v2, vertex);
      if (dst == -1 && !is_training) {
        continue; // debug, ignore all features only appear in test, could be the new year, hazard
      }
      assert(dst != -1);
      int k = src;
      int f = dst;
      map<int, vector<int> >::iterator lb = src_features->lower_bound(k);

      if(lb != src_features->end() && !(src_features->key_comp()(k, lb->first))) {
        // key already exists
        lb->second.push_back(f);
      }
      else {
        // the key does not exist in the map
        vector<int> f_vec;
        f_vec.push_back(f);
        src_features->insert(lb, map<int, vector<int> >::value_type(k, f_vec));
      }
    }
    fclose(fin);

    // add author to features
    // also make sure all papers in pairs should have features
    for (vector<pair<int, int> >::const_iterator it = pairs->begin(); it != pairs->end();
        ++it) {
      int paper = it->first;
      int author = it->second;
      if (dst_features->find(author) == dst_features->end()) {
        vector<int> a_vec;
        a_vec.push_back(author);
        (*dst_features)[author] = a_vec;
      }
      assert(src_features->find(paper) != src_features->end());
    }

    // _check_test_data();
  }

  void _check_test_data() {
    printf("\nChecking test data..\n");
    int i = -1;
    for (vector<pair<int, int> >::const_iterator it = test_pairs.begin(); it != test_pairs.end();
        ++it) {
      i++;
      int src = it->first;
      int dst = it->second;
      float label = test_pairs_label[i];
      printf("%d %s %.2f\n", src, vertex[dst].name, label);
    }

    for (map<int, vector<int> >::const_iterator it = test_src_features.begin();
        it != test_src_features.end(); ++it) {
      const int &paper = it->first;
      const vector<int> &features = it->second;
      printf("%d:", paper);
      for (size_t i = 0; i < features.size(); i++) {
        printf("\t%s", vertex[features[i]].name);
      }
      printf("\n");
    }
  }

  /*
   * loading group informtion for train/test file
   */
  void load_group(string group_file, bool is_training) {
    char line_buffer[MAX_LINE_LEN];
    int num_lines = 0, val;
    FILE *fin = fopen(group_file.c_str(), "rb");
    if (fin == NULL) {
      printf("ERROR: group file %s not found!\n", group_file.c_str());
      exit(1);
    }
    while (fgets(line_buffer, sizeof(line_buffer), fin)) num_lines++;

    fin = fopen(group_file.c_str(), "rb");
    vector<int> * group;
    if (is_training)
      group = &train_group;
    else
      group = &test_group;
    while (fgets(line_buffer, MAX_LINE_LEN, fin)) {
      val = atoi(line_buffer);
      group->push_back(val);
    }
    fclose(fin);
  }

  // constructing group informtion from train/test pairs
  // require train_pairs, test_pairs, and organized according to paper
  void construct_group(bool test_only = true) {
    int prev_src, cur_size, i;

    if (!test_only) {
      // train group
      printf("Constructing train group from train pairs...\r");
      fflush(stdout);
      prev_src = train_pairs[0].first;
      cur_size = 1, i = 0;
      for (vector<pair<int, int> >::const_iterator it = train_pairs.begin() + 1; it != train_pairs.end();
          ++it) {
        i++;
        int src = it->first;
        if (src == prev_src) {
          cur_size++;
        } else {
          train_group.push_back(cur_size);
          cur_size = 1;
          prev_src = src;
        }
      }
      train_group.push_back(cur_size);
    }

    // test_group
    printf("Constructing test group from test pairs...\r");
    fflush(stdout);
    prev_src = test_pairs[0].first;
    cur_size = 1, i = 0;
    string prev_type_name;
    bool set_task_start_end = test_pairs_type.size() > 0? true: false;
    if (set_task_start_end) {
      prev_type_name = test_pairs_type[0];
      test_task_group_start_end[prev_type_name] = make_pair(0, -1);
    }
    for (vector<pair<int, int> >::const_iterator it = test_pairs.begin() + 1; it != test_pairs.end();
        ++it) {
      i++;
      int src = it->first;
      if (src == prev_src) {
        cur_size++;
      } else {
        test_group.push_back(cur_size);
        cur_size = 1;
        prev_src = src;
      }
      if (set_task_start_end) {
        string &cur_type_name = test_pairs_type[i];
        if (cur_type_name != prev_type_name) {
          if (cur_size != 1) printf("%d\t%d\n", i, cur_size);
          assert(cur_size == 1);  // task switch can only occur at the same time as group switch
          test_task_group_start_end[prev_type_name].second = test_group.size();
          // make sure the task types' continouty by assuring it never appears before
          assert(test_task_group_start_end.find(cur_type_name) == test_task_group_start_end.end());
          test_task_group_start_end[cur_type_name] = make_pair(test_group.size(), -1);
          prev_type_name = cur_type_name;
        }
      }
    }
    test_group.push_back(cur_size);
    if (set_task_start_end) test_task_group_start_end[prev_type_name].second = test_group.size();

    /* test construction and print
    for (size_t i = 0; i < test_group.size(); i++) {
      if (test_group[i] != 50)
        printf("%d, %d\n", i, test_group[i]);
      assert(test_group[i] == 50);
      // printf("%d\n", test_group[i]);
    }
    */
    printf("Number of test groups %ld                  \n", test_group.size());

    /* test and print test_task_group_start_end */
    if (set_task_start_end) {
      printf("[test tasks]\n");
      for (map<string, pair<int, int> >::const_iterator it = test_task_group_start_end.begin();
          it != test_task_group_start_end.end(); it ++) {
        printf("\t%s, start group: %d, end group %d\n", it->first.c_str(),
          it->second.first,  it->second.second);
      }
    }

    printf("Done constructing group                              \n");
  }

  /*
   * Split the train/test data based on group of test data
   * using_traing: set to true if using training data for split, otherwise use test data for it
   * target_ratio: the ratio to split out
   */
  void traintest_split(bool using_traing, double target_ratio) {
    vector<int>             *group, *target_group;
    vector<pair<int, int> > *pairs, *target_pairs;
    vector<real>            *pairs_label, *target_pairs_label;
    map<int, vector<int> >  *src_features, *target_src_features;
    map<int, vector<int> >  *dst_features, *target_dst_features;
    vector<int>             group_new;
    vector<pair<int, int> > pairs_new;
    vector<real>            pairs_label_new;
    uint                    cur, nex, train_size, test_size;
    clock_t                 start, end;
    printf("Train test spliting...\r");
    fflush(stdout);
    start = clock();

    if (using_traing) {
      group = &train_group;
      pairs = &train_pairs;
      pairs_label = &train_pairs_label;
      src_features = &train_src_features;
      dst_features = &train_dst_features;
      target_group = &test_group;
      target_pairs = &test_pairs;
      target_pairs_label = &test_pairs_label;
      target_src_features = &test_src_features;
      target_dst_features = &test_dst_features;
    } else {
      group = &test_group;
      pairs = &test_pairs;
      pairs_label = &test_pairs_label;
      src_features = &test_src_features;
      dst_features = &test_dst_features;
      target_group = &train_group;
      target_pairs = &train_pairs;
      target_pairs_label = &train_pairs_label;
      target_src_features = &train_src_features;
      target_dst_features = &train_dst_features;
    }
    target_group->reserve(group->size() * target_ratio + 1);
    target_pairs->reserve(pairs->size() * target_ratio + 1);
    target_pairs_label->reserve(pairs_label->size() * target_ratio + 1);
    group_new.reserve(group->size() * (1 - target_ratio) + 1);
    pairs_new.reserve(pairs->size() * (1 - target_ratio) + 1);
    pairs_label_new.reserve(pairs_label->size() * (1 - target_ratio) + 1);
    assert(pairs->size() == pairs_label->size());

    // split group by ratio
    cur = 0;
    for (vector<int>::const_iterator it = group->begin(); it != group->end(); it++) {
      nex = cur + *it;
      if (rand() / (double) RAND_MAX < target_ratio) {
        // split it
        target_group->push_back(*it);
        for (size_t i = cur; i < nex; i++) {
          target_pairs->push_back((*pairs)[i]);
          target_pairs_label->push_back((*pairs_label)[i]);
        }
      } else {
        // keep it
        group_new.push_back(*it);
        for (size_t i = cur; i < nex; i++) {
          pairs_new.push_back((*pairs)[i]);
          pairs_label_new.push_back((*pairs_label)[i]);
        }
      }
      cur = nex;
    }
    assert(nex == pairs->size());

    // rewrite original info
    *group = group_new;
    *pairs = pairs_new;
    *pairs_label = pairs_label_new;

    // copy features, simply
    *target_src_features = *src_features;
    *target_dst_features = *dst_features;

    end = clock();
    if (using_traing) {
      train_size = group_new.size();
      test_size = target_group->size();
    } else {
      train_size = target_group->size();
      test_size = group_new.size();
    }
    printf("Train test splited in %.2f (s), with %u / %u training / test samples. \n",
      (double)(end - start) / CLOCKS_PER_SEC, train_size, test_size);
  }

  void add_test_to_train() {
    printf("[WARNING!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!] Adding test PA data into training.\n");
    for (size_t i = 0; i < test_pairs.size(); i++) {
      if (test_pairs_label[i] == 0) continue;
      train_pairs.push_back(test_pairs[i]);
      train_pairs_label.push_back(test_pairs_label[i]);
    }
    for (map<int, vector<int> >::iterator it = test_src_features.begin();
        it != test_src_features.end(); it++) {
      train_src_features[it->first] = it->second;
    }
    for (map<int, vector<int> >::iterator it = test_dst_features.begin();
        it != test_dst_features.end(); it++) {
      train_dst_features[it->first] = it->second;
    }
  }

  int get_num_vertices() {
    return num_vertices;
  }

  int get_num_edges() {
    return num_edges;
  }

  int get_vertex_id(char *name) {
    return search_hash_table(name, vertex);
  }

  const Graph * get_graph() {
    return &graph;
  }

  const Vertex * get_vertex() {
    return vertex;
  }

  const int * get_vertex_type() {
    return vertex_type;
  }

  const map<int, string> & get_node_type2name() {
    return node_type2name;
  }

  const map<string, int> & get_node_name2type() {
    return node_name2type;
  }

  const map<int, string> & get_edge_type2name() {
    return edge_type2name;
  }

  const map<string, int> & get_edge_name2type() {
    return edge_name2type;
  }

  const int * get_edge_type() {
    return edge_type;
  }

  const double * get_edge_weight() {
    return edge_weight;
  }

  const int * get_edge_source_id() {
    return edge_source_id;
  }

  const int * get_edge_target_id() {
    return edge_target_id;
  }

  const vector<int> * get_train_group() {
    return &train_group;
  }

  const vector<pair<int, int> > * get_train_pairs() {
    return &train_pairs;
  }

  const vector<real> * get_train_pairs_label() {
    return &train_pairs_label;
  }

  const map<int, vector<int> > * get_train_src_features() {
    return &train_src_features;
  }

  const map<int, vector<int> > * get_train_dst_features() {
    return &train_dst_features;
  }

  const vector<int> * get_test_group() {
    return &test_group;
  }

  const vector<pair<int, int> > * get_test_pairs() {
    return &test_pairs;
  }

  const vector<real> * get_test_pairs_label() {
    return &test_pairs_label;
  }

  const vector<string> * get_test_pairs_type() {
    return &test_pairs_type;
  }

  const vector<int> * get_test_pairs_etype() {
    return &test_pairs_etype;
  }

  const map<int, vector<int> > * get_test_src_features() {
    return &test_src_features;
  }

  const map<int, vector<int> > * get_test_dst_features() {
    return &test_dst_features;
  }

  const map<string, pair<int, int> > * get_test_task_group_start_end() {
    return &test_task_group_start_end;
  }
};
