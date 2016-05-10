#pragma once
#include <string>
#include <string.h>
#include <map>
#include <math.h>
#include <pthread.h>
#include <cassert>
#include <unistd.h>
#include "./common.h"
#include "./config.h"
#include "./utility.h"
#include "./sampler.h"
#include "./data_helper.h"
using namespace std;

class EmbeddingModel {
 protected:
  int                     dim;
  int                     num_negative;
  real                    rho;
  real                    eta;
  real                    sigma;
  real                    lambda;
  real                    gamma;
  real                    init_rho;
  real                    init_eta;
  real                    init_sigma;
  real                    init_lambda;
  real                    init_gamma;
  int64                   total_samples;
  int64                   current_sample_count;
  int64                   current_sample_count_emb;  // count only by emb only
  int                     num_threads;
  bool                    using_edge_type_bias;
  bool                    using_transformation_vector;
  bool                    using_transformation_matrix;

  int64                   samples_before_switch_emb;
  int64                   samples_before_switch_other;
  bool                    *task_switchs_for_embedding;

  struct Context {
    EmbeddingModel *model_ptr;
    int id;
  };

  const Config            *conf_p;
  const Graph             *graph;
  const bool              *node_type_to_edge_type;
  const Vertex            *vertex;
  const int               *vertex_type;
  const double            *vertex_degree_of_etype;
  map<string, int>        node_name2type;
  map<int, string>        node_type2name;
  map<string, int>        edge_name2type;
  map<int, string>        edge_type2name;
  int                     num_node_type;
  int                     num_vertices;
  const int               *edge_type;
  const int               *edge_source_id;
  const int               *edge_target_id;
  int64                   num_edges;
  int                     num_edge_type;
  GSLRandUniform          uniform;

  bool                    fit_not_finished;
  bool                    *edge_type_using_context;
  bool                    *edge_type_apply_transform;
  int                     ls;
  int                     band_width;

  real                    *emb_vertex;            // main embedding vector for each node
                                                  // indexed by [vid * dim + k]
  real                    *emb_context;           // context embedding vector for certain nodes
                                                  // indexed by [vid * dim + k]
  real                    *weight_edge_type;      // weights for each meta-path
  real                    *W_m_band_chuck;        // memory chunk trick
  real                    **W_m_band;             // weight matrix diagonal band for each meta-path
  real                    ***w_mn;                // weight vectors for each (path, node-type)
  real                    *bias_edge_type;        // bias for each meta-path

  double                  *ll_edge_type;          // log-likelihood under different paths
  int64                   *ll_edge_type_cnt;      // count of times ll_edge_type being added up

  Sigmoid                 *sigmoid;
  DataHelper              *data_helper;
  NodeSampler             *node_sampler;
  EdgeSampler             *edge_sampler;
  GSLRandUniform          gsl_rand;

  // Initialize the vertex embedding and the context embedding
  // There are some over-allocation in here, meaning some parameters might not be used but allocated
  // Not worry as long as memory is not a bound
  void init_vector();

  inline real logl_noweight(real *vec_u, real *vec_v, const int &label) {
    // Note: this function does not include edge/node type weights
    real x = 0, f;
    for (int c = 0; c < dim; c++) x += vec_u[c] * vec_v[c];
    f = (*sigmoid)(x);
    return label > 0? fast_log(f+LOG_MIN): fast_log(1-f+LOG_MIN);
  }

  // Update embeddings & return likelihood, skip-gram negative sampling objective
  inline real update(real *vec_u, real *vec_v, real *vec_error, const int &label,
    const real &e_type_bias = 0, real *e_type_bias_err = NULL);

  // updaet embedding with vector weighting for embeddings
  inline real update_with_weight(real *vec_u, real *vec_v, real *vec_error,
    const int &label, const real &e_type_bias, real *e_type_bias_err,
    real *w_mn_u, real *w_mn_v, real *w_mn_err_u, real *w_mn_err_v);

  // update embedding with matrix weighting for embeddings
  inline real update_with_weight(real *vec_u, real *vec_v, real *vec_error, const int &label,
    const real &e_type_bias, real *e_type_bias_err, real *W_m_uv, real *W_m_err_uv);

  inline void train_on_sample(const int &id, int64 &u, int64 &v, const int64 &curedge,
    double &ll, uint64 &seed, real *vec_error, real *e_type_bias_err_vec = NULL,
    real ***w_mn_err = NULL, real **W_m_err = NULL);

  static void *fit_thread_helper(void* context) {
      Context *c = (Context *)context;
      EmbeddingModel* p = static_cast<EmbeddingModel*>(c->model_ptr);
      p->fit_thread(c->id);
      return NULL;
  }

  void fit_thread(int id);

 public:
  EmbeddingModel(DataHelper *data_helper,
                 NodeSampler *node_sampler,
                 EdgeSampler *edge_sampler,
                 int dim, const Config *conf_p) :
                dim(dim),
                eta(0),
                lambda(0),
                gamma(0),
                using_edge_type_bias(false),
                using_transformation_vector(false),
                using_transformation_matrix(false),
                samples_before_switch_emb(0),
                samples_before_switch_other(0),
                conf_p(conf_p),
                data_helper(data_helper),
                node_sampler(node_sampler),
                edge_sampler(edge_sampler) {
    sigmoid = new Sigmoid();

    graph = data_helper->get_graph();
    node_type_to_edge_type = graph->node_type_to_edge_type;
    vertex = graph->vertex;
    vertex_type = graph->vertex_type;
    vertex_degree_of_etype = graph->vertex_degree_of_etype;
    node_name2type = data_helper->get_node_name2type();
    node_type2name = data_helper->get_node_type2name();
    edge_name2type = data_helper->get_edge_name2type();
    edge_type2name = data_helper->get_edge_type2name();
    num_node_type = *graph->num_node_type_p;
    edge_type = graph->edge_type;
    edge_source_id = graph->edge_source_id;
    edge_target_id = graph->edge_target_id;
    num_vertices = *graph->num_vertices_p;
    num_edges = *graph->num_edges_p;
    num_edge_type = *graph->num_edge_type_p;

    init_rho = rho = conf_p->rho;
    init_eta = eta = conf_p->eta;
    init_sigma = sigma = conf_p->sigma;
    init_lambda = lambda = conf_p->lambda;
    init_gamma = gamma = conf_p->gamma;

    this->num_threads = conf_p->num_threads;
    this->num_negative = conf_p->num_negative;
    this->total_samples = conf_p->total_samples;

    edge_type_using_context = new bool[num_edge_type];
    for (int m = 0; m < num_edge_type; m++) edge_type_using_context[m] = true;
    const bool filter_edge_type_using_context = true;
    if (filter_edge_type_using_context) {
      printf("[WARNING!!!!!!!!!!!!!!!!!] Enabling the following edge type for using context:\n");
      for (int m = 0; m < num_edge_type; m++) {
        map<int, string>::const_iterator it = edge_type2name.find(m);
        assert(it != edge_type2name.end());
        if (it->second[0] == it->second[2]) {
          edge_type_using_context[m] = true;
          printf("%s\n", it->second.c_str());
        }
      }
    }

    edge_type_apply_transform = new bool[num_edge_type];
    for (int m = 0; m < num_edge_type; m++) edge_type_apply_transform[m] = true;
    const bool filter_edge_type_apply_transformation = false;
    if (filter_edge_type_apply_transformation) {
      printf("[WARNING!!!!!!!!!!!!!!!!!] Filtering the following edge type from transformation:\n");
      for (int m = 0; m < num_edge_type; m++) {
        map<int, string>::const_iterator it = edge_type2name.find(m);
        assert(it != edge_type2name.end());
        if (it->second == "A2W" ||
            it->second == "A2V" ||
            it->second == "A2Y" ||
            it->second == "A2P") {
          edge_type_apply_transform[m] = false;
          printf("%s\n", it->second.c_str());
        }
      }
    }

    init_task_schduler();

    init_vector();
  }

  void init_task_schduler() {
    current_sample_count = 0;
    current_sample_count_emb = 0;
    fit_not_finished = true;

    real emb_task_sampling_rate = conf_p->omega;
    if (emb_task_sampling_rate >= 0) {
      const int rounds_task_switch = 100;  // dark parameter
      int samples_per_round_thread = total_samples / rounds_task_switch / num_threads;
      samples_before_switch_emb = samples_per_round_thread * emb_task_sampling_rate;
      samples_before_switch_other = samples_per_round_thread * (1 - emb_task_sampling_rate);
    } else {
      samples_before_switch_emb = -1;
      samples_before_switch_other = -1;
    }

    printf("[INFO!] embedding task sampling rate %f, samples_before_switch_emb %lld, "
      "samples_before_switch_other %lld.\n", emb_task_sampling_rate, samples_before_switch_emb,
      samples_before_switch_other);

    task_switchs_for_embedding = new bool[num_threads];
    for (int i = 0; i < num_threads; i++) task_switchs_for_embedding[i] = true;
  }

  void fit() {
    time_t start, end;
    time(&start);

    if (num_threads == 0) printf("[DEBUG!] num_threads is set to zero!!!!!!!!!!!!!!!!!!!!!!!!\n");
    pthread_t *pt = (pthread_t *)malloc(num_threads * sizeof(pthread_t));
    printf("--------------------------------\n");
    Context *context[num_threads];
    for (int a = 0; a < num_threads; a++) {
      context[a] = new Context;
      context[a]->model_ptr = this;
      context[a]->id = a;
      pthread_create(&pt[a], NULL, fit_thread_helper, (void *)(context[a]));
    }
    for (int a = 0; a < num_threads; a++) {
      pthread_join(pt[a], NULL);
      free(context[a]);
    }
    printf("\n");

    time(&end);
    printf("Embedding training finished in %ld seconds.\n", (end - start));
  }

  void save(string embedding_file, bool is_binary) {
    if (embedding_file.size() == 0) {
      printf("[WARNING] embedding_file not saved due to no path given.\n");
      return;
    }
    bool is_type_constraint = false;  // debug, dark parameter
    vector<int> desired_node_types;
    // desired_node_types.push_back(node_name2type.at("P"));
    // desired_node_types.push_back(node_name2type.at("A"));
    printf("[INFO] saving embedding to file..\n");
    if (is_type_constraint)
      printf("[WARNING!!!!!!!!!!!!!] node type contraint is used when saving embedding_file.\n");
    FILE *fo = fopen(embedding_file.c_str(), "wb");
    assert(fo != NULL);
    fprintf(fo, "%d %d\n", num_vertices, dim);
    for (int a = 0; a < num_vertices; a++) {
      if (is_type_constraint){
        int n_type = vertex_type[a];
        bool skip = true;
        for (vector<int>::const_iterator it = desired_node_types.begin();
            it != desired_node_types.end(); it++) {
          if (*it == n_type) {
            skip = false;
            break;
          }
        }
        if (skip) continue;
      }
      fprintf(fo, "%s ", vertex[a].name);
      if (is_binary) for (int b = 0; b < dim; b++) fwrite(&emb_vertex[a * dim + b], sizeof(real), 1, fo);
      else for (int b = 0; b < dim; b++) fprintf(fo, "%lf ", emb_vertex[a * dim + b]);
      fprintf(fo, "\n");
    }
    fclose(fo);
  }

  void load(string embedding_file, bool is_binary) {
    printf("[INFO] loading embedding from file..\n");

    char _name[MAX_STRING];
    int _num_vertices, _dim;
    map<string, int> name2vid;
    for (int a = 0; a < num_vertices; a++) {
      string name(vertex[a].name);
      name2vid[name] = a;
    }

    FILE *fi = fopen(embedding_file.c_str(), "rb");
    assert(fi != NULL);
    fscanf(fi, "%d %d\n", &_num_vertices, &_dim);
    assert(_num_vertices == num_vertices);
    assert(_dim == dim);
    for (int a = 0; a < num_vertices; a++) {
      fscanf(fi, "%s", _name);
      int v = name2vid[_name];
      assert(strcmp(vertex[v].name, _name) == 0);
      _name[0] = fgetc(fi);
      if (is_binary) {
        for (int b = 0; b < dim; b++)
          fread(&emb_vertex[v * dim + b], sizeof(real), 1, fi);
      } else {
        for (int b = 0; b < dim; b++)
          fscanf(fi, "%f", &emb_vertex[v * dim + b]);
      }
    }
    fclose(fi);
  }
};
