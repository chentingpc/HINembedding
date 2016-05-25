#include "emb_model.h"


void EmbeddingModel::init_vector() {
  printf("Initialize embedding vectors... \r");
  fflush(stdout);

  clock_t start, end;
  start = clock();
  int64 a, b;
  srand(time(NULL));

  // speed tips:
  // 1. posix_memalign can help a littble bit, less than 5% probably.
  // 2. 1d indexing compared to 2d indexing help a little bit, less than 5% probably.
  // 3. memory countinuousity helps more, more than 15% probably.
  a = posix_memalign((void **)&emb_vertex, 128, (int64)num_vertices * dim * sizeof(real));
  if (emb_vertex == NULL) { printf("Error: memory allocation failed\n"); exit(1); }
  for (b = 0; b < dim; b++) for (a = 0; a < num_vertices; a++)
    emb_vertex[a * dim + b] = (rand() / (real)RAND_MAX - 0.5) / dim;

  bool filter_edge_type_using_context = false;
  for (int m = 0; m < num_edge_type; m++)
    if (edge_type_using_context[m]) { filter_edge_type_using_context = true; break; }
  if (filter_edge_type_using_context) {
    a = posix_memalign((void **)&emb_context, 128, (int64)num_vertices * dim * sizeof(real));
    if (emb_context == NULL) { printf("Error: memory allocation failed\n"); exit(1); }
    for (b = 0; b < dim; b++) for (a = 0; a < num_vertices; a++)
      emb_context[a * dim + b] = (rand() / (real)RAND_MAX - 0.5) / dim;
  }

  // todo - following can be optimized by using memory chunk trick
  // initilize w vector for edge type
  w_mn = new real**[num_edge_type];
  for (int m = 0; m < num_edge_type; m++) {
    w_mn[m] = new real*[num_node_type];
    for (int n = 0; n < num_node_type; n++) {
      if (node_type_to_edge_type[n * num_edge_type + m]) {
        w_mn[m][n] = new real[dim];
        // debug
        for (int c = 0; c < dim; c++) {
          w_mn[m][n][c] = 1.;
          // w_mn[m][n][c] = rand() / (real)RAND_MAX + 0.5;
        }
      } else {
        w_mn[m][n] = NULL;
      }
    }
  }
  // initialize W_m_band, with memory chunk trick
  ls = 1;  // hazard
  band_width = (2 * ls + 1);
  W_m_band_chuck = new real[num_edge_type * dim * band_width];
  memset(W_m_band_chuck, 0, sizeof(real) * num_edge_type * dim * band_width);
  W_m_band = new real*[num_edge_type];
  for (int m = 0; m < num_edge_type; m++) {
    W_m_band[m] = &W_m_band_chuck[m * dim * band_width];
    for (int d = 0; d < dim; d++) {
      real *row = &W_m_band[m][d * band_width];
      for (int i = 0; i < band_width; i++) {  // debug
        if (i == ls) {  // diagonal
          if (!edge_type_apply_transform[m]) row[i] = 1.;
          else row[i] = rand() / (real)RAND_MAX + 0.5;
        } else {
          if (!edge_type_apply_transform[m]) row[i] = 0.;
          else row[i] = rand() / (real)RAND_MAX;
        }
      }
    }
  }
  // initilize bias for edge type
  bias_edge_type = new real[num_edge_type];
  for (int i = 0; i < num_edge_type; i++) bias_edge_type[i] = 0.;

  ll_edge_type = new double[num_edge_type];
  memset(ll_edge_type, 0, sizeof(double) * num_edge_type);

  ll_edge_type_cnt = new int64[num_edge_type];
  memset(ll_edge_type_cnt, 0, sizeof(int64) * num_edge_type);

  end = clock();
  printf("Embedding vectors initialized in %.2f (s).\n", (double)(end-start) / CLOCKS_PER_SEC);
}

inline real EmbeddingModel::update(real *vec_u, real *vec_v, real *vec_error, const int &label,
  const real &e_type_bias, real *e_type_bias_err) {
  real x = 0, f, g, g_w_rho;
  real reg_rho = lambda * rho;
  for (int c = 0; c < dim; c++) x += vec_u[c] * vec_v[c];
  f = (*sigmoid)(x + e_type_bias);
  g = (label - f);
  g_w_rho = g * rho;
  for (int c = 0; c < dim; c++) vec_error[c] += g_w_rho * vec_v[c] - reg_rho * vec_u[c];
  for (int c = 0; c < dim; c++) vec_v[c] += g_w_rho * vec_u[c] - reg_rho * vec_v[c];
  if (e_type_bias_err != NULL)
    *e_type_bias_err += g;
  return label > 0? fast_log(f+LOG_MIN): fast_log(1-f+LOG_MIN);
}

// dark parameter, debug
#define REG_ON_WU

inline real EmbeddingModel::update_with_weight(real *vec_u, real *vec_v, real *vec_error,
    const int &label, const real &e_type_bias, real *e_type_bias_err,
    real *w_mn_u, real *w_mn_v, real *w_mn_err_u, real *w_mn_err_v) {
  real x = 0, f, g;
  for (int c = 0; c < dim; c++)
    x += (vec_u[c] * w_mn_u[c]) * (vec_v[c] * w_mn_v[c]);
  f = (*sigmoid)(x + e_type_bias);
  g = (label - f);
  for (int c = 0; c < dim; c++) {
    real temp = g * (vec_v[c] * w_mn_v[c]);
#ifdef REG_ON_WU
    // regularization on WU
    real wu = w_mn_u[c] * vec_u[c];
    real reg_w_mn = lambda * wu * vec_u[c];
    real reg_vec = lambda * wu * w_mn_u[c];
#else
    // regularization on W and U
    real reg_w_mn = gamma * w_mn_u[c];
    real reg_vec = lambda * vec_u[c];
#endif
    w_mn_err_u[c] += temp * vec_u[c] - reg_w_mn;
    vec_error[c] += rho * (temp * w_mn_u[c] - reg_vec);
  }
  for (int c = 0; c < dim; c++) {
    real temp = g * (w_mn_u[c] * vec_u[c]);
    // regularization on WV
#ifdef REG_ON_WU
    real wv = w_mn_v[c] * vec_v[c];
    real reg_w_mn = lambda * wv * vec_v[c];
    real reg_vec = lambda * wv * w_mn_v[c];
#else
    // regularization on W and V, debug
    real reg_w_mn = gamma * w_mn_v[c];
    real reg_vec = lambda * vec_v[c];
#endif
    w_mn_err_v[c] += temp * vec_v[c] - reg_w_mn;
    vec_v[c] += rho * (temp * w_mn_v[c] - reg_vec);
  }
  if (e_type_bias_err != NULL)
    *e_type_bias_err += g;

  return label > 0? fast_log(f+LOG_MIN): fast_log(1-f+LOG_MIN);
}

inline real EmbeddingModel::update_with_weight(real *vec_u, real *vec_v, real *vec_error,
    const int &label, const real &e_type_bias, real *e_type_bias_err,
    real *W_m_uv, real *W_m_err_uv) {
  real x = 0, f, g, g_w_rho, rho_lambda;
  real *local_buffer = new real[dim * band_width];
  for (int c = 0; c < dim; c++) {
    int cban = c * band_width;
    real *W_m_uv_row = &W_m_uv[cban];
    real *local_buffer_row = &local_buffer[cban];
    for (int l = 0, k = c + (l - ls); l < band_width; l++, k++) {
      if (k < 0 || k >= dim) continue;
      real cross = vec_u[c] * vec_v[k];
      local_buffer_row[l] = cross - 0 * W_m_uv_row[l];  // hazard
      x += cross * W_m_uv_row[l];
    }
  }
  f = (*sigmoid)(x + e_type_bias);
  g = (label - f);
  g_w_rho = g * rho;
  rho_lambda = rho * lambda;
  for (int c = 0; c < dim; c++) {
    real g_link = 0;
    int cban = c * band_width;
    real *W_m_uv_row = &W_m_uv[cban];
    real *W_m_err_uv_row = &W_m_err_uv[cban];
    real *local_buffer_row = &local_buffer[cban];
    for (int l = 0, k = c + (l - ls); l < band_width; l++, k++) {
      if (k < 0 || k >= dim) continue;
      g_link += W_m_uv_row[l] * vec_v[k];
      W_m_err_uv_row[l] += g * local_buffer_row[l];
    }
    g_link *= g_w_rho;
    vec_error[c] += g_link - rho_lambda * vec_u[c];
  }
  for (int k = 0; k < dim; k++) {
    real g_link = 0;
    for (int l = 0, c = k + (l - ls), j = 2 * ls; l < band_width; l++, c++, j--) {
      if (c < 0 || c >= dim) continue;
      g_link += vec_u[c] * W_m_uv[c * band_width + j];  // here j=(k-c)+ls
    }
    g_link *= g_w_rho;
    vec_v[k] += g_link - rho_lambda * vec_v[k];
  }
  if (e_type_bias_err != NULL)
    *e_type_bias_err += g;
  delete []local_buffer;

  return label > 0? fast_log(f+LOG_MIN): fast_log(1-f+LOG_MIN);
}

inline void EmbeddingModel::train_on_sample(const int &id, int64 &u, int64 &v,
    const int64 &curedge, double &ll, uint64 &seed, real *vec_error, real *e_type_bias_err_vec,
    real ***w_mn_err, real **W_m_err) {
  int64 lu, lv, target;
  real *src_vec, *dst_vec;
  int label, src_type, dst_type, e_type;
  real e_type_bias = 0;
  real *e_type_bias_err = NULL;
  real *w_mn_u, *w_mn_v, *w_mn_err_u, *w_mn_err_v;
  real *W_m_err_uv, *W_m_uv;

  src_type = vertex_type[u];
  dst_type = vertex_type[v];
  e_type = edge_type[curedge];
  if (e_type_bias_err_vec != NULL) {
    e_type_bias = bias_edge_type[e_type];
    e_type_bias_err = &e_type_bias_err_vec[e_type];
  }
  if (w_mn_err != NULL) {
    w_mn_u = w_mn[e_type][src_type];
    w_mn_v = w_mn[e_type][dst_type];
    w_mn_err_u = w_mn_err[e_type][src_type];
    w_mn_err_v = w_mn_err[e_type][dst_type];
  }
  if (W_m_err != NULL) {
    W_m_err_uv = W_m_err[e_type];
    W_m_uv = W_m_band[e_type];
  }

  lu = u * dim;
  src_vec = &emb_vertex[lu];
  memset(vec_error, 0, sizeof(real) * dim);

  // NEGATIVE SAMPLING
  real ll_acc = 0;
  for (int d = 0; d != num_negative + 1; d++) {
    if (d == 0) {
      target = v;
      label = 1;
    } else {
      // target = node_sampler->sample(seed);
      // sample both u, v randomly, works worse
      // u = node_sampler->sample(seed, e_type, src_type);
      // lu = u * dim;
      // src_vec = &emb_vertex[lu];
      target = node_sampler->sample(seed, e_type, dst_type);
      label = 0;
      // random flipping sign, helpful
      // if (uniform() < 0.01)
      //  label = 1;
    }
    lv = target * dim;
    if (edge_type_using_context[e_type])
      dst_vec = &emb_context[lv];
    else
      dst_vec = &emb_vertex[lv];
    real ll_local;
    if (w_mn_err != NULL) {
      ll_local = update_with_weight(src_vec, dst_vec, vec_error, label,
                                    e_type_bias, e_type_bias_err,
                                    w_mn_u, w_mn_v, w_mn_err_u, w_mn_err_v);
    } else if (W_m_err != NULL) {
      ll_local = update_with_weight(src_vec, dst_vec, vec_error, label,
                                    e_type_bias, e_type_bias_err, W_m_uv, W_m_err_uv);
    } else {
#ifdef MAX_MARGIN_EMBEDDING
      // max-margin ranking objective for updating embedding (homo-net)
      if (d == 0) continue;
      real score_pos = 0, score_neg = 0;
      real *dst_vec_pos, *dst_vec_neg, *src_err;
      dst_vec_pos = &emb_vertex[v * dim];
      dst_vec_neg = dst_vec;
      src_err = vec_error;
      for (int c = 0; c < dim; c++) score_pos += src_vec[c] * dst_vec_pos[c];
      for (int c = 0; c < dim; c++) score_neg += src_vec[c] * dst_vec_neg[c];
      const real margin = -1;
      real margin_temp = score_pos - score_neg + margin;
      ll_local = margin_temp > 0? 0: margin_temp;

      real score_pos_err = margin_temp >= 0? 0: 1;
      real score_neg_err = -score_pos_err;
      for (int k = 0; k < dim; k++) {
        src_err[k] = rho * (score_pos_err * (dst_vec_pos[k] - dst_vec_neg[k]) -
                            lambda * src_err[k]);
        dst_vec_pos[k] += rho * (score_pos_err * src_vec[k] - lambda * dst_vec_pos[k]);
        dst_vec_neg[k] += rho * (score_neg_err * src_vec[k] - lambda * dst_vec_neg[k]);
      }
      for (int k = 0; k < dim; k++) src_vec[k] += src_err[k];
#else
      ll_local = update(src_vec, dst_vec, vec_error, label, e_type_bias, e_type_bias_err);
#endif
    }
    ll += ll_local;
    ll_acc += ll_local;
  }
  for (int c = 0; c < dim; c++) src_vec[c] += vec_error[c];

  if (id == 0) {
    ll_edge_type[e_type] += ll_acc;
    ll_edge_type_cnt[e_type]++;
  }
}

void EmbeddingModel::fit_thread(int id) {
  if (samples_before_switch_emb == 0) {
    printf("[WARNING!] turn down emb_model training..\n");
    if (id == 0) fit_not_finished = false;
    return;
  }
  // dark parameters, debug
  if (init_eta > 0) {
    using_transformation_vector = true;
    using_transformation_matrix = false;
    printf("[WARNING!!!!!!!!!!!!] using_transformation_vector %d,"
      "using_transformation_matrix %d, using_edge_type_bias %d\n",
      using_transformation_vector, using_transformation_matrix, using_edge_type_bias);
    printf("Exit here, comment the code if you really want to use \"bilinear\" model\n");
    exit(-1);
  }
  if (init_sigma > 0) {
    using_edge_type_bias = true;
  }

  int64 u, v;
  int64 count = 0, last_count = 0, ll_count = 0, curedge;
  int64 samples_task_round = 0;
  int direction = conf_p->path_direction_default;
  double prog = 0., ll = 0.;
  uint64 seed = static_cast<int64>(id);
  real lr_w = init_eta;
  real lr_bias = init_sigma;
  real *vec_error = new real[dim];
  real *e_type_bias_err_vec = NULL, ***w_mn_err = NULL, *W_m_err_chunk = NULL, **W_m_err = NULL;

  {  // init error holder for relational weights & bias, if not done, will follow homo-net embedding
    if (using_transformation_vector) {
      w_mn_err = new real**[num_edge_type];
      for (int m = 0; m < num_edge_type; m++) {
        w_mn_err[m] = new real*[num_node_type];
        for (int n = 0; n < num_node_type; n++) {
          if (node_type_to_edge_type[n * num_edge_type + m]) {
            w_mn_err[m][n] = new real[dim];
            memset(w_mn_err[m][n], 0, sizeof(real) * dim);
          } else { w_mn_err[m][n] = NULL; }
        }
      }
    }
    if (using_transformation_matrix) {
      W_m_err_chunk = new real[num_edge_type * dim * band_width];
      memset(W_m_err_chunk, 0, sizeof(real) * num_edge_type * dim * band_width);
      W_m_err = new real*[num_edge_type];
      for (int m = 0; m < num_edge_type; m++) {
        W_m_err[m] = &W_m_err_chunk[m * dim * band_width];
      }
    }
    if (using_edge_type_bias) {
      e_type_bias_err_vec = new real[num_edge_type];
      memset(e_type_bias_err_vec, 0, sizeof(real) * num_edge_type);
    }
  }

  while (current_sample_count < total_samples) {
    static const int count_interval = 100000;
    if (count - last_count > count_interval) {
      int64 incremental = count - last_count;
      current_sample_count += incremental;
      current_sample_count_emb += incremental;
      last_count = count;
      if (id == 0) {  // reset logistics, update learning rates
        real sample_ratio = current_sample_count_emb / (double)(current_sample_count + 1);
        prog = (real)current_sample_count / (real)(total_samples + 1);
        printf("Prog: %.2lf%%, emb_sample_ratio: %f, LogL: %.4lf\n", prog * 100, sample_ratio,
          ll / ll_count);
        fflush(stdout);
        rho = init_rho * (1. - prog);
        eta = init_eta * (1. - prog);
        sigma = init_sigma * (1. - prog);
        if (rho < init_rho * 0.001) rho = init_rho * 0.001;
        if (eta < init_eta * 0.001) eta = init_eta * 0.001;
        if (sigma < init_sigma * 0.001) sigma = init_sigma * 0.001;
        lr_w = eta;
        lr_bias = sigma;
        ll = ll_count = 0;
        static const int compress_to_cnt = count_interval / 5;  // downweight previous for smoothing
        for (int m = 0; m < num_edge_type; m++) {
          ll_edge_type[m] *= static_cast<double>(compress_to_cnt) / ll_edge_type_cnt[m];
          ll_edge_type_cnt[m] = compress_to_cnt;
        }
      }

      { // all threads asynchronously modify the weights and bias.
        if (using_transformation_vector) {
          // update weights for transforming embedding
          real _w_mn_learn_rate = 1. / count_interval * lr_w;
          for (int m = 0; m < num_edge_type; m++) {
            if (!edge_type_apply_transform[m]) continue;
            for (int n = 0; n < num_node_type; n++) {
              if (w_mn_err[m][n] != NULL) {
                real *w_mn_mn = w_mn[m][n];
                real *w_mn_err_mn = w_mn_err[m][n];
                for (int c = 0; c < dim; c++) w_mn_mn[c] += _w_mn_learn_rate * w_mn_err_mn[c];
                memset(w_mn_err_mn, 0, sizeof(real) * dim);
              }
            }
          }
        }
        if (using_transformation_matrix) {
          // update interaction weight matrix
          real _w_m_learn_rate = 1. / count_interval * lr_w;
          for (int m = 0; m < num_edge_type; m++) {
            if (!edge_type_apply_transform[m]) continue;
            for (int c = 0; c < dim; c++)
              for (int l = 0; l < band_width; l++)
                W_m_band[m][c * band_width + l] += W_m_err[m][c * band_width + l] * _w_m_learn_rate;
          }
          memset(W_m_err_chunk, 0, sizeof(real) * num_edge_type * dim * band_width);
        }
        if (using_edge_type_bias) {
          // update global bias
          real _b_learn_rate = 1. / count_interval * lr_bias;
          for (int m = 0; m < num_edge_type; m++) {
            bias_edge_type[m] += e_type_bias_err_vec[m] * _b_learn_rate;
          }
          memset(e_type_bias_err_vec, 0, sizeof(real) * num_edge_type);
        }
      }
    }

    if (samples_task_round == samples_before_switch_emb) {
      samples_task_round = 0;
      task_switchs_for_embedding[id] = false;
      while (!task_switchs_for_embedding[id] && fit_not_finished) {
        usleep(100);
      }
    }

    curedge = edge_sampler->sample();
    u = edge_source_id[curedge];
    v = edge_target_id[curedge];

    if (use_path_conf)
      direction = path_direction[edge_type[curedge]];

    if (direction == PATH_DIRECTION_BIDIRECTION) {
      if (gsl_rand() < 0.5)
        train_on_sample(id, u, v, curedge, ll, seed, vec_error,
                        e_type_bias_err_vec, w_mn_err, W_m_err);
      else
        train_on_sample(id, v, u, curedge, ll, seed, vec_error,
                        e_type_bias_err_vec, w_mn_err, W_m_err);
    } else if (direction == PATH_DIRECTION_NORMAL) {
      train_on_sample(id, u, v, curedge, ll, seed, vec_error,
                      e_type_bias_err_vec, w_mn_err, W_m_err);
    } else if (direction == PATH_DIRECTION_REVERSE) {
      train_on_sample(id, v, u, curedge, ll, seed, vec_error,
                      e_type_bias_err_vec, w_mn_err, W_m_err);
    } else {
      printf("[ERROR!] direction %d not recognized\n", direction);
      exit(-1);
    }
    count++;
    samples_task_round++;
    ll_count++;
  }

  if (id == 0) {
    if (using_transformation_vector) {
      // print out the transformation vector
      sleep(2);
      printf("------------------------------- w_mn -------------------------------\n");
      for (int n = 0; n < num_node_type; n++) {
        printf("\n\n[node_type %s]\n\n", node_type2name[n].c_str());
        for (int m = 0; m < num_edge_type; m++) {
          if (w_mn[m][n] != NULL) {
            printf("\t[edge_type %s]", edge_type2name[m].c_str());
            real *w_mn_mn = w_mn[m][n];
            for (int c = 0; c < dim; c++) printf(", %f", w_mn_mn[c]);
            printf("\n");
          }
        }
      }
    }
    if (using_transformation_matrix) {
      sleep(2);
      printf("------------------------------- W_m -------------------------------\n");
      for (int m = 0; m < num_edge_type; m++) {
        printf("\n\n[edge_type %s] each col is a dim\n\n", edge_type2name[m].c_str());
        for (int l = 0; l < band_width; l++) {
          for (int c = 0; c < dim; c++) {
            printf("%f, ", W_m_band[m][c * band_width + l]);
          }
          printf("\n");
        }
      }
    }
    if (using_edge_type_bias) {
      sleep(2);
      printf("------------------------------- bias_edge_type -------------------------------\n");
      for (int m = 0; m < num_edge_type; m++) {
        printf("\n[bias_edge_type %s] %f\n", edge_type2name[m].c_str(), bias_edge_type[m]);
      }
    }
    fflush(stdout);
  }

  fit_not_finished = false;
  pthread_exit(NULL);
}
