#pragma once
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <string.h>
#include <vector>
#include "./common.h"
#include "./utility.h"
using namespace std;

class Config {
  int arg_pos(char *str, int argc, char **argv) {
    int a;
    for (a = 1; a < argc; a++) if (!strcmp(str, argv[a])) {
      if (a == argc - 1) {
        printf("Argument missing for %s\n", str);
        exit(1);
      }
      return a;
    }
    return -1;
  }

  void print_conf() {
    // print some important configurations
    printf("--------------------------------\n");
    printf("Threads: %d\n", num_threads);
    printf("Samples: %lldM\n", total_samples / 1000000);
    printf("Negative: %d\n", num_negative);
    printf("Dimension: %d\n", dim);
    printf("Initial rho: %lf\n", rho);
    if (num_train_threads >= 0) printf("Train Threads %d\n", num_train_threads);
    printf("path_normalization %d\n", path_normalization);
    printf("eta %f\n", eta);
    printf("epsilon %f\n", epsilon);
    printf("sigma %f\n", sigma);
    printf("mu %f\n", mu);
    printf("lambda %f\n", lambda);
    printf("gamma %f\n", gamma);
    printf("omega %f\n", omega);
    printf("PA_loss %d\n", PA_loss);
    printf("PA_neg_sampling_pow %f\n", PA_neg_sampling_pow);
    printf("PA_neg_base_deg %f\n", PA_neg_base_deg);
    printf("map_topk %d\n", map_topk);
    printf("--------------------------------\n");

   }

  void edge_type_preprocessing() {
    char line_buffer[MAX_LINE_LEN];

    // edge type conf preparation and path selection
    if (path_file.size() > 0) {
      // set up valid_paths according to path file
      if (path_conf_file.size() > 0) {
        printf("[ERROR!] path_file and path_conf_file are in conflict, only one should be given.\n");
        exit(-1);
      }
      int line_number = path_line;  // select the path of this line, line number starting from 0
      printf("[WARNING!!!!!!!!!!!!!!!!!!!!!!!!!!!] do path selection using line %d of file %s!!!\n",
        line_number, path_file.c_str());
      FILE *fp = fopen(path_file.c_str(), "r");
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
    } else if (path_conf_file.size() > 0) {
      // read edge type conf file
      use_path_conf = true;
      FILE *fp = fopen(path_conf_file.c_str(), "r");
      assert(fp != NULL);
      while (fgets(line_buffer, sizeof(line_buffer), fp)) {
        int pos = 0;
        while (line_buffer[++pos] != '\n');
        assert(pos < MAX_LINE_LEN);  line_buffer[pos] = '\0';
        vector<string> tuples = split(line_buffer, ' ');
        string path = tuples[0];
        float weihgt = atof(tuples[1].c_str());
        int direction, order;
        if (tuples[2] == "normal")
          direction = PATH_DIRECTION_NORMAL;
        else if (tuples[2] == "reverse")
          direction = PATH_DIRECTION_REVERSE;
        else if (tuples[2] == "bidirection")
          direction = PATH_DIRECTION_BIDIRECTION;
        else {
          printf("[ERROR!] unrecognized direction in path conf file.\n");
          exit(-1);
        }
        if (tuples[3] == "single")
          order = PATH_ORDER_SINGLE;
        else if (tuples[3] == "context")
          order = PATH_ORDER_CONTEXT;
        else {
          printf("[ERROR!] unrecognized proximity/order in path conf file.\n");
          exit(-1);
        }
        float sampling_pow = atof(tuples[4].c_str());
        float base_deg = atof(tuples[5].c_str());
        valid_paths.push_back(path);
        path_weight.push_back(weihgt);
        path_direction.push_back(direction);
        path_order.push_back(order);
        path_sampling_pow.push_back(sampling_pow);
        path_base_deg.push_back(base_deg);
      }
      fclose(fp);
    }
  }

 public:
  string network_file, node_type_file;
  string train_group_file, test_group_file;
  string train_file, test_file, train_feature_file, test_feature_file;
  string embedding_infile, embedding_outfile, pred_file, path_file, path_conf_file;
  int is_binary = 0, path_normalization = 1, num_threads = 1, num_train_threads = 0, dim = 100, num_negative = 5;
  int64 total_samples = 1;
  int map_topk = 10;
  real rho = 0.025, eta = 0, epsilon = 0, sigma = 0, mu = 0;
  real lambda = -1, gamma = -1;
  int PA_loss = 1, path_line = 0, net_randomize = 0;
  real omega = -1; // embedding task sampling rate
  real supf_dropout = 0;
  int is_sampling_neg_train = 1;  // whether to use random generated negative samples
  real train_percent = 1;   // when is_sampling_neg_train = true, use to reduce train group in train file
                            // when is_sampling_neg_train = false, use to split train group from test file
  float PA_neg_sampling_pow = 0., PA_neg_base_deg = 1;  // default is uniform distribution

  int option0 = 0;

  vector<string> valid_paths;
  vector<float> path_weight;
  vector<int> path_direction;
  vector<int> path_order;
  vector<float> path_sampling_pow;
  vector<float> path_base_deg;
  double path_sum_default = 10000.;
  int path_direction_default = PATH_DIRECTION_BIDIRECTION;
  int path_order_default = PATH_ORDER_CONTEXT;
  float path_sampling_pow_default = NEG_SAMPLING_POWER;
  float path_base_deg_default = 1;
  bool use_path_conf = false;

  Config(int argc, char **argv) {
    int i;
    if (argc == 1) {
      printf("<Warning> This option menu may be not complete </Warning>\n\n");
      printf("Options:\n");
      printf("Parameters for training:\n");
      printf("\t-network <file>\n");
      printf("\t\tUse network data from <file> to train the model\n");
      printf("\t-output <file>\n");
      printf("\t\tUse <file> to save the learnt embeddings\n");
      printf("\t-binary <int>\n");
      printf("\t\tSave the learnt embeddings in binary moded; default is 0 (off)\n");
      printf("\t-size <int>\n");
      printf("\t\tSet dimension of vertex embeddings; default is 100\n");
      printf("\t-negative <int>\n");
      printf("\t\tNumber of negative examples; default is 5\n");
      printf("\t-samples <int>\n");
      printf("\t\tSet the number of training samples as <int>Million; default is 1\n");
      printf("\t-threads <int>\n");
      printf("\t\tUse <int> threads (default 1)\n");
      printf("\t-rho <float>\n");
      printf("\t\tSet the starting learning rate; default is 0.025\n");
    }
    if ((i = arg_pos((char *)"-network", argc, argv)) > 0) network_file = string(argv[i + 1]);
    if ((i = arg_pos((char *)"-node2type", argc, argv)) > 0) node_type_file = string(argv[i + 1]);
    if ((i = arg_pos((char *)"-train", argc, argv)) > 0) train_file = string(argv[i + 1]);
    if ((i = arg_pos((char *)"-test", argc, argv)) > 0) test_file = string(argv[i + 1]);
    if ((i = arg_pos((char *)"-train_f", argc, argv)) > 0) train_feature_file = string(argv[i + 1]);
    if ((i = arg_pos((char *)"-test_f", argc, argv)) > 0) test_feature_file = string(argv[i + 1]);
    if ((i = arg_pos((char *)"-input", argc, argv)) > 0) embedding_infile = string(argv[i + 1]);
    if ((i = arg_pos((char *)"-output", argc, argv)) > 0) embedding_outfile = string(argv[i + 1]);
    if ((i = arg_pos((char *)"-pred", argc, argv)) > 0) pred_file = string(argv[i + 1]);
    if ((i = arg_pos((char *)"-path_conf_file", argc, argv)) > 0) path_conf_file = string(argv[i + 1]);  // file contains valid paths to use
    if ((i = arg_pos((char *)"-path_file", argc, argv)) > 0) path_file = string(argv[i + 1]);  // file contains valid paths to use
    if ((i = arg_pos((char *)"-path_line", argc, argv)) > 0) path_line = atoi(argv[i + 1]);  // use paths in the line in path_file
    if ((i = arg_pos((char *)"-threads", argc, argv)) > 0) num_threads = atoi(argv[i + 1]);
    if ((i = arg_pos((char *)"-binary", argc, argv)) > 0) is_binary = atoi(argv[i + 1]);
    if ((i = arg_pos((char *)"-size", argc, argv)) > 0) dim = atoi(argv[i + 1]);
    if ((i = arg_pos((char *)"-negative", argc, argv)) > 0) num_negative = atoi(argv[i + 1]);
    if ((i = arg_pos((char *)"-samples", argc, argv)) > 0) total_samples = atoi(argv[i + 1]);
    if ((i = arg_pos((char *)"-map_topk", argc, argv)) > 0) map_topk = atoi(argv[i + 1]);
    if ((i = arg_pos((char *)"-PA_loss", argc, argv)) > 0) PA_loss = atoi(argv[i + 1]);
    if ((i = arg_pos((char *)"-PA_neg_sampling_pow", argc, argv)) > 0) PA_neg_sampling_pow = atof(argv[i + 1]);
    if ((i = arg_pos((char *)"-PA_neg_base_deg", argc, argv)) > 0) PA_neg_base_deg = atof(argv[i + 1]);
    if ((i = arg_pos((char *)"-omega", argc, argv)) > 0) omega = atof(argv[i + 1]);
    if ((i = arg_pos((char *)"-rho", argc, argv)) > 0) rho = atof(argv[i + 1]);
    if ((i = arg_pos((char *)"-eta", argc, argv)) > 0) eta = atof(argv[i + 1]);
    if ((i = arg_pos((char *)"-epsilon", argc, argv)) > 0) epsilon = atof(argv[i + 1]);
    if ((i = arg_pos((char *)"-sigma", argc, argv)) > 0) sigma = atof(argv[i + 1]);
    if ((i = arg_pos((char *)"-mu", argc, argv)) > 0) mu = atof(argv[i + 1]);
    if ((i = arg_pos((char *)"-lambda", argc, argv)) > 0) lambda = atof(argv[i + 1]);
    if ((i = arg_pos((char *)"-gamma", argc, argv)) > 0) gamma = atof(argv[i + 1]);
    // following parameters better leave as default
    if ((i = arg_pos((char *)"-option0", argc, argv)) > 0) option0 = atoi(argv[i + 1]);
    if ((i = arg_pos((char *)"-train_group", argc, argv)) > 0) train_group_file = string(argv[i + 1]);
    if ((i = arg_pos((char *)"-test_group", argc, argv)) > 0) test_group_file = string(argv[i + 1]);
    if ((i = arg_pos((char *)"-path_normalization", argc, argv)) > 0) path_normalization = atoi(argv[i + 1]);
    if ((i = arg_pos((char *)"-supf_dropout", argc, argv)) > 0) supf_dropout = atof(argv[i + 1]);
    if ((i = arg_pos((char *)"-net_randomize", argc, argv)) > 0) net_randomize = atoi(argv[i + 1]);
    if ((i = arg_pos((char *)"-train_threads", argc, argv)) > 0) num_train_threads = atoi(argv[i + 1]);
    if ((i = arg_pos((char *)"-train_percent", argc, argv)) > 0) train_percent = atof(argv[i + 1]);
    if ((i = arg_pos((char *)"-sample_neg_train", argc, argv)) > 0) is_sampling_neg_train = atoi(argv[i + 1]);
    total_samples *= 1000000;

    // simple preprocessing
    if (omega >= 0) {
      if (num_train_threads == 0)
        num_train_threads = num_threads;
      if (num_train_threads != num_threads) {
        printf("[ERROR!] num_train_threads %d, num_threads %d should be equal when omega is set.\n",
          num_train_threads, num_threads);
        exit(-1);
      }
    }

    // other preprocessing
    edge_type_preprocessing();
    print_conf();
  }

};
