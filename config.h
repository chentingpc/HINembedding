#pragma once
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <string.h>
#include "./common.h"
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

 public:
  string network_file, node_type_file;
  string train_group_file, test_group_file;
  string train_file, test_file, train_feature_file, test_feature_file;
  string embedding_infile, embedding_outfile, pred_file, path_file;
  int is_binary = 0, path_normalization = 0, num_threads = 1, num_train_threads = 0, dim = 100, num_negative = 5;
  int64 total_samples = 1;
  int map_topk = 10;
  real rho = 0.025, eta = 0, epsilon = 0, sigma = 0;
  real lambda = -1, gamma = -1;
  int is_sampling_neg_train = 1, PA_loss = 0, path_line = 0, net_randomize = 0;
  real omega = -1; // embedding task sampling rate
  real supf_dropout = 0;
  real train_percent = 1;   // when is_sampling_neg_train = true, use to reduce train group in train file
                            // when is_sampling_neg_train = false, use to split train group from test file
  int option0 = 0;

  Config(int argc, char **argv) {
    int i;
    if (argc == 1) {
      printf("<Warning> This option menue is outdated </Warning>\n\n");
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
      printf("\nExamples:\n");
      printf("./line -train net.txt -output vec.txt -binary 1 -size 200 -negative 5 -samples 100 -rho 0.025 -threads 20\n\n");
    }
    if ((i = arg_pos((char *)"-network", argc, argv)) > 0) network_file = string(argv[i + 1]);
    if ((i = arg_pos((char *)"-node2type", argc, argv)) > 0) node_type_file = string(argv[i + 1]);
    if ((i = arg_pos((char *)"-train_group", argc, argv)) > 0) train_group_file = string(argv[i + 1]);
    if ((i = arg_pos((char *)"-test_group", argc, argv)) > 0) test_group_file = string(argv[i + 1]);
    if ((i = arg_pos((char *)"-train", argc, argv)) > 0) train_file = string(argv[i + 1]);
    if ((i = arg_pos((char *)"-test", argc, argv)) > 0) test_file = string(argv[i + 1]);
    if ((i = arg_pos((char *)"-train_f", argc, argv)) > 0) train_feature_file = string(argv[i + 1]);
    if ((i = arg_pos((char *)"-test_f", argc, argv)) > 0) test_feature_file = string(argv[i + 1]);
    if ((i = arg_pos((char *)"-input", argc, argv)) > 0) embedding_infile = string(argv[i + 1]);
    if ((i = arg_pos((char *)"-output", argc, argv)) > 0) embedding_outfile = string(argv[i + 1]);
    if ((i = arg_pos((char *)"-pred", argc, argv)) > 0) pred_file = string(argv[i + 1]);
    if ((i = arg_pos((char *)"-path_file", argc, argv)) > 0) path_file = string(argv[i + 1]);  // file contains valid paths to use
    if ((i = arg_pos((char *)"-path_line", argc, argv)) > 0) path_line = atoi(argv[i + 1]);  // use paths in the line in path_file
    if ((i = arg_pos((char *)"-path_normalization", argc, argv)) > 0) path_normalization = atoi(argv[i + 1]);
    if ((i = arg_pos((char *)"-PA_loss", argc, argv)) > 0) PA_loss = atoi(argv[i + 1]);
    if ((i = arg_pos((char *)"-net_randomize", argc, argv)) > 0) net_randomize = atoi(argv[i + 1]);
    if ((i = arg_pos((char *)"-binary", argc, argv)) > 0) is_binary = atoi(argv[i + 1]);
    if ((i = arg_pos((char *)"-size", argc, argv)) > 0) dim = atoi(argv[i + 1]);
    if ((i = arg_pos((char *)"-negative", argc, argv)) > 0) num_negative = atoi(argv[i + 1]);
    if ((i = arg_pos((char *)"-samples", argc, argv)) > 0) total_samples = atoi(argv[i + 1]);
    if ((i = arg_pos((char *)"-sample_neg_train", argc, argv)) > 0) is_sampling_neg_train = atoi(argv[i + 1]);
    if ((i = arg_pos((char *)"-train_percent", argc, argv)) > 0) train_percent = atof(argv[i + 1]);
    if ((i = arg_pos((char *)"-rho", argc, argv)) > 0) rho = atof(argv[i + 1]);
    if ((i = arg_pos((char *)"-eta", argc, argv)) > 0) eta = atof(argv[i + 1]);
    if ((i = arg_pos((char *)"-epsilon", argc, argv)) > 0) epsilon = atof(argv[i + 1]);
    if ((i = arg_pos((char *)"-sigma", argc, argv)) > 0) sigma = atof(argv[i + 1]);
    if ((i = arg_pos((char *)"-lambda", argc, argv)) > 0) lambda = atof(argv[i + 1]);
    if ((i = arg_pos((char *)"-gamma", argc, argv)) > 0) gamma = atof(argv[i + 1]);
    if ((i = arg_pos((char *)"-omega", argc, argv)) > 0) omega = atof(argv[i + 1]);
    if ((i = arg_pos((char *)"-supf_dropout", argc, argv)) > 0) supf_dropout = atof(argv[i + 1]);
    if ((i = arg_pos((char *)"-map_topk", argc, argv)) > 0) map_topk = atoi(argv[i + 1]);
    if ((i = arg_pos((char *)"-threads", argc, argv)) > 0) num_threads = atoi(argv[i + 1]);
    if ((i = arg_pos((char *)"-train_threads", argc, argv)) > 0) num_train_threads = atoi(argv[i + 1]);
    if ((i = arg_pos((char *)"-option0", argc, argv)) > 0) option0 = atoi(argv[i + 1]);
    total_samples *= 1000000;

    if (omega >= 0) {
      if (num_train_threads == 0)
        num_train_threads = num_threads;
      if (num_train_threads != num_threads) {
        printf("[ERROR!] num_train_threads %d, num_threads %d should be equal when omega is set.\n",
          num_train_threads, num_threads);
        exit(-1);
      }
    }
  }
};
