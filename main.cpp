#include "./common.h"
#include "./config.h"
#include "./data_helper.h"
#include "./sampler.h"
#include "./emb_model.h"
#include "./sup_model.h"
#include <time.h>

int main(int argc, char **argv) {
  time_t start, end;
  time(&start);

  Config conf(argc, argv);

  DataHelper data_helper = DataHelper(conf.network_file, conf.node_type_file,
                                      conf.path_normalization, &conf);
  if (conf.test_file.length() > 0) {
    data_helper.load_test(conf.test_file);
    data_helper.construct_group();
  }
  NodeSampler node_sampler = NodeSampler(data_helper.get_graph(), &conf);
  EdgeSampler edge_sampler = EdgeSampler(data_helper.get_graph());
  SupervisedModel model = SupervisedModel(&data_helper, &node_sampler, &edge_sampler,
                                          conf.dim, &conf);
  if (conf.embedding_infile.size() > 0)
    model.load(conf.embedding_infile, conf.is_binary);
  model.fit();
  model.save(conf.embedding_outfile, conf.is_binary, conf.pred_file);

  time(&end);
  printf("The program finishes in %ld seconds.\n", (end - start));
  return 0;
}
