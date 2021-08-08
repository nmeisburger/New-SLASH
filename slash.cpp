#include "src/Slash.h"

#include <mpi.h>

#include <iostream>
#include <random>

#include "src/Config.h"
#include "src/DataLoader.h"
#include "src/DistributedLog.h"

class InitHelper {
 public:
  InitHelper(std::string logPrefix) {
    MPI_Init(0, 0);
    Logging::InitLogging(logPrefix);
    LOG << "Initializing SLASH" << std::endl;
  }
  ~InitHelper() {
    LOG << "Finished, starting cleanup" << std::endl;
    MPI_Finalize();
    Logging::StopLogging();
  }
};

double Magnitude(float* v, uint64_t len) {
  double val = 0;
  for (uint64_t i = 0; i < len; i++) {
    val += v[i] * v[i];
  }
  return sqrt(val);
}

double SparseMultiply(uint32_t* iA, float* vA, uint64_t lA, uint32_t* iB, float* vB, uint64_t lB) {
  uint64_t a = 0, b = 0;
  double val = 0;
  while (a < lA && b < lB) {
    if (iA[a] == iB[b]) {
      val += vA[a] * vB[b];
      a++;
      b++;
    } else if (iA[a] < iB[b]) {
      a++;
    } else {
      b++;
    }
  }
  return val;
}

template <typename Label_t>
void Eval(SvmDataset<Label_t>& data, SvmDataset<Label_t>& queries, QueryResult<Label_t>& results,
          uint64_t K) {
  double totalSim = 0;
  uint64_t cnt = 0;
  for (uint64_t query = 0; query < queries.len; query++) {
    double tmpSim = 0;
    uint64_t tmpCnt = 0;
    for (uint64_t x = 0; x < std::min(K, results.len(query)); x++) {
      auto result = results[query][x];
      double innerProduct =
          SparseMultiply(queries.Indices(query), queries.Values(query), queries.Len(query),
                         data.Indices(result), data.Values(result), data.Len(result));
      double queryMagnitude = Magnitude(queries.Values(query), queries.Len(query));
      double dataMagnitude = Magnitude(data.Values(result), data.Len(result));

      tmpSim += innerProduct / (queryMagnitude * dataMagnitude);
      tmpCnt++;
    }
    totalSim += tmpSim / tmpCnt;
    cnt++;
  }

  LOG << "Average Cosine Similarity @" << K << " = " << totalSim / cnt << std::endl;
}

int main(int argc, char** argv) {
  if (argc != 2) {
    std::cerr << "Invalid arguments, usage '$ ./slash <config file name>'" << std::endl;
    return 1;
  }

  ConfigReader config(argv[1]);

  InitHelper _i_(config.StrVal("logfile"));

  uint64_t K = config.IntVal("K");
  uint64_t L = config.IntVal("L");
  uint64_t range_pow = config.IntVal("range_pow");
  uint64_t reservoir_size = config.IntVal("reservoir_size");

  Slash slash(K, L, range_pow, reservoir_size);

  uint64_t N = config.IntVal("data_len");
  uint64_t Q = config.IntVal("query_len");
  uint64_t topk = config.IntVal("topk");
  uint64_t avg_dim = config.IntVal("avg_dim");
  uint64_t batch_size = config.IntVal("batch_size");

  std::string data_file = config.StrVal("data_file");
  std::string query_file = config.StrVal("query_file");

  slash.InsertSVM(data_file, N, Q, avg_dim, batch_size);

  auto results = slash.QuerySVM(query_file, Q, avg_dim, topk);

  LOG << "Reading data for evaluation" << std::endl;
  SvmDataset<uint32_t> data =
      SvmDataset<uint32_t>::ReadSvmDataset(data_file, (uint32_t)0, N, avg_dim, Q);
  SvmDataset<uint32_t> queries =
      SvmDataset<uint32_t>::ReadSvmDataset(query_file, (uint32_t)0, Q, avg_dim, 0);

  LOG << "Evaluating" << std::endl;

  for (uint32_t i = 0; i < config.Len("evaluate"); i++) {
    Eval<uint32_t>(data, queries, results, config.IntVal("evaluate", i));
  }

  return 0;
}