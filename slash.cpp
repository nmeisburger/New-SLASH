#include <mpi.h>

#include <iostream>
#include <random>

#include "src/DOPH.h"
#include "src/DataLoader.h"
#include "src/DistributedLog.h"
#include "src/HashTable.h"

class InitHelper {
 public:
  InitHelper() {
    MPI_Init(0, 0);
    Logging::InitLogging("slash");
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

int main() {
  InitHelper _i_;

  uint64_t K = 4, L = 32, RP = 9, R = 128;

  uint64_t N = 10000, Q = 1000, topk = 20;

  std::string file = "/Users/nmeisburger/files/Research/data/webspam_wc_normalized_trigram.svm";

  LOG << "Creating hash table and hash function" << std::endl;
  HashTable<uint32_t, uint32_t> ht(L, R, RP);
  DOPH<uint32_t, uint32_t> hf(K, L, RP);

  LOG << "Reading data" << std::endl;
  SvmDataset<uint32_t> data = SvmDataset<uint32_t>::ReadSvmDataset(file, (uint32_t)0, N, 4000, Q);
  SvmDataset<uint32_t> queries =
      SvmDataset<uint32_t>::ReadSvmDataset(file, (uint32_t)0, Q, 4000, 0);

  LOG << "Inserting data" << std::endl;
  auto hashes = hf.Hash(data);
  ht.Insert(data.len, data.start, hashes);

  LOG << "Querying" << std::endl;
  auto qHashes = hf.Hash(queries);
  auto results = ht.Query(queries.len, qHashes, topk);


  LOG << "Evaluating" << std::endl;
  Eval<uint32_t>(data, queries, results, 1);
  Eval<uint32_t>(data, queries, results, 2);
  Eval<uint32_t>(data, queries, results, 4);

  return 0;
}