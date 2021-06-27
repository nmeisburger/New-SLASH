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

int main() {
  InitHelper _i_;

  uint64_t K = 4, L = 128, RP = 10, R = 128;

  uint64_t N = 1000, Q = 1000;

  std::string file = "/Users/nmeisburger/files/Research/data/webspam_wc_normalized_trigram.svm";

  LOG << "Creating hash table and hash function" << std::endl;
  HashTable<uint32_t, uint32_t> ht(L, R, RP);
  DOPH<uint32_t, uint32_t> hf(K, L, RP);

  LOG << "Reading data" << std::endl;
  SvmDataset<uint32_t> data = SvmDataset<uint32_t>::ReadSvmDataset(file, (uint32_t)0, N, 5000, Q);
  SvmDataset<uint32_t> queries =
      SvmDataset<uint32_t>::ReadSvmDataset(file, (uint32_t)0, Q, 5000, 0);

  LOG << "Inserting data" << std::endl;
  auto hashes = hf.Hash(data);
  ht.Insert(data.len, data.start, hashes);

  LOG << "Querying" << std::endl;
  auto qHashes = hf.Hash(queries);
  auto results = ht.Query(queries.len, qHashes, 10);

  LOG << "Evaluating" << std::endl;
  Eval<uint32_t>(data, queries, results, 10);

  return 0;
}

template <typename Label_t>
void Eval(SvmDataset<Label_t>& data, SvmDataset<Label_t>& queries, QueryResult<Label_t>& results,
          uint64_t K) {
  double totalSim;
  uint64_t cnt;
  for (uint64_t query = 0; query < queries.len; query++) {
    for (uint64_t x = 0; x < std::min(K, results.len(query)); x++) {
      auto result = results[query][x];
      double innerProduct =
          SparseMultiply(queries.Indices(query), queries.Values(query), queries.Len(query),
                         data.Indices(result), data.Values(result), data.Len(result));
      double queryMagnitude = Magnitude(queries.Values(query), queries.Len(query));
      double dataMagnitude = Magnitude(data.Values(result), data.Len(result));

      totalSim += innerProduct / (queryMagnitude * dataMagnitude);
      cnt++;
    }
  }

  std::cout << "Average Cosine Similarity @" << K << " = " << totalSim / cnt << std::endl;
}

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