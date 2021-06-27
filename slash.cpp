#include <iostream>
#include <random>

#include "src/DataLoader.h"
#include "src/DistributedLog.h"
#include "src/HashTable.h"

int main() {
  Logging::InitLogging("slash");

  LOG << "Does it work" << std::endl;

  HashTable<uint32_t, uint32_t> x(10, 10, 10);

  Logging::StopLogging();

  return 0;
}

template <typename T>
void Eval(SvmDataset<T>& data, SvmDataset<T>& queries, QueryResult<T>& results, uint64_t K) {
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