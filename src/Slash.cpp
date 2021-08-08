#include "Slash.h"

#include <mpi.h>

#include <chrono>

#include "DataLoader.h"
#include "DistributedLog.h"

Slash::Slash(uint64_t K, uint64_t L, uint64_t range_pow, uint64_t reservoir_size) {
  hasher = new DOPH<uint32_t, uint32_t>(K, L, range_pow);
  hash_tables = new HashTable<uint32_t, uint32_t>(L, reservoir_size, range_pow);
}

void Slash::InsertSVM(std::string datafile, uint64_t N, uint64_t offset, uint64_t avg_dim,
                      uint64_t batch_size) {
  int rank, world_size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  uint64_t base_n = N / world_size;
  uint64_t local_n = base_n;
  if (static_cast<uint64_t>(rank) < N % world_size) {
    local_n++;
  }
  uint64_t local_offset = base_n * rank + std::min<uint64_t>(rank, N % world_size);

  LOG << "Inserting: local_n = " << local_n << " local_offset = " << local_offset << std::endl;
  auto dataset = SvmDataset<uint32_t>::ReadSvmDataset(datafile, local_offset, local_n, avg_dim,
                                                      local_offset + offset);

  uint64_t num_batches = (local_n + batch_size - 1) / batch_size;
  auto start = std::chrono::high_resolution_clock::now();
  for (uint64_t batch = 0; batch < num_batches; batch++) {
    uint64_t start = batch * batch_size;
    uint64_t cnt = std::min(local_n, (batch + 1) * batch_size) - start;
    auto hashes = hasher->Hash(dataset, start, cnt);
    hash_tables->Insert(cnt, dataset.start + start, hashes);
  }
  auto end = std::chrono::high_resolution_clock::now();

  LOG << "Inserted " << local_n << " vectors in << " << num_batches << " batches in "
      << std::chrono::duration_cast<std::chrono::seconds>(end - start).count() << " seconds"
      << std::endl;
}

QueryResult<uint32_t> Slash::QuerySVM(std::string queryfile, uint64_t Q, uint64_t avg_dim,
                                      uint64_t topk) {
  LOG << "Querying" << std::endl;
  SvmDataset<uint32_t> queries =
      SvmDataset<uint32_t>::ReadSvmDataset(queryfile, (uint32_t)0, Q, avg_dim, 0);

  auto start = std::chrono::high_resolution_clock::now();
  auto qHashes = hasher->Hash(queries, 0, Q);
  auto res = hash_tables->Query(queries.len, qHashes, topk);
  auto end = std::chrono::high_resolution_clock::now();

  LOG << "Perfomred " << Q << " queries in "
      << std::chrono::duration_cast<std::chrono::seconds>(end - start).count() << " seconds"
      << std::endl;
  return res;
}