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

  LOG << "Inserted " << local_n << " vectors in " << num_batches << " batches in "
      << std::chrono::duration_cast<std::chrono::seconds>(end - start).count() << " seconds"
      << std::endl;
}

QueryResult<uint32_t> Slash::QuerySVMSingleMachine(std::string queryfile, uint64_t Q,
                                                   uint64_t avg_dim, uint64_t topk) {
  LOG << "Querying" << std::endl;
  SvmDataset<uint32_t> queries =
      SvmDataset<uint32_t>::ReadSvmDataset(queryfile, (uint32_t)0, Q, avg_dim, 0);

  auto start = std::chrono::high_resolution_clock::now();
  auto qHashes = hasher->Hash(queries, 0, Q);
  auto res = hash_tables->Query(queries.len, qHashes, topk);
  auto end = std::chrono::high_resolution_clock::now();

  LOG << "Performed " << Q << " queries in "
      << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count()
      << " milliseconds" << std::endl;
  return res;
}

constexpr uint64_t BufLocID(uint64_t query, uint64_t index, uint64_t K) {
  return query * K * 2 + index * 2;
}

constexpr uint64_t BufLocCnt(uint64_t query, uint64_t index, uint64_t K) {
  return query * K * 2 + index * 2 + 1;
}

QueryResult<uint32_t> Slash::QuerySVM(std::string queryfile, uint64_t Q, uint64_t avg_dim,
                                      uint64_t topk) {
  LOG << "Querying" << std::endl;
  SvmDataset<uint32_t> queries =
      SvmDataset<uint32_t>::ReadSvmDataset(queryfile, (uint32_t)0, Q, avg_dim, 0);

  auto start = std::chrono::high_resolution_clock::now();
  auto qHashes = hasher->Hash(queries, 0, Q);
  auto res = hash_tables->QueryWithCounts(queries.len, qHashes, topk);
  auto end = std::chrono::high_resolution_clock::now();

  LOG << "Performed " << Q << " queries in "
      << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count()
      << " milliseconds" << std::endl;

  uint32_t* send_buf = new uint32_t[Q * topk * 2];

  for (uint32_t q = 0; q < res.len(); q++) {
    uint32_t i = 0;
    for (; i < res.len(q); i++) {
      send_buf[BufLocID(q, i, topk)] = res[q][i].first;
      send_buf[BufLocCnt(q, i, topk)] = res[q][i].second;
    }
    for (; i < topk; i++) {
      send_buf[BufLocID(q, i, topk)] = std::numeric_limits<uint32_t>::max();
      send_buf[BufLocCnt(q, i, topk)] = std::numeric_limits<uint32_t>::max();
    }
  }

  uint32_t num_iter = std::ceil(std::log(world_size) / std::log(2));
  uint32_t* recv_buf = new uint32_t[Q * topk * 2];
  MPI_Status status;
  for (uint32_t iter = 0; iter < num_iter; iter++) {
    if (rank % ((int)std::pow(2, iter + 1)) == 0 && (rank + std::pow(2, iter)) < world_size) {
      int source = rank + std::pow(2, iter);
      LOG << "Iter: " << iter << " Receiving from: " << source << std::endl;
      MPI_Recv(recv_buf, Q * topk * 2, MPI_UNSIGNED, source, iter, MPI_COMM_WORLD, &status);

      uint32_t* new_send_buf = new uint32_t[Q * topk * 2];

      for (uint64_t q = 0; q < Q; q++) {
        uint32_t loc = 0, loc_self = 0, loc_recv = 0;
        while (loc < topk) {
          if (recv_buf[BufLocCnt(q, loc_recv, topk)] > send_buf[BufLocCnt(q, loc_self, topk)]) {
            new_send_buf[BufLocID(q, loc, topk)] = recv_buf[BufLocID(q, loc_recv, topk)];
            new_send_buf[BufLocCnt(q, loc, topk)] = recv_buf[BufLocCnt(q, loc_recv, topk)];
            loc++;
            loc_recv++;
          } else {
            new_send_buf[BufLocID(q, loc, topk)] = send_buf[BufLocID(q, loc_self, topk)];
            new_send_buf[BufLocCnt(q, loc, topk)] = send_buf[BufLocCnt(q, loc_self, topk)];
            loc++;
            loc_self++;
          }
        }
      }
      delete[] send_buf;

      send_buf = new_send_buf;

    } else if (rank % ((int)std::pow(2, iter + 1)) == ((int)std::pow(2, iter))) {
      int destination = rank - ((int)std::pow(2, iter));
      LOG << "Iter: " << iter << " Sending to: " << destination << std::endl;
      MPI_Send(send_buf, Q * topk * 2, MPI_UNSIGNED, destination, iter, MPI_COMM_WORLD);
    }
  }

  delete[] recv_buf;

  QueryResult<uint32_t> result(Q, topk);

  if (rank == 0) {
    for (uint32_t q = 0; q < Q; q++) {
      uint32_t loc = 0;
      uint32_t id = send_buf[BufLocID(q, loc, topk)];
      while (id != std::numeric_limits<uint32_t>::max()) {
        result[q][loc++] = id;
        if (loc >= topk) {
          break;
        }
        id = send_buf[BufLocID(q, loc, topk)];
      }
      result.len(q) = loc;
    }
  }

  return result;
}