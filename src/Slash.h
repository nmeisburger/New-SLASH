#pragma once

#include "DOPH.h"
#include "HashTable.h"

class Slash {
 public:
  Slash(uint64_t K, uint64_t L, uint64_t range_pow, uint64_t reservoir_size);

  void InsertSVM(std::string datafile, uint64_t N, uint64_t offset, uint64_t avg_dim,
                 uint64_t batch_size);

  QueryResult<uint32_t> QuerySVM(std::string queryfile, uint64_t Q, uint64_t avg_dim,
                                 uint64_t topk);

  ~Slash() {
    delete hasher;
    delete hash_tables;
  }

 private:
  DOPH<uint32_t, uint32_t>* hasher;
  HashTable<uint32_t, uint32_t>* hash_tables;
};