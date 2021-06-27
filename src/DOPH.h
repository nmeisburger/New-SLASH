#pragma once

#include <random>

#include "DataLoader.h"

template <typename Label_t, typename Hash_t>
class DOPH {
 private:
  uint64_t K, L, numHashes, logNumHashes, rangePow, range, binsize;

  uint32_t* randSeeds;
  uint32_t seed, dhSeed;

  constexpr uint64_t HashIdx(uint64_t i, uint64_t table) { return i * L + table; }

  uint32_t RandDoubleHash(uint32_t binid, uint32_t cnt);

  Hash_t* ComputeMinHashes(uint32_t* nonzeros, uint32_t len);

 public:
  DOPH(uint64_t _K, uint64_t _L, uint64_t _rangePow);

  Hash_t* Hash(const SvmDataset<Label_t>& dataset);

  ~DOPH();
};