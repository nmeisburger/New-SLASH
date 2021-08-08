#include "DOPH.h"

#define NULL_HASH ((uint32_t)-1)

constexpr uint32_t ODD(uint32_t x) { return x << 31 ? x : x + 1; }

template class DOPH<uint32_t, uint32_t>;

template <typename Label_t, typename Hash_t>
DOPH<Label_t, Hash_t>::DOPH(uint64_t _K, uint64_t _L, uint64_t _rangePow)
    : K(_K), L(_L), numHashes(_K * _L), rangePow(_rangePow), range(1 << rangePow) {
  binsize = std::ceil(range / numHashes);

  logNumHashes = std::floor(log2(numHashes));

  randSeeds = new uint32_t[numHashes];

  srand(10);
  for (uint64_t i = 0; i < numHashes; i++) {
    randSeeds[i] = ODD(rand());
  }

  srand(11);
  seed = ODD(rand());
  srand(12);
  dhSeed = ODD(rand());
}

template <typename Label_t, typename Hash_t>
Hash_t* DOPH<Label_t, Hash_t>::Hash(const SvmDataset<Label_t>& dataset, uint64_t offset,
                                    uint64_t num) {
  Hash_t* finalHashes = new Hash_t[num * L];

#pragma omp parallel for default(none) shared(dataset, offset, num, finalHashes)
  for (uint64_t n = offset; n < offset + num; n++) {
    uint32_t start = dataset.markers[n];
    Hash_t* allHashes = ComputeMinHashes(dataset.indices + start, dataset.markers[n + 1] - start);

    for (uint64_t tb = 0; tb < L; tb++) {
      Hash_t index = 0;
      for (uint64_t k = 0; k < K; k++) {
        Hash_t h = allHashes[K * tb + k];
        h *= randSeeds[K * tb + k];
        h ^= h >> 13;
        h ^= randSeeds[K * tb + k];
        index += h * allHashes[K * tb + k];
      }
      finalHashes[HashIdx(n - offset, tb)] = (index << 2) >> (32 - rangePow);
    }
    delete[] allHashes;
  }
  return finalHashes;
}

template <typename Label_t, typename Hash_t>
uint32_t DOPH<Label_t, Hash_t>::RandDoubleHash(uint32_t binid, uint32_t cnt) {
  uint32_t val = ((binid + 1) << 10) + cnt;
  return (dhSeed * val << 3) >> (32 - logNumHashes);
}

template <typename Label_t, typename Hash_t>
Hash_t* DOPH<Label_t, Hash_t>::ComputeMinHashes(uint32_t* nonzeros, uint32_t len) {
  Hash_t* hashes = new Hash_t[numHashes];

  for (uint64_t i = 0; i < numHashes; i++) {
    hashes[i] = NULL_HASH;
  }

  for (uint32_t i = 0; i < len; i++) {
    Hash_t h = nonzeros[i];
    h *= seed;
    h ^= h >> 13;
    h *= 0x85ebca6b;
    Hash_t curhash = ((h * nonzeros[i]) << 5) >> (32 - rangePow);
    uint32_t binid = std::min<uint64_t>(curhash / binsize, numHashes - 1);
    if (curhash < hashes[binid]) {
      hashes[binid] = curhash;
    }
  }

  Hash_t* finalHashes = new Hash_t[numHashes];

  for (uint64_t bin = 0; bin < numHashes; bin++) {
    Hash_t next = hashes[bin];
    if (next != NULL_HASH) {
      finalHashes[bin] = next;
      continue;
    }
    uint32_t cnt = 0;
    while (next == NULL_HASH) {
      cnt++;
      uint32_t index = RandDoubleHash(bin, cnt);
      next = hashes[index];
      if (cnt > 100) {
        next = (Hash_t)-1;
        break;
      }
    }
    finalHashes[bin] = next;
  }
  delete[] hashes;
  return finalHashes;
}

template <typename Label_t, typename Hash_t>
DOPH<Label_t, Hash_t>::~DOPH() {
  delete[] randSeeds;
}