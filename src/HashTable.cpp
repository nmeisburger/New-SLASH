#include "HashTable.h"

#include <assert.h>

#include <algorithm>
#include <iostream>
#include <unordered_map>
#include <vector>

template class HashTable<uint32_t, uint32_t>;

template <typename Label_t, typename Hash_t>
HashTable<Label_t, Hash_t>::HashTable(uint64_t _numTables, uint64_t _reservoirSize,
                                      uint64_t _rangePow, uint64_t _maxRand)
    : numTables(_numTables),
      reservoirSize(_reservoirSize),
      rangePow(_rangePow),
      range(1 << _rangePow),
      maxRand(_maxRand) {
  data = new Label_t[numTables * range * reservoirSize];
  genRand = new uint32_t[maxRand];

  mask = range - 1;

  srand(32);
  for (uint64_t i = 1; i < maxRand; i++) {
    genRand[i] = ((uint32_t)rand()) % (i + 1);
  }

  counters = new std::atomic<uint32_t>[numTables * range]();
}

template <typename Label_t, typename Hash_t>
void HashTable<Label_t, Hash_t>::Insert(uint64_t n, Label_t* labels, Hash_t* hashes) {
#pragma omp parallel for default(none) shared(n, labels, hashes)
  for (uint64_t i = 0; i < n; i++) {
    for (uint64_t table = 0; table < numTables; table++) {
      Hash_t rowIndex = HashMod(hashes[HashIdx(i, table)]);
      uint32_t counter = counters[CounterIdx(table, rowIndex)]++;

      if (counter < reservoirSize) {
        data[DataIdx(table, rowIndex, counter)] = labels[i];
      } else {
        counter = genRand[counter % maxRand];
        if (counter < reservoirSize) {
          data[DataIdx(table, rowIndex, counter)] = labels[i];
        }
      }
    }
  }
}

template <typename Label_t, typename Hash_t>
void HashTable<Label_t, Hash_t>::Insert(uint64_t n, Label_t start, Hash_t* hashes) {
#pragma omp parallel for default(none) shared(n, start, hashes)
  for (uint64_t i = 0; i < n; i++) {
    for (uint64_t table = 0; table < numTables; table++) {
      Hash_t rowIndex = HashMod(hashes[HashIdx(i, table)]);
      uint32_t counter = counters[CounterIdx(table, rowIndex)]++;

      if (counter < reservoirSize) {
        data[DataIdx(table, rowIndex, counter)] = start + i;
      } else {
        counter = genRand[counter % maxRand];
        if (counter < reservoirSize) {
          data[DataIdx(table, rowIndex, counter)] = start + i;
        }
      }
    }
  }
}

template <typename Label_t, typename Hash_t>
QueryResult<Label_t> HashTable<Label_t, Hash_t>::Query(uint64_t n, Hash_t* hashes, uint64_t k) {
  QueryResult<Label_t> result(n, k);
#pragma omp parallel for default(none) shared(n, hashes, k, result)
  for (uint64_t query = 0; query < n; query++) {
    std::unordered_map<Label_t, uint32_t> contents(reservoirSize * numTables);
    for (uint64_t table = 0; table < numTables; table++) {
      Hash_t rowIndex = HashMod(hashes[HashIdx(query, table)]);
      uint32_t counter = counters[CounterIdx(table, rowIndex)];

      for (uint64_t i = 0; i < std::min<uint64_t>(counter, reservoirSize); i++) {
        contents[data[DataIdx(table, rowIndex, i)]]++;
      }
    }

    std::pair<Label_t, uint32_t>* pairs = new std::pair<Label_t, uint32_t>[contents.size()]();
    uint64_t cnt = 0;
    for (const auto& x : contents) {
      pairs[cnt++] = x;  // std::move(x)?
    }

    std::sort(pairs, pairs + cnt, [](const auto& a, const auto& b) { return a.second > b.second; });

    uint64_t len = std::min(k, cnt);
    result.len(query) = len;
    for (uint64_t i = 0; i < len; i++) {
      result[query][i] = pairs[i].first;
    }
    delete[] pairs;
  }

  return result;
}

template <typename Label_t, typename Hash_t>
void HashTable<Label_t, Hash_t>::Dump() {
  for (uint64_t table = 0; table < numTables; table++) {
    std::cout << "Table: " << table << std::endl;
    for (uint64_t row = 0; row < range; row++) {
      uint32_t cnt = counters[CounterIdx(table, row)];
      std::cout << "[ " << row << " :: " << cnt << " ]";
      for (uint64_t i = 0; i < std::min<uint64_t>(cnt, reservoirSize); i++) {
        std::cout << "\t" << data[DataIdx(table, row, i)];
      }
      std::cout << std::endl;
    }
    std::cout << std::endl;
  }
}

template <typename Label_t, typename Hash_t>
HashTable<Label_t, Hash_t>::~HashTable() {
  delete[] data;
  delete[] counters;
  delete[] genRand;
}
