#pragma once

#include <atomic>

constexpr uint64_t MaxRand = 10000;

template <typename Label_t>
class QueryResult {
 private:
  Label_t* results;
  uint64_t* lens;
  uint64_t n, k;

 public:
  QueryResult(uint64_t _n, uint64_t _k) : n(_n), k(_k) {
    results = new Label_t[n * k]();
    lens = new uint64_t[n]();
  }

  QueryResult(const QueryResult& other) = delete;
  QueryResult& operator=(const QueryResult& other) = delete;
  QueryResult(QueryResult&& other) = default;
  QueryResult& operator=(QueryResult&& other) = default;

  uint64_t len() const { return n; }

  uint64_t& len(uint64_t i) { return lens[i]; }

  Label_t* operator[](uint64_t i) { return results + i * k; }

  ~QueryResult() {
    delete[] results;
    delete[] lens;
  }
};

template <typename Label_t, typename Hash_t>
class HashTable {
 private:
  uint64_t numTables, reservoirSize, rangePow, range, maxRand;
  Hash_t mask;

  Label_t* data;
  std::atomic<uint32_t>* counters;

  uint32_t* genRand;

  constexpr uint64_t CounterIdx(uint64_t table, uint64_t row) { return table * range + row; }

  constexpr uint64_t DataIdx(uint64_t table, uint64_t row, uint64_t offset) {
    return table * range * reservoirSize + row * reservoirSize + offset;
  }

  constexpr uint64_t HashIdx(uint64_t i, uint64_t table) { return i * numTables + table; }

  constexpr Hash_t HashMod(Hash_t hash) { return hash & mask; }

 public:
  HashTable(uint64_t _numTables, uint64_t _reservoirSize, uint64_t _rangePow, uint64_t _maxRand);

  void Insert(uint64_t n, Label_t* labels, Hash_t* hashes);

  QueryResult<Label_t> Query(uint64_t n, Hash_t* hashes, uint64_t k);

  ~HashTable();
};
