#pragma once

#include <fstream>
#include <iostream>
#include <sstream>

template <typename Label_t>
class SvmDataset {
 private:
  bool sequentiallyLabeled;

  static void ReadSvmDatasetHelper(const std::string& filename, SvmDataset& result, uint64_t n,
                                   uint64_t offset = 0) {
    std::ifstream file(filename);
    std::string line;

    uint64_t totalRead = 0;
    uint64_t totalLines = 0;
    uint64_t totalDim = 0;

    while (std::getline(file, line)) {
      if (totalLines++ < offset) {
        continue;
      }

      result.markers[totalRead++] = totalDim;
      std::stringstream stream(line);

      std::string item;
      stream >> item;  // ignore label
      while (stream >> item) {
        size_t pos = item.find(":");
        long index = atol(item.substr(0, pos).c_str());
        float value = atof(item.substr(pos + 1).c_str());

        result.indices[totalDim] = index;
        result.values[totalDim] = value;
        totalDim++;
      }
      if (totalRead >= n) {
        break;
      }
    }

    if (totalRead < n) {
      std::cout << "Only read " << totalRead << " out of " << n << " lines from file " << filename
                << std::endl;
      exit(1);
    }

    std::cout << "Read " << totalRead << " vectors with a total dimension " << totalDim
              << std::endl;
    result.markers[totalRead] = totalDim;
  }

 public:
  uint64_t len;
  uint32_t* indices;
  float* values;
  uint32_t* markers;

  union {
    Label_t* labels;
    Label_t start;
  };

  SvmDataset(uint64_t _len, uint64_t avgDim, Label_t _start)
      : sequentiallyLabeled(true), len(_len), start(_start) {
    indices = new uint32_t[len * avgDim];
    values = new float[len * avgDim];
    markers = new uint32_t[len + 1];
  }

  SvmDataset(uint64_t _len, uint64_t avgDim, Label_t* _labels)
      : sequentiallyLabeled(false), len(_len), labels(_labels) {
    indices = new uint32_t[len * avgDim];
    values = new float[len * avgDim];
    markers = new uint32_t[len + 1];
  }

  bool IsSequentiallyLabeled() const { return sequentiallyLabeled; }

  uint32_t* Indices(uint64_t i) { return indices + markers[i]; }

  float* Values(uint64_t i) { return values + markers[i]; }

  uint64_t Len(uint64_t i) { return markers[i + 1] - markers[i]; }

  static SvmDataset ReadSvmDataset(const std::string& filename, Label_t* labels, uint64_t n,
                                   uint64_t avgDim, uint64_t offset = 0) {
    SvmDataset data(n, avgDim, labels);
    ReadSvmDatasetHelper(filename, data, n, offset);
    return data;
  }

  static SvmDataset ReadSvmDataset(const std::string& filename, Label_t start, uint64_t n,
                                   uint64_t avgDim, uint64_t offset = 0) {
    SvmDataset data(n, avgDim, start);
    ReadSvmDatasetHelper(filename, data, n, offset);
    return data;
  }

  void Dump() {
    for (uint64_t i = 0; i < len; i++) {
      if (sequentiallyLabeled) {
        std::cout << start + i << " ";
      } else {
        std::cout << labels[i] << " ";
      }
      for (uint32_t j = markers[i]; j < markers[i + 1]; j++) {
        std::cout << indices[j] << ":" << values[j] << " ";
      }
      std::cout << std::endl;
    }
  }

  ~SvmDataset() {
    delete[] indices;
    delete[] values;
    delete[] markers;

    if (!sequentiallyLabeled) {
      delete[] labels;
    } else {
      if (!std::is_trivially_destructible<Label_t>::value) {
        start.~Label_t();
      }
    }
  }
};