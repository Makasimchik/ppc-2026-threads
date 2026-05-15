#include "titaev_m_sortirovka_betchera/omp/include/ops_omp.hpp"

#include <omp.h>

#include <algorithm>
#include <cstring>
#include <vector>

namespace titaev_m_sortirovka_betchera {

namespace {
uint64_t DoubleToOrderedUint(double value) {
  uint64_t x = 0;
  std::memcpy(&x, &value, sizeof(double));
  constexpr uint64_t kSignMask = (1ULL << 63);
  if ((x & kSignMask) != 0ULL) {
    x = ~x;
  } else {
    x ^= kSignMask;
  }
  return x;
}

double OrderedUintToDouble(uint64_t x) {
  constexpr uint64_t kSignMask = (1ULL << 63);
  if ((x & kSignMask) != 0ULL) {
    x ^= kSignMask;
  } else {
    x = ~x;
  }
  double result = 0.0;
  std::memcpy(&result, &x, sizeof(double));
  return result;
}
}  // namespace

TitaevSortirovkaBetcheraOMP::TitaevSortirovkaBetcheraOMP(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput().clear();
}

bool TitaevSortirovkaBetcheraOMP::ValidationImpl() {
  return !GetInput().empty();
}

bool TitaevSortirovkaBetcheraOMP::PreProcessingImpl() {
  GetOutput() = GetInput();
  return true;
}

void TitaevSortirovkaBetcheraOMP::ConvertToKeys(const InType &input, std::vector<uint64_t> &keys) {
  const size_t n = input.size();
#pragma omp parallel for
  for (long long i = 0; i < (long long)n; i++) {
    keys[i] = DoubleToOrderedUint(input[i]);
  }
}

void TitaevSortirovkaBetcheraOMP::RadixSortParallel(std::vector<uint64_t> &keys) {
  const size_t n = keys.size();
  if (n <= 1) {
    return;
  }

  constexpr int kBits = 8;
  constexpr int kBuckets = 1 << kBits;
  constexpr int kPasses = 64 / kBits;

  std::vector<uint64_t> tmp(n);

  for (int pass = 0; pass < kPasses; pass++) {
    std::vector<std::vector<size_t>> local_counts(omp_get_max_threads(), std::vector<size_t>(kBuckets, 0));

#pragma omp parallel
    {
      int tid = omp_get_thread_num();
#pragma omp for
      for (long long i = 0; i < (long long)n; i++) {
        size_t bucket = (keys[i] >> (pass * kBits)) & (kBuckets - 1);
        local_counts[tid][bucket]++;
      }
    }

    std::vector<size_t> common_counts(kBuckets, 0);
    for (int b = 0; b < kBuckets; b++) {
      for (int t = 0; t < omp_get_max_threads(); t++) {
        common_counts[b] += local_counts[t][b];
      }
    }

    std::vector<size_t> prefixes(kBuckets, 0);
    for (int b = 1; b < kBuckets; b++) {
      prefixes[b] = prefixes[b - 1] + common_counts[b - 1];
    }

    std::vector<std::vector<size_t>> thread_offsets(omp_get_max_threads(), std::vector<size_t>(kBuckets, 0));
    for (int b = 0; b < kBuckets; b++) {
      size_t current_offset = prefixes[b];
      for (int t = 0; t < omp_get_max_threads(); t++) {
        thread_offsets[t][b] = current_offset;
        current_offset += local_counts[t][b];
      }
    }

#pragma omp parallel
    {
      int tid = omp_get_thread_num();
#pragma omp for
      for (long long i = 0; i < (long long)n; i++) {
        size_t bucket = (keys[i] >> (pass * kBits)) & (kBuckets - 1);
        tmp[thread_offsets[tid][bucket]++] = keys[i];
      }
    }
    keys.swap(tmp);
  }
}

void TitaevSortirovkaBetcheraOMP::ConvertFromKeys(const std::vector<uint64_t> &keys, OutType &output) {
  const size_t n = keys.size();
  output.resize(n);
#pragma omp parallel for
  for (long long i = 0; i < (long long)n; i++) {
    output[i] = OrderedUintToDouble(keys[i]);
  }
}

void TitaevSortirovkaBetcheraOMP::BatcherStepParallel(OutType &result, size_t n, size_t step, size_t stage) {
#pragma omp parallel for
  for (long long i = 0; i < (long long)n; i++) {
    size_t j = (size_t)i ^ stage;
    if (j > (size_t)i && j < n) {
      const bool ascending = ((size_t)i & step) == 0;
      if (ascending) {
        if (result[i] > result[j]) {
          std::swap(result[i], result[j]);
        }
      } else {
        if (result[i] < result[j]) {
          std::swap(result[i], result[j]);
        }
      }
    }
  }
}

void TitaevSortirovkaBetcheraOMP::BatcherSortParallel() {
  auto &result = GetOutput();
  const size_t n = result.size();
  for (size_t step = 1; step < n; step <<= 1) {
    for (size_t stage = step; stage > 0; stage >>= 1) {
      BatcherStepParallel(result, n, step, stage);
    }
  }
}

bool TitaevSortirovkaBetcheraOMP::RunImpl() {
  auto &input = GetInput();
  const size_t n = input.size();
  if (n <= 1) {
    return true;
  }

  std::vector<uint64_t> keys(n);
  ConvertToKeys(input, keys);
  RadixSortParallel(keys);
  ConvertFromKeys(keys, GetOutput());

  if ((n & (n - 1)) == 0) {
    BatcherSortParallel();
  }
  return true;
}

bool TitaevSortirovkaBetcheraOMP::PostProcessingImpl() {
  return true;
}

}  // namespace titaev_m_sortirovka_betchera
