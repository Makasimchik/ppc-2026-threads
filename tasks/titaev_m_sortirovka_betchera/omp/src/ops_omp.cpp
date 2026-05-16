#include "titaev_m_sortirovka_betchera/omp/include/ops_omp.hpp"

#include <omp.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <numeric>
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
#pragma omp parallel for default(none) shared(keys, input, n)
  for (int64_t i = 0; i < static_cast<int64_t>(n); i++) {
    keys[static_cast<size_t>(i)] = DoubleToOrderedUint(input[static_cast<size_t>(i)]);
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

  for (int pass = 0; pass < kPasses; ++pass) {  // ++pass
    std::vector<std::vector<size_t>> local_counts(omp_get_max_threads(), std::vector<size_t>(kBuckets, 0));

#pragma omp parallel default(none) shared(keys, local_counts, n, pass, kBits, kBuckets)
    {
      int tid = omp_get_thread_num();
#pragma omp for
      for (int64_t i = 0; i < static_cast<int64_t>(n); ++i) {
        size_t bucket = (keys[static_cast<size_t>(i)] >> (pass * kBits)) & (kBuckets - 1);
        local_counts[tid][bucket]++;
      }
    }

    std::vector<size_t> common_counts(kBuckets, 0);
    for (const auto &thread_local_count : local_counts) {
      for (size_t b_idx = 0; b_idx < kBuckets; ++b_idx) {
        common_counts[b_idx] += thread_local_count[b_idx];
      }
    }

    std::vector<size_t> prefixes(kBuckets, 0);
    for (size_t b_idx = 1; b_idx < kBuckets; ++b_idx) {
      prefixes[b_idx] = prefixes[b_idx - 1] + common_counts[b_idx - 1];
    }

    std::vector<std::vector<size_t>> thread_offsets(omp_get_max_threads(), std::vector<size_t>(kBuckets, 0));
    for (size_t b_idx = 0; b_idx < kBuckets; ++b_idx) {
      size_t current_offset = prefixes[b_idx];
      for (int t_idx = 0; t_idx < omp_get_max_threads(); ++t_idx) {
        thread_offsets[static_cast<size_t>(t_idx)][b_idx] = current_offset;
        current_offset += local_counts[static_cast<size_t>(t_idx)][b_idx];
      }
    }

#pragma omp parallel default(none) shared(keys, tmp, thread_offsets, n, pass, kBits, kBuckets)
    {
      int tid = omp_get_thread_num();
#pragma omp for
      for (int64_t i = 0; i < static_cast<int64_t>(n); ++i) {
        size_t bucket = (keys[static_cast<size_t>(i)] >> (pass * kBits)) & (kBuckets - 1);
        tmp[thread_offsets[static_cast<size_t>(tid)][bucket]++] = keys[static_cast<size_t>(i)];
      }
    }
    keys.swap(tmp);
  }
}

void TitaevSortirovkaBetcheraOMP::ConvertFromKeys(const std::vector<uint64_t> &keys, OutType &output) {
  const size_t n = keys.size();
  output.resize(n);
#pragma omp parallel for default(none) shared(keys, output, n)
  for (int64_t i = 0; i < static_cast<int64_t>(n); ++i) {
    output[static_cast<size_t>(i)] = OrderedUintToDouble(keys[static_cast<size_t>(i)]);
  }
}

void TitaevSortirovkaBetcheraOMP::BatcherStepParallel(OutType &result, size_t n, size_t step, size_t stage) {
#pragma omp parallel for default(none) shared(result, n, step, stage)
  for (int64_t i = 0; i < static_cast<int64_t>(n); ++i) {
    size_t j = static_cast<size_t>(i) ^ stage;
    if (j > static_cast<size_t>(i) && j < n) {
      const bool ascending = (static_cast<size_t>(i) & step) == 0;
      if (ascending ? (result[static_cast<size_t>(i)] > result[j]) : (result[static_cast<size_t>(i)] < result[j])) {
        std::swap(result[static_cast<size_t>(i)], result[j]);
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
