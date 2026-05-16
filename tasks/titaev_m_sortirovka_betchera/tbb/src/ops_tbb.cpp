#include "titaev_m_sortirovka_betchera/tbb/include/ops_tbb.hpp"

#include <tbb/tbb.h>

#include <algorithm>
#include <cstdint>
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

void RadixPass(int pass, size_t n, const std::vector<uint64_t> &src, std::vector<uint64_t> &dest) {
  constexpr size_t kBuckets = 256;
  using LocalCounts = std::vector<size_t>;
  tbb::enumerable_thread_specific<LocalCounts> thread_counts(LocalCounts(kBuckets, 0));

  tbb::parallel_for(tbb::blocked_range<size_t>(0, n), [&](const tbb::blocked_range<size_t> &range) {
    auto &my_counts = thread_counts.local();
    for (size_t i = range.begin(); i < range.end(); ++i) {
      size_t bucket = (src[i] >> (static_cast<size_t>(pass) * 8)) & 255;
      my_counts[bucket]++;
    }
  });

  std::vector<size_t> common_counts(kBuckets, 0);
  for (const auto &lc : thread_counts) {
    for (size_t b_idx = 0; b_idx < kBuckets; ++b_idx) {
      common_counts[b_idx] += lc[b_idx];
    }
  }

  std::vector<size_t> prefixes(kBuckets, 0);
  for (size_t b_idx = 1; b_idx < kBuckets; ++b_idx) {
    prefixes[b_idx] = prefixes[b_idx - 1] + common_counts[b_idx - 1];
  }

  for (size_t b_idx = 0; b_idx < kBuckets; ++b_idx) {
    size_t offset = prefixes[b_idx];
    for (size_t i = 0; i < n; ++i) {
      if (((src[i] >> (static_cast<size_t>(pass) * 8)) & 255) == b_idx) {
        dest[offset++] = src[i];
      }
    }
  }
}
}  // namespace

TitaevSortirovkaBetcheraTBB::TitaevSortirovkaBetcheraTBB(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput().clear();
}

bool TitaevSortirovkaBetcheraTBB::ValidationImpl() {
  return !GetInput().empty();
}
bool TitaevSortirovkaBetcheraTBB::PreProcessingImpl() {
  GetOutput() = GetInput();
  return true;
}

void TitaevSortirovkaBetcheraTBB::ConvertToKeys(const InType &input, std::vector<uint64_t> &keys) {
  tbb::parallel_for(tbb::blocked_range<size_t>(0, input.size()), [&](const tbb::blocked_range<size_t> &r) {
    for (size_t i = r.begin(); i < r.end(); ++i) {
      keys[i] = DoubleToOrderedUint(input[i]);
    }
  });
}

void TitaevSortirovkaBetcheraTBB::RadixSortParallel(std::vector<uint64_t> &keys) {
  const size_t n = keys.size();
  if (n <= 1) {
    return;
  }
  std::vector<uint64_t> tmp(n);
  for (int p = 0; p < 8; ++p) {
    if (p % 2 == 0) {
      RadixPass(p, n, keys, tmp);
    } else {
      RadixPass(p, n, tmp, keys);
    }
  }
}

void TitaevSortirovkaBetcheraTBB::ConvertFromKeys(const std::vector<uint64_t> &keys, OutType &output) {
  output.resize(keys.size());
  tbb::parallel_for(tbb::blocked_range<size_t>(0, keys.size()), [&](const tbb::blocked_range<size_t> &r) {
    for (size_t i = r.begin(); i < r.end(); ++i) {
      output[i] = OrderedUintToDouble(keys[i]);
    }
  });
}

void TitaevSortirovkaBetcheraTBB::BatcherSortParallel() {
  auto &res = GetOutput();
  size_t n = res.size();
  for (size_t step = 1; step < n; step <<= 1) {
    for (size_t stage = step; stage > 0; stage >>= 1) {
      tbb::parallel_for(tbb::blocked_range<size_t>(0, n), [&](const tbb::blocked_range<size_t> &r) {
        for (size_t i = r.begin(); i < r.end(); ++i) {
          size_t j = i ^ stage;
          if (j > i && j < n) {
            bool asc = (i & step) == 0;
            if (asc ? (res[i] > res[j]) : (res[i] < res[j])) {
              std::swap(res[i], res[j]);
            }
          }
        }
      });
    }
  }
}

bool TitaevSortirovkaBetcheraTBB::RunImpl() {
  auto &input = GetInput();
  if (input.size() <= 1) {
    return true;
  }
  std::vector<uint64_t> keys(input.size());
  ConvertToKeys(input, keys);
  RadixSortParallel(keys);
  ConvertFromKeys(keys, GetOutput());
  if ((input.size() & (input.size() - 1)) == 0) {
    BatcherSortParallel();
  }
  return true;
}

bool TitaevSortirovkaBetcheraTBB::PostProcessingImpl() {
  return true;
}
}  // namespace titaev_m_sortirovka_betchera
