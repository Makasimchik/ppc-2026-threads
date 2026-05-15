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
  std::vector<uint64_t> tmp(n);
  for (int pass = 0; pass < 8; pass++) {
    std::vector<std::vector<size_t>> l_counts(omp_get_max_threads(), std::vector<size_t>(256, 0));
#pragma omp parallel
    {
      int tid = omp_get_thread_num();
#pragma omp for
      for (long long i = 0; i < (long long)n; i++) {
        l_counts[tid][(keys[i] >> (pass * 8)) & 255]++;
      }
    }
    std::vector<size_t> g_counts(256, 0);
    for (int b = 0; b < 256; b++) {
      for (int t = 0; t < (int)l_counts.size(); t++) {
        g_counts[b] += l_counts[t][b];
      }
    }
    std::vector<size_t> pref(256, 0);
    for (int b = 1; b < 256; b++) {
      pref[b] = pref[b - 1] + g_counts[b - 1];
    }
    std::vector<std::vector<size_t>> t_offs(omp_get_max_threads(), std::vector<size_t>(256));
    for (int b = 0; b < 256; b++) {
      size_t curr = pref[b];
      for (int t = 0; t < (int)t_offs.size(); t++) {
        t_offs[t][b] = curr;
        curr += l_counts[t][b];
      }
    }
#pragma omp parallel
    {
      int tid = omp_get_thread_num();
#pragma omp for
      for (long long i = 0; i < (long long)n; i++) {
        tmp[t_offs[tid][(keys[i] >> (pass * 8)) & 255]++] = keys[i];
      }
    }
    keys.swap(tmp);
  }
}

void TitaevSortirovkaBetcheraOMP::ConvertFromKeys(const std::vector<uint64_t> &keys, OutType &output) {
  output.resize(keys.size());
#pragma omp parallel for
  for (long long i = 0; i < (long long)keys.size(); i++) {
    output[i] = OrderedUintToDouble(keys[i]);
  }
}

void TitaevSortirovkaBetcheraOMP::BatcherStepParallel(OutType &res, size_t n, size_t step, size_t stage) {
#pragma omp parallel for
  for (long long i = 0; i < (long long)n; i++) {
    size_t j = (size_t)i ^ stage;
    if (j > (size_t)i && j < n) {
      bool asc = ((size_t)i & step) == 0;
      if (asc ? (res[i] > res[j]) : (res[i] < res[j])) {
        std::swap(res[i], res[j]);
      }
    }
  }
}

void TitaevSortirovkaBetcheraOMP::BatcherSortParallel() {
  auto &res = GetOutput();
  for (size_t step = 1; step < res.size(); step <<= 1) {
    for (size_t stage = step; stage > 0; stage >>= 1) {
      BatcherStepParallel(res, res.size(), step, stage);
    }
  }
}

bool TitaevSortirovkaBetcheraOMP::RunImpl() {
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

bool TitaevSortirovkaBetcheraOMP::PostProcessingImpl() {
  return true;
}

}  // namespace titaev_m_sortirovka_betchera
