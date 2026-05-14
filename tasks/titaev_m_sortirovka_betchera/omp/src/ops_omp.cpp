#include "titaev_m_sortirovka_betchera/omp/include/ops_omp.hpp"

#include <omp.h>

#include <algorithm>
#include <cstring>

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

bool TitaevSortirovkaBetcheraOMP::ValidationImpl() {
  return task_data != nullptr && task_data->inputs_count[0] >= 0 &&
         task_data->outputs_count[0] == task_data->inputs_count[0];
}

bool TitaevSortirovkaBetcheraOMP::PreProcessingImpl() {
  size_t n = task_data->inputs_count[0];
  if (n > 0) {
    auto *in_ptr = reinterpret_cast<double *>(task_data->inputs[0]);
    GetInput().assign(in_ptr, in_ptr + n);
  }
  GetOutput().resize(n);
  return true;
}

void TitaevSortirovkaBetcheraOMP::ConvertToKeys(const InType &input, std::vector<uint64_t> &keys) {
  const size_t n = input.size();
#pragma omp parallel for
  for (int i = 0; i < (int)n; i++) {
    keys[i] = DoubleToOrderedUint(input[i]);
  }
}

void TitaevSortirovkaBetcheraOMP::RadixSort(std::vector<uint64_t> &keys) {
  const size_t n = keys.size();
  if (n <= 1) {
    return;
  }
  std::vector<uint64_t> tmp(n);
  for (int pass = 0; pass < 8; pass++) {
    size_t count[256] = {0};
    int shift = pass * 8;
    for (size_t i = 0; i < n; i++) {
      count[(keys[i] >> shift) & 0xFF]++;
    }
    for (int i = 1; i < 256; i++) {
      count[i] += count[i - 1];
    }
    for (int i = (int)n - 1; i >= 0; i--) {
      tmp[--count[(keys[i] >> shift) & 0xFF]] = keys[i];
    }
    keys.swap(tmp);
  }
}

void TitaevSortirovkaBetcheraOMP::ConvertFromKeys(const std::vector<uint64_t> &keys, OutType &output) {
  const size_t n = keys.size();
#pragma omp parallel for
  for (int i = 0; i < (int)n; i++) {
    output[i] = OrderedUintToDouble(keys[i]);
  }
}

void TitaevSortirovkaBetcheraOMP::BatcherStep(OutType &result, size_t n, size_t step, size_t stage) {
#pragma omp parallel for
  for (int i = 0; i < (int)n; i++) {
    size_t j = i ^ stage;
    if (j > (size_t)i && j < n) {
      bool ascending = (i & step) == 0;
      if (ascending ? (result[i] > result[j]) : (result[i] < result[j])) {
        std::swap(result[i], result[j]);
      }
    }
  }
}

void TitaevSortirovkaBetcheraOMP::BatcherSort() {
  auto &result = GetOutput();
  const size_t n = result.size();
  for (size_t step = 1; step < n; step <<= 1) {
    for (size_t stage = step; stage > 0; stage >>= 1) {
      BatcherStep(result, n, step, stage);
    }
  }
}

bool TitaevSortirovkaBetcheraOMP::RunImpl() {
  size_t n = GetInput().size();
  if (n <= 1) {
    if (n == 1) {
      GetOutput() = GetInput();
    }
    return true;
  }
  std::vector<uint64_t> keys(n);
  ConvertToKeys(GetInput(), keys);
  RadixSort(keys);
  ConvertFromKeys(keys, GetOutput());
  if ((n & (n - 1)) == 0) {
    BatcherSort();
  }
  return true;
}

bool TitaevSortirovkaBetcheraOMP::PostProcessingImpl() {
  auto *out_ptr = reinterpret_cast<double *>(task_data->outputs[0]);
  std::copy(GetOutput().begin(), GetOutput().end(), out_ptr);
  return true;
}

}  // namespace titaev_m_sortirovka_betchera
