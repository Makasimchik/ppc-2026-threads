#include "titaev_m_sortirovka_betchera/stl/include/ops_stl.hpp"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <execution>
#include <numeric>
#include <vector>

#include "titaev_m_sortirovka_betchera/common/include/common.hpp"

namespace titaev_m_sortirovka_betchera {

namespace {
uint64_t DoubleToOrderedUint(double value) {
  uint64_t x_val = 0;
  std::memcpy(&x_val, &value, sizeof(double));
  constexpr uint64_t kSignMask = (1ULL << 63);
  if ((x_val & kSignMask) != 0ULL) {
    x_val = ~x_val;
  } else {
    x_val ^= kSignMask;
  }
  return x_val;
}

double OrderedUintToDouble(uint64_t x_val) {
  constexpr uint64_t kSignMask = (1ULL << 63);
  if ((x_val & kSignMask) != 0ULL) {
    x_val ^= kSignMask;
  } else {
    x_val = ~x_val;
  }
  double res_val = 0.0;
  std::memcpy(&res_val, &x_val, sizeof(double));
  return res_val;
}
}  // namespace

TitaevSortirovkaBetcheraSTL::TitaevSortirovkaBetcheraSTL(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput().clear();
}

bool TitaevSortirovkaBetcheraSTL::ValidationImpl() {
  return !GetInput().empty();
}

bool TitaevSortirovkaBetcheraSTL::PreProcessingImpl() {
  GetOutput() = GetInput();
  return true;
}

void TitaevSortirovkaBetcheraSTL::ConvertToKeys(const InType &input, std::vector<uint64_t> &keys) {
  const size_t n = input.size();
  std::vector<size_t> indices(n);
  std::iota(indices.begin(), indices.end(), 0);
  std::for_each(std::execution::par, indices.begin(), indices.end(),
                [&](size_t i) { keys[i] = DoubleToOrderedUint(input[i]); });
}

void TitaevSortirovkaBetcheraSTL::RadixSortParallel(std::vector<uint64_t> &keys) {
  if (keys.size() <= 1) {
    return;
  }
  std::sort(std::execution::par, keys.begin(), keys.end());
}

void TitaevSortirovkaBetcheraSTL::ConvertFromKeys(const std::vector<uint64_t> &keys, OutType &output) {
  const size_t n = keys.size();
  output.resize(n);
  std::vector<size_t> indices(n);
  std::iota(indices.begin(), indices.end(), 0);
  std::for_each(std::execution::par, indices.begin(), indices.end(),
                [&](size_t i) { output[i] = OrderedUintToDouble(keys[i]); });
}

void TitaevSortirovkaBetcheraSTL::BatcherSortParallel() {
  auto &res_vec = GetOutput();
  const size_t total_n = res_vec.size();

  for (size_t step_size = 1; step_size < total_n; step_size <<= 1) {
    for (size_t stage_dist = step_size; stage_dist > 0; stage_dist >>= 1) {
      std::vector<size_t> indices(total_n);
      std::iota(indices.begin(), indices.end(), 0);

      std::for_each(std::execution::par, indices.begin(), indices.end(), [&](size_t i) {
        size_t j_idx = i ^ stage_dist;
        if (j_idx > i && j_idx < total_n) {
          bool is_asc = (i & step_size) == 0;
          if (is_asc ? (res_vec[i] > res_vec[j_idx]) : (res_vec[i] < res_vec[j_idx])) {
            std::swap(res_vec[i], res_vec[j_idx]);
          }
        }
      });
    }
  }
}

bool TitaevSortirovkaBetcheraSTL::RunImpl() {
  auto &input_data = GetInput();
  const size_t data_size = input_data.size();
  if (data_size <= 1) {
    return true;
  }

  std::vector<uint64_t> keys_vec(data_size);
  ConvertToKeys(input_data, keys_vec);
  RadixSortParallel(keys_vec);
  ConvertFromKeys(keys_vec, GetOutput());

  if ((data_size & (data_size - 1)) == 0) {
    BatcherSortParallel();
  }
  return true;
}

bool TitaevSortirovkaBetcheraSTL::PostProcessingImpl() {
  return true;
}

}  // namespace titaev_m_sortirovka_betchera
