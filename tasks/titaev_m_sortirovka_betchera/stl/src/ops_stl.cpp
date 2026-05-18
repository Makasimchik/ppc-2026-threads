#include "titaev_m_sortirovka_betchera/stl/include/ops_stl.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <thread>
#include <vector>

#include "titaev_m_sortirovka_betchera/common/include/common.hpp"

namespace titaev_m_sortirovka_betchera {

uint64_t TitaevSortirovkaBetcheraSTL::PackDouble(double value) {
  uint64_t bits = 0;
  std::memcpy(&bits, &value, sizeof(double));
  constexpr uint64_t kSignMask = (1ULL << 63);
  if ((bits & kSignMask) != 0ULL) {
    bits = ~bits;
  } else {
    bits ^= kSignMask;
  }
  return bits;
}

double TitaevSortirovkaBetcheraSTL::UnpackDouble(uint64_t bits) {
  constexpr uint64_t kSignMask = (1ULL << 63);
  uint64_t x_val = bits;
  if ((x_val & kSignMask) != 0ULL) {
    x_val ^= kSignMask;
  } else {
    x_val = ~x_val;
  }
  double value = 0.0;
  std::memcpy(&value, &x_val, sizeof(double));
  return value;
}

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

void TitaevSortirovkaBetcheraSTL::RadixSortSequential(std::vector<uint64_t> &keys_vec) {
  const size_t count_n = keys_vec.size();
  if (count_n <= 1) {
    return;
  }
  std::vector<uint64_t> temp_vec(count_n);
  for (int pass_idx = 0; pass_idx < 8; ++pass_idx) {
    size_t count_buckets[256] = {0};
    size_t shift_val = static_cast<size_t>(pass_idx) * 8;
    for (size_t i = 0; i < count_n; ++i) {
      count_buckets[(keys_vec[i] >> shift_val) & 255]++;
    }
    for (size_t i = 1; i < 256; ++i) {
      count_buckets[i] += count_buckets[i - 1];
    }
    for (size_t i = count_n; i > 0; --i) {
      temp_vec[--count_buckets[(keys_vec[i - 1] >> shift_val) & 255]] = keys_vec[i - 1];
    }
    keys_vec.swap(temp_vec);
  }
}

void TitaevSortirovkaBetcheraSTL::CompareAndSwap(OutType &arr_vec, size_t i_idx, size_t j_idx, bool is_ascending) {
  if (is_ascending ? (arr_vec[i_idx] > arr_vec[j_idx]) : (arr_vec[i_idx] < arr_vec[j_idx])) {
    std::swap(arr_vec[i_idx], arr_vec[j_idx]);
  }
}

void TitaevSortirovkaBetcheraSTL::BatcherStepThreads(OutType &arr_vec, size_t count_n, size_t step_size,
                                                     size_t stage_dist) {
  if (count_n < 1000) {
    for (size_t i = 0; i < count_n; ++i) {
      size_t j_idx = i ^ stage_dist;
      if (j_idx > i && j_idx < count_n) {
        CompareAndSwap(arr_vec, i, j_idx, (i & step_size) == 0);
      }
    }
    return;
  }

  std::thread worker_thread([&]() {
    for (size_t i = 0; i < count_n / 2; ++i) {
      size_t j_idx = i ^ stage_dist;
      if (j_idx > i && j_idx < count_n) {
        CompareAndSwap(arr_vec, i, j_idx, (i & step_size) == 0);
      }
    }
  });
  for (size_t i = count_n / 2; i < count_n; ++i) {
    size_t j_idx = i ^ stage_dist;
    if (j_idx > i && j_idx < count_n) {
      CompareAndSwap(arr_vec, i, j_idx, (i & step_size) == 0);
    }
  }
  worker_thread.join();
}

void TitaevSortirovkaBetcheraSTL::BatcherMergeParallel(OutType &arr_vec, size_t count_n) {
  for (size_t step = 1; step < count_n; step <<= 1) {
    for (size_t stage = step; stage > 0; stage >>= 1) {
      BatcherStepThreads(arr_vec, count_n, step, stage);
    }
  }
}

bool TitaevSortirovkaBetcheraSTL::RunImpl() {
  auto &input_data = GetInput();
  size_t original_size = input_data.size();
  if (original_size <= 1) {
    return true;
  }

  size_t count_n = 1;
  while (count_n < original_size) {
    count_n <<= 1;
  }

  std::vector<uint64_t> keys(count_n, PackDouble(std::numeric_limits<double>::max()));
  for (size_t i = 0; i < original_size; ++i) {
    keys[i] = PackDouble(input_data[i]);
  }

  size_t half_n = count_n / 2;
  std::vector<uint64_t> left_keys(keys.begin(), keys.begin() + static_cast<std::ptrdiff_t>(half_n));
  std::vector<uint64_t> right_keys(keys.begin() + static_cast<std::ptrdiff_t>(half_n), keys.end());

  std::thread radix_thread([&]() { RadixSortSequential(left_keys); });
  RadixSortSequential(right_keys);
  radix_thread.join();

  auto &output_data = GetOutput();
  output_data.resize(count_n);
  for (size_t i = 0; i < half_n; ++i) {
    output_data[i] = UnpackDouble(left_keys[i]);
    output_data[i + half_n] = UnpackDouble(right_keys[i]);
  }

  BatcherMergeParallel(output_data, count_n);
  output_data.resize(original_size);
  return true;
}

bool TitaevSortirovkaBetcheraSTL::PostProcessingImpl() {
  return true;
}

}  // namespace titaev_m_sortirovka_betchera
