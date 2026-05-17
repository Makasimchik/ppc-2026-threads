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

namespace {
uint64_t PackDouble(double value) {
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

double UnpackDouble(uint64_t bits) {
  constexpr uint64_t kSignMask = (1ULL << 63);
  uint64_t x_val = bits;
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

void TitaevSortirovkaBetcheraSTL::RadixSortSequential(std::vector<uint64_t> &keys) {
  const size_t count_n = keys.size();
  std::vector<uint64_t> tmp(count_n);
  for (int pass_idx = 0; pass_idx < 8; ++pass_idx) {
    std::vector<size_t> count_vec(256, 0);
    for (size_t i = 0; i < count_n; ++i) {
      size_t bucket = (keys[i] >> (static_cast<size_t>(pass_idx) * 8)) & 255;
      count_vec[bucket]++;
    }
    for (size_t i = 1; i < 256; ++i) {
      count_vec[i] += count_vec[i - 1];
    }
    for (size_t i = count_n; i > 0; --i) {
      size_t bucket = (keys[i - 1] >> (static_cast<size_t>(pass_idx) * 8)) & 255;
      tmp[--count_vec[bucket]] = keys[i - 1];
    }
    keys.swap(tmp);
  }
}

void TitaevSortirovkaBetcheraSTL::CompareAndSwap(OutType &arr, size_t i, size_t j, bool ascending) {
  if (ascending ? (arr[i] > arr[j]) : (arr[i] < arr[j])) {
    std::swap(arr[i], arr[j]);
  }
}

void TitaevSortirovkaBetcheraSTL::BatcherStepThreads(OutType &arr, size_t count_n, size_t step, size_t stage) {
  const size_t num_threads = std::thread::hardware_concurrency();
  std::vector<std::thread> threads;
  size_t chunk_size = (count_n + num_threads - 1) / num_threads;

  for (size_t t_idx = 0; t_idx < num_threads; ++t_idx) {
    size_t start = t_idx * chunk_size;
    size_t end = std::min(start + chunk_size, count_n);
    if (start >= end) {
      break;
    }

    threads.emplace_back([&arr, start, end, stage, step, count_n]() {
      for (size_t i = start; i < end; ++i) {
        size_t j = i ^ stage;
        if (j > i && j < count_n) {
          bool ascending = (i & step) == 0;
          CompareAndSwap(arr, i, j, ascending);
        }
      }
    });
  }
  for (auto &th : threads) {
    if (th.joinable()) {
      th.join();
    }
  }
}

void TitaevSortirovkaBetcheraSTL::BatcherMergeParallel(OutType &arr, size_t count_n) {
  for (size_t step = 1; step < count_n; step <<= 1) {
    for (size_t stage = step; stage > 0; stage >>= 1) {
      BatcherStepThreads(arr, count_n, step, stage);
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

  size_t half = count_n / 2;
  std::vector<uint64_t> left_keys(keys.begin(), keys.begin() + static_cast<std::ptrdiff_t>(half));
  std::vector<uint64_t> right_keys(keys.begin() + static_cast<std::ptrdiff_t>(half), keys.end());

  std::thread radix_thread([&]() { RadixSortSequential(left_keys); });
  RadixSortSequential(right_keys);
  radix_thread.join();

  auto &output = GetOutput();
  output.resize(count_n);
  for (size_t i = 0; i < half; ++i) {
    output[i] = UnpackDouble(left_keys[i]);
    output[i + half] = UnpackDouble(right_keys[i]);
  }

  BatcherMergeParallel(output, count_n);
  output.resize(original_size);
  return true;
}

bool TitaevSortirovkaBetcheraSTL::PostProcessingImpl() {
  return true;
}

}  // namespace titaev_m_sortirovka_betchera
