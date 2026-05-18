#include "titaev_m_sortirovka_betchera/stl/include/ops_stl.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <future>
#include <limits>
#include <vector>

#include "titaev_m_sortirovka_betchera/common/include/common.hpp"

namespace titaev_m_sortirovka_betchera {

uint64_t TitaevSortirovkaBetcheraSTL::DoubleToBits(double val) {
  uint64_t bits = 0;
  std::memcpy(&bits, &val, sizeof(double));
  uint64_t mask = 1ULL << 63;
  return ((bits & mask) != 0ULL) ? ~bits : (bits ^ mask);
}

double TitaevSortirovkaBetcheraSTL::BitsToDouble(uint64_t bits) {
  uint64_t mask = 1ULL << 63;
  uint64_t x_val = ((bits & mask) != 0ULL) ? (bits ^ mask) : ~bits;
  double res = 0.0;
  std::memcpy(&res, &x_val, sizeof(double));
  return res;
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

void TitaevSortirovkaBetcheraSTL::SerialRadixSort(std::vector<uint64_t> &data) {
  const size_t data_size = data.size();
  if (data_size <= 1) {
    return;
  }
  std::vector<uint64_t> buffer(data_size);
  for (int byte_idx = 0; byte_idx < 8; ++byte_idx) {
    std::vector<size_t> counts(256, 0);
    size_t shift = static_cast<size_t>(byte_idx) * 8;
    for (const auto &val : data) {
      counts[(val >> shift) & 255]++;
    }
    for (size_t i = 1; i < 256; ++i) {
      counts[i] += counts[i - 1];
    }
    for (size_t i = data_size; i > 0; --i) {
      buffer[--counts[(data[i - 1] >> shift) & 255]] = data[i - 1];
    }
    data.swap(buffer);
  }
}

void TitaevSortirovkaBetcheraSTL::BatcherMergeStep(OutType &output, size_t size_n, size_t step, size_t stage) {
  auto future_task = std::async(std::launch::async, [&output, step, stage, size_n]() {
    for (size_t i = 0; i < size_n / 2; ++i) {
      size_t j = i ^ stage;
      if (j > i && j < size_n) {
        bool asc = (i & step) == 0;
        if (asc ? (output[i] > output[j]) : (output[i] < output[j])) {
          std::swap(output[i], output[j]);
        }
      }
    }
  });
  for (size_t i = size_n / 2; i < size_n; ++i) {
    size_t j = i ^ stage;
    if (j > i && j < size_n) {
      bool asc = (i & step) == 0;
      if (asc ? (output[i] > output[j]) : (output[i] < output[j])) {
        std::swap(output[i], output[j]);
      }
    }
  }
  future_task.get();
}

void TitaevSortirovkaBetcheraSTL::ParallelBatcherMerge(OutType &output, size_t size_n) {
  for (size_t step = 1; step < size_n; step <<= 1) {
    for (size_t stage = step; stage > 0; stage >>= 1) {
      BatcherMergeStep(output, size_n, step, stage);
    }
  }
}

bool TitaevSortirovkaBetcheraSTL::RunImpl() {
  auto &input_vec = GetInput();
  size_t original_count = input_vec.size();
  size_t size_n = 1;
  while (size_n < original_count) {
    size_n <<= 1;
  }

  std::vector<uint64_t> keys(size_n, DoubleToBits(std::numeric_limits<double>::max()));
  for (size_t i = 0; i < original_count; ++i) {
    keys[i] = DoubleToBits(input_vec[i]);
  }

  size_t mid = size_n / 2;
  std::vector<uint64_t> left_part(keys.begin(), keys.begin() + static_cast<std::ptrdiff_t>(mid));
  std::vector<uint64_t> right_part(keys.begin() + static_cast<std::ptrdiff_t>(mid), keys.end());

  auto left_future = std::async(std::launch::async, [&left_part]() { SerialRadixSort(left_part); });
  SerialRadixSort(right_part);
  left_future.get();

  auto &res_out = GetOutput();
  res_out.resize(size_n);
  for (size_t i = 0; i < mid; ++i) {
    res_out[i] = BitsToDouble(left_part[i]);
    res_out[i + mid] = BitsToDouble(right_part[i]);
  }

  ParallelBatcherMerge(res_out, size_n);
  res_out.resize(original_count);
  return true;
}

bool TitaevSortirovkaBetcheraSTL::PostProcessingImpl() {
  return true;
}

}  // namespace titaev_m_sortirovka_betchera
