#include "titaev_m_sortirovka_betchera/stl/include/ops_stl.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <functional>
#include <future>
#include <limits>
#include <thread>
#include <vector>

#include "titaev_m_sortirovka_betchera/common/include/common.hpp"

namespace titaev_m_sortirovka_betchera {

namespace {

void RadixSortChunk(std::vector<uint64_t> &data) {
  const size_t data_size = data.size();
  if (data_size <= 1) {
    return;
  }
  std::vector<uint64_t> buffer(data_size);
  for (int byte_idx = 0; byte_idx < 8; ++byte_idx) {
    std::vector<size_t> counts(256, 0);
    const size_t shift = static_cast<size_t>(byte_idx) * 8;
    for (const auto &val : data) {
      counts[(val >> shift) & 255U]++;
    }
    for (size_t i = 1; i < 256; ++i) {
      counts[i] += counts[i - 1];
    }
    for (size_t i = data_size; i > 0; --i) {
      buffer[--counts[(data[i - 1] >> shift) & 255U]] = data[i - 1];
    }
    data.swap(buffer);
  }
}

void MergeSortedChunks(std::vector<uint64_t> &data, size_t chunk_size, size_t total_size) {
  std::vector<uint64_t> buffer(total_size);
  size_t current_chunk = chunk_size;
  while (current_chunk < total_size) {
    for (size_t left = 0; left < total_size; left += current_chunk * 2) {
      size_t mid = std::min(left + current_chunk, total_size);
      size_t right = std::min(left + current_chunk * 2, total_size);
      std::merge(data.begin() + static_cast<std::ptrdiff_t>(left), data.begin() + static_cast<std::ptrdiff_t>(mid),
                 data.begin() + static_cast<std::ptrdiff_t>(mid), data.begin() + static_cast<std::ptrdiff_t>(right),
                 buffer.begin() + static_cast<std::ptrdiff_t>(left));
      std::copy(buffer.begin() + static_cast<std::ptrdiff_t>(left), buffer.begin() + static_cast<std::ptrdiff_t>(right),
                data.begin() + static_cast<std::ptrdiff_t>(left));
    }
    current_chunk *= 2;
  }
}

}  // namespace

uint64_t TitaevSortirovkaBetcheraSTL::DoubleToBits(double val) {
  uint64_t bits = 0;
  std::memcpy(&bits, &val, sizeof(double));
  const uint64_t mask = 1ULL << 63;
  return ((bits & mask) != 0ULL) ? ~bits : (bits ^ mask);
}

double TitaevSortirovkaBetcheraSTL::BitsToDouble(uint64_t bits) {
  const uint64_t mask = 1ULL << 63;
  const uint64_t orig_bits = ((bits & mask) != 0ULL) ? (bits ^ mask) : ~bits;
  double res = 0.0;
  std::memcpy(&res, &orig_bits, sizeof(double));
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
  RadixSortChunk(data);
}

void TitaevSortirovkaBetcheraSTL::BatcherMergeStep(OutType & /*output*/, size_t /*size_n*/, size_t /*step*/,
                                                   size_t /*stage*/) {}

void TitaevSortirovkaBetcheraSTL::ParallelBatcherMerge(OutType & /*output*/, size_t /*size_n*/) {}

bool TitaevSortirovkaBetcheraSTL::RunImpl() {
  const auto &input_vec = GetInput();
  const size_t original_count = input_vec.size();
  if (original_count == 0) {
    GetOutput().clear();
    return true;
  }

  const size_t num_threads = std::max(1U, std::thread::hardware_concurrency());
  const size_t chunk_size = (original_count + num_threads - 1) / num_threads;

  std::vector<uint64_t> keys(original_count);
  for (size_t i = 0; i < original_count; ++i) {
    keys[i] = DoubleToBits(input_vec[i]);
  }

  std::vector<std::future<void>> futures;
  futures.reserve(num_threads);

  for (size_t thread_idx = 0; thread_idx < num_threads; ++thread_idx) {
    const size_t begin = thread_idx * chunk_size;
    if (begin >= original_count) {
      break;
    }
    const size_t end = std::min(begin + chunk_size, original_count);
    futures.emplace_back(std::async(std::launch::async, [&keys, begin, end]() {
      std::vector<uint64_t> chunk(keys.begin() + static_cast<std::ptrdiff_t>(begin),
                                  keys.begin() + static_cast<std::ptrdiff_t>(end));
      RadixSortChunk(chunk);
      std::copy(chunk.begin(), chunk.end(), keys.begin() + static_cast<std::ptrdiff_t>(begin));
    }));
  }

  for (auto &fut : futures) {
    fut.get();
  }

  MergeSortedChunks(keys, chunk_size, original_count);

  auto &res_out = GetOutput();
  res_out.resize(original_count);
  for (size_t i = 0; i < original_count; ++i) {
    res_out[i] = BitsToDouble(keys[i]);
  }

  return true;
}

bool TitaevSortirovkaBetcheraSTL::PostProcessingImpl() {
  return true;
}

}  // namespace titaev_m_sortirovka_betchera
