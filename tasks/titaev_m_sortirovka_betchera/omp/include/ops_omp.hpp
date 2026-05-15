#pragma once

#include <algorithm>
#include <cstring>
#include <limits>
#include <vector>

#include "task/include/task.hpp"
#include "titaev_m_sortirovka_betchera/common/include/common.hpp"

namespace titaev_m_sortirovka_betchera {

class TitaevSortirovkaBetcheraOMP : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kOMP;
  }

  explicit TitaevSortirovkaBetcheraOMP(const InType &in);
  virtual ~TitaevSortirovkaBetcheraOMP();

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

  static uint64_t PackDouble(double v) noexcept;
  static double UnpackDouble(uint64_t k) noexcept;

  static void LSDRadixSort(std::vector<double> &array);
  static void BatcherOddEvenMerge(std::vector<double> &arr, size_t n);
  static void CompareSwap(std::vector<double> &arr, size_t i, size_t j);
};

inline TitaevSortirovkaBetcheraOMP::TitaevSortirovkaBetcheraOMP(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
}

inline TitaevSortirovkaBetcheraOMP::~TitaevSortirovkaBetcheraOMP() = default;

inline bool TitaevSortirovkaBetcheraOMP::ValidationImpl() {
  return true;
}

inline bool TitaevSortirovkaBetcheraOMP::PreProcessingImpl() {
  return true;
}

inline bool TitaevSortirovkaBetcheraOMP::PostProcessingImpl() {
  return true;
}

inline bool TitaevSortirovkaBetcheraOMP::RunImpl() {
  const auto &input = GetInput();
  if (input.empty()) {
    GetOutput() = std::vector<double>();
    return true;
  }
  size_t original_size = input.size();
  size_t pow2 = 1;
  while (pow2 < original_size) {
    pow2 <<= 1;
  }

  std::vector<double> data(pow2, std::numeric_limits<double>::max());
  for (size_t i = 0; i < original_size; ++i) {
    data[i] = input[i];
  }

  size_t half = pow2 / 2;
  if (half > 0) {
    double *d_ptr = data.data();
#pragma omp parallel sections shared(d_ptr, half, pow2)
    {
#pragma omp section
      {
        std::vector<double> l(half);
        for (size_t i = 0; i < half; ++i) {
          l[i] = d_ptr[i];
        }
        LSDRadixSort(l);
        for (size_t i = 0; i < half; ++i) {
          d_ptr[i] = l[i];
        }
      }
#pragma omp section
      {
        size_t rs = pow2 - half;
        std::vector<double> r(rs);
        for (size_t i = 0; i < rs; ++i) {
          r[i] = d_ptr[half + i];
        }
        LSDRadixSort(r);
        for (size_t i = 0; i < rs; ++i) {
          d_ptr[half + i] = r[i];
        }
      }
    }
    BatcherOddEvenMerge(data, pow2);
  } else {
    LSDRadixSort(data);
  }

  std::vector<double> result(original_size);
  for (size_t i = 0; i < original_size; ++i) {
    result[i] = data[i];
  }
  GetOutput() = std::move(result);
  return true;
}

}  // namespace titaev_m_sortirovka_betchera
