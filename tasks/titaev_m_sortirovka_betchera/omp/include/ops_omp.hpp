#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
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

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

  static void LSDRadixSort(std::vector<double> &array);
  static void BatcherMerge(std::vector<double> &arr, size_t n);
  static uint64_t PackDouble(double v);
  static double UnpackDouble(uint64_t k);
};

}  // namespace titaev_m_sortirovka_betchera
