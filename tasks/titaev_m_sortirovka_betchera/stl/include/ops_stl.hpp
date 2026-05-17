#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "task/include/task.hpp"
#include "titaev_m_sortirovka_betchera/common/include/common.hpp"

namespace titaev_m_sortirovka_betchera {

class TitaevSortirovkaBetcheraSTL : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kSTL;
  }
  explicit TitaevSortirovkaBetcheraSTL(const InType &in);

  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  static void RadixSortSequential(std::vector<uint64_t> &keys);
  static void BatcherMergeParallel(OutType &arr, size_t count_n);
  static void BatcherStepThreads(OutType &arr, size_t count_n, size_t step, size_t stage);
  static void CompareAndSwap(OutType &arr, size_t i, size_t j, bool ascending);
};

}  // namespace titaev_m_sortirovka_betchera
