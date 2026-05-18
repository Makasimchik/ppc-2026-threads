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
  static void RadixSortSequential(std::vector<uint64_t> &keys_vec);
  static void BatcherMergeParallel(OutType &arr_vec, size_t count_n);
  static void BatcherStepThreads(OutType &arr_vec, size_t count_n, size_t step_size, size_t stage_dist);
  static void BatcherTask(OutType &arr_vec, size_t start, size_t end, size_t step, size_t stage, size_t count_n);
  static void CompareAndSwap(OutType &arr_vec, size_t i_idx, size_t j_idx, bool is_ascending);
  static uint64_t PackDouble(double value);
  static double UnpackDouble(uint64_t bits);
};

}  // namespace titaev_m_sortirovka_betchera
