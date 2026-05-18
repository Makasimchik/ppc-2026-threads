#pragma once

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
  static uint64_t DoubleToBits(double val);
  static double BitsToDouble(uint64_t bits);
  static void SerialRadixSort(std::vector<uint64_t> &data);
  void ParallelBatcherMerge(OutType &output, size_t size_n);
};

}  // namespace titaev_m_sortirovka_betchera
