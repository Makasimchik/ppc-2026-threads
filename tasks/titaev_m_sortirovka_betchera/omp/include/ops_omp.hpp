#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include "task/include/task.hpp"
#include "titaev_m_sortirovka_betchera/common/include/common.hpp"

namespace titaev_m_sortirovka_betchera {

class TitaevSortirovkaBetcheraOMP : public ppc::core::Task {
 public:
  explicit TitaevSortirovkaBetcheraOMP(std::shared_ptr<ppc::core::TaskData> taskData) : Task(std::move(taskData)) {}
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  std::vector<double> input_;
  std::vector<double> output_;

  static void ConvertToKeys(const std::vector<double> &input, std::vector<uint64_t> &keys);
  static void RadixSort(std::vector<uint64_t> &keys);
  static void ConvertFromKeys(const std::vector<uint64_t> &keys, std::vector<double> &output);
  void BatcherSort();
  static void BatcherStep(std::vector<double> &result, size_t n, size_t step, size_t stage);
};

}  // namespace titaev_m_sortirovka_betchera
