#include <gtest/gtest.h>

#include <algorithm>
#include <memory>
#include <vector>

#include "titaev_m_sortirovka_betchera/omp/include/ops_omp.hpp"
#include "titaev_m_sortirovka_betchera/seq/include/ops_seq.hpp"

namespace titaev_m_sortirovka_betchera {

static void RunOmpTest(int size) {
  std::vector<double> in(size);
  std::vector<double> out(size, 0.0);
  for (int i = 0; i < size; i++) {
    in[i] = size - i;
  }

  auto taskData = std::make_shared<ppc::core::TaskData>();
  taskData->inputs.push_back(reinterpret_cast<uint8_t *>(in.data()));
  taskData->inputs_count.push_back(in.size());
  taskData->outputs.push_back(reinterpret_cast<uint8_t *>(out.data()));
  taskData->outputs_count.push_back(out.size());

  TitaevSortirovkaBetcheraOMP task(taskData);
  ASSERT_TRUE(task.ValidationImpl());
  task.PreProcessingImpl();
  task.RunImpl();
  task.PostProcessingImpl();
  ASSERT_TRUE(std::is_sorted(out.begin(), out.end()));
}

TEST(titaev_m_sortirovka_betchera_omp, Test_Small) {
  RunOmpTest(100);
}
TEST(titaev_m_sortirovka_betchera_omp, Test_PowerOfTwo) {
  RunOmpTest(512);
}
TEST(titaev_m_sortirovka_betchera_omp, Test_Large) {
  RunOmpTest(1000);
}

}  // namespace titaev_m_sortirovka_betchera
