#include <gtest/gtest.h>

#include <algorithm>
#include <cstddef>
#include <random>

#include "titaev_m_sortirovka_betchera/common/include/common.hpp"
#include "titaev_m_sortirovka_betchera/omp/include/ops_omp.hpp"
#include "titaev_m_sortirovka_betchera/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace titaev_m_sortirovka_betchera {

class TitaevBatcherRadixPerfTests : public ppc::util::BaseRunPerfTests<InType, OutType> {
 protected:
  static constexpr size_t kSize = 200000;
  InType input;

  void SetUp() override {
    std::mt19937 gen(42);  // Фиксированный seed для стабильности
    std::uniform_real_distribution<double> dist(-10000.0, 10000.0);
    input.resize(kSize);
    for (size_t i = 0; i < kSize; i++) {
      input[i] = dist(gen);
    }
  }

  bool CheckTestOutputData(OutType &output) final {
    if (output.size() != input.size()) {
      return false;
    }
    for (size_t i = 1; i < output.size(); i++) {
      if (output[i] < output[i - 1]) {
        return false;
      }
    }
    return true;
  }

  InType GetTestInputData() final {
    return input;
  }
};

TEST_P(TitaevBatcherRadixPerfTests, RunPerformanceModes) {
  ExecuteTest(GetParam());
}

namespace {

// Чтобы не падать при ошибках JSON, оборачиваем макрос в безопасную проверку
inline std::string GetSettings() {
  std::string s = PPC_SETTINGS_titaev_m_sortirovka_betchera;
  return s.empty() ? "{}" : s;
}

const auto kPerfTasks =
    ppc::util::MakeAllPerfTasks<InType, TitaevSortirovkaBetcheraSEQ, TitaevSortirovkaBetcheraOMP>(GetSettings());

INSTANTIATE_TEST_SUITE_P(Performance, TitaevBatcherRadixPerfTests, ppc::util::TupleToGTestValues(kPerfTasks),
                         TitaevBatcherRadixPerfTests::CustomPerfTestName);

}  // namespace
}  // namespace titaev_m_sortirovka_betchera
