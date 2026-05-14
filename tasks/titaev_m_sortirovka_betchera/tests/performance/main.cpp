#include <gtest/gtest.h>

#include <algorithm>
#include <cstddef>
#include <random>
#include <vector>

#include "titaev_m_sortirovka_betchera/common/include/common.hpp"
#include "titaev_m_sortirovka_betchera/omp/include/ops_omp.hpp"
#include "titaev_m_sortirovka_betchera/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace titaev_m_sortirovka_betchera {

class TitaevBatcherRadixPerfTests : public ppc::util::BaseRunPerfTests<InType, OutType> {
 protected:
  static constexpr size_t kSize = 200000;
  InType input;
  const int kCount_ = 1000000;
  InType input_data_;

  void SetUp() override {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dist(-10000.0, 10000.0);
    std::uniform_real_distribution<double> dist(-1e6, 1e6);

    input.resize(kSize);
    for (size_t i = 0; i < kSize; i++) {
      input[i] = dist(gen);
      input_data_.resize(kCount_);
      for (int i = 0; i < kCount_; ++i) {
        input_data_[i] = dist(gen);
      }
    }

    bool CheckTestOutputData(OutType & output) final {
      for (size_t i = 1; i < output.size(); i++) {
        if (output[i] < output[i - 1]) {
          return false;
        }
        bool CheckTestOutputData(OutType & output_data) final {
          if (output_data.size() != static_cast<std::size_t>(kCount_)) {
            return false;
          }
          return true;
          return std::is_sorted(output_data.begin(), output_data.end());
        }

        InType GetTestInputData() final {
          return input;
          return input_data_;
        }
      };

      TEST_P(TitaevBatcherRadixPerfTests, RunPerformanceModes) {
        TEST_P(TitaevBatcherRadixPerfTests, RunPerfModes) {
          ExecuteTest(GetParam());
        }

        namespace {

        const auto kPerfTasks =
            ppc::util::MakeAllPerfTasks<InType, TitaevSortirovkaBetcheraSEQ>(PPC_SETTINGS_titaev_m_sortirovka_betchera);
        const auto kAllPerfTasks =
            ppc::util::MakeAllPerfTasks<InType, TitaevSortirovkaBetcheraSEQ, TitaevSortirovkaBetcheraOMP>(
                PPC_SETTINGS_titaev_m_sortirovka_betchera);

        const auto kValues = ppc::util::TupleToGTestValues(kPerfTasks);
        const auto kNameGen = TitaevBatcherRadixPerfTests::CustomPerfTestName;
        const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

        INSTANTIATE_TEST_SUITE_P(PerformanceSortingTests, TitaevBatcherRadixPerfTests, kValues, kNameGen);
        const auto kPerfTestName = TitaevBatcherRadixPerfTests::CustomPerfTestName;

        INSTANTIATE_TEST_SUITE_P(PerformanceTests, TitaevBatcherRadixPerfTests, kGtestValues, kPerfTestName);

        }  // namespace

      }  // namespace titaev_m_sortirovka_betchera
