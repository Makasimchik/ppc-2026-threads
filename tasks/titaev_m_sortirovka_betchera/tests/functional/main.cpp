#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <functional>
#include <memory>
#include <random>
#include <string>
#include <tuple>
#include <vector>

#include "task/include/task.hpp"
#include "titaev_m_sortirovka_betchera/common/include/common.hpp"
#include "titaev_m_sortirovka_betchera/omp/include/ops_omp.hpp"
#include "titaev_m_sortirovka_betchera/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"

namespace titaev_m_sortirovka_betchera {

GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(TitaevBatcherRadixFuncTests);

using ParamType =
    std::tuple<std::function<std::shared_ptr<ppc::task::Task<InType, OutType>>(const InType &)>, std::string, TestType>;

class TitaevBatcherRadixFuncTests : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const testing::TestParamInfo<ParamType> &info) {
    return std::get<1>(info.param);
    static std::string PrintTestParam(const testing::TestParamInfo<std::tuple<InType, OutType, std::string>> &info) {
      return std::get<2>(info.param);
    }

   protected:
    InType input;

    void SetUp() override {
      ParamType full_param = GetParam();
      TestType param = std::get<2>(full_param);

      const int size = std::get<0>(param);

      std::mt19937_64 gen((size * 17) + 3);
      std::uniform_real_distribution<double> dist(-5000.0, 5000.0);

      input.resize(size);
      for (int i = 0; i < size; i++) {
        input[i] = dist(gen);
      }
      TestType param = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());
      input_data_ = std::get<0>(param);
      expected_data_ = std::get<1>(param);
    }

    bool CheckTestOutputData(OutType & output) final {
      if (output.size() != input.size()) {
        bool CheckTestOutputData(OutType & output_data) final {
          if (output_data.size() != expected_data_.size()) {
            return false;
          }
          for (size_t i = 1; i < output.size(); i++) {
            if (output[i] < output[i - 1]) {
              for (std::size_t i = 0; i < output_data.size(); ++i) {
                if (std::abs(output_data[i] - expected_data_[i]) > 1e-9) {
                  return false;
                }
              }
              return true;
            }

            InType GetTestInputData() final {
              return input;
              return input_data_;
            }

           private:
            InType input_data_;
            OutType expected_data_;
          };

          namespace {

          std::shared_ptr<ppc::task::Task<InType, OutType>> MakeSeqTask(const InType &in) {
            return std::make_shared<TitaevSortirovkaBetcheraSEQ>(in);
            TEST_P(TitaevBatcherRadixFuncTests, SortingTests) {
              ExecuteTest(GetParam());
            }

            const ParamType kParamSmall{MakeSeqTask, "titaev_m_sortirovka_betchera_seq_size_100",
                                        TestType{100, "size_100"}};
            const ParamType kParamMedium{MakeSeqTask, "titaev_m_sortirovka_betchera_seq_size_500",
                                         TestType{500, "size_500"}};
            const ParamType kParamLarge{MakeSeqTask, "titaev_m_sortirovka_betchera_seq_size_1000",
                                        TestType{1000, "size_1000"}};

INSTANTIATE_TEST_SUITE_P(FunctionalSortingTests, TitaevBatcherRadixFuncTests,
                         ::testing::Values(kParamSmall, kParamMedium, kParamLarge),
const std::array<TestType, 12> kTestParam = {
    TestType{InType{8.8}, OutType{8.8}, "Single"},
    TestType{InType{}, OutType{}, "Empty"},
    TestType{InType{6.6, 3.3}, OutType{3.3, 6.6}, "ReverseTwo"},
    TestType{InType{-0.2, -150.0, -60.5, -4.4, -9.9}, OutType{-150.0, -60.5, -9.9, -4.4, -0.2}, "Negative"},
    TestType{InType{15.7, 0.6, 98.2, 3.75, 7.83, 46.0}, OutType{0.6, 3.75, 7.83, 15.7, 46.0, 98.2}, "Positive"},
    TestType{InType{9e12, 1e3, 5e9, 7e15, 2e11}, OutType{1e3, 5e9, 2e11, 9e12, 7e15}, "Large"},
    TestType{InType{1e-20, 5e-6, 3e-12, 2e-3, 4e-9}, OutType{1e-20, 3e-12, 4e-9, 5e-6, 2e-3}, "Small"},
    TestType{InType{-8.0, -2.0, 0.5, 8.0, 9.0}, OutType{-8.0, -2.0, 0.5, 8.0, 9.0}, "Sorted"},
    TestType{InType{-3.3, 6.6, -10.9, 0.0, 2.2, -1.1}, OutType{-10.9, -3.3, -1.1, 0.0, 2.2, 6.6}, "DifferentSigns"},
    TestType{InType{7.7, 3.3, 7.7, 3.3, 7.7}, OutType{3.3, 3.3, 7.7, 7.7, 7.7}, "Duplicates"},
    TestType{InType{36.6, 25.5, 10.0, 8.9, 6.7, 4.5, 2.2}, OutType{2.2, 4.5, 6.7, 8.9, 10.0, 25.5, 36.6},
             "ReverseSeven"},
    TestType{InType{0.0, -0.0}, OutType{-0.0, 0.0}, "ZeroAndNegativeZero"}};

const auto kTestTasksList = std::tuple_cat(
    ppc::util::AddFuncTask<TitaevSortirovkaBetcheraSEQ, InType>(kTestParam, PPC_SETTINGS_titaev_m_sortirovka_betchera),
    ppc::util::AddFuncTask<TitaevSortirovkaBetcheraOMP, InType>(kTestParam, PPC_SETTINGS_titaev_m_sortirovka_betchera));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

INSTANTIATE_TEST_SUITE_P(FunctionalTests, TitaevBatcherRadixFuncTests, kGtestValues,
                         TitaevBatcherRadixFuncTests::PrintTestParam);

          }  // namespace

          }  // namespace titaev_m_sortirovka_betchera
