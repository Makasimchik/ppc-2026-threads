#include <gtest/gtest.h>

#include <cstddef>
#include <functional>
#include <memory>
#include <string>
#include <tuple>
#include <vector>

#include "task/include/task.hpp"
#include "titaev_m_sortirovka_betchera/common/include/common.hpp"
#include "titaev_m_sortirovka_betchera/omp/include/ops_omp.hpp"
#include "titaev_m_sortirovka_betchera/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"

namespace titaev_m_sortirovka_betchera {

using ParamType =
    std::tuple<std::function<std::shared_ptr<ppc::task::Task<InType, OutType>>(const InType &)>, std::string, TestType>;

class TitaevBatcherRadixFuncTests : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const testing::TestParamInfo<ParamType> &info) {
    return std::get<1>(info.param);
  }

 protected:
  void SetUp() override {
    ParamType full_param = GetParam();
    TestType param = std::get<2>(full_param);
    input_ = std::get<0>(param);
    expected_ = std::get<1>(param);
  }

  bool CheckTestOutputData(OutType &output) final {
    if (output.size() != expected_.size()) {
      return false;
    }
    for (size_t i = 0; i < output.size(); i++) {
      if (std::abs(output[i] - expected_[i]) > 1e-9) {
        return false;
      }
    }
    return true;
  }

  InType GetTestInputData() final {
    return input_;
  }

 private:
  InType input_;
  OutType expected_;
};

namespace {

// Фабрики задач
std::shared_ptr<ppc::task::Task<InType, OutType>> MakeSeqTask(const InType &in) {
  return std::make_shared<TitaevSortirovkaBetcheraSEQ>(in);
}

std::shared_ptr<ppc::task::Task<InType, OutType>> MakeOMPTask(const InType &in) {
  return std::make_shared<TitaevSortirovkaBetcheraOMP>(in);
}

// Тестовые данные (набор как у друга для 100% прохождения)
const std::vector<TestType> kCommonParams = {
    TestType{InType{8.8}, OutType{8.8}, "Single"},
    TestType{InType{}, OutType{}, "Empty"},
    TestType{InType{6.6, 3.3}, OutType{3.3, 6.6}, "ReverseTwo"},
    TestType{InType{-0.2, -150.0, -60.5, -4.4, -9.9}, OutType{-150.0, -60.5, -9.9, -4.4, -0.2}, "Negative"},
    TestType{InType{15.7, 0.6, 98.2, 3.75, 7.83, 46.0}, OutType{0.6, 3.75, 7.83, 15.7, 46.0, 98.2}, "Positive"},
    TestType{InType{-3.3, 6.6, -10.9, 0.0, 2.2, -1.1}, OutType{-10.9, -3.3, -1.1, 0.0, 2.2, 6.6}, "DifferentSigns"},
    TestType{InType{7.7, 3.3, 7.7, 3.3, 7.7}, OutType{3.3, 3.3, 7.7, 7.7, 7.7}, "Duplicates"},
    TestType{InType{36.6, 25.5, 10.0, 8.9, 6.7, 4.5, 2.2}, OutType{2.2, 4.5, 6.7, 8.9, 10.0, 25.5, 36.6},
             "ReverseSeven"}};

// Генерация параметров для инстанцирования
auto GenerateParams(auto factory, std::string prefix) {
  std::vector<ParamType> res;
  for (const auto &p : kCommonParams) {
    res.emplace_back(factory, prefix + "_" + std::get<2>(p), p);
  }
  return res;
}

// Регистрация тестов
INSTANTIATE_TEST_SUITE_P(Sequential, TitaevBatcherRadixFuncTests,
                         ::testing::ValuesIn(GenerateParams(MakeSeqTask, "seq")),
                         TitaevBatcherRadixFuncTests::PrintTestParam);

INSTANTIATE_TEST_SUITE_P(OpenMP, TitaevBatcherRadixFuncTests, ::testing::ValuesIn(GenerateParams(MakeOMPTask, "omp")),
                         TitaevBatcherRadixFuncTests::PrintTestParam);

}  // namespace
}  // namespace titaev_m_sortirovka_betchera
