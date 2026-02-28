#include <gtest/gtest.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>
#include <functional>
#include <memory>

#include "titaev_m_sortirovka_betchera/common/include/common.hpp"
#include "titaev_m_sortirovka_betchera/seq/include/ops_seq.hpp"

#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"

namespace titaev_m_sortirovka_betchera {
GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(TitaevBatcherRadixFuncTests);

// ParamType, с которым реально работает BaseRunFuncTests
using ParamType = std::tuple<
    std::function<std::shared_ptr<ppc::task::Task<InType, OutType>>(InType)>,
    std::string,
    TestType>;

class TitaevBatcherRadixFuncTests
    : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  // gtest сюда передаёт TestParamInfo<ParamType>
  static std::string PrintTestParam(
      const testing::TestParamInfo<ParamType>& info) {
    return std::get<1>(info.param);  // имя теста
  }

 protected:
  InType input_;

  void SetUp() override {
    // GetParam() имеет тип ParamType
    ParamType full_param = GetParam();
    TestType param = std::get<2>(full_param);  // третий элемент — наш TestType

    const int size = std::get<0>(param);

    std::mt19937_64 gen(size * 17 + 3);
    std::uniform_real_distribution<double> dist(-5000.0, 5000.0);

    input_.resize(size);
    for (int i = 0; i < size; i++) {
      input_[i] = dist(gen);
    }
  }

  bool CheckTestOutputData(OutType& output) final {
    if (output.size() != input_.size())
      return false;
    for (size_t i = 1; i < output.size(); i++) {
      if (output[i] < output[i - 1])
        return false;
    }
    return true;
  }

  InType GetTestInputData() final {
    return input_;
  }
};

namespace {

// Фабрика задачи
std::shared_ptr<ppc::task::Task<InType, OutType>> MakeSeqTask(InType in) {
  return std::make_shared<TitaevSortirovkaBetcheraSEQ>(in);
}

// Набор параметров для gtest
const ParamType kParamSmall{
    MakeSeqTask,
    "titaev_m_sortirovka_betchera_seq_size_100",
    TestType{100, "size_100"}};

const ParamType kParamMedium{
    MakeSeqTask,
    "titaev_m_sortirovka_betchera_seq_size_500",
    TestType{500, "size_500"}};

const ParamType kParamLarge{
    MakeSeqTask,
    "titaev_m_sortirovka_betchera_seq_size_1000",
    TestType{1000, "size_1000"}};

INSTANTIATE_TEST_SUITE_P(
    FunctionalSortingTests,
    TitaevBatcherRadixFuncTests,
    ::testing::Values(
        kParamSmall,
        kParamMedium,
        kParamLarge),
    TitaevBatcherRadixFuncTests::PrintTestParam);

}  // namespace
}  // namespace titaev_m_sortirovka_betchera
