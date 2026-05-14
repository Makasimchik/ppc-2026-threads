#include <gtest/gtest.h>

#include <algorithm>
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
  }

  bool CheckTestOutputData(OutType &output) final {
    if (output.size() != input.size()) {
      return false;
    }
    return std::is_sorted(output.begin(), output.end());
  }

  InType GetTestInputData() final {
    return input;
  }
};

namespace {

std::shared_ptr<ppc::task::Task<InType, OutType>> MakeSeqTask(const InType &in) {
  auto taskData = std::make_shared<ppc::core::TaskData>();
  taskData->inputs.push_back(const_cast<uint8_t *>(reinterpret_cast<const uint8_t *>(in.data())));
  taskData->inputs_count.push_back(in.size());

  // Создаем статический буфер для выхода, чтобы данные жили во время выполнения задачи
  static OutType res_seq;
  res_seq.resize(in.size());
  taskData->outputs.push_back(reinterpret_cast<uint8_t *>(res_seq.data()));
  taskData->outputs_count.push_back(res_seq.size());

  return std::make_shared<TitaevSortirovkaBetcheraSEQ>(taskData);
}

std::shared_ptr<ppc::task::Task<InType, OutType>> MakeOmpTask(const InType &in) {
  auto taskData = std::make_shared<ppc::core::TaskData>();
  taskData->inputs.push_back(const_cast<uint8_t *>(reinterpret_cast<const uint8_t *>(in.data())));
  taskData->inputs_count.push_back(in.size());

  static OutType res_omp;
  res_omp.resize(in.size());
  taskData->outputs.push_back(reinterpret_cast<uint8_t *>(res_omp.data()));
  taskData->outputs_count.push_back(res_omp.size());

  return std::make_shared<TitaevSortirovkaBetcheraOMP>(taskData);
}

const ParamType kParamSmall{MakeSeqTask, "seq_size_100", TestType{100, "size_100"}};
const ParamType kParamMedium{MakeSeqTask, "seq_size_500", TestType{500, "size_500"}};
const ParamType kParamLarge{MakeSeqTask, "seq_size_1000", TestType{1000, "size_1000"}};

const ParamType kParamSmallOMP{MakeOmpTask, "omp_size_100", TestType{100, "size_100"}};
const ParamType kParamMediumOMP{MakeOmpTask, "omp_size_500", TestType{500, "size_500"}};
const ParamType kParamLargeOMP{MakeOmpTask, "omp_size_1000", TestType{1000, "size_1000"}};

INSTANTIATE_TEST_SUITE_P(FunctionalSortingTests, TitaevBatcherRadixFuncTests,
                         ::testing::Values(kParamSmall, kParamMedium, kParamLarge, kParamSmallOMP, kParamMediumOMP,
                                           kParamLargeOMP),
                         TitaevBatcherRadixFuncTests::PrintTestParam);

}  // namespace
}  // namespace titaev_m_sortirovka_betchera
