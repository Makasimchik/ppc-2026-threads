#include <gtest/gtest.h>

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

using TaskFactory = std::function<std::shared_ptr<ppc::task::Task<InType, OutType>>(InType)>;
using ParamType = std::tuple<TaskFactory, std::string, TestType>;

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

    std::random_device rd;
    std::mt19937_64 gen(rd());
    std::uniform_real_distribution<double> dist(-1000.0, 1000.0);

    input.resize(static_cast<size_t>(size));
    for (int i = 0; i < size; i++) {
      input[static_cast<size_t>(i)] = dist(gen);
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

static std::shared_ptr<ppc::task::Task<InType, OutType>> CreateSeqTask(InType in) {
  return std::make_shared<TitaevSortirovkaBetcheraSEQ>(in);
}

static std::shared_ptr<ppc::task::Task<InType, OutType>> CreateOmpTask(InType in) {
  return std::make_shared<TitaevSortirovkaBetcheraOMP>(in);
}

static const std::vector<ParamType> kSeqParams = {{CreateSeqTask, "seq_size_128", TestType{128, "size_128"}},
                                                  {CreateSeqTask, "seq_size_512", TestType{512, "size_512"}},
                                                  {CreateSeqTask, "seq_size_1024", TestType{1024, "size_1024"}}};

static const std::vector<ParamType> kOmpParams = {{CreateOmpTask, "omp_size_128", TestType{128, "size_128"}},
                                                  {CreateOmpTask, "omp_size_512", TestType{512, "size_512"}},
                                                  {CreateOmpTask, "omp_size_1024", TestType{1024, "size_1024"}}};

INSTANTIATE_TEST_SUITE_P(SequentialTests, TitaevBatcherRadixFuncTests, ::testing::ValuesIn(kSeqParams),
                         TitaevBatcherRadixFuncTests::PrintTestParam);

INSTANTIATE_TEST_SUITE_P(OpenMPTests, TitaevBatcherRadixFuncTests, ::testing::ValuesIn(kOmpParams),
                         TitaevBatcherRadixFuncTests::PrintTestParam);

}  // namespace titaev_m_sortirovka_betchera
