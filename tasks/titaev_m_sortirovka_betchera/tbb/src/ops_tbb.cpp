#include "titaev_m_sortirovka_betchera/tbb/include/ops_tbb.hpp"

#include <tbb/tbb.h>

#include <atomic>
#include <numeric>
#include <util/include/util.hpp>
#include <vector>

#include "titaev_m_sortirovka_betchera/common/include/common.hpp"
#include "oneapi/tbb/parallel_for.h"

namespace titaev_m_sortirovka_betchera {

TitaevSortirovkaBetcheraTBB::TitaevSortirovkaBetcheraTBB(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0;
}

bool TitaevSortirovkaBetcheraTBB::ValidationImpl() {
  return (GetInput() > 0) && (GetOutput() == 0);
}

bool TitaevSortirovkaBetcheraTBB::PreProcessingImpl() {
  GetOutput() = 2 * GetInput();
  return GetOutput() > 0;
}

bool TitaevSortirovkaBetcheraTBB::RunImpl() {
  for (InType i = 0; i < GetInput(); i++) {
    for (InType j = 0; j < GetInput(); j++) {
      for (InType k = 0; k < GetInput(); k++) {
        std::vector<InType> tmp(i + j + k, 1);
        GetOutput() += std::accumulate(tmp.begin(), tmp.end(), 0);
        GetOutput() -= i + j + k;
      }
    }
  }

  const int num_threads = ppc::util::GetNumThreads();
  GetOutput() *= num_threads;

  std::atomic<int> counter(0);
  tbb::parallel_for(0, ppc::util::GetNumThreads(), [&](int /*i*/) { counter++; });

  GetOutput() /= counter;
  return GetOutput() > 0;
}

bool TitaevSortirovkaBetcheraTBB::PostProcessingImpl() {
  GetOutput() -= GetInput();
  return GetOutput() > 0;
}

}  // namespace titaev_m_sortirovka_betchera
