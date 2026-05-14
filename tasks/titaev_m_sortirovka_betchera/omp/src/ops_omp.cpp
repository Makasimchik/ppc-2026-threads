#include "titaev_m_sortirovka_betchera/omp/include/ops_omp.hpp"

#include <omp.h>

#include <algorithm>
#include <cstring>
#include <limits>
#include <vector>

namespace titaev_m_sortirovka_betchera {

TitaevSortirovkaBetcheraOMP::TitaevSortirovkaBetcheraOMP(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
}

bool TitaevSortirovkaBetcheraOMP::ValidationImpl() {
  return !GetInput().empty();
}

bool TitaevSortirovkaBetcheraOMP::PreProcessingImpl() {
  GetOutput() = {};
  return true;
}

uint64_t TitaevSortirovkaBetcheraOMP::PackDouble(double v) {
  uint64_t bits;
  std::memcpy(&bits, &v, sizeof(bits));
  if ((bits & (1ULL << 63)) != 0ULL) {
    bits = ~bits;
  } else {
    bits ^= (1ULL << 63);
  }
  return bits;
}

double TitaevSortirovkaBetcheraOMP::UnpackDouble(uint64_t k) {
  if ((k & (1ULL << 63)) != 0ULL) {
    k ^= (1ULL << 63);
  } else {
    k = ~k;
  }
  double v;
  std::memcpy(&v, &k, sizeof(v));
  return v;
}

void TitaevSortirovkaBetcheraOMP::LSDRadixSort(std::vector<double> &array) {
  const size_t n = array.size();
  if (n <= 1) {
    return;
  }

  std::vector<uint64_t> keys(n);
  for (size_t i = 0; i < n; ++i) {
    keys[i] = PackDouble(array[i]);
  }

  std::vector<uint64_t> tmp_keys(n);
  std::vector<double> tmp_vals(n);

  for (int pass = 0; pass < 8; ++pass) {
    size_t cnt[256] = {0};
    int shift = pass * 8;
    for (size_t i = 0; i < n; ++i) {
      cnt[(keys[i] >> shift) & 255]++;
    }
    for (int i = 1; i < 256; ++i) {
      cnt[i] += cnt[i - 1];
    }
    for (int i = (int)n - 1; i >= 0; --i) {
      size_t pos = --cnt[(keys[i] >> shift) & 255];
      tmp_keys[pos] = keys[i];
      tmp_vals[pos] = array[i];
    }
    keys.swap(tmp_keys);
    array.swap(tmp_vals);
  }
  for (size_t i = 0; i < n; ++i) {
    array[i] = UnpackDouble(keys[i]);
  }
}

void TitaevSortirovkaBetcheraOMP::BatcherMerge(std::vector<double> &arr, size_t n) {
  for (size_t po = n / 2; po > 0; po >>= 1) {
#pragma omp parallel for shared(arr, n, po) default(none)
    for (size_t i = (po == n / 2 ? 0 : po); i < n - po; i += (po == n / 2 ? n : 2 * po)) {
      for (size_t j = 0; j < (po == n / 2 ? po : po); ++j) {
        if (arr[i + j] > arr[i + j + po]) {
          std::swap(arr[i + j], arr[i + j + po]);
        }
      }
    }
  }
}

bool TitaevSortirovkaBetcheraOMP::RunImpl() {
  auto data = GetInput();
  size_t original_size = data.size();
  if (original_size <= 1) {
    GetOutput() = data;
    return true;
  }

  size_t pow2 = 1;
  while (pow2 < original_size) {
    pow2 <<= 1;
  }
  data.resize(pow2, std::numeric_limits<double>::max());

  size_t half = pow2 / 2;
  std::vector<double> left(data.begin(), data.begin() + half);
  std::vector<double> right(data.begin() + half, data.end());

#pragma omp parallel sections shared(left, right)
  {
#pragma omp section
    LSDRadixSort(left);
#pragma omp section
    LSDRadixSort(right);
  }

  std::copy(left.begin(), left.end(), data.begin());
  std::copy(right.begin(), right.end(), data.begin() + half);

  BatcherMerge(data, pow2);

  data.resize(original_size);
  GetOutput() = data;
  return true;
}

bool TitaevSortirovkaBetcheraOMP::PostProcessingImpl() {
  return true;
}

}  // namespace titaev_m_sortirovka_betchera
