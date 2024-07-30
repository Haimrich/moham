#ifndef MOHAM_COMMON_H_
#define MOHAM_COMMON_H_

#ifdef _OPENMP
#include <omp.h>
#else
#define omp_get_thread_num() 0
#define omp_get_num_threads() 1
#define omp_get_max_threads() 1
#define omp_set_num_threads(X)
#endif

#include <vector>
#include <algorithm>

#include "mapping/loop.hpp"

namespace moham
{

  template <class Iter>
  class Iterange
  {
    Iter start_;
    Iter end_;

  public:
    Iterange(Iter start, Iter end) : start_(start), end_(end) {}

    Iter begin() { return start_; }
    Iter end() { return end_; }
  };


  using LoopRange = Iterange<std::vector<loop::Descriptor>::const_iterator>;

  enum class LayerType 
    {
        CONV,
        TRANSPOSED_CONV,
        DEPTHWISE_CONV,
        SEPARABLE_CONV,
        DENSE,
        MATMUL,
        OTHER
    };

  template <typename T, typename U>
  T GCD(T a, U b) 
  {
    if (b == 0) return a;
    return GCD(b, a % b);
  }
  
}

#endif // MOHAM_COMMON_H_