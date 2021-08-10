#ifndef NEW_REDUCE_HPP
#define NEW_REDUCE_HPP

#include "util/valloc.hpp"

namespace detail
{

  //
  //
  // Basic Reducer
  //
  //
  template <typename Op, typename T>
  struct Reducer {
    using op = Op;
    using val_type = T;
    Reducer() {}
    Reducer(T *target_in) : target(target_in), val(op::identity()) {}
    T *target = nullptr;
    T val = op::identity();

#if defined(RAJA_ENABLE_CUDA)
    // CUDA related attributes.
    T * cudatarget = nullptr;
    T cudaval = op::identity();
    RAJA::detail::SoAPtr<T, RAJA::cuda::device_mempool_type> device_mem;
    unsigned int * device_count = nullptr;
#endif

#if defined(RAJA_ENABLE_HIP)
    // HIP related attributes.
    T * hiptarget = nullptr;
    T hipval = op::identity;
    RAJA::detail::SoAPtr<T, RAJA::hip::device_mempool_type> device_mem;
    unsigned int * device_count = nullptr;
#endif
  };

} //  namespace detail

#include "sequential/reduce.hpp"
#include "openmp/reduce.hpp"
#include "omp-target/reduce.hpp"
#include "cuda/reduce.hpp"
#include "hip/reduce.hpp"

template <template <typename, typename, typename> class Op, typename T>
auto constexpr Reduce(T *target)
{
  return detail::Reducer<Op<T, T, T>, T>(target);
}

template <typename T>
auto constexpr ReduceLoc(ValLocMin<T> *target) 
{
  using R = ValLocMin<T>;
  return detail::Reducer<RAJA::operators::minimum<R,R,R>, R>(target);
}
template <typename T>
auto constexpr ReduceLoc(ValLocMax<T> *target)
{
  using R = ValLocMax<T>;
  return detail::Reducer<RAJA::operators::maximum<R,R,R>, R>(target);
}

#endif //  NEW_REDUCE_HPP
