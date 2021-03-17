#ifndef NEW_REDUCE_HPP
#define NEW_REDUCE_HPP

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
  };

} //  namespace detail

#include "sequential/reduce.hpp"
#include "openmp/reduce.hpp"
#include "omp-target/reduce.hpp"

template <template <typename, typename, typename> class Op, typename T>
auto Reduce(T *target)
{
  return detail::Reducer<Op<T, T, T>, T>(target);
}

#endif //  NEW_REDUCE_HPP
