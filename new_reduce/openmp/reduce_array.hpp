#ifndef NEW_REDUCE_ARRAY_OMP_REDUCE_HPP
#define NEW_REDUCE_ARRAY_OMP_REDUCE_HPP

#include "../util/policy.hpp"

namespace detail {

#if defined(RAJA_ENABLE_OPENMP)

  // Init
  template<typename EXEC_POL, typename OP, typename T>
  camp::concepts::enable_if< is_openmp_policy<EXEC_POL> >
  init(ReducerArray<OP, T>& red) {
    //printf("init ReducerArray %p\n", &red);
    //red.val_array = new T[red.size];
    //for (size_t i = 0; i < red.size; i++){
    //  red.val_array[i] = OP::identity();
    //}
  }

  // Combine
  template<typename EXEC_POL, typename OP, typename T>
  camp::concepts::enable_if< is_openmp_policy<EXEC_POL> >
  combine(ReducerArray<OP, T>& out, const ReducerArray<OP, T>& in) {
    //printf("Combine ReducerArray %p, %p\n", &out, &in);
    for (size_t i = 0; i < out.size; i++){
      out.val_array[i] = typename ReducerArray<OP,T>::op{}(out.val_array[i], in.val_array[i]);
    }
  }

  // Resolve
  template<typename EXEC_POL, typename OP, typename T>
  camp::concepts::enable_if< is_openmp_policy<EXEC_POL> >
  resolve(ReducerArray<OP, T>& red) {
    for (size_t i = 0; i < red.size; i++){
      red.target[i] = red.val_array[i];
    }
    //delete red.val_array;
    //free(red.val_array);
  }

#endif

} //  namespace detail

#endif //  NEW_REDUCE_ARRAY_OMP_REDUCE_HPP
