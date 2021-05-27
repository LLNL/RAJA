#ifndef NEW_REDUCE_OMP_TARGET_REDUCE_HPP
#define NEW_REDUCE_OMP_TARGET_REDUCE_HPP

namespace detail {

#if defined(RAJA_ENABLE_TARGET_OPENMP)
  // Init
  template<typename EXEC_POL, typename OP, typename T>
  camp::concepts::enable_if< std::is_same< EXEC_POL, RAJA::omp_target_parallel_for_exec_nt> >
  init(Reducer<OP, T>& red) {
    red.val = Reducer<OP,T>::op::identity();
  }
  // Combine
  template<typename EXEC_POL, typename OP, typename T>
  camp::concepts::enable_if< std::is_same< EXEC_POL, RAJA::omp_target_parallel_for_exec_nt> >
  combine(Reducer<OP, T>& out, const Reducer<OP, T>& in) {
    out.val = typename Reducer<OP,T>::op{}(out.val, in.val);
  }
  // Resolve
  template<typename EXEC_POL, typename OP, typename T>
  camp::concepts::enable_if< std::is_same< EXEC_POL, RAJA::omp_target_parallel_for_exec_nt> >
  resolve(Reducer<OP, T>& red) {
    *red.target = red.val;
  }
#endif

} //  namespace detail

#endif //  NEW_REDUCE_OMP_TARGET_REDUCE_HPP
