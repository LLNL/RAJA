#ifndef NEW_REDUCE_OMP_REDUCE_HPP
#define NEW_REDUCE_OMP_REDUCE_HPP

namespace detail {

#if defined(RAJA_ENABLE_OPENMP)
  // Init
  template<typename EXEC_POL, typename OP, typename T>
  std::enable_if_t< std::is_same< EXEC_POL, RAJA::omp_parallel_for_exec>::value , bool> // Returning bool because void param loop machine broken.
  init(Reducer<OP, T>& red) {
    red.val = Reducer<OP,T>::op::identity();
    return true;
  }
  // Combine
  template<typename EXEC_POL, typename OP, typename T>
  std::enable_if_t< std::is_same< EXEC_POL, RAJA::omp_parallel_for_exec>::value , bool> // Returning bool because void param loop machine broken.
  combine(Reducer<OP, T>& out, const Reducer<OP, T>& in) {
    out.val = typename Reducer<OP,T>::op{}(out.val, in.val);
    return true;
  }
  // Resolve
  template<typename EXEC_POL, typename OP, typename T>
  std::enable_if_t< std::is_same< EXEC_POL, RAJA::omp_parallel_for_exec>::value , bool> // Returning bool because void param loop machine broken.
  resolve(Reducer<OP, T>& red) {
    *red.target = red.val;
    return true;
  }
#endif

} //  namespace detail

#endif //  NEW_REDUCE_OMP_REDUCE_HPP
