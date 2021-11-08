#ifndef NEW_REDUCE_SYCL_REDUCE_HPP
#define NEW_REDUCE_SYCL_REDUCE_HPP

namespace detail {

#if defined(RAJA_ENABLE_SYCL)
  // Init
  template<typename EXEC_POL, typename OP, typename T>
  camp::concepts::enable_if< std::is_same< EXEC_POL, RAJA::sycl_exec_nontrivial<256, false>> >
  init(Reducer<OP, T>& red) {
    red.sycl_res = new camp::resources::Resource{camp::resources::Sycl()};
    ::RAJA::sycl::detail::setQueue(red.sycl_res);
    red.val = Reducer<OP,T>::op::identity();
  }
  // Combine
  template<typename EXEC_POL, typename OP, typename T>
  camp::concepts::enable_if< std::is_same< EXEC_POL, RAJA::sycl_exec_nontrivial<256, false>> >
  combine(Reducer<OP, T>& out, const Reducer<OP, T>& in) {
    out.val = typename Reducer<OP,T>::op{}(out.val, in.val);
  }
  // Resolve
  template<typename EXEC_POL, typename OP, typename T>
  camp::concepts::enable_if< std::is_same< EXEC_POL, RAJA::sycl_exec_nontrivial<256, false>> >
  resolve(Reducer<OP, T>& red) {
    *red.target = red.val;
  }
#endif

} //  namespace detail

#endif //  NEW_REDUCE_SYCL_REDUCE_HPP
