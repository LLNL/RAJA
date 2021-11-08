#ifndef NEW_REDUCE_FORALL_SYCL_HPP
#define NEW_REDUCE_FORALL_SYCL_HPP

#include <RAJA/RAJA.hpp>

#include "RAJA/policy/sycl/MemUtils_SYCL.hpp"
#include "RAJA/policy/sycl/policy.hpp"
#include "RAJA/policy/sycl/forall.hpp"

#if defined(RAJA_ENABLE_SYCL)
namespace detail {

  template <size_t BlockSize, bool Async, typename B, typename ParamPack>
  void forall_param(RAJA::sycl_exec_nontrivial<BlockSize, Async>&&, int N, B const &body, ParamPack f_params)
  {
    using EXEC_POL = RAJA::sycl_exec_nontrivial<BlockSize, Async>;

    // Compute the number of blocks
    sycl_dim_t blockSize{BlockSize};
    sycl_dim_t gridSize = ::RAJA::policy::sycl::impl::getGridDim(static_cast<size_t>(N), BlockSize);  // is N correct? or need segment length from f_params?

    init<EXEC_POL>(f_params);

    cl::sycl::queue * q = ::RAJA::sycl::detail::getQueue();

    // Global resource was not set, use the resource that was passed to forall
    // Determine if the default SYCL res is being used
    if (!q) { 
      q = red.sycl_res.get_queue(); // is red correct?
    }

    q->submit([&](cl::sycl::handler& h) {

      h.parallel_for( cl::sycl::nd_range<1>{gridSize, blockSize},
                      [=]  (cl::sycl::nd_item<1> it) {

        IndexType ii = it.get_global_id(0);
        if (ii < N) {
          body(begin[ii]);
        }
        combine<EXEC_POL>(f_params);
      });
    });

    if (!Async) { q->wait(); }

    resolve<EXEC_POL>(f_params);
  }

} //  namespace detail
#endif

#endif //  NEW_REDUCE_SYCL_HPP
