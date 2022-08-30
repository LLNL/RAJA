#ifndef NEW_REDUCE_FORALL_CUDA_HPP
#define NEW_REDUCE_FORALL_CUDA_HPP

#include <RAJA/RAJA.hpp>

#if defined(RAJA_ENABLE_CUDA)

#include "RAJA/util/SoAArray.hpp"
#include "RAJA/util/SoAPtr.hpp"
#include "RAJA/util/basic_mempool.hpp"

#include "RAJA/policy/cuda/MemUtils_CUDA.hpp"
#include "RAJA/policy/cuda/policy.hpp"
#include "RAJA/policy/cuda/forall.hpp"

namespace detail {

using cuda_dim_t = dim3;
using cuda_dim_member_t = camp::decay<decltype(std::declval<cuda_dim_t>().x)>;

  template <typename EXEC_POL,
            size_t BlockSize,
            typename Ts,
            typename LOOP_BODY,
            typename Fps>
  __launch_bounds__(BlockSize, 1) __global__
      void forallp_cuda_kernel(
                              Ts extra,
                              LOOP_BODY loop_body,
                              Fps f_params)
  {
    Ts global_idx = static_cast<Ts>(RAJA::policy::cuda::impl::getGlobalIdx_1D_1D());
    if ( global_idx < extra )
    {
      invoke( f_params, loop_body, global_idx );
    }
    combine<EXEC_POL>(f_params);
  }

  template <size_t BlockSize, bool Async, typename B, typename ParamPack>
  void forall_param(RAJA::cuda_exec<BlockSize, Async>&&, int N, B const &body, ParamPack f_params)
  {
    using EXEC_POL = RAJA::cuda_exec<BlockSize, Async>;

    auto func = forallp_cuda_kernel<
      EXEC_POL,
      BlockSize,
      int,
      camp::decay<B>,
      camp::decay<decltype(f_params)>
      >;

    RAJA::cuda::detail::cudaInfo launch_info;
    launch_info.gridDim = RAJA::policy::cuda::impl::getGridDim(static_cast<cuda_dim_member_t>(N), BlockSize);
    launch_info.blockDim = cuda_dim_t{BlockSize, 1, 1};
    launch_info.res = RAJA::resources::Cuda::get_default();
    init<EXEC_POL>(f_params, launch_info);

    size_t shmem = 1000;

    //
    // Launch the kernels
    //
    void *args[] = {(void*)&N, (void*)&body, (void*)&f_params};
    RAJA::cuda::launch(
        (const void*)func,
        launch_info.gridDim,  //gridSize,
        launch_info.blockDim, //blockSize,
        args,
        shmem,
        launch_info.res
    );

    resolve<EXEC_POL>(f_params);
  }

} //  namespace detail
#endif

#endif //  NEW_REDUCE_CUDA_HPP
