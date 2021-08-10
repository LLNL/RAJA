#ifndef NEW_REDUCE_FORALL_CUDA_HPP
#define NEW_REDUCE_FORALL_CUDA_HPP

#include <RAJA/RAJA.hpp>

#include "RAJA/util/SoAArray.hpp"
#include "RAJA/util/SoAPtr.hpp"
#include "RAJA/util/basic_mempool.hpp"

#include "RAJA/policy/cuda/MemUtils_CUDA.hpp"
#include "RAJA/policy/cuda/policy.hpp"
#include "RAJA/policy/cuda/forall.hpp"

#if defined(RAJA_ENABLE_CUDA)
namespace detail {

using cuda_dim_t = dim3;
using cuda_dim_member_t = camp::decay<decltype(std::declval<cuda_dim_t>().x)>;

  template <size_t BlockSize,
            typename Ts,
            typename LOOP_BODY,
            typename Fps>
  __launch_bounds__(BlockSize, 1) __global__
      void forallp_cuda_kernel(
                              Ts extra,
                              LOOP_BODY loop_body,
                              Fps t)
  {
    Ts ii = static_cast<Ts>(RAJA::policy::cuda::impl::getGlobalIdx_1D_1D());
    if ( ii < extra )
    {
      cuda_invoke( t, loop_body, ii );
    }
    combine<RAJA::cuda_exec<256>>(t);
  }

  template <typename EXEC_POL, typename B, typename... Params>
  std::enable_if_t< std::is_same< EXEC_POL, RAJA::cuda_exec<256>>::value >
  forall_param(EXEC_POL&&, int N, B const &body, Params... params)
  {
    FORALL_PARAMS_T<Params...> f_params(params...);

    auto func = forallp_cuda_kernel<
      256 /*BlockSize*/,
      int,
      camp::decay<B>,
      camp::decay<decltype(f_params)>
      >;

    RAJA::cuda::detail::cudaInfo cudastuff;
    cudastuff.gridDim = RAJA::policy::cuda::impl::getGridDim(static_cast<cuda_dim_member_t>(N), 256);
    cudastuff.blockDim = cuda_dim_t{256, 1, 1};
    cudastuff.stream = 0;

    init<EXEC_POL>(f_params, cudastuff);

    printf("----------\n");

    size_t shmem = 1000;

    //
    // Launch the kernels
    //
    //size_t blocksz = 256;
    void *args[] = {(void*)&N, (void*)&body, (void*)&f_params};
    RAJA::cuda::launch(
        (const void*)func,
        cudastuff.gridDim,  //gridSize,
        cudastuff.blockDim, //blockSize,
        args,
        shmem,
        cudastuff.stream   //stream
    );

    cudaDeviceSynchronize(); // TODO : remove, this is only here for printing degub info.
    printf("----------\n");

    resolve<RAJA::cuda_exec<256>>(f_params);
  }

} //  namespace detail
#endif

#endif //  NEW_REDUCE_CUDA_HPP
