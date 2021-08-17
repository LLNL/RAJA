#ifndef NEW_REDUCE_FORALL_HIP_HPP
#define NEW_REDUCE_FORALL_HIP_HPP

#include <RAJA/RAJA.hpp>

#include "RAJA/util/SoAArray.hpp"
#include "RAJA/util/SoAPtr.hpp"
#include "RAJA/util/basic_mempool.hpp"

#include "RAJA/policy/hip/MemUtils_HIP.hpp"
#include "RAJA/policy/hip/policy.hpp"
#include "RAJA/policy/hip/forall.hpp"

#if defined(RAJA_ENABLE_HIP)
namespace detail {

using hip_dim_t = dim3;
using hip_dim_member_t = camp::decay<decltype(std::declval<hip_dim_t>().x)>;

  template <size_t BlockSize,
            typename Ts,
            typename LOOP_BODY,
            typename Fps>
  __launch_bounds__(BlockSize, 1) __global__
      void forallp_hip_kernel(
                              Ts extra,
                              LOOP_BODY loop_body,
                              Fps t)
  {
    Ts ii = static_cast<Ts>(RAJA::policy::hip::impl::getGlobalIdx_1D_1D());
    if ( ii < extra )
    {
      hip_invoke( t, loop_body, ii );
    }
    combine<RAJA::hip_exec<256>>(t);
  }

  template <typename EXEC_POL, typename B, typename... Params>
  std::enable_if_t< std::is_same< EXEC_POL, RAJA::hip_exec<256>>::value >
  forall_param(EXEC_POL&&, int N, B const &body, Params... params)
  {
    ForallParamPack<Params...> f_params(params...);

    auto func = forallp_hip_kernel<
      256 /*BlockSize*/,
      int,
      camp::decay<B>,
      camp::decay<decltype(f_params)>
      >;

    RAJA::hip::detail::hipInfo hipstuff;
    hipstuff.gridDim = RAJA::policy::hip::impl::getGridDim(static_cast<hip_dim_member_t>(N), 256);
    hipstuff.blockDim = hip_dim_t{256, 1, 1};
    hipstuff.stream = 0;

    init<EXEC_POL>(f_params, hipstuff);

    size_t shmem = 1000;

    //
    // Launch the kernels
    //
    //size_t blocksz = 256;
    void *args[] = {(void*)&N, (void*)&body, (void*)&f_params};
    RAJA::hip::launch(
        (const void*)func,
        hipstuff.gridDim,  //gridSize,
        hipstuff.blockDim, //blockSize,
        args,
        shmem,
        hipstuff.stream   //stream
    );

    resolve<RAJA::hip_exec<256>>(f_params);
  }

} //  namespace detail
#endif

#endif //  NEW_REDUCE_HIP_HPP
