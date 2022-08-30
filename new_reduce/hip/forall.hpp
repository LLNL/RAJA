#ifndef NEW_REDUCE_FORALL_HIP_HPP
#define NEW_REDUCE_FORALL_HIP_HPP

#include <RAJA/RAJA.hpp>

#if defined(RAJA_ENABLE_HIP)

#include "RAJA/util/SoAArray.hpp"
#include "RAJA/util/SoAPtr.hpp"
#include "RAJA/util/basic_mempool.hpp"

#include "RAJA/policy/hip/MemUtils_HIP.hpp"
#include "RAJA/policy/hip/policy.hpp"
#include "RAJA/policy/hip/forall.hpp"

namespace detail {

using hip_dim_t = dim3;
using hip_dim_member_t = camp::decay<decltype(std::declval<hip_dim_t>().x)>;

  template <typename EXEC_POL,
            size_t BlockSize,
            typename Ts,
            typename LOOP_BODY,
            typename Fps>
  __launch_bounds__(BlockSize, 1) __global__
      void forallp_hip_kernel(
                              Ts extra,
                              LOOP_BODY loop_body,
                              Fps f_params)
  {
    Ts global_idx = static_cast<Ts>(RAJA::policy::hip::impl::getGlobalIdx_1D_1D());
    if ( global_idx < extra )
    {
      invoke( f_params, loop_body, global_idx );
    }
    combine<EXEC_POL>(f_params);
  }

  template <size_t BlockSize, bool Async, typename B, typename... Params>
  void forall_param(RAJA::hip_exec<BlockSize, Async>&&, int N, B const &body, Params... params)
  {
    using EXEC_POL = RAJA::hip_exec<BlockSize, Async>;
    ForallParamPack<Params...> f_params(params...);

    auto func = forallp_hip_kernel<
      EXEC_POL,
      BlockSize,
      int,
      camp::decay<B>,
      camp::decay<decltype(f_params)>
      >;

    RAJA::hip::detail::hipInfo launch_info;
    launch_info.gridDim = RAJA::policy::hip::impl::getGridDim(static_cast<hip_dim_member_t>(N), BlockSize);
    launch_info.blockDim = hip_dim_t{BlockSize, 1, 1};
    launch_info.res = RAJA::resources::Hip::get_default();
    init<EXEC_POL>(f_params, launch_info);

    size_t shmem = 1000;

    //
    // Launch the kernels
    //
    void *args[] = {(void*)&N, (void*)&body, (void*)&f_params};
    RAJA::hip::launch(
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

#endif //  NEW_REDUCE_HIP_HPP
