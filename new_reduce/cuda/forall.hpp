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
    //using RAJA::internal::thread_privatize;
    //auto privatizer = thread_privatize(loop_body);
    //auto& body = privatizer.get_priv();
    //auto ii = static_cast<IndexType>(getGlobalIdx_1D_1D());
    //if (ii < length) {
    //  loop_body(idx[ii]);
    //  //body(idx[ii]);
    //}
    Ts ii = static_cast<Ts>(RAJA::policy::cuda::impl::getGlobalIdx_1D_1D());
    if ( ii < extra )
    {
      invoke( t, loop_body, ii ); //, extra );
    }
    //loop_body(  extra,
    //            t
    //         );

    resolve<RAJA::cuda_exec<256>>(t);
    //grid_reduce<Combiner>(temp, identity, device, device_count);
    //grid_reduce<RAJA::operators::plus, double>(t/*, identity, device, device_count*/);
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

    //cudaStream_t stream;
    //cudaStreamCreate( &stream );

    //
    // Compute the number of blocks
    //
    //cuda_dim_t blockSize{256, 1, 1};
    //cuda_dim_t gridSize = RAJA::policy::cuda::impl::getGridDim(static_cast<cuda_dim_member_t>(N), 256);

    RAJA::cuda::detail::cudaInfo cudastuff;
    cudastuff.gridDim = RAJA::policy::cuda::impl::getGridDim(static_cast<cuda_dim_member_t>(N), 256);
    cudastuff.blockDim = cuda_dim_t{256, 1, 1};
    cudastuff.stream = 0;

    init<EXEC_POL>(f_params, cudastuff);

    //for (int i = 0; i < N; ++i) {
    //  invoke(f_params, body, i);
    //}

    //
    // Setup shared memory buffers
    //
    size_t shmem = 1000;

    //
    // Privatize the loop_body, using make_launch_body to setup reductions
    //
    //B cudabody = RAJA::cuda::make_launch_body(
    //    gridSize, blockSize, shmem, 0 /*stream*/, /*std::forward<B>*/(body));

    // call init on each Reducer object
    //auto objs = camp::make_tuple(params...);
    //camp::make_idx_seq_t<camp::tuple_size<camp::decay<TupleLike>>::value>{},
    //objs.initcuda()...;
    
    // set up memory on device
    //RAJA::detail::SoAPtr<T, RAJA::cuda::device_mempool_type> device_mem;
    //device_mem.allocate(1);
    //unsigned int * device_count = RAJA::cuda::device_zeroed_mempool_type::getInstance().template malloc<unsigned int>(1);

    //
    // Launch the kernels
    //
    size_t blocksz = 256;
    void *args[] = {(void*)&N,/*(void*)&init_val,*/ (void*)&body, /*(void*)&std::begin(init_val),*/ (void*)&f_params};
    //void *args[] = {(void*)&body, (void*)&begin, (void*)&len};
    RAJA::cuda::launch(
        (const void*)func,
        cudastuff.gridDim,  //gridSize,
        cudastuff.blockDim, //blockSize,
        args,
        shmem,
        cudastuff.stream   //stream
    );

  }

} //  namespace detail
#endif

#endif //  NEW_REDUCE_CUDA_HPP
