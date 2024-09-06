#ifndef NEW_REDUCE_CUDA_REDUCE_HPP
#define NEW_REDUCE_CUDA_REDUCE_HPP

#if defined(RAJA_CUDA_ACTIVE)

#include <cuda.h>
#include "RAJA/policy/cuda/MemUtils_CUDA.hpp"
#include "RAJA/policy/cuda/reduce.hpp"
#include "RAJA/pattern/params/reducer.hpp"

#include "RAJA/policy/cuda/policy.hpp"

namespace RAJA {
namespace expt {
namespace detail {

  // Init
  template<typename EXEC_POL, template <typename, typename, typename> class OP, typename T>
  camp::concepts::enable_if< type_traits::is_cuda_policy<EXEC_POL> >
  init(Reducer<OP, T, ValOp<T,OP>>& red, RAJA::cuda::detail::cudaInfo& ci)
  {
    red.devicetarget = RAJA::cuda::pinned_mempool_type::getInstance().template malloc<T>(1);
    red.device_mem.allocate(ci.gridDim.x * ci.gridDim.y * ci.gridDim.z);
    red.device_count = RAJA::cuda::device_zeroed_mempool_type::getInstance().template malloc<unsigned int>(1);
  }

  // Combine
  template<typename EXEC_POL, template <typename, typename, typename> class OP, typename T>
  RAJA_HOST_DEVICE
  camp::concepts::enable_if< type_traits::is_cuda_policy<EXEC_POL> >
  combine(Reducer<OP, T, ValOp<T,OP>>& red)
  {
    RAJA::cuda::impl::expt::grid_reduce<typename EXEC_POL::IterationGetter, OP, T>(red.devicetarget,
                                                                            red.getVal(),
                                                                            red.device_mem,
                                                                            red.device_count);
  }

  // Resolve
  template<typename EXEC_POL, template <typename, typename, typename> class OP, typename T>
  camp::concepts::enable_if< type_traits::is_cuda_policy<EXEC_POL> >
  resolve(Reducer<OP, T, ValOp<T,OP>>& red, RAJA::cuda::detail::cudaInfo& ci)
  {
    // complete reduction
    ci.res.wait();

    red.target->val = OP<T,T,T>{}(red.target->val, *red.devicetarget);

    // free memory
    RAJA::cuda::device_zeroed_mempool_type::getInstance().free(red.device_count);
    red.device_count = nullptr;
    red.device_mem.deallocate();
    RAJA::cuda::pinned_mempool_type::getInstance().free(red.devicetarget);
    red.devicetarget = nullptr;
  }

//  // Init
//  template<typename EXEC_POL, template <typename, typename, typename> class OP, typename T>
//  camp::concepts::enable_if< type_traits::is_cuda_policy<EXEC_POL> >
//  init(Reducer<OP, T, T>& red, RAJA::cuda::detail::cudaInfo& ci)
//  {
//    red.devicetarget = RAJA::cuda::pinned_mempool_type::getInstance().template malloc<T>(1);
//    red.device_mem.allocate(ci.gridDim.x * ci.gridDim.y * ci.gridDim.z);
//    red.device_count = RAJA::cuda::device_zeroed_mempool_type::getInstance().template malloc<unsigned int>(1);
//  }
//
//  // Combine
//  template<typename EXEC_POL, template <typename, typename, typename> class OP, typename T>
//  RAJA_HOST_DEVICE
//  camp::concepts::enable_if< type_traits::is_cuda_policy<EXEC_POL> >
//  combine(Reducer<OP, T, T>& red)
//  //combine(Reducer<OP, T>& red)
//  {
//    RAJA::cuda::impl::expt::grid_reduce<typename EXEC_POL::IterationGetter>(red);
//    //RAJA::cuda::impl::expt::grid_reduce<typename EXEC_POL::IterationGetter>(red.devicetarget,
//    //                                                                        red.val,
//    //                                                                        red.device_mem,
//    //                                                                        red.device_count);
//  }
//
//  // Resolve
//  template<typename EXEC_POL, template <typename, typename, typename> class OP, typename T>
//  //template<typename EXEC_POL, typename OP, typename T>
//  camp::concepts::enable_if< type_traits::is_cuda_policy<EXEC_POL> >
//  resolve(Reducer<OP, T, T>& red, RAJA::cuda::detail::cudaInfo& ci)
//  //resolve(Reducer<OP, T>& red, RAJA::cuda::detail::cudaInfo& ci)
//  {
//    // complete reduction
//    ci.res.wait();
//    *red.target = OP<T,T,T>{}(*red.target, *red.devicetarget);
//    printf( "RCC DANGER\n" );
//
//    // free memory
//    RAJA::cuda::device_zeroed_mempool_type::getInstance().free(red.device_count);
//    red.device_count = nullptr;
//    red.device_mem.deallocate();
//    RAJA::cuda::pinned_mempool_type::getInstance().free(red.devicetarget);
//    red.devicetarget = nullptr;
//  }


} //  namespace detail
} //  namespace expt
} //  namespace RAJA

#endif

#endif //  NEW_REDUCE_CUDA_REDUCE_HPP
