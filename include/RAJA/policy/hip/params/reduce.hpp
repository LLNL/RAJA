#ifndef NEW_REDUCE_HIP_REDUCE_HPP
#define NEW_REDUCE_HIP_REDUCE_HPP

#if defined(RAJA_ENABLE_HIP)

#include <hip/hip_runtime.h>
#include "RAJA/policy/hip/MemUtils_HIP.hpp"
#include "RAJA/policy/hip/reduce.hpp"
#include "RAJA/pattern/params/reducer.hpp"

namespace RAJA {
namespace expt {
namespace detail {

  // Init
  template<typename EXEC_POL, typename OP, typename T>
  camp::concepts::enable_if< type_traits::is_hip_policy<EXEC_POL> >
  init(Reducer<OP, T>& red, const RAJA::hip::detail::hipInfo & cs)
  {
    hipMalloc( (void**)(&(red.devicetarget)), sizeof(T));
    red.device_mem.allocate(cs.gridDim.x * cs.gridDim.y * cs.gridDim.z);
    red.device_count = RAJA::hip::device_zeroed_mempool_type::getInstance().template malloc<unsigned int>(1);
  }

  // Combine
  template<typename EXEC_POL, typename OP, typename T>
  RAJA_HOST_DEVICE
  camp::concepts::enable_if< type_traits::is_hip_policy<EXEC_POL> >
  combine(Reducer<OP, T>& red) {
    RAJA::hip::impl::expt::grid_reduce(red);
  }

  // Resolve
  template<typename EXEC_POL, typename OP, typename T>
  camp::concepts::enable_if< type_traits::is_hip_policy<EXEC_POL> >
  resolve(Reducer<OP, T>& red) {
    hipDeviceSynchronize();
    hipMemcpy(&red.val, red.devicetarget, sizeof(T), hipMemcpyDeviceToHost);
    *red.target = OP{}(red.val, *red.target);
  }

} //  namespace detail
} //  namespace expt
} //  namespace RAJA

#endif

#endif //  NEW_REDUCE_HIP_REDUCE_HPP
