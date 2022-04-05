#ifndef NEW_REDUCE_HPP
#define NEW_REDUCE_HPP

#include "RAJA/pattern/params/params_base.hpp"
#include "RAJA/util/SoAPtr.hpp"

#if defined(RAJA_ENABLE_CUDA)
#define DEVICE cuda
#include "RAJA/policy/cuda/MemUtils_CUDA.hpp"
#elif defined(RAJA_ENABLE_HIP)
#define DEVICE hip
#include "RAJA/policy/hip/MemUtils_HIP.hpp"
#endif

namespace RAJA
{
namespace expt
{
namespace detail
{

  //
  //
  // Basic Reducer
  //
  //
  template <typename Op, typename T>
  struct Reducer : public ForallParamBase {
  //struct Reducer : public ForallParamBase {
    using op = Op;
    using val_type = T;
    RAJA_HOST_DEVICE Reducer() {}
    Reducer(T *target_in) : target(target_in), val(op::identity()) {}
    T *target = nullptr;
    T val = op::identity();

#if defined(RAJA_ENABLE_CUDA) || defined(RAJA_ENABLE_HIP)
    // Device related attributes.
    T * devicetarget = nullptr;
    RAJA::detail::SoAPtr<T, RAJA::DEVICE::device_mempool_type> device_mem;
    unsigned int * device_count = nullptr;
#endif

    static constexpr size_t num_lambda_args = 1;
    RAJA_HOST_DEVICE auto get_lambda_arg_tup() { return camp::make_tuple(&val); }
  };

} // namespace detail

template <template <typename, typename, typename> class Op, typename T>
auto constexpr Reduce(T *target)
{
  return detail::Reducer<Op<T, T, T>, T>(target);
}
} // namespace expt


} //  namespace RAJA

#endif //  NEW_REDUCE_HPP
