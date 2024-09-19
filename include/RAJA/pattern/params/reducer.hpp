#ifndef NEW_REDUCE_HPP
#define NEW_REDUCE_HPP

#include "RAJA/pattern/params/params_base.hpp"
#include "RAJA/util/SoAPtr.hpp"

#if defined(RAJA_CUDA_ACTIVE)
#include "RAJA/policy/cuda/MemUtils_CUDA.hpp"
#elif defined(RAJA_HIP_ACTIVE)
#include "RAJA/policy/hip/MemUtils_HIP.hpp"
#elif defined(RAJA_SYCL_ACTIVE)
#include "RAJA/policy/sycl/MemUtils_SYCL.hpp"
#endif

namespace RAJA
{

namespace operators
{

template <typename T, typename IndexType>
struct limits<RAJA::expt::ValLoc<T, IndexType>> {
  RAJA_INLINE RAJA_HOST_DEVICE static constexpr RAJA::expt::ValLoc<T, IndexType> min()
  {
    return RAJA::expt::ValLoc<T, IndexType>(RAJA::operators::limits<T>::min());
  }
  RAJA_INLINE RAJA_HOST_DEVICE static constexpr RAJA::expt::ValLoc<T, IndexType> max()
  {
    return RAJA::expt::ValLoc<T, IndexType>(RAJA::operators::limits<T>::max());
  }
};

} //  namespace operators

} //  namespace RAJA

namespace RAJA
{

namespace expt
{
namespace detail
{

#if defined(RAJA_CUDA_ACTIVE)
  using device_mem_pool_t = RAJA::cuda::device_mempool_type;
#elif defined(RAJA_HIP_ACTIVE)
  using device_mem_pool_t = RAJA::hip::device_mempool_type;
#elif defined(RAJA_SYCL_ACTIVE)
  using device_mem_pool_t = RAJA::sycl::device_mempool_type;
#endif

  //
  //
  // Basic Reducer
  //
  //
  template <template <typename, typename, typename> class Op, typename T, typename VType, typename DataType = T, typename IndexType = RAJA::Index_type, bool passthru = false>
  struct Reducer : public ForallParamBase {
    using op = Op<T,T,T>;
    using value_type = T;

    RAJA_HOST_DEVICE Reducer() = default;
    RAJA_HOST_DEVICE Reducer(value_type *target_in) : valop_m(VType{}), target(target_in){}
    RAJA_HOST_DEVICE Reducer(DataType *data_in, IndexType *index_in) : valop_m(VType(*data_in, *index_in)), target(&valop_m.val), passthruval(data_in), passthruindex(index_in) {}

    RAJA_HOST_DEVICE constexpr Reducer(Reducer const &) = default;
    RAJA_HOST_DEVICE constexpr Reducer(Reducer &&) = default;
    RAJA_HOST_DEVICE Reducer& operator=(Reducer const &) = default;
    RAJA_HOST_DEVICE Reducer& operator=(Reducer &&) = default;

    VType valop_m = VType{};
    value_type *target = nullptr;
    DataType *passthruval = nullptr;
    IndexType *passthruindex = nullptr; 

    template <typename U = VType, std::enable_if_t<std::is_same<U,ValOp<T,Op>>::value>* = nullptr>
    RAJA_HOST_DEVICE
    value_type &
    getVal() { return valop_m.val; }

    template <typename U = VType, std::enable_if_t<std::is_same<U,ValOp<ValLoc<T,IndexType>,Op>>::value>* = nullptr>
    RAJA_HOST_DEVICE
    value_type &
    getVal() { return valop_m.val.val; }

#if defined(RAJA_CUDA_ACTIVE) || defined(RAJA_HIP_ACTIVE) || defined(RAJA_SYCL_ACTIVE)
    // Device related attributes.
    value_type * devicetarget = nullptr;
    RAJA::detail::SoAPtr<value_type, device_mem_pool_t> device_mem;
    unsigned int * device_count = nullptr;
#endif

    using ARG_TUP_T = camp::tuple<VType*>;
    RAJA_HOST_DEVICE ARG_TUP_T get_lambda_arg_tup() { return camp::make_tuple(&valop_m); }

    using ARG_LIST_T = typename ARG_TUP_T::TList;
    static constexpr size_t num_lambda_args = camp::tuple_size<ARG_TUP_T>::value ;
  };

} // namespace detail

template <template <typename, typename, typename> class Op, typename T, typename VType = ValOp<T, Op>>
auto constexpr Reduce(T *target)
{
  return detail::Reducer<Op, T, VType>(target);
}

template <template <typename, typename, typename> class Op, typename T, typename IndexType = RAJA::Index_type, bool passingthru = true,
          std::enable_if_t<std::is_integral<T>::value || std::is_floating_point<T>::value> * = nullptr>
auto constexpr ReduceLoc(T *target, IndexType *index)
{
  return detail::Reducer<Op, ValLoc<T, IndexType>, ValLocOp<T, IndexType, Op>, T, IndexType, passingthru>(target, index);
}

} // namespace expt


} //  namespace RAJA

#endif //  NEW_REDUCE_HPP
