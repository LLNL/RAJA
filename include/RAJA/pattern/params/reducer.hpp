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

  // Basic data type Reducer
  // T must be a basic data type
  // IndexType not used here
  // VType must be ValOp<T, Op>
  template <typename Op, typename T, typename VType>
  struct Reducer : public ForallParamBase {
    //using op = Op<T,T,T>;
    using value_type = T; // This is a basic data type

    Reducer() = default;

    // Basic data type constructor
    RAJA_HOST_DEVICE Reducer(value_type *target_in) : valop_m(VType{}), target(target_in){}

    Reducer(Reducer const &) = default;
    Reducer(Reducer &&) = default;
    Reducer& operator=(Reducer const &) = default;
    Reducer& operator=(Reducer &&) = default;

    // Internal ValOp object that is used within RAJA::forall/launch
    VType valop_m = VType{};

    // Points to the user specified result variable
    value_type *target = nullptr;

    // Used in the resolve() call to set target in each backend
    RAJA_HOST_DEVICE void setTarget(value_type in) { *target = in; }

    RAJA_HOST_DEVICE
    value_type &
    getVal() { return valop_m.val; }

#if defined(RAJA_CUDA_ACTIVE) || defined(RAJA_HIP_ACTIVE) || defined(RAJA_SYCL_ACTIVE)
    // Device related attributes.
    value_type * devicetarget = nullptr;
    RAJA::detail::SoAPtr<value_type, device_mem_pool_t> device_mem;
    unsigned int * device_count = nullptr;
#endif

    // These are types and parameters extracted from this struct, and given to the forall.
    using ARG_TUP_T = camp::tuple<VType*>;
    RAJA_HOST_DEVICE ARG_TUP_T get_lambda_arg_tup() { return camp::make_tuple(&valop_m); }

    using ARG_LIST_T = typename ARG_TUP_T::TList;
    static constexpr size_t num_lambda_args = camp::tuple_size<ARG_TUP_T>::value ;
  };

  // Partial specialization of Reducer for ValLoc
  // T must be a ValLoc
  // TTOp is the template template version of Op, required by ValOp
  // VType must be ValOp<ValLoc<>,Op>
  template <typename Op, typename T, template <typename, typename, typename> class TTOp>
  struct Reducer<Op, T, ValOp<ValLoc<typename T::value_type, typename T::index_type>, TTOp>> : public ForallParamBase {
    using target_value_type = typename T::value_type; // This should be a basic data type extracted from the T=ValLoc passed in
    using target_index_type = typename T::index_type;
    using value_type = T; // Note that this struct operates on ValLoc
    using op = TTOp<value_type,value_type,value_type>;
    using VType = ValOp<ValLoc<target_value_type,target_index_type>, TTOp>;

    Reducer() = default;

    // ValLoc constructor
    // Note that the passthru variables point to the val and loc within the user defined target ValLoc
    RAJA_HOST_DEVICE Reducer(value_type *target_in) : valop_m(VType{}), target(target_in), passthruval(&target_in->val), passthruindex(&target_in->loc) {}

    // Pass through constructor for ReduceLoc<>(data, index) case
    // The passthru variables point to vars defined by the user
    RAJA_HOST_DEVICE Reducer(target_value_type *data_in, target_index_type *index_in) : valop_m(VType(*data_in, *index_in)), target(&valop_m.val), passthruval(data_in), passthruindex(index_in) {}

    Reducer(Reducer const &) = default;
    Reducer(Reducer &&) = default;
    Reducer& operator=(Reducer const &) = default;
    Reducer& operator=(Reducer &&) = default;

    VType valop_m = VType{};

    value_type *target = nullptr;

    target_value_type *passthruval = nullptr;
    target_index_type *passthruindex = nullptr;

    // This single-argument setTarget() matches the basic data type Reducer::setTarget()'s
    // function signature to make the init/combine/resolve backend calls easier
    // to maintain, but operates on the passthru variables instead
    RAJA_HOST_DEVICE void setTarget(value_type in) { *passthruval = in.val; *passthruindex = in.loc; }

    RAJA_HOST_DEVICE
    value_type &
    getVal() { return valop_m.val; }

#if defined(RAJA_CUDA_ACTIVE) || defined(RAJA_HIP_ACTIVE) || defined(RAJA_SYCL_ACTIVE)
    // Device related attributes.
    value_type * devicetarget = nullptr;
    RAJA::detail::SoAPtr<value_type, device_mem_pool_t> device_mem;
    unsigned int * device_count = nullptr;
#endif

    // These are types and parameters extracted from this struct, and given to the forall.
    using ARG_TUP_T = camp::tuple<VType*>;
    RAJA_HOST_DEVICE ARG_TUP_T get_lambda_arg_tup() { return camp::make_tuple(&valop_m); }

    using ARG_LIST_T = typename ARG_TUP_T::TList;
    static constexpr size_t num_lambda_args = camp::tuple_size<ARG_TUP_T>::value ;
  };

} // namespace detail

// Standard use case.
template <template <typename, typename, typename> class Op, typename T, typename IndexType = RAJA::Index_type,
          std::enable_if_t<std::is_integral<T>::value || std::is_floating_point<T>::value> * = nullptr>
auto constexpr Reduce(T *target)
{
  return detail::Reducer<Op<T,T,T>, T, ValOp<T, Op>>(target);
}

// User-defined ValLoc case.
template <template <typename, typename, typename> class Op, typename T, typename IndexType>
auto constexpr Reduce(ValLoc<T, IndexType> *target)
{
  using VL = ValLoc<T,IndexType>;
  return detail::Reducer<Op<VL,VL,VL>, VL, ValOp<ValLoc<T, IndexType>, Op>>(target);
}

// Pass through use case where reduction value and location are separate, non-ValLoc types supplied by the user.
template <template <typename, typename, typename> class Op, typename T, typename IndexType,
          std::enable_if_t<std::is_integral<T>::value || std::is_floating_point<T>::value> * = nullptr>
auto constexpr ReduceLoc(T *target, IndexType *index)
{
  using VL = ValLoc<T,IndexType>;
  return detail::Reducer<Op<VL,VL,VL>, VL, ValOp<ValLoc<T, IndexType>, Op>>(target, index);
}

} // namespace expt


} //  namespace RAJA

#endif //  NEW_REDUCE_HPP
