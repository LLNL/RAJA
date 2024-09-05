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

namespace expt
{

template <typename T>
struct ValLoc
{
  using index_type = RAJA::Index_type;
  using value_type = T;

  RAJA_HOST_DEVICE ValLoc() {}
  RAJA_HOST_DEVICE ValLoc(value_type v) : val(v) {}
  RAJA_HOST_DEVICE ValLoc(value_type v, RAJA::Index_type l) : val(v), loc(l) {}

  RAJA_HOST_DEVICE void min(value_type v, index_type l)
  {
    if (v < val)
    {
      val = v;
      loc = l;
    }
  }
  RAJA_HOST_DEVICE void max(value_type v, index_type l)
  {
    if (v > val)
    {
      val = v;
      loc = l;
    }
  }

  bool constexpr operator<(const ValLoc& rhs) const { return val < rhs.val; }
  bool constexpr operator>(const ValLoc& rhs) const { return val > rhs.val; }

  value_type getVal() { return val; }
  RAJA::Index_type getLoc() { return loc; }

private:
  value_type val;
  index_type loc = -1;
};

} //  namespace expt

namespace operators
{

template <typename T>
struct limits<RAJA::expt::ValLoc<T>>
{
  RAJA_INLINE RAJA_HOST_DEVICE static constexpr RAJA::expt::ValLoc<T> min()
  {
    return RAJA::expt::ValLoc<T>(RAJA::operators::limits<T>::min());
  }
  RAJA_INLINE RAJA_HOST_DEVICE static constexpr RAJA::expt::ValLoc<T> max()
  {
    return RAJA::expt::ValLoc<T>(RAJA::operators::limits<T>::max());
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
template <typename Op, typename T>
struct Reducer : public ForallParamBase
{
  using op = Op;
  using value_type = T;

  RAJA_HOST_DEVICE Reducer() {}
  Reducer(value_type* target_in) : target(target_in), val(op::identity()) {}

  value_type* target = nullptr;
  value_type val = op::identity();

#if defined(RAJA_CUDA_ACTIVE) || defined(RAJA_HIP_ACTIVE) ||                   \
    defined(RAJA_SYCL_ACTIVE)
  // Device related attributes.
  value_type* devicetarget = nullptr;
  RAJA::detail::SoAPtr<value_type, device_mem_pool_t> device_mem;
  unsigned int* device_count = nullptr;
#endif

  using ARG_TUP_T = camp::tuple<value_type*>;
  RAJA_HOST_DEVICE ARG_TUP_T get_lambda_arg_tup()
  {
    return camp::make_tuple(&val);
  }

  using ARG_LIST_T = typename ARG_TUP_T::TList;
  static constexpr size_t num_lambda_args = camp::tuple_size<ARG_TUP_T>::value;
};

} // namespace detail

template <template <typename, typename, typename> class Op, typename T>
auto constexpr Reduce(T* target)
{
  return detail::Reducer<Op<T, T, T>, T>(target);
}


namespace detail
{

//
//
// Basic ReducerLoc
//
//
template <typename Op, typename T>
struct ReducerLoc : public Reducer<Op, T>
{
  using Base = Reducer<Op, T>;
  using value_type = typename Base::value_type;
  ReducerLoc(value_type* target_in)
  {
    Base::target = target_in;
    Base::val = value_type(Op::identity());
  }
};

} // namespace detail

template <template <typename, typename, typename> class Op, typename T>
auto constexpr ReduceLoc(T* target)
{
  return detail::ReducerLoc<Op<T, T, T>, T>(target);
}
} // namespace expt


} //  namespace RAJA

#endif //  NEW_REDUCE_HPP
