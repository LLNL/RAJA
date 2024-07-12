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

#include "RAJA/pattern/params/refloc_base.hpp"

namespace RAJA
{

namespace expt
{

template<typename T>
struct ValLoc {
  using index_type = RAJA::Index_type;
  using value_type = T;

  RAJA_HOST_DEVICE constexpr ValLoc() {}
  RAJA_HOST_DEVICE constexpr explicit ValLoc(value_type v) : val(v) {}
  RAJA_HOST_DEVICE constexpr ValLoc(value_type v, RAJA::Index_type l) : val(v), loc(l) {}

  RAJA_HOST_DEVICE constexpr bool operator<(const ValLoc& rhs) const { return val < rhs.val; }
  RAJA_HOST_DEVICE constexpr bool operator>(const ValLoc& rhs) const { return val > rhs.val; }

  RAJA_HOST_DEVICE constexpr value_type getVal() const {return val;}
  RAJA_HOST_DEVICE constexpr RAJA::Index_type getLoc() const {return loc;}

private:
  value_type val;
  index_type loc = -1;
};

template<typename T, template <typename, typename, typename> class Op>
struct ValOp {
  using value_type = T;
  using op_type = Op<T,T,T>;

  RAJA_HOST_DEVICE constexpr ValOp() {}
  RAJA_HOST_DEVICE constexpr explicit ValOp(value_type v) : val(v) {}

  template <typename U = op_type, std::enable_if_t<std::is_same<U, RAJA::operators::minimum<T,T,T>>::value> * = nullptr>
  RAJA_HOST_DEVICE constexpr ValOp & min(value_type v) { if (v < val) { val = v; } return *this; }
  template <typename U = op_type, std::enable_if_t<std::is_same<U, RAJA::operators::maximum<T,T,T>>::value> * = nullptr>
  RAJA_HOST_DEVICE constexpr ValOp & max(value_type v) { if (v > val) { val = v; } return *this; }

  template <typename U = op_type, std::enable_if_t<std::is_same<U, RAJA::operators::plus<T,T,T>>::value> * = nullptr>
  RAJA_HOST_DEVICE constexpr ValOp & operator+=(const value_type& rhs) { val += rhs; return *this; }

  template <typename U = op_type, std::enable_if_t<std::is_same<U, RAJA::operators::bit_and<T,T,T>>::value> * = nullptr>
  RAJA_HOST_DEVICE constexpr ValOp & operator&=(const value_type& rhs) { val &= rhs; return *this; }

  template <typename U = op_type, std::enable_if_t<std::is_same<U, RAJA::operators::bit_or<T,T,T>>::value> * = nullptr>
  RAJA_HOST_DEVICE constexpr ValOp & operator|=(const value_type& rhs) { val |= rhs; return *this; }

  template <typename U = op_type, std::enable_if_t<std::is_same<U, RAJA::operators::bit_and<T,T,T>>::value> * = nullptr>
  RAJA_HOST_DEVICE ValOp & operator&=(value_type& rhs) { val &= rhs; return *this; }

  template <typename U = op_type, std::enable_if_t<std::is_same<U, RAJA::operators::bit_or<T,T,T>>::value> * = nullptr>
  RAJA_HOST_DEVICE ValOp & operator|=(value_type& rhs) { val |= rhs; return *this; }

  RAJA_HOST_DEVICE constexpr bool operator<(const ValOp& rhs) const { val < rhs.val; return *this; }
  RAJA_HOST_DEVICE constexpr bool operator>(const ValOp& rhs) const { val > rhs.val; return *this; }

  RAJA_HOST_DEVICE constexpr value_type get() const {return val;}

//private:
  value_type val;
};

template<typename T, template <typename, typename, typename> class Op>
struct ValOp <ValLoc<T>, Op> {
  using index_type = RAJA::Index_type;
  using value_type = ValLoc<T>;
  using op_type = Op<value_type,value_type,value_type>;
  using valloc_value_type = typename value_type::value_type;
  using valloc_index_type = typename value_type::index_type;

  RAJA_HOST_DEVICE constexpr ValOp() {}
  RAJA_HOST_DEVICE constexpr explicit ValOp(value_type v) : val(v) {}
  RAJA_HOST_DEVICE constexpr explicit ValOp(valloc_value_type v) : val(v) {}
  RAJA_HOST_DEVICE constexpr ValOp(valloc_value_type v, valloc_index_type l) : val(v, l) {}

  template <typename U = op_type, std::enable_if_t<std::is_same<U, RAJA::operators::minimum<value_type,value_type,value_type>>::value> * = nullptr>
  RAJA_HOST_DEVICE constexpr ValOp & min(value_type v) { if (v < val) { val = v; } return *this; }
  template <typename U = op_type, std::enable_if_t<std::is_same<U, RAJA::operators::maximum<value_type,value_type,value_type>>::value> * = nullptr>
  RAJA_HOST_DEVICE constexpr ValOp & max(value_type v) { if (v > val) { val = v; } return *this; }

  template <typename U = op_type, std::enable_if_t<std::is_same<U, RAJA::operators::minimum<value_type,value_type,value_type>>::value> * = nullptr>
  RAJA_HOST_DEVICE constexpr ValOp & minloc(valloc_value_type v, valloc_index_type l) { return min(value_type(v,l)); }

  template <typename U = op_type, std::enable_if_t<std::is_same<U, RAJA::operators::maximum<value_type,value_type,value_type>>::value> * = nullptr>
  RAJA_HOST_DEVICE constexpr ValOp & maxloc(valloc_value_type v, valloc_index_type l) { return max(value_type(v,l)); }

  RAJA_HOST_DEVICE constexpr bool operator<(const ValOp& rhs) const { return val < rhs.val; }
  RAJA_HOST_DEVICE constexpr bool operator>(const ValOp& rhs) const { return val > rhs.val; }

  RAJA_HOST_DEVICE constexpr value_type get() const {return val;}
  RAJA_HOST_DEVICE constexpr valloc_value_type getVal() const {return val.getVal();}
  RAJA_HOST_DEVICE constexpr valloc_index_type getLoc() const {return val.getLoc();}

private:
  value_type val;
};

template<typename T, template <typename, typename, typename> class Op>
using ValLocOp = ValOp<ValLoc<T>, Op>;

} //  namespace expt

namespace operators
{

template <typename T, template <typename, typename, typename> class Op>
struct limits<RAJA::expt::ValOp<T, Op>, void> {
  RAJA_INLINE RAJA_HOST_DEVICE static constexpr RAJA::expt::ValOp<T, Op> min()
  {
    return RAJA::expt::ValOp<T, Op>(RAJA::operators::limits<T>::min());
  }

  RAJA_INLINE RAJA_HOST_DEVICE static constexpr RAJA::expt::ValOp<T, Op> max()
  {
    return RAJA::expt::ValOp<T, Op>(RAJA::operators::limits<T>::max());
  }
};


template <typename T>
struct limits<RAJA::expt::ValLoc<T>> {
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
  struct Reducer : public ForallParamBase {
    //using op = Op<T,T,T>;
    using op = Op;
    using value_type = T;
    //using value_op = ValOp<T, Op>;

    RAJA_HOST_DEVICE Reducer() {}
    Reducer(value_type *target_in) : target(target_in), val(op::identity()) {}
    //Reducer(value_type *target_in) : target(target_in), val(op::identity()), vop(value_op(*target_in)) {}
    //Reducer(value_op *target_in) : target(target_in->get()), val(op::identity()), vop(value_op(*target_in)) {}

    value_type *target = nullptr;
    value_type val = op::identity();
    //value_op *vop = nullptr;

#if defined(RAJA_CUDA_ACTIVE) || defined(RAJA_HIP_ACTIVE) || defined(RAJA_SYCL_ACTIVE)
    // Device related attributes.
    value_type * devicetarget = nullptr;
    RAJA::detail::SoAPtr<value_type, device_mem_pool_t> device_mem;
    unsigned int * device_count = nullptr;
#endif

    using ARG_TUP_T = camp::tuple<value_type*>;
    RAJA_HOST_DEVICE ARG_TUP_T get_lambda_arg_tup() { return camp::make_tuple(&val); }

    using ARG_LIST_T = typename ARG_TUP_T::TList;
    static constexpr size_t num_lambda_args = camp::tuple_size<ARG_TUP_T>::value ;

  };

} // namespace detail

template <template <typename, typename, typename> class Op, typename T>
auto constexpr Reduce(ValOp<ValLoc<T>,Op> *target)
{
  return detail::Reducer<Op<ValOp<ValLoc<T>,Op>, ValOp<ValLoc<T>,Op>, ValOp<ValLoc<T>,Op>>, ValOp<ValLoc<T>,Op>>(target);
}

template <template <typename, typename, typename> class Op, typename T>
auto constexpr Reduce(ValOp<T,Op> *target)
{
  return detail::Reducer<Op<ValOp<T,Op>, ValOp<T,Op>, ValOp<T,Op>>, ValOp<T,Op>>(target);
}

template <template <typename, typename, typename> class Op, typename T>
auto constexpr Reduce(T *target)
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
  struct ReducerLoc : public Reducer<Op, T> {
    using Base = Reducer<Op, T>;
    using value_type = typename Base::value_type;
    ReducerLoc(value_type *target_in) {
      Base::target = target_in;
      Base::val = value_type(Op::identity());
    }
  };

} // namespace detail

template <template <typename, typename, typename> class Op, typename T>
auto constexpr ReduceLoc(T *target)
{
  return detail::ReducerLoc<Op<T, T, T>, T>(target);
}
} // namespace expt


} //  namespace RAJA

#endif //  NEW_REDUCE_HPP
