#ifndef RAJA_PARAMS_BASE
#define RAJA_PARAMS_BASE


namespace RAJA
{
namespace expt
{

  template<typename T, typename IndexType = RAJA::Index_type>
  struct ValLoc {
    using index_type = IndexType;
    using value_type = T;

    RAJA_HOST_DEVICE constexpr ValLoc() {}
    RAJA_HOST_DEVICE constexpr explicit ValLoc(value_type v) : val(v) {}
    RAJA_HOST_DEVICE constexpr ValLoc(value_type v, index_type l) : val(v), loc(l) {}

    RAJA_HOST_DEVICE constexpr void min(value_type v, index_type l) { if (v < val) { val = v; loc = l; } }

    RAJA_HOST_DEVICE constexpr void max(value_type v, index_type l) { if (v > val) { val = v; loc = l; } }

    RAJA_HOST_DEVICE constexpr bool operator<(const ValLoc& rhs) const { return val < rhs.val; }
    RAJA_HOST_DEVICE constexpr bool operator>(const ValLoc& rhs) const { return val > rhs.val; }

    RAJA_HOST_DEVICE constexpr explicit operator T() const { return val; }

    RAJA_HOST_DEVICE constexpr value_type getVal() const {return val;}
    RAJA_HOST_DEVICE constexpr index_type getLoc() const {return loc;}

  //private:
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
    value_type val = op_type::identity();
  };

  template<typename T, typename IndexType, template <typename, typename, typename> class Op>
  struct ValOp <ValLoc<T,IndexType>, Op> {
    using index_type = IndexType;
    using value_type = ValLoc<T,index_type>;
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

  //private:
    value_type val = op_type::identity();
  };

  template<typename T, typename IndexType, template <typename, typename, typename> class Op>
  using ValLocOp = ValOp<ValLoc<T, IndexType>, Op>;

namespace detail
{

  struct ForallParamBase {

    // Some of this can be made virtual in c++20, for now must be defined in each child class
    // if any arguments to the forall lambda are needed (e.g. KernelName is excluded.)
    using ARG_TUP_T = camp::tuple<>; 
    using ARG_LIST_T = typename ARG_TUP_T::TList;
    RAJA_HOST_DEVICE ARG_TUP_T get_lambda_arg_tup() { return camp::make_tuple(); }
    static constexpr size_t num_lambda_args = camp::tuple_size<ARG_TUP_T>::value;
  
  };

} // namespace detail

} // namespace expt

} //  namespace RAJA

#endif //  RAJA_PARAMS_BASE
