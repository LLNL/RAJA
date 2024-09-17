#ifndef RAJA_PARAMS_BASE
#define RAJA_PARAMS_BASE


namespace RAJA
{
namespace expt
{

  template<typename T, typename IndexType = RAJA::Index_type>
  struct ValLoc {
    public:
    using index_type = IndexType;
    using value_type = T;

    RAJA_HOST_DEVICE constexpr ValLoc() {}
    RAJA_HOST_DEVICE constexpr explicit ValLoc(value_type v) : val(v) {}
    RAJA_HOST_DEVICE constexpr ValLoc(value_type v, index_type l) : val(v), loc(l) {}

    RAJA_HOST_DEVICE constexpr bool operator<(const ValLoc& rhs) const { return val < rhs.val; }
    RAJA_HOST_DEVICE constexpr bool operator>(const ValLoc& rhs) const { return val > rhs.val; }

    RAJA_HOST_DEVICE constexpr value_type getVal() const {return val;}
    RAJA_HOST_DEVICE constexpr index_type getLoc() const {return loc;}

    RAJA_HOST_DEVICE void set(T inval, IndexType inindex) {val = inval; loc = inindex;}

    value_type val;
    index_type loc = -1;
  };

  template<typename T, template <typename, typename, typename> class Op>
  struct ValOp {
    public:
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

    value_type val = op_type::identity();
  };

  template<typename T, typename IndexType, template <typename, typename, typename> class Op>
  struct ValOp <ValLoc<T,IndexType>, Op> {
    public:
    using index_type = IndexType;
    using value_type = ValLoc<T,index_type>;
    using op_type = Op<value_type,value_type,value_type>;
    using valloc_value_type = typename value_type::value_type;
    using valloc_index_type = typename value_type::index_type;

    RAJA_HOST_DEVICE constexpr ValOp() {}
    RAJA_HOST_DEVICE constexpr explicit ValOp(value_type v) : val(v) {}
    RAJA_HOST_DEVICE constexpr explicit ValOp(valloc_value_type v) : val(v) {}
    RAJA_HOST_DEVICE constexpr ValOp(valloc_value_type v, valloc_index_type l) : val(v, l) {}

    // Normal min and max for loc reducers.
    template <typename U = op_type, std::enable_if_t<std::is_same<U, RAJA::operators::minimum<value_type,value_type,value_type>>::value> * = nullptr>
    RAJA_HOST_DEVICE constexpr T min(T v) { if (v < val.val) { val.val = v; return val.val; } return v; }

    template <typename U = op_type, std::enable_if_t<std::is_same<U, RAJA::operators::maximum<value_type,value_type,value_type>>::value> * = nullptr>
    RAJA_HOST_DEVICE constexpr T max(T v) { if (v > val.val) { val.val = v; return val.val; } return v; }

    template <typename U = op_type, std::enable_if_t<std::is_same<U, RAJA::operators::minimum<value_type,value_type,value_type>>::value> * = nullptr>
    RAJA_HOST_DEVICE constexpr ValOp & minloc(valloc_value_type v, valloc_index_type l) { return min(value_type(v,l)); }

    template <typename U = op_type, std::enable_if_t<std::is_same<U, RAJA::operators::maximum<value_type,value_type,value_type>>::value> * = nullptr>
    RAJA_HOST_DEVICE constexpr ValOp & maxloc(valloc_value_type v, valloc_index_type l) { return max(value_type(v,l)); }

    RAJA_HOST_DEVICE constexpr bool operator<(const ValOp& rhs) const { return val < rhs.val; }
    RAJA_HOST_DEVICE constexpr bool operator>(const ValOp& rhs) const { return val > rhs.val; }

    value_type val = op_type::identity();

    private:
    // These are called by maxloc and minloc, and should not be available to the user.
    template <typename U = op_type, std::enable_if_t<std::is_same<U, RAJA::operators::minimum<value_type,value_type,value_type>>::value> * = nullptr>
    RAJA_HOST_DEVICE constexpr ValOp & min(value_type v) { if (v < val) { val = v; } return *this; }

    template <typename U = op_type, std::enable_if_t<std::is_same<U, RAJA::operators::maximum<value_type,value_type,value_type>>::value> * = nullptr>
    RAJA_HOST_DEVICE constexpr ValOp & max(value_type v) { if (v > val) { val = v; } return *this; }
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
