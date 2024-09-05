#ifndef RAJA_PARAMS_BASE
#define RAJA_PARAMS_BASE


namespace RAJA
{
namespace expt
{
namespace detail
{

struct ForallParamBase
{

  // Some of this can be made virtual in c++20, for now must be defined in each
  // child class if any arguments to the forall lambda are needed (e.g.
  // KernelName is excluded.)
  using ARG_TUP_T  = camp::tuple<>;
  using ARG_LIST_T = typename ARG_TUP_T::TList;
  RAJA_HOST_DEVICE ARG_TUP_T get_lambda_arg_tup() { return camp::make_tuple(); }
  static constexpr size_t num_lambda_args = camp::tuple_size<ARG_TUP_T>::value;
};

} // namespace detail

} // namespace expt

} //  namespace RAJA

#endif //  RAJA_PARAMS_BASE
