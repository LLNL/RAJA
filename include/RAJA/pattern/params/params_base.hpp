#ifndef RAJA_PARAMS_BASE
#define RAJA_PARAMS_BASE


namespace RAJA
{
namespace expt
{
namespace detail
{

  struct ForallParamBase {

    // This can be made virtual in c++20
    static constexpr size_t num_lambda_args = 0;
    RAJA_HOST_DEVICE auto get_lambda_arg_tup() { return camp::make_tuple(); }
  
  };

} // namespace detail

} // namespace expt

} //  namespace RAJA

#endif //  RAJA_PARAMS_BASE
