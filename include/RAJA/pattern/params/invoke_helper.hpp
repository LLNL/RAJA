namespace RAJA
{
namespace internal
{
template<camp::idx_t Idx, typename FP>
RAJA_HOST_DEVICE constexpr auto get_lambda_args(FP& fpp)
    -> decltype(*camp::get<Idx>(fpp.lambda_args()))
{
  return (*camp::get<Idx>(fpp.lambda_args()));
}
}  // namespace internal
}  // namespace RAJA