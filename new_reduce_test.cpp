
/* #define RAJA_ENABLE_TARGET_OPENMP 1 */
//#include <cuda_runtime.h>
#include <iostream>
#include <numeric>
#include <RAJA/RAJA.hpp>
#include <RAJA/util/Timer.hpp>
#include <camp/tuple.hpp>

namespace RAJA {
namespace reduce {
namespace detail {
// Custom ValueLoc struct for the purposes of testing, exists in the same namespace as ValueLoc.
template <typename T, typename IndexType>
class MaxLoc
{
public:
  T max_val = operators::limits<T>::min();
  IndexType loc = DefaultLoc<IndexType>().value();

#if __NVCC__ && defined(CUDART_VERSION) && CUDART_VERSION < 9020 || defined(__HIPCC__)
  RAJA_HOST_DEVICE constexpr MaxLoc() {}
  RAJA_HOST_DEVICE constexpr MaxLoc(MaxLoc const &other) : max_val{other.max_val}, loc{other.loc} {}
  RAJA_HOST_DEVICE
  MaxLoc &operator=(MaxLoc const &other) { max_val = other.max_val; loc = other.loc; return *this;}
#else
  constexpr MaxLoc() = default;
  constexpr MaxLoc(MaxLoc const &) = default;
  MaxLoc &operator=(MaxLoc const &) = default;
#endif

  RAJA_HOST_DEVICE constexpr MaxLoc(T const &max_val_) : max_val{max_val_}, loc{DefaultLoc<IndexType>().value()} {}
  RAJA_HOST_DEVICE constexpr MaxLoc(T const &max_val_, IndexType const &loc_)
      : max_val{max_val_}, loc{loc_}
  {
  }

  RAJA_HOST_DEVICE void max(T val, IndexType val_loc)
  {
    if (val > max_val)
    {
      max_val = val;
      loc = val_loc;
    }
  }

  RAJA_HOST_DEVICE operator T() const { return max_val; }
  RAJA_HOST_DEVICE IndexType getLoc() { return loc; }
  RAJA_HOST_DEVICE bool operator<(MaxLoc const &rhs) const
  {
    return max_val < rhs.max_val;
  }
  RAJA_HOST_DEVICE bool operator>(MaxLoc const &rhs) const
  {
    return max_val > rhs.max_val;
  }
};
} // namespace detail
} // namespace reduce
} // namespace RAJA


CAMP_SUPPRESS_HD_WARN
template <typename Fn,
          camp::idx_t... Sequence,
          typename TupleLike,
          typename... Ts>
CAMP_HOST_DEVICE
constexpr auto invoke_with_order(TupleLike &&t,
                                                  Fn &&f,
                                                  camp::idx_seq<Sequence...>,
                                                  Ts &&... extra)
{
  return f(extra..., camp::get<Sequence>(t)...);
}

CAMP_SUPPRESS_HD_WARN
template <typename TupleLike, typename Fn, typename... Ts>
CAMP_HOST_DEVICE
constexpr auto invoke(TupleLike &&t, Fn &&f, Ts &&... extra)
{
  return invoke_with_order(
      camp::forward<TupleLike>(t),
      camp::forward<Fn>(f),
      camp::make_idx_seq_t<camp::tuple_size<camp::decay<TupleLike>>::value>{},
      camp::forward<Ts>(extra)...);
}

template <typename Op, typename T>
struct Reducer {
  using op = Op;
  Reducer(T &target_in) : target(target_in), val(op::identity()) {}
  T &target;
  T val;
};

template <template <typename, typename, typename> class Op, typename T>
auto Reduce(T &target)
{
  // Getting the type of the MaxLoc struct to get the correct Op.
  using t = decltype(target.max_val);
  return Reducer<Op<t, t, t>, T>(target);
}

template <typename... Params, typename T, camp::idx_t... Idx>
void apply_combiner(T &l, T const &r, camp::idx_seq<Idx...>)
{
  camp::sink((camp::get<Idx>(l) = typename Params::op{}(camp::get<Idx>(l),
                                                        camp::get<Idx>(r)))...);
}
int use_dev = 0;

template <typename B, typename... Params>
void forall_param(int N, B const &body, Params... params)
{
  auto init_val = camp::make_tuple(params.val...);

    // TODO: Some type mismatching going on with this pragma.
#pragma omp declare reduction(                                                \
    combine                                                                   \
    : decltype(init_val)                                                      \
    : apply_combiner <Params...>(omp_out,                                     \
                                 omp_in,                                      \
                                 camp::make_idx_seq_t <sizeof...(Params)>{})) \
    initializer(omp_priv = camp::make_tuple(Params::op::identity()...))

#pragma omp target data if (use_dev) map(tofrom : init_val) map(to : body)
  {
    // TODO: Create custom reduction for a MaxLoc or ValueLoc kind of struct.
#pragma omp target teams distribute parallel for reduction(combine : init_val) if (use_dev)
    for (int i = 0; i < N; ++i) {
      invoke(init_val, body, i);
    }
  }
  camp::tie(params.target...) = init_val;
}


int main(int argc, char *argv[])
{
  if (argc < 3) {
    std::cout << "Execution Format: ./executable N use_dev\n";
    std::cout << "Example of usage: ./new_reduce 5000 0\n";
    exit(0);
  }
  use_dev = atoi(argv[2]);
  int N = atoi(argv[1]);
  //double r = 0;
  //double m = 5000;
  RAJA::reduce::detail::MaxLoc<double, int> ma;
  double *a = new double[N]();
  double *b = new double[N]();

  std::iota(a, a + N, 0);
  std::iota(b, b + N, 0);

#pragma omp target enter data map(to : a[:N], b[:N])
  RAJA::Timer t;
  t.reset();
  t.start();
  forall_param(N,
                [=](int i, RAJA::reduce::detail::MaxLoc<double, int> &ma) {
                  ma.max(a[i], i);
                },
                Reduce<RAJA::operators::maximum>(ma));
  t.stop();
  std::cout << "use_dev: " << use_dev << " time: " << t.elapsed()
            << std::endl;
  //std::cout << r << std::endl;
  //std::cout << m << " " << ma << std::endl;
  std::cout << ma.max_val << std::endl;
  /*r = 0;
  m = 5000;
  ma = 0;

  RAJA::ReduceSum<RAJA::omp_target_reduce, double> rr(0);
  RAJA::ReduceMin<RAJA::omp_target_reduce, double> rm(5000);
  RAJA::ReduceMax<RAJA::omp_target_reduce, double> rma(0);

  RAJA::Timer rt;
  rt.start();
  RAJA::forall<RAJA::omp_target_parallel_for_exec_nt>(RAJA::RangeSegment(0, N),
                                                      [=](int i) {
                                                        rr += a[i] * b[i];
                                                        rm.min(a[i]);
                                                        rma.max(a[i]);
                                                      });
  rt.stop();

  std::cout << rr.get() << " " << rt.elapsed() << std::endl;
  std::cout << rm.get() << " " << rma.get() << std::endl;

  std::cout << t.elapsed() << " " << rt.elapsed() << std::endl;

*/
  return 0;
}
