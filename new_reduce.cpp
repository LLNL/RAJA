
/* #define RAJA_ENABLE_TARGET_OPENMP 1 */
//#include <cuda_runtime.h>
#include <iostream>
#include <numeric>
#include <RAJA/RAJA.hpp>
#include <RAJA/util/Timer.hpp>
#include <camp/tuple.hpp>


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
  return Reducer<Op<T, T, T>, T>(target);
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

#pragma omp declare reduction(                                                \
    combine                                                                   \
    : decltype(init_val)                                                      \
    : apply_combiner <Params...>(omp_out,                                     \
                                 omp_in,                                      \
                                 camp::make_idx_seq_t <sizeof...(Params)>{})) \
    initializer(omp_priv = camp::make_tuple(Params::op::identity()...))

  //printf("%p, %p\n", &init_val, &body);
#pragma omp target data if (use_dev) map(tofrom : init_val) map(to : body)
  {
    //printparams(params...);
    /* #pragma omp target */
    /*     printf("%p, %p\n", &init_val, &body); */
#pragma omp target if (use_dev) teams distribute parallel for reduction(combine : init_val)
    /* #pragma omp target teams distribute parallel for schedule(static, 1)
     * reduction(combine: identity) */
    for (int i = 0; i < N; ++i) {
      /* if (i==0) */
      //printf("%p, %p\n", &init_val, &body);
      invoke(init_val, body, i);
    }
  }
  camp::tie(params.target...) = init_val;
}

// template<typename ...Ts>
// void fmt(Ts...args) {
//   std::cout <<
// }

int main(int argc, char *argv[])
{
  if (argc < 3) {
    std::cout << "Execution Format: ./executable N use_dev\n";
    std::cout << "Example of usage: ./new_reduce 5000 0\n";
    exit(0);
  }
  use_dev = atoi(argv[2]);
  int N = atoi(argv[1]);  // 500000;
  double r = 0;
  double m = 5000;
  double ma = 0;
  double r_host = 0;
  double *a = new double[N]();
  double *b = new double[N]();
  /* cudaMallocManaged(&a, N*sizeof(double)); */
  /* cudaMallocManaged(&b, N*sizeof(double)); */
  /* double *a = (double *)omp_target_alloc(sizeof(double) * N, 0); */
  /* double *b = (double *)omp_target_alloc(sizeof(double) * N, 0); */
  std::iota(a, a + N, 0);
  std::iota(b, b + N, 0);
  /* #pragma omp target teams distribute parallel for is_device_ptr(a,b) */
  /* for(int i=0; i<N; ++i) { */
  /*   a[i] = i; */
  /*   b[i] = i; */
  /* } */

#pragma omp target enter data map(to : a[:N], b[:N])
  RAJA::Timer t;
//  for (int j = 0; j < 20; ++j) {
    //use_dev = 0;
    t.reset();
    t.start();
//#pragma omp target data use_device_ptr(a, b) if (use_dev)
    forall_param(N,
                 [=](int i, double &r, double &m, double &ma) {
                   r += a[i] * b[i];
                   m = a[i] < m ? a[i] : m;
                   ma = a[i] > m ? a[i] : m;
                 },
                 Reduce<RAJA::operators::plus>(r),
                 Reduce<RAJA::operators::minimum>(m),
                 Reduce<RAJA::operators::maximum>(ma));
    t.stop();
    std::cout << "use_dev: " << use_dev << " time: " << t.elapsed()
              << std::endl;
    std::cout << r << std::endl;
    std::cout << m << " " << ma << std::endl;
    r = 0;
    m = 5000;
    ma = 0;
//  }
/*
  for (int j = 0; j < 20; ++j) {
    t.reset();
    t.start();
#pragma omp target map(r, m, ma)
#pragma omp teams distribute parallel for \
    reduction(+:r) \
    reduction(min:m) \
    reduction(max:ma)
    for (int i = 0; i < N; ++i) {
      r += a[i] * b[i];
      m = a[i] < m ? a[i] : m;
      ma = a[i] > m ? a[i] : m;
    }
    t.stop();
    std::cout << "use_dev: " << use_dev << " time: " << t.elapsed()
              << std::endl;
    std::cout << r << std::endl;
    std::cout << m << " " << ma << std::endl;
    r = 0;
    m = 5000;
    ma = 0;
  }
*/
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

  /* for(int i=0; i<N; ++i) { */
  /*   r_host += a[i] * b[i]; */
  /* } */


  std::cout << rr.get() << " " << rt.elapsed() << std::endl;
  std::cout << rm.get() << " " << rma.get() << std::endl;

  std::cout << t.elapsed() << " " << rt.elapsed() << std::endl;


  return 0;
}
