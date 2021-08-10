#include <iostream>
#include <RAJA/RAJA.hpp>
#include <RAJA/util/Timer.hpp>

#include <numeric>

#include "new_reduce/reduce_basic.hpp"
#include "new_reduce/forall_param.hpp"

#if defined(RAJA_ENABLE_CUDA)
#define DeviceMallocManaged cudaMallocManaged
#elif defined(RAJA_ENABLE_HIP)
#define DeviceMallocManaged hipMallocManaged
#endif

int main(int argc, char *argv[])
{
  if (argc < 2) {
    std::cout << "Execution Format: ./executable N\n";
    std::cout << "Example of usage: ./new_reduce 5000\n";
    exit(0);
  }
  int N = atoi(argv[1]);

  double r = 0;
  double m = 5000;
  double ma = 0;

#if defined(RAJA_ENABLE_CUDA) or defined(RAJA_ENABLE_HIP)
  DeviceMallocManaged( (void**)(&r), sizeof(double));
  DeviceMallocManaged( (void**)(&m), sizeof(double));
  DeviceMallocManaged( (void**)(&ma), sizeof(double));

  double *a = new double[N]();
  double *b = new double[N]();

  DeviceMallocManaged( (void**)(&a), N*sizeof(double));
  DeviceMallocManaged( (void**)(&b), N*sizeof(double));
#else
  double *a = new double[N]();
  double *b = new double[N]();
#endif

  std::iota(a, a + N, 0);
  std::iota(b, b + N, 0);

#if defined(RAJA_ENABLE_TARGET_OPENMP)
  {
    std::cout << "OMP Target Reduction NEW\n";
    #pragma omp target enter data map(to : a[:N], b[:N])

    RAJA::Timer t;
    t.reset();
    t.start();

    forall_param<RAJA::omp_target_parallel_for_exec_nt>(N,
                 [=](int i, double &r_, double &m_, double &ma_) {
                   r_ += a[i] * b[i];
                   m_ = a[i] < m_ ? a[i] : m_;
                   ma_ = a[i] > m_ ? a[i] : m_;
                 },
                 Reduce<RAJA::operators::plus>(&r),
                 Reduce<RAJA::operators::minimum>(&m),
                 Reduce<RAJA::operators::maximum>(&ma));
    t.stop();
    
    std::cout << "t : " << t.elapsed() << "\n";
    std::cout << "r : " << r << "\n";
    std::cout << "m : "  << m  <<"\n";
    std::cout << "ma : " << ma <<"\n";
  }
#endif

#if 1
#if defined(RAJA_ENABLE_OPENMP)
  {
    std::cout << "OMP Reduction NEW\n";

    RAJA::Timer t;
    t.reset();
    t.start();

    forall_param<RAJA::omp_parallel_for_exec>(N,
                 [=](int i, double &r_, double &m_, double &ma_) {
                   r_ += a[i] * b[i];
                   m_ = a[i] < m_ ? a[i] : m_;
                   ma_ = a[i] > m_ ? a[i] : m_;
                 },
                 Reduce<RAJA::operators::plus>(&r),
                 Reduce<RAJA::operators::minimum>(&m),
                 Reduce<RAJA::operators::maximum>(&ma));
    t.stop();
    
    std::cout << "t : " << t.elapsed() << "\n";
    std::cout << "r : " << r << "\n";
    std::cout << "m : "  << m  <<"\n";
    std::cout << "ma : " << ma <<"\n";
  }
#endif
#endif

#if 1
#if defined(RAJA_ENABLE_CUDA)
  {
    std::cout << "CUDA Reduction NEW Single\n";

    RAJA::Timer t;
    t.reset();
    t.start();

    forall_param<RAJA::cuda_exec<256>>(N,
                 [=] RAJA_HOST_DEVICE (int i, double &r_) {
                   r_ += a[i] * b[i];
                 },
                 Reduce<RAJA::operators::plus>(&r));
    t.stop();
    
    std::cout << "t : " << t.elapsed() << "\n";
    std::cout << "r : " << r << "\n";
  }
#endif
#endif

#if 1
#if defined(RAJA_ENABLE_CUDA)
  {
    std::cout << "CUDA Reduction NEW Multi\n";

    RAJA::Timer t;
    t.reset();
    t.start();

    forall_param<RAJA::cuda_exec<256>>(N,
                 [=] RAJA_HOST_DEVICE (int i, double &r_, double &m_, double &ma_) {
                   r_ += a[i] * b[i];
                   m_ = a[i] < m_ ? a[i] : m_;
                   ma_ = a[i] > m_ ? a[i] : m_;
                 },
                 Reduce<RAJA::operators::plus>(&r),
                 Reduce<RAJA::operators::minimum>(&m),
                 Reduce<RAJA::operators::maximum>(&ma)
                 );
    t.stop();
    
    std::cout << "t : " << t.elapsed() << "\n";
    std::cout << "r : " << r << "\n";
    std::cout << "m : "  << m  <<"\n";
    std::cout << "ma : " << ma <<"\n";
  }
#endif
#endif

#if 1
#if defined(RAJA_ENABLE_HIP)
  {
    std::cout << "HIP Reduction NEW Single\n";

    RAJA::Timer t;
    t.reset();
    t.start();

    forall_param<RAJA::hip_exec<256>>(N,
                 [=] RAJA_HOST_DEVICE (int i, double &r_) {
                   r_ += a[i] * b[i];
                 },
                 Reduce<RAJA::operators::plus>(&r));
    t.stop();
    
    std::cout << "t : " << t.elapsed() << "\n";
    std::cout << "r : " << r << "\n";
  }
#endif
#endif

#if 1
  {
    std::cout << "Sequential Reduction NEW\n";

    RAJA::Timer t;
    t.reset();
    t.start();

    forall_param<RAJA::seq_exec>(N,
                 [=](int i, double &r_, double &m_, double &ma_) {
                   r_ += a[i] * b[i];
                   m_ = a[i] < m_ ? a[i] : m_;
                   ma_ = a[i] > m_ ? a[i] : m_;
                 },
                 Reduce<RAJA::operators::plus>(&r),
                 Reduce<RAJA::operators::minimum>(&m),
                 Reduce<RAJA::operators::maximum>(&ma));
    t.stop();
    
    std::cout << "t : " << t.elapsed() << "\n";
    std::cout << "r : " << r << "\n";
    std::cout << "m : "  << m  <<"\n";
    std::cout << "ma : " << ma <<"\n";
  }
#endif
  {
    std::cout << "Basic Reduction RAJA\n";
    RAJA::ReduceSum<RAJA::seq_reduce, double> rr(0);
    RAJA::ReduceMin<RAJA::seq_reduce, double> rm(5000);
    RAJA::ReduceMax<RAJA::seq_reduce, double> rma(0);

    RAJA::Timer t;
    t.start();
    RAJA::forall<RAJA::loop_exec>(RAJA::RangeSegment(0, N),
                                                        [=](int i) {
                                                          rr += a[i] * b[i];
                                                          rm.min(a[i]);
                                                          rma.max(a[i]);
                                                        });
    t.stop();

    std::cout << "t : " << t.elapsed() << "\n";
    std::cout << "r : " << rr.get() << "\n";
    std::cout << "m : "  << rm.get()  <<"\n";
    std::cout << "ma : " << rma.get() <<"\n";
  }

  return 0;
}
