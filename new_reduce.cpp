#include <iostream>

#define RAJA_EXPT_FORALL
#include <RAJA/RAJA.hpp>
#include <RAJA/util/Timer.hpp>

#include <numeric>

#include "new_reduce/reduce_basic.hpp"
#include "new_reduce/reduce_array.hpp"
#include "new_reduce/kernel_name.hpp"
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

#if 0
#if defined(RAJA_ENABLE_OPENMP)
  {
    std::cout << "OMP Reduction NEW\n";

    RAJA::Timer t;
    t.reset();
    t.start();

    forall_param<RAJA::omp_parallel_for_exec>(N,
                 Reduce<RAJA::operators::plus>(&r),
                 Reduce<RAJA::operators::minimum>(&m),
                 Reduce<RAJA::operators::maximum>(&ma),
                 [=](int i, double &r_, double &m_, double &ma_) {
                   r_ += a[i] * b[i];
                   m_ = a[i] < m_ ? a[i] : m_;
                   ma_ = a[i] > ma_ ? a[i] : ma_;
                 });
    t.stop();
    
    std::cout << "t : " << t.elapsed() << "\n";
    std::cout << "r : " << r << "\n";
    std::cout << "m : "  << m  <<"\n";
    std::cout << "ma : " << ma <<"\n";
  }
#endif
#endif

#if 0
#if defined(RAJA_ENABLE_OPENMP)
  {
    std::cout << "OMP ARRAY Reduction NEW\n";

    RAJA::Timer t;
    t.reset();
    t.start();

    const size_t arr_sz = 5;
    double m_array[arr_sz];

    forall_param<RAJA::omp_parallel_for_exec>(N,
                 Reduce<RAJA::operators::plus>(&r),
                 Reduce<RAJA::operators::minimum>(&m),
                 Reduce<RAJA::operators::maximum>(&ma),
                 ReduceArray<RAJA::operators::plus>(m_array, arr_sz),
                 [=](int i, double &r_, double &m_, double &ma_, double *l_array) {
                   r_ += a[i] * b[i];
                   m_ = a[i] < m_ ? a[i] : m_;
                   ma_ = a[i] > m_ ? a[i] : m_;
                   l_array[1] += 2; 
                   l_array[3] += 2; 
                 });
    t.stop();
    
    std::cout << "t : " << t.elapsed() << "\n";
    std::cout << "r : " << r << "\n";
    std::cout << "m : "  << m  <<"\n";
    std::cout << "ma : " << ma <<"\n";
    for (size_t i = 0; i < arr_sz; i++) {
      std::cout << "m_array[" << i << "] : " << m_array[i] << "\n";
    }
  }
#endif
#endif

#if 0
#if defined(RAJA_ENABLE_CUDA)
  {
    std::cout << "CUDA Reduction NEW Single\n";

    RAJA::Timer t;
    t.reset();
    t.start();

    forall_param<RAJA::cuda_exec<256>>(N,
                 Reduce<RAJA::operators::plus>(&r),
                 [=] RAJA_HOST_DEVICE (int i, double &r_) {
                   r_ += a[i] * b[i];
                 });
    t.stop();
    
    std::cout << "t : " << t.elapsed() << "\n";
    std::cout << "r : " << r << "\n";
  }
#endif
#endif

#if 0
#if defined(RAJA_ENABLE_CUDA)
  {
    std::cout << "CUDA Reduction NEW Multi\n";

    RAJA::Timer t;
    t.reset();
    t.start();

    forall_param<RAJA::cuda_exec<256>>(N,
                 Reduce<RAJA::operators::plus>(&r),
                 Reduce<RAJA::operators::minimum>(&m),
                 KernelName("Test"),
                 Reduce<RAJA::operators::maximum>(&ma),
                 [=] RAJA_HOST_DEVICE (int i, double &r_, double &m_, double &ma_) {
                   r_ += a[i] * b[i];
                   m_ = a[i] < m_ ? a[i] : m_;
                   ma_ = a[i] > m_ ? a[i] : m_;
                 }
                 );
    t.stop();
    
    std::cout << "t : " << t.elapsed() << "\n";
    std::cout << "r : " << r << "\n";
    std::cout << "m : "  << m  <<"\n";
    std::cout << "ma : " << ma <<"\n";
  }
#endif
#endif

#if 0
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

#if 0
  {
    std::cout << "Sequential Reduction NEW\n";

    RAJA::Timer t;
    t.reset();
    t.start();

    forall_param<RAJA::seq_exec>(N,
                 Reduce<RAJA::operators::plus>(&r),
                 Reduce<RAJA::operators::minimum>(&m),
                 Reduce<RAJA::operators::maximum>(&ma),
                 [=](int i, double &r_, double &m_, double &ma_) {
                   r_ += a[i] * b[i];
                   m_ = a[i] < m_ ? a[i] : m_;
                   ma_ = a[i] > m_ ? a[i] : m_;
                 }
                 );
    t.stop();
    
    std::cout << "t : " << t.elapsed() << "\n";
    std::cout << "r : " << r << "\n";
    std::cout << "m : "  << m  <<"\n";
    std::cout << "ma : " << ma <<"\n";
  }
#endif
#if 1
  {
    std::cout << "Basic TBB Reduction RAJA w/ NEW REDUCE\n";

    using VLD = RAJA::expt::ValLoc<double>;
    VLD vlm;
    VLD vlma;

    RAJA::Timer t;
    t.start();
    RAJA::forall<RAJA::tbb_for_dynamic>(
                   RAJA::RangeSegment(0, N),
                     RAJA::expt::Reduce<RAJA::operators::plus>(&r),
                     RAJA::expt::Reduce<RAJA::operators::minimum>(&m),
                     RAJA::expt::Reduce<RAJA::operators::maximum>(&ma),
                     RAJA::expt::Reduce<RAJA::operators::maximum>(&vlma),
                     RAJA::expt::Reduce<RAJA::operators::minimum>(&vlm),
                     //int{}
                     [=](int i, double &r_, double &m_, double &ma_, VLD &vlma_, VLD &vlm_) {
                       r_ += a[i] * b[i];
                       m_ = a[i] < m_ ? a[i] : m_;
                       ma_ = a[i] > m_ ? a[i] : m_;
                       vlma_.max(i, i);
                       vlm_.min(i, i);
                     }
                 );
    t.stop();

    std::cout << "t : " << t.elapsed() << "\n";
    std::cout << "r : " << r << "\n";
    std::cout << "m : "  << m  <<"\n";
    std::cout << "ma : " << ma <<"\n";
    std::cout << "vlm val : " << vlm.getVal() <<"\n";
    std::cout << "vlm loc : " << vlm.getLoc() <<"\n";
    std::cout << "vlma val : " << vlma.getVal() <<"\n";
    std::cout << "vlma loc : " << vlma.getLoc() <<"\n";
  }
#endif
#if 0
  {
    std::cout << "Basic Reduction RAJA\n";
    RAJA::ReduceSum<RAJA::seq_reduce, double> rr(0);
    RAJA::ReduceMin<RAJA::seq_reduce, double> rm(5000);
    RAJA::ReduceMax<RAJA::seq_reduce, double> rma(0);

    RAJA::Timer t;
    t.start();
    RAJA::forall<RAJA::seq_exec>(
                   RAJA::RangeSegment(0, N),
                     [=](int i) {
                       rr += a[i] * b[i];
                       rm.min(a[i]);
                       rma.max(a[i]);
                     }
                 );
    t.stop();

    std::cout << "t : " << t.elapsed() << "\n";
    std::cout << "r : " << rr.get() << "\n";
    std::cout << "m : "  << rm.get()  <<"\n";
    std::cout << "ma : " << rma.get() <<"\n";
  }
#endif
#if 1
  {
    std::cout << "Basic Reduction RAJA w/ NEW REDUCE\n";

    using VLD = RAJA::expt::ValLoc<double>;
    VLD vlm;
    VLD vlma;
    //auto rlvl = RAJA::expt::ReduceLoc<RAJA::operators::maximum>(&vl);
    r = 0;
    RAJA::Timer t;
    t.start();
    RAJA::forall<RAJA::seq_exec>(
                   RAJA::RangeSegment(0, N),
                     RAJA::expt::Reduce<RAJA::operators::plus>(&r),
                     RAJA::expt::Reduce<RAJA::operators::minimum>(&m),
                     RAJA::expt::Reduce<RAJA::operators::maximum>(&ma),
                     RAJA::expt::ReduceLoc<RAJA::operators::maximum>(&vlma),
                     RAJA::expt::ReduceLoc<RAJA::operators::minimum>(&vlm),
                     //int{}
                     [=](int i, double &r_, double &m_, double &ma_, VLD &vlma_, VLD &vlm_) {
                       r_ += a[i] * b[i];
                       m_ = a[i] < m_ ? a[i] : m_;
                       ma_ = a[i] > m_ ? a[i] : m_;
                       vlma_.max(i, i);
                       vlm_.min(i, i);
                     }
                 );
    t.stop();

    std::cout << "t : " << t.elapsed() << "\n";
    std::cout << "r : " << r << "\n";
    std::cout << "m : "  << m  <<"\n";
    std::cout << "ma : " << ma <<"\n";
    std::cout << "vlm val : " << vlm.getVal() <<"\n";
    std::cout << "vlm loc : " << vlm.getLoc() <<"\n";
    std::cout << "vlma val : " << vlma.getVal() <<"\n";
    std::cout << "vlma loc : " << vlma.getLoc() <<"\n";
  }
#endif
#if 0
  {
    std::cout << "Basic Reduction RAJA w/ NEW REDUCE LOC\n";

    using VLD = RAJA::expt::ValLoc<double>;
    VLD vlm;
    VLD vlma;

    RAJA::Timer t;
    t.start();
    RAJA::forall<RAJA::seq_exec>(
                   RAJA::RangeSegment(0, N),
                     RAJA::expt::ReduceLoc<RAJA::operators::minimum>(&vlm),
                     RAJA::expt::ReduceLoc<RAJA::operators::maximum>(&vlma),
                     [=](int i, VLD &vlm_, VLD &vlma_) {
                       vlm_.min(i,i);
                       vlma_.max(i,i);
                     }
                 );
    t.stop();

    std::cout << "t : " << t.elapsed() << "\n";
    //std::cout << "r : " << r << "\n";
    //std::cout << "m : "  << m  <<"\n";
    //std::cout << "ma : " << ma <<"\n";
    std::cout << "vlm val : " << vlm.getVal() <<"\n";
    std::cout << "vlm loc : " << vlm.getLoc() <<"\n";
    std::cout << "vlma val : " << vlma.getVal() <<"\n";
    std::cout << "vlma loc : " << vlma.getLoc() <<"\n";
    //std::cout << "vlma val : " << vlma.getVal() <<"\n";
    //std::cout << "vlma loc : " << vlma.getLoc() <<"\n";
  }
#endif

#if 0
#if defined(RAJA_ENABLE_OPENMP)
  {
    std::cout << "Basic OMP Reduction RAJA w/ NEW REDUCE\n";

    double r = 0;
    RAJA::Timer t;
    t.start();
    RAJA::forall<RAJA::omp_parallel_for_exec>(
                   RAJA::RangeSegment(0, N),
                     RAJA::expt::Reduce<RAJA::operators::plus>(&r),
                     //RAJA::expt::Reduce<RAJA::operators::minimum>(&m),
                     //RAJA::expt::Reduce<RAJA::operators::maximum>(&ma),
                     [=](int i, double &r_) {
                     //[=](int i, double &r_, double &m_, double &ma_) {
                       r_ += a[i] * b[i];
                       //m_ = a[i] < m_ ? a[i] : m_;
                       //ma_ = a[i] > m_ ? a[i] : m_;
                     }
                 );
    t.stop();

    std::cout << "t : " << t.elapsed() << "\n";
    std::cout << "r : " << r << "\n";
    //std::cout << "m : "  << m  <<"\n";
    //std::cout << "ma : " << ma <<"\n";
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
    r = 0;

    RAJA::forall<RAJA::cuda_exec<256>>(
                 RAJA::RangeSegment(0, N),
                 RAJA::expt::Reduce<RAJA::operators::plus>(&r),
                 RAJA::expt::Reduce<RAJA::operators::minimum>(&m),
                 RAJA::expt::KernelName("Test"),
                 RAJA::expt::Reduce<RAJA::operators::maximum>(&ma),
                 [=] RAJA_HOST_DEVICE (int i, double &r_, double &m_, double &ma_) {
                   r_ += a[i] * b[i];
                   m_ = a[i] < m_ ? a[i] : m_;
                   ma_ = a[i] > m_ ? a[i] : m_;
                 }
                 );
    t.stop();
    
    std::cout << "t : " << t.elapsed() << "\n";
    std::cout << "r : " << r << "\n";
    std::cout << "m : "  << m  <<"\n";
    std::cout << "ma : " << ma <<"\n";
  }
#endif
#endif

int sample_sz = 1;
RAJA::Timer::ElapsedType old_t_sum = 0;
RAJA::Timer::ElapsedType new_t_sum = 0;
for (int sample = 0; sample < sample_sz; sample++){
#if 0
  {
    std::cout << "Basic OMP Reduction RAJA\n";
    RAJA::ReduceSum<RAJA::omp_reduce, double> rr(0);
    RAJA::ReduceMin<RAJA::omp_reduce, double> rm(5000);
    RAJA::ReduceMax<RAJA::omp_reduce, double> rma(0);

    RAJA::Timer t;
    t.start();
    RAJA::forall<RAJA::omp_parallel_for_exec>(
                   RAJA::RangeSegment(0, N),
                     [=](int i) {
                       rr += a[i] * b[i];
                       rm.min(a[i]);
                       rma.max(a[i]);
                     }
                 );
    t.stop();

    //std::cout << N << " " << t.elapsed() << "\n";
    old_t_sum += t.elapsed();
    std::cout << "t : " << t.elapsed() << "\n";
    std::cout << "r : " << rr.get() << "\n";
    std::cout << "m : "  << rm.get()  <<"\n";
    std::cout << "ma : " << rma.get() <<"\n";
  }
#endif
#if 0
  {
    r = 0;
    m = 5000;
    ma = 0;
    std::cout << "Basic OMP Reduction RAJA w/ NEW REDUCE\n";

    RAJA::Timer t;
    t.start();
    RAJA::forall<RAJA::omp_parallel_for_exec>(
                   RAJA::RangeSegment(0, N),
                     RAJA::expt::Reduce<RAJA::operators::plus>(&r),
                     RAJA::expt::Reduce<RAJA::operators::minimum>(&m),
                     RAJA::expt::Reduce<RAJA::operators::maximum>(&ma),
                     [=](int i, double &r_, double &m_, double &ma_) {
                       r_ += a[i] * b[i];
                       m_ = a[i] < m_ ? a[i] : m_;
                       ma_ = a[i] > m_ ? a[i] : m_;
                     }
                 );
    t.stop();

    //std::cout << N << " " << t.elapsed() << "\n";
    new_t_sum += t.elapsed();
    std::cout << "t : " << t.elapsed() << "\n";
    std::cout << "r : " << r << "\n";
    std::cout << "m : "  << m  <<"\n";
    std::cout << "ma : " << ma <<"\n";
  }
#endif
} //  sample loop
std::cout << "AVERAGES:\n";
std::cout << old_t_sum / sample_sz << "\n";
std::cout << new_t_sum / sample_sz << "\n";

  return 0;
}
