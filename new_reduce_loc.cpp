#include <iostream>
#include <RAJA/RAJA.hpp>
#include <RAJA/util/Timer.hpp>

#include "new_reduce/reduce_basic.hpp"
#include "new_reduce/forall_param.hpp"

template<typename T, bool min = true>
class ValLoc {
  T val = min ? RAJA::operators::limits<T>::max() : RAJA::operators::limits<T>::min();
  RAJA::Index_type loc;
public:

  ValLoc() : loc(-1) {}
  ValLoc(T v) : val(v), loc(-1) {}
  ValLoc(T v, RAJA::Index_type l) : val(v), loc(l) {}

  ValLoc constexpr operator () (T v, RAJA::Index_type l) {
    if (min) 
      {if (v < val) { val = v; loc = l; } } 
    else
      {if (v > val) { val = v; loc = l; } }
    return *this;
  }

  bool constexpr operator < (const ValLoc& rhs) const { return val < rhs.val; }
  bool constexpr operator <=(const ValLoc& rhs) const { return val < rhs.val; }
  bool constexpr operator > (const ValLoc& rhs) const { return val > rhs.val; }
  bool constexpr operator >=(const ValLoc& rhs) const { return val > rhs.val; }

  T getVal() {return val;}
  RAJA::Index_type getLoc() {return loc;}
};

template<typename T>
ValLoc<T>& make_valloc(T v, RAJA::Index_type l) { return ValLoc<T>(v, l); }

template<typename T>
using ValLocMin = ValLoc<T, true>;

template<typename T>
using ValLocMax = ValLoc<T, false>;


namespace RAJA
{

namespace operators
{

template <typename T>
struct limits<ValLocMin<T>> {
  RAJA_INLINE RAJA_HOST_DEVICE static constexpr ValLocMin<T> min()
  {
    return ValLocMin<T>(RAJA::operators::limits<T>::min());
  }
  RAJA_INLINE RAJA_HOST_DEVICE static constexpr ValLocMin<T> max()
  {
    return ValLocMin<T>(RAJA::operators::limits<T>::max());
  }
};
template <typename T>
struct limits<ValLocMax<T>> {
  RAJA_INLINE RAJA_HOST_DEVICE static constexpr ValLocMax<T> min()
  {
    return ValLocMax<T>(RAJA::operators::limits<T>::min());
  }
  RAJA_INLINE RAJA_HOST_DEVICE static constexpr ValLocMax<T> max()
  {
    return ValLocMax<T>(RAJA::operators::limits<T>::max());
  }
};
}}

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

  double *a = new double[N]();
  double *b = new double[N]();

  std::iota(a, a + N, 0);
  std::iota(b, b + N, 0);

//#if defined(RAJA_ENABLE_TARGET_OPENMP)
//  {
//    std::cout << "OMP Target Reduction NEW\n";
//    #pragma omp target enter data map(to : a[:N], b[:N])
//
//    RAJA::Timer t;
//    t.reset();
//    t.start();
//
//    forall_param<RAJA::omp_target_parallel_for_exec_nt>(N,
//                 [=](int i, double &r_, double &m_, double &ma_) {
//                   r_ += a[i] * b[i];
//                   m_ = a[i] < m_ ? a[i] : m_;
//                   ma_ = a[i] > m_ ? a[i] : m_;
//                 },
//                 Reduce<RAJA::operators::plus>(&r),
//                 Reduce<RAJA::operators::minimum>(&m),
//                 Reduce<RAJA::operators::maximum>(&ma));
//    t.stop();
//    
//    std::cout << "t : " << t.elapsed() << "\n";
//    std::cout << "r : " << r << "\n";
//    std::cout << "m : "  << m  <<"\n";
//    std::cout << "ma : " << ma <<"\n";
//  }
//#endif

#if defined(RAJA_ENABLE_OPENMP)
  {
    std::cout << "OMP Reduction NEW\n";

    RAJA::Timer t;
    t.reset();
    t.start();

    ValLocMin<double> mL;
    ValLocMax<double> ML;

    forall_param<RAJA::omp_parallel_for_exec>(N,
                 [=](int i, ValLocMin<double> &mL_, ValLocMax<double> &ML_) {
                   mL_(a[i], i);
                   ML_(a[i], i);
                 },
                 //ReduceLoc(&mL),
                 //ReduceLoc(&ML));
                 Reduce<RAJA::operators::minimum>(&mL),
                 Reduce<RAJA::operators::maximum>(&ML));
    t.stop();
    
    std::cout << "t : " << t.elapsed() << "\n";
    std::cout << "mL : " << mL.getLoc() <<"\n";
    std::cout << "ML : " << ML.getLoc() <<"\n";
  }
#endif

//#if defined(RAJA_ENABLE_OPENMP)
//  {
//    std::cout << "OMP Reduction NEW\n";
//
//    RAJA::Timer t;
//    t.reset();
//    t.start();
//
//    ValueLoc<double, int> mL(-1);
//
//    forall_param<RAJA::omp_parallel_for_exec>(N,
//                 [=](int i, ValueLoc<double, int> &mL_, double &r_, double &m_, double &ma_) {
//                   r_ += a[i] * b[i];
//
//                   mL_ = ValueLoc<double, int>(a[i], i);
//
//                   m_ = a[i] < m_ ? a[i] : m_;
//                   ma_ = a[i] > m_ ? a[i] : m_;
//                 },
//                 Reduce<RAJA::operators::minimum>(&mL),
//                 Reduce<RAJA::operators::plus>(&r),
//                 Reduce<RAJA::operators::minimum>(&m),
//                 Reduce<RAJA::operators::maximum>(&ma));
//    t.stop();
//    
//    std::cout << "t : " << t.elapsed() << "\n";
//    std::cout << "r : " << r << "\n";
//    std::cout << "m : "  << m  <<"\n";
//    std::cout << "ma : " << ma <<"\n";
//    std::cout << "mL : " << mL <<"\n";
//  }
//#endif

//  {
//    std::cout << "Sequential Reduction NEW\n";
//
//    RAJA::Timer t;
//    t.reset();
//    t.start();
//
//    forall_param<RAJA::seq_exec>(N,
//                 [=](int i, double &r_, double &m_) {
//                 //[=](int i, double &r_, double &m_, double &ma_) {
//                   r_ += a[i] * b[i];
//                   m_ = a[i] < m_ ? a[i] : m_;
//                   //ma_ = a[i] > m_ ? a[i] : m_;
//                 },
//                 Reduce<RAJA::operators::plus>(&r),
//                 Reduce<RAJA::operators::minimum>(&m));//,
//                 //Reduce<RAJA::operators::maximum>(&ma));
//    t.stop();
//    
//    std::cout << "t : " << t.elapsed() << "\n";
//    std::cout << "r : " << r << "\n";
//    std::cout << "m : "  << m  <<"\n";
//    //std::cout << "ma : " << ma <<"\n";
//  }

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
