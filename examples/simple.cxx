#include <omp.h>
#include <cstdint>
#include "RAJA/RAJA.hxx"
#include <iostream>

RAJA_INDEX_VALUE(TimeInd, "Time Index")

RAJA_INDEX_VALUE(YInd, "Y Spatial Index")

RAJA_INDEX_VALUE(XInd, "X Spatial Index")

int
main(int argc, char **argv) {
  std::cout << "Starting\nMaximum number of threads = "
    << omp_get_max_threads()
    << std::endl;
  using testpolicy = RAJA::NestedPolicy<RAJA::ExecList<RAJA::omp_collapse_nowait_exec,
        RAJA::omp_collapse_nowait_exec,
        RAJA::omp_collapse_nowait_exec>,
        RAJA::OMP_Parallel<> >;

  RAJA::Index_type t_size = 16;
  RAJA::Index_type x_size = 16;
  RAJA::Index_type y_size = 16;

  double * const memory = new double[t_size*x_size*y_size];
  const double value = 4.0/(t_size * x_size * y_size);

  RAJA::View<double, RAJA::Layout<int, RAJA::PERM_IJK, TimeInd, XInd, YInd>> val_view(memory, t_size, x_size, y_size);

  RAJA::ReduceSum<RAJA::omp_reduce_ordered, double> oldsum(0);
  RAJA::ReduceSum<RAJA::omp_reduce, double> newsum(0);

  RAJA::forallN<testpolicy, TimeInd, XInd, YInd >
    (RAJA::RangeSegment(0, t_size),
     RAJA::RangeSegment(0, x_size),
     RAJA::RangeSegment(0, y_size),
     [=](TimeInd t, XInd x, YInd y) {
     val_view(t, x, y) = value;
     });

  RAJA::forallN<testpolicy, TimeInd, XInd, YInd >
    (RAJA::RangeSegment(0, t_size),
     RAJA::RangeSegment(0, x_size),
     RAJA::RangeSegment(0, y_size),
     [=](TimeInd t, XInd x, YInd y) {
     oldsum += val_view(t, x, y);
     newsum += val_view(t, x, y);
     });

  RAJA::ReduceSum<RAJA::omp_reduce, double> sum(0.0);

  RAJA::forall<RAJA::omp_parallel_for_exec>(
      0, (t_size*x_size*y_size), [=] (int i) {
      sum += memory[i];
  });

  std::cout << "Done\noldsum = " << oldsum << "\nnewsum = "
    << newsum << "\nnon-nestedsum = " << sum << "\n";

}
