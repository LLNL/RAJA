#include "RAJA/RAJA.hpp"

int main(int RAJA_UNUSED_ARG(argc), char **RAJA_UNUSED_ARG(argv[]))
{
  using namespace RAJA;
  using EXEC_POL_SEQ = RAJA::KernelPolicy<
      RAJA::statement::For<1, RAJA::seq_exec,
        RAJA::statement::For<0, RAJA::seq_exec,
                                RAJA::statement::Lambda<0>
        >
      >
  >;

  using VALLOC_INT = RAJA::expt::ValLoc<int>;
  using VALOP_INT_SUM = RAJA::expt::ValOp<int, RAJA::operators::plus>;
  using VALOP_INT_MIN = RAJA::expt::ValOp<int, RAJA::operators::minimum>;
  using VALOP_INT_MAX = RAJA::expt::ValOp<int, RAJA::operators::maximum>;
  using VALOPLOC_INT_MIN = RAJA::expt::ValLocOp<int, RAJA::Index_type, RAJA::operators::minimum>;
  using VALOPLOC_INT_MAX = RAJA::expt::ValLocOp<int, RAJA::Index_type, RAJA::operators::maximum>;

  int N = 1000;

  RAJA::resources::Host host_res;
  int* A = host_res.allocate<int>(N * N);
  RAJA::View<int, RAJA::Layout<2>> Aview(A, N, N);

  int seq_min = std::numeric_limits<int>::max();
  int seq_max = std::numeric_limits<int>::min();
  int seq_sum = 0;
  VALLOC_INT seq_minloc(std::numeric_limits<int>::max(), -1);
  VALLOC_INT seq_maxloc(std::numeric_limits<int>::min(), -1);
  RAJA::Index_type seq_minloc2(-1);
  RAJA::Index_type seq_maxloc2(-1);
  int seq_min2 = std::numeric_limits<int>::max();
  int seq_max2 = std::numeric_limits<int>::min();

  for (int row = 0; row < N; ++row) {
    for (int col = 0; col < N; ++col) {
      A[col + row * N] = - row - col;
    }
  }
  A[0] = 1000;

  RAJA::TypedRangeSegment<int> row_range(0, N);
  RAJA::TypedRangeSegment<int> col_range(0, N);

  RAJA::kernel_param<EXEC_POL_SEQ>(
    // segments
    RAJA::make_tuple(col_range, row_range),
    // params
    RAJA::make_tuple(
      RAJA::expt::Reduce<RAJA::operators::plus   >(&seq_sum),
      RAJA::expt::Reduce<RAJA::operators::minimum>(&seq_min),
      RAJA::expt::Reduce<RAJA::operators::maximum>(&seq_max),
      RAJA::expt::Reduce<RAJA::operators::minimum>(&seq_minloc),
      RAJA::expt::Reduce<RAJA::operators::maximum>(&seq_maxloc),
      RAJA::expt::ReduceLoc<RAJA::operators::minimum>(&seq_min2, &seq_minloc2),
      RAJA::expt::ReduceLoc<RAJA::operators::maximum>(&seq_max2, &seq_maxloc2)
    ),
    // lambda 1
    [=] RAJA_HOST_DEVICE (int row,
      int col,
      VALOP_INT_SUM &_sum,
      VALOP_INT_MIN &_min,
      VALOP_INT_MAX &_max,
      VALOPLOC_INT_MIN &_minloc,
      VALOPLOC_INT_MAX &_maxloc,
      VALOPLOC_INT_MIN &_minloc2,
      VALOPLOC_INT_MAX &_maxloc2)
    {
      _min = _min.min(Aview(row, col));
      _max = _max.max(Aview(row, col));
      _sum += Aview(row, col);

      // loc
      _minloc.minloc(Aview(row, col), row * N + col);
      _maxloc.maxloc(Aview(row, col), row * N + col);
      _minloc2.minloc(Aview(row, col), row * N + col);
      _maxloc2.maxloc(Aview(row, col), row * N + col);
    }
  );

  std::cout << "SEQ MIN VAL = " << seq_min << std::endl;
  std::cout << "SEQ MAX VAL = " << seq_max << std::endl;
  std::cout << "SEQ SUM VAL = " << seq_sum << std::endl;
  std::cout << "SEQ MIN VAL = " << seq_minloc.getVal() << " MIN VALLOC  = " << seq_minloc.getLoc() << std::endl;
  std::cout << "SEQ MAX VAL = " << seq_maxloc.getVal() << " MAX VALLOC  = " << seq_maxloc.getLoc() << std::endl;
  std::cout << "SEQ MIN VAL LOC 2 = "<< seq_minloc2 << std::endl;
  std::cout << "SEQ MAX VAL LOC 2 = "<< seq_maxloc2 << std::endl;

#if defined RAJA_ENABLE_CUDA
  RAJA::resources::Cuda cuda_res;
  int *dA = cuda_res.allocate<int>(N * N);
  cuda_res.memcpy(dA, A, sizeof(int) * N * N);

  using EXEC_POL_CUDA_2 =
  RAJA::KernelPolicy<
    RAJA::statement::CudaKernel<
      RAJA::statement::For<1, RAJA::cuda_thread_x_loop,
        RAJA::statement::For<0, RAJA::cuda_thread_y_loop,
                                RAJA::statement::Lambda<0>
        >
      >
    >
  >;

  RAJA::View<int, RAJA::Layout<2>> dAview(dA, N, N);
  int cuda_min = std::numeric_limits<int>::max();
  int cuda_max = std::numeric_limits<int>::min();
  int cuda_sum = 0;
  VALLOC_INT cuda_minloc(std::numeric_limits<int>::max(), -1);
  VALLOC_INT cuda_maxloc(std::numeric_limits<int>::min(), -1);
  RAJA::Index_type cuda_minloc2(-1);
  RAJA::Index_type cuda_maxloc2(-1);
  int cuda_min2 = std::numeric_limits<int>::max();
  int cuda_max2 = std::numeric_limits<int>::min();
  auto res = RAJA::kernel_param<EXEC_POL_CUDA_2>(
    // segments
    RAJA::make_tuple(col_range, row_range),
    // params
    RAJA::make_tuple(
      RAJA::expt::Reduce<RAJA::operators::plus   >(&cuda_sum),
      RAJA::expt::Reduce<RAJA::operators::minimum>(&cuda_min),
      RAJA::expt::Reduce<RAJA::operators::maximum>(&cuda_max),
      RAJA::expt::Reduce<RAJA::operators::minimum>(&cuda_minloc),
      RAJA::expt::Reduce<RAJA::operators::maximum>(&cuda_maxloc),
      RAJA::expt::ReduceLoc<RAJA::operators::minimum>(&cuda_min2, &cuda_minloc2),
      RAJA::expt::ReduceLoc<RAJA::operators::maximum>(&cuda_max2, &cuda_maxloc2)
    ),
    // lambda 1
    [=] RAJA_HOST_DEVICE (int row,
      int col,
      VALOP_INT_SUM &_cuda_sum,
      VALOP_INT_MIN &_cuda_min,
      VALOP_INT_MAX &_cuda_max,
      VALOPLOC_INT_MIN &_cuda_minloc,
      VALOPLOC_INT_MAX &_cuda_maxloc,
      VALOPLOC_INT_MIN &_cuda_minloc2,
      VALOPLOC_INT_MAX &_cuda_maxloc2)
    {
      _cuda_min = _cuda_min.min(dAview(row, col));
      _cuda_max = _cuda_max.max(dAview(row, col));
      _cuda_sum += dAview(row, col);

      // loc
      _cuda_minloc.minloc(dAview(row, col), row * N + col);
      _cuda_maxloc.maxloc(dAview(row, col), row * N + col);
      _cuda_minloc2.minloc(dAview(row, col), row * N + col);
      _cuda_maxloc2.maxloc(dAview(row, col), row * N + col);
    }
  );
  std::cout << "CUDA MIN VAL = " << cuda_min << std::endl;
  std::cout << "CUDA MAX VAL = " << cuda_max << std::endl;
  std::cout << "CUDA SUM VAL = " << cuda_sum << std::endl;
  std::cout << "CUDA MIN VAL = " << cuda_minloc.getVal() << " MIN VALLOC  = " << cuda_minloc.getLoc() << std::endl;
  std::cout << "CUDA MAX VAL = " << cuda_maxloc.getVal() << " MAX VALLOC  = " << cuda_maxloc.getLoc() << std::endl;
  std::cout << "CUDA MIN VAL LOC 2 = "<< cuda_minloc2 << std::endl;
  std::cout << "CUDA MAX VAL LOC 2 = "<< cuda_maxloc2 << std::endl;
#endif

};
