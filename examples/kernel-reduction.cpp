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
      A[col + row * N] = (row * N + col) % 100 + 1;
    }
  }
  A[0] = 1000;
  A[N * N / 2] = 0;

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
    [=] (int row,
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

  int seq_for_min = std::numeric_limits<int>::max();
  int seq_for_max = std::numeric_limits<int>::min();
  int seq_for_sum = 0;
  VALLOC_INT seq_for_minloc(std::numeric_limits<int>::max(), -1);
  VALLOC_INT seq_for_maxloc(std::numeric_limits<int>::min(), -1);
  RAJA::Index_type seq_for_minloc2(-1);
  RAJA::Index_type seq_for_maxloc2(-1);
  int seq_for_min2 = std::numeric_limits<int>::max();
  int seq_for_max2 = std::numeric_limits<int>::min();

  using EXEC_POL_SEQ_FOR = RAJA::KernelPolicy<
  RAJA::statement::For<0, RAJA::seq_exec,
                            RAJA::statement::Lambda<0>
    >
  >;

  RAJA::kernel_param<EXEC_POL_SEQ_FOR>(
    // segments
    RAJA::make_tuple(col_range),
    // params
    RAJA::make_tuple(
      RAJA::expt::Reduce<RAJA::operators::plus   >(&seq_for_sum),
      RAJA::expt::Reduce<RAJA::operators::minimum>(&seq_for_min),
      RAJA::expt::Reduce<RAJA::operators::maximum>(&seq_for_max),
      RAJA::expt::Reduce<RAJA::operators::minimum>(&seq_for_minloc),
      RAJA::expt::Reduce<RAJA::operators::maximum>(&seq_for_maxloc),
      RAJA::expt::ReduceLoc<RAJA::operators::minimum>(&seq_for_min2, &seq_for_minloc2),
      RAJA::expt::ReduceLoc<RAJA::operators::maximum>(&seq_for_max2, &seq_for_maxloc2)
    ),
    // lambda 1
    [=] (
      int col,
      VALOP_INT_SUM &_sum,
      VALOP_INT_MIN &_min,
      VALOP_INT_MAX &_max,
      VALOPLOC_INT_MIN &_minloc,
      VALOPLOC_INT_MAX &_maxloc,
      VALOPLOC_INT_MIN &_minloc2,
      VALOPLOC_INT_MAX &_maxloc2)
    {
      for( int row = 0; row < N; ++row)
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
    }
  );

  std::cout << "SEQ + FOR MIN VAL = " << seq_for_min << std::endl;
  std::cout << "SEQ + FOR MAX VAL = " << seq_for_max << std::endl;
  std::cout << "SEQ + FOR SUM VAL = " << seq_for_sum << std::endl;
  std::cout << "SEQ + FOR MIN VAL = " << seq_for_minloc.getVal() << " MIN VALLOC  = " << seq_for_minloc.getLoc() << std::endl;
  std::cout << "SEQ + FOR MAX VAL = " << seq_for_maxloc.getVal() << " MAX VALLOC  = " << seq_for_maxloc.getLoc() << std::endl;
  std::cout << "SEQ + FOR MIN VAL LOC 2 = "<< seq_for_minloc2 << std::endl;
  std::cout << "SEQ + FOR MAX VAL LOC 2 = "<< seq_for_maxloc2 << std::endl;

  int for_seq_min = std::numeric_limits<int>::max();
  int for_seq_max = std::numeric_limits<int>::min();
  int for_seq_sum = 0;
  VALLOC_INT for_seq_minloc(std::numeric_limits<int>::max(), -1);
  VALLOC_INT for_seq_maxloc(std::numeric_limits<int>::min(), -1);
  RAJA::Index_type for_seq_minloc2(-1);
  RAJA::Index_type for_seq_maxloc2(-1);
  int for_seq_min2 = std::numeric_limits<int>::max();
  int for_seq_max2 = std::numeric_limits<int>::min();

  RAJA::forall<RAJA::seq_exec>(col_range,
    RAJA::expt::Reduce<RAJA::operators::plus>(&for_seq_sum),
    RAJA::expt::Reduce<RAJA::operators::minimum>(&for_seq_min),
    RAJA::expt::Reduce<RAJA::operators::maximum>(&for_seq_max),
    RAJA::expt::Reduce<RAJA::operators::minimum>(&for_seq_minloc),
    RAJA::expt::Reduce<RAJA::operators::maximum>(&for_seq_maxloc),
    RAJA::expt::ReduceLoc<RAJA::operators::minimum>(&for_seq_min2, &for_seq_minloc2),
    RAJA::expt::ReduceLoc<RAJA::operators::maximum>(&for_seq_max2, &for_seq_maxloc2),
    [=] (int col,
         VALOP_INT_SUM &_sum,
         VALOP_INT_MIN &_min,
         VALOP_INT_MAX &_max,
         VALOPLOC_INT_MIN &_minloc,
         VALOPLOC_INT_MAX &_maxloc,
         VALOPLOC_INT_MIN &_minloc2,
         VALOPLOC_INT_MAX &_maxloc2
    ) {
    for( int row = 0; row < N; ++row)
    {
      _min = _min.min(Aview(row, col));
      _max = _max.max(Aview(row, col));
      _sum += Aview(row, col);

      _minloc.minloc(Aview(row, col), row * N + col);
      _maxloc.maxloc(Aview(row, col), row * N + col);
      _minloc2.minloc(Aview(row, col), row * N + col);
      _maxloc2.maxloc(Aview(row, col), row * N + col);
    }
  });

  std::cout << "FOR SEQ MIN VAL = " << for_seq_min << std::endl;
  std::cout << "FOR SEQ MAX VAL = " << for_seq_max << std::endl;
  std::cout << "FOR SEQ SUM VAL = " << for_seq_sum << std::endl;


#if defined RAJA_ENABLE_CUDA
  using ResourceType = RAJA::resources::Cuda;
  using EXEC_POL =
  RAJA::KernelPolicy<
    RAJA::statement::CudaKernel<
      RAJA::statement::For<1, RAJA::cuda_thread_x_loop,
        RAJA::statement::For<0, RAJA::cuda_thread_y_loop,
                                RAJA::statement::Lambda<0>
        >
      >
    >
  >;
#elif defined RAJA_ENABLE_HIP
  using ResourceType = RAJA::resources::Hip;
  using EXEC_POL =
  RAJA::KernelPolicy<
    RAJA::statement::HipKernel<
      RAJA::statement::For<1, RAJA::hip_thread_x_loop,
        RAJA::statement::For<0, RAJA::hip_thread_y_loop,
                                RAJA::statement::Lambda<0>
        >
      >
    >
  >;
#endif

#if defined RAJA_ENABLE_CUDA or defined RAJA_ENABLE_HIP
  ResourceType res;
  int *dA = res.allocate<int>(N * N);
  res.memcpy(dA, A, sizeof(int) * N * N);


  RAJA::View<int, RAJA::Layout<2>> dAview(dA, N, N);
  int gpu_min = std::numeric_limits<int>::max();
  int gpu_max = std::numeric_limits<int>::min();
  int gpu_sum = 0;
  VALLOC_INT gpu_minloc(std::numeric_limits<int>::max(), -1);
  VALLOC_INT gpu_maxloc(std::numeric_limits<int>::min(), -1);
  RAJA::Index_type gpu_minloc2(-1);
  RAJA::Index_type gpu_maxloc2(-1);
  int gpu_min2 = std::numeric_limits<int>::max();
  int gpu_max2 = std::numeric_limits<int>::min();
  RAJA::kernel_param<EXEC_POL>(
    // segments
    RAJA::make_tuple(col_range, row_range),
    // params
    RAJA::make_tuple(
      RAJA::expt::Reduce<RAJA::operators::plus   >(&gpu_sum),
      RAJA::expt::Reduce<RAJA::operators::minimum>(&gpu_min),
      RAJA::expt::Reduce<RAJA::operators::maximum>(&gpu_max),
      RAJA::expt::Reduce<RAJA::operators::minimum>(&gpu_minloc),
      RAJA::expt::Reduce<RAJA::operators::maximum>(&gpu_maxloc),
      RAJA::expt::ReduceLoc<RAJA::operators::minimum>(&gpu_min2, &gpu_minloc2),
      RAJA::expt::ReduceLoc<RAJA::operators::maximum>(&gpu_max2, &gpu_maxloc2)
    ),
    // lambda 1
    [=] RAJA_HOST_DEVICE (int row,
      int col,
      VALOP_INT_SUM &_gpu_sum,
      VALOP_INT_MIN &_gpu_min,
      VALOP_INT_MAX &_gpu_max,
      VALOPLOC_INT_MIN &_gpu_minloc,
      VALOPLOC_INT_MAX &_gpu_maxloc,
      VALOPLOC_INT_MIN &_gpu_minloc2,
      VALOPLOC_INT_MAX &_gpu_maxloc2)
      {
      _gpu_sum += dAview(row, col);
      _gpu_min = _gpu_min.min(dAview(row, col));
      _gpu_max = _gpu_max.max(dAview(row, col));

      // loc
      _gpu_minloc.minloc(dAview(row, col), row * N + col);
      _gpu_maxloc.maxloc(dAview(row, col), row * N + col);
      _gpu_minloc2.minloc(dAview(row, col), row * N + col);
      _gpu_maxloc2.maxloc(dAview(row, col), row * N + col);
    }
  );
  std::cout << "GPU MIN VAL = " << gpu_min << std::endl;
  std::cout << "GPU MAX VAL = " << gpu_max << std::endl;
  std::cout << "GPU SUM VAL = " << gpu_sum << std::endl;
  std::cout << "GPU MIN VAL = " << gpu_minloc.getVal() << " MIN VALLOC  = " << gpu_minloc.getLoc() << std::endl;
  std::cout << "GPU MAX VAL = " << gpu_maxloc.getVal() << " MAX VALLOC  = " << gpu_maxloc.getLoc() << std::endl;
  std::cout << "GPU MIN VAL LOC 2 = "<< gpu_minloc2 << std::endl;
  std::cout << "GPU MAX VAL LOC 2 = "<< gpu_maxloc2 << std::endl;

#endif
#if defined RAJA_ENABLE_OPENMP
using EXEC_POL_OPENMP =
  RAJA::KernelPolicy<
    RAJA::statement::For<1, RAJA::seq_exec,  // row
      RAJA::statement::For<0, RAJA::omp_parallel_for_exec,            // col
        RAJA::statement::Lambda<0>
      >
    >
  >;
//using EXEC_POL_OPENMP =
//  RAJA::KernelPolicy<
//    RAJA::statement::Collapse<RAJA::omp_parallel_collapse_exec,
//      RAJA::ArgList<1,0>,
//      RAJA::statement::Lambda<0>
//    >
//  >;

int openmp_min = std::numeric_limits<int>::max();
int openmp_max = std::numeric_limits<int>::min();
int openmp_sum = 0;
VALLOC_INT openmp_minloc(std::numeric_limits<int>::max(), -1);
VALLOC_INT openmp_maxloc(std::numeric_limits<int>::min(), -1);
RAJA::Index_type openmp_minloc2(-1);
RAJA::Index_type openmp_maxloc2(-1);
int openmp_min2 = std::numeric_limits<int>::max();
int openmp_max2 = std::numeric_limits<int>::min();

RAJA::kernel_param<EXEC_POL_OPENMP>(
  RAJA::make_tuple(col_range, row_range),
  RAJA::make_tuple(
    RAJA::expt::Reduce<RAJA::operators::plus   >(&openmp_sum),
    RAJA::expt::Reduce<RAJA::operators::minimum>(&openmp_min),
    RAJA::expt::Reduce<RAJA::operators::maximum>(&openmp_max),
    RAJA::expt::Reduce<RAJA::operators::minimum>(&openmp_minloc),
    RAJA::expt::Reduce<RAJA::operators::maximum>(&openmp_maxloc),
    RAJA::expt::ReduceLoc<RAJA::operators::minimum>(&openmp_min2, &openmp_minloc2),
    RAJA::expt::ReduceLoc<RAJA::operators::maximum>(&openmp_max2, &openmp_maxloc2)
  ),
  // lambda 1
  [=] RAJA_HOST_DEVICE (
      int row,
      int col,
      VALOP_INT_SUM &_gpu_sum,
      VALOP_INT_MIN &_gpu_min ,
      VALOP_INT_MAX &_gpu_max,
      VALOPLOC_INT_MIN &_gpu_minloc,
      VALOPLOC_INT_MAX &_gpu_maxloc,
      VALOPLOC_INT_MIN &_gpu_minloc2,
      VALOPLOC_INT_MAX &_gpu_maxloc2
    )
    {
    _gpu_sum += Aview(row, col);
    _gpu_min = _gpu_min.min(Aview(row, col));
    _gpu_max = _gpu_max.max(Aview(row, col));

    // loc
    _gpu_minloc.minloc(Aview(row, col), row * N + col);
    _gpu_maxloc.maxloc(Aview(row, col), row * N + col);
    _gpu_minloc2.minloc(Aview(row, col), row * N + col);
    _gpu_maxloc2.maxloc(Aview(row, col), row * N + col);
  }

  );

  std::cout << "Seq + OpenMP MIN VAL = " << openmp_min << std::endl;
  std::cout << "Seq + OpenMP MAX VAL = " << openmp_max << std::endl;
  std::cout << "Seq + OpenMP SUM VAL = " << openmp_sum << std::endl;
  std::cout << "Seq + OpenMP MIN VAL = " << openmp_minloc.getVal() << " MIN VALLOC  = " << openmp_minloc.getLoc() << std::endl;
  std::cout << "Seq + OpenMP MAX VAL = " << openmp_maxloc.getVal() << " MAX VALLOC  = " << openmp_maxloc.getLoc() << std::endl;
  std::cout << "Seq + OpenMP MIN VAL LOC 2 = "<< openmp_minloc2 << std::endl;
  std::cout << "Seq + OpenMP MAX VAL LOC 2 = "<< openmp_maxloc2 << std::endl;



  using EXEC_POL_OPENMP_2 =
  RAJA::KernelPolicy<
      RAJA::statement::For<0, RAJA::omp_parallel_for_exec,
        RAJA::statement::Lambda<0>
      >
  >;


int openmp_min_2 = std::numeric_limits<int>::max();
int openmp_max_2 = std::numeric_limits<int>::min();
int openmp_sum_2 = 0;
VALLOC_INT openmp_minloc_2(std::numeric_limits<int>::max(), -1);
VALLOC_INT openmp_maxloc_2(std::numeric_limits<int>::min(), -1);
RAJA::Index_type openmp_minloc_2_loc(-1);
RAJA::Index_type openmp_maxloc_2_loc(-1);
int openmp_min2_2 = std::numeric_limits<int>::max();
int openmp_max2_2 = std::numeric_limits<int>::min();

RAJA::kernel_param<EXEC_POL_OPENMP_2>(
  RAJA::make_tuple(RAJA::RangeSegment(0, N * N)),
  RAJA::make_tuple(
    RAJA::expt::Reduce<RAJA::operators::plus   >(&openmp_sum_2),
    RAJA::expt::Reduce<RAJA::operators::minimum>(&openmp_min_2),
    RAJA::expt::Reduce<RAJA::operators::maximum>(&openmp_max_2),
    RAJA::expt::Reduce<RAJA::operators::minimum>(&openmp_minloc_2),
    RAJA::expt::Reduce<RAJA::operators::maximum>(&openmp_maxloc_2),
    RAJA::expt::ReduceLoc<RAJA::operators::minimum>(&openmp_min2_2, &openmp_minloc_2_loc),
    RAJA::expt::ReduceLoc<RAJA::operators::maximum>(&openmp_max2_2, &openmp_maxloc_2_loc)
  ),
  // lambda 1
  [=] RAJA_HOST_DEVICE (
      int idx,
      VALOP_INT_SUM &_gpu_sum,
      VALOP_INT_MIN &_gpu_min,
      VALOP_INT_MAX &_gpu_max,
      VALOPLOC_INT_MIN &_gpu_minloc,
      VALOPLOC_INT_MAX &_gpu_maxloc,
      VALOPLOC_INT_MIN &_gpu_minloc2,
      VALOPLOC_INT_MAX &_gpu_maxloc2
      )
      {
      _gpu_sum += A[idx];
      _gpu_min.min(A[idx]);
      _gpu_max.max(A[idx]);

      // loc
      _gpu_minloc.minloc(A[idx], idx);
      _gpu_maxloc.maxloc(A[idx], idx);
      _gpu_minloc2.minloc(A[idx], idx);
      _gpu_maxloc2.maxloc(A[idx], idx);
    }
  );

  std::cout << "OpenMP MIN VAL = " << openmp_min_2 << std::endl;
  std::cout << "OpenMP MAX VAL = " << openmp_max_2 << std::endl;
  std::cout << "OpenMP SUM VAL = " << openmp_sum_2 << std::endl;
  std::cout << "OpenMP MIN VAL = " << openmp_minloc_2.getVal() << " MIN VALLOC  = " << openmp_minloc_2.getLoc() << std::endl;
  std::cout << "OpenMP MAX VAL = " << openmp_maxloc_2.getVal() << " MAX VALLOC  = " << openmp_maxloc_2.getLoc() << std::endl;
  std::cout << "OpenMP MIN VAL LOC 2 = "<< openmp_minloc_2_loc << std::endl;
  std::cout << "OpenMP MAX VAL LOC 2 = "<< openmp_maxloc_2_loc << std::endl;

#endif

};
