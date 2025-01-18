#include "RAJA/RAJA.hpp"
#include "RAJA/index/RangeSegment.hpp"
#include "memoryManager.hpp"


int main(int RAJA_UNUSED_ARG(argc), char **RAJA_UNUSED_ARG(argv[]))
{

// matrix min, really dumb example
using EXEC_POL8 =
    RAJA::KernelPolicy<
      RAJA::statement::CudaKernel<
        RAJA::statement::For<1, RAJA::cuda_block_x_loop,    // row
          RAJA::statement::For<0, RAJA::cuda_thread_x_loop, // col
            RAJA::statement::Lambda<0>  // min addition do I need an extra , RAJA::Params<0> here?
          >
        >
      >
    >;
  // _matmult_3lambdakernel_cuda_end

  using VALOPLOC_INT_MIN = RAJA::expt::ValLocOp<int, RAJA::Index_type, RAJA::operators::minimum>;
  using VALOP_INT_MIN = RAJA::expt::ValOp<int, RAJA::operators::minimum>;
  // RAJA::expt::Reduce<RAJA::operators::minimum>(&cuda_min),
  int cuda_min = 0;

  int seq_sum = 0;
  int N = 10000;

  RAJA::TypedRangeSegment<int> row_range(0, N);
  RAJA::TypedRangeSegment<int> col_range(0, N);

  RAJA::resources::Cuda cuda_res;
  int *A = memoryManager::allocate<int>(N * N);
  for (int row = 0; row < N; ++row) {
    for (int col = 0; col < N; ++col) {
      A[col + row * N] = -row;
    }
  }

  RAJA::View<int, RAJA::Layout<2>> Aview(A, N, N);

  // doesn't compile: 
  //      no known conversion from 
  //      'RAJA::expt::detail::Reducer<RAJA::operators::minimum<int>, int, RAJA::expt::ValOp<int, RAJA::operators::minimum>>' 
  //       to 'VALOP_INT_MIN &'
  RAJA::kernel_param<EXEC_POL8>(
    // segments
    RAJA::make_tuple(col_range, row_range),
    // params
    RAJA::make_tuple(RAJA::expt::Reduce<RAJA::operators::minimum>(&cuda_min)),
    //RAJA::tuple<double>(0.0),
    // lambda 1
    [=] RAJA_DEVICE (int col, int row, VALOP_INT_MIN &_cuda_min) {
      _cuda_min.min(Aview(row, col));
      //double& a){
        //a += Aview(row, col);
    }

  );

  // compiles
  RAJA::forall<RAJA::cuda_exec<256>>(cuda_res, RAJA::RangeSegment(0, N),
    RAJA::expt::Reduce<RAJA::operators::minimum>(&cuda_min),
    [=] RAJA_DEVICE (int i, VALOP_INT_MIN &_cuda_min) {
      _cuda_min.min(Aview(i, 0));
    }

  );

  std::cout << "MIN VAL = " << cuda_min << std::endl;
  //checkResult<double>(Cview, N);
};
