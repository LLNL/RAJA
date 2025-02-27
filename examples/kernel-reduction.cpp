#include "RAJA/RAJA.hpp"
#include "RAJA/index/RangeSegment.hpp"


int main(int RAJA_UNUSED_ARG(argc), char **RAJA_UNUSED_ARG(argv[]))
{
  using namespace RAJA;
// matrix min, really dumb example
using EXEC_POL_CUDA =
    RAJA::KernelPolicy<
      RAJA::statement::CudaKernel<
        //RAJA::statement::For<1, RAJA::cuda_block_x_loop,    // row
          RAJA::statement::For<0, RAJA::cuda_thread_x_loop, // col
            RAJA::statement::Lambda<0>  // min addition do I need an extra , RAJA::Params<0> here?
          >
       //>
      >
    >;

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

using EXEC_SEQ =
    RAJA::KernelPolicy<
             statement::For<0, seq_exec, statement::Lambda<0>>
    >;
  // _matmult_3lambdakernel_cuda_end

  using VALOPLOC_INT_MIN = RAJA::expt::ValLocOp<int, RAJA::Index_type, RAJA::operators::minimum>;
  using VALOP_INT_MIN = RAJA::expt::ValOp<int, RAJA::operators::minimum>;
  using VALOP_INT_MAX = RAJA::expt::ValOp<int, RAJA::operators::maximum>;
  // RAJA::expt::Reduce<RAJA::operators::minimum>(&cuda_min),
  //int cuda_min = 0;
  //int cuda_max = -10;
//
  int seq_sum = 0;
  int N = 1000;

  RAJA::TypedRangeSegment<int> row_range(0, N);
  RAJA::TypedRangeSegment<int> col_range(0, N);

  RAJA::resources::Host host_res;
  RAJA::resources::Cuda cuda_res;
  int* A = host_res.allocate<int>(N * N);
  int *dA = cuda_res.allocate<int>(N * N);
  for (int row = 0; row < N; ++row) {
    for (int col = 0; col < N; ++col) {
      A[col + row * N] = -row;
    }
  }
  A[0] = 1000;
  cuda_res.memcpy(dA, A, sizeof(int) * N * N);

  RAJA::View<int, RAJA::Layout<2>> Aview(dA, N, N);
  int cuda_min = 0;
  int cuda_max = -42;
  /*RAJA::kernel_param<EXEC_POL8>(
    // segments
    RAJA::make_tuple(col_range),//, row_range),
    // params
    RAJA::make_tuple(RAJA::expt::Reduce<RAJA::operators::minimum>(&cuda_min),
                            RAJA::expt::Reduce<RAJA::operators::maximum>(&cuda_max)),
    //RAJA::tuple<double>(0.0),
    // lambda 1
    [=] RAJA_DEVICE (int row, VALOP_INT_MIN &_cuda_min, VALOP_INT_MAX &_cuda_max) {
      _cuda_min.min(Aview(row, 0));
      _cuda_max.max(Aview(row, 0));
      printf("updated min to %d\n", _cuda_min.val);
      printf("updated max to %d\n", _cuda_max.val);
    }
  );*/
  auto res = RAJA::kernel_param<EXEC_POL_CUDA>(
    // segments
    RAJA::make_tuple(col_range),//, row_range),
    // params
    RAJA::make_tuple(RAJA::expt::Reduce<RAJA::operators::minimum>(&cuda_min),
                            RAJA::expt::Reduce<RAJA::operators::maximum>(&cuda_max)),
    // lambda 1
    [=] RAJA_HOST_DEVICE (int row, VALOP_INT_MIN &_cuda_min, VALOP_INT_MAX &_cuda_max) {
      _cuda_min = _cuda_min.min(Aview(row, 0));
      _cuda_max = _cuda_max.max(Aview(row, 0));;
    }
  );


  //RAJA::kernel_param<EXEC_POL_CUDA_2>(
  //  // segments
  //  RAJA::make_tuple(col_range, row_range),
  //  // params
  //  RAJA::make_tuple(RAJA::expt::Reduce<RAJA::operators::minimum>(&cuda_min),
  //  RAJA::expt::Reduce<RAJA::operators::maximum>(&cuda_max)),
  //  // lambda 1
  //  [=] RAJA_HOST_DEVICE (int row, int col, VALOP_INT_MIN &_cuda_min, VALOP_INT_MAX &_cuda_max) {
  //    _cuda_min = _cuda_min.min(Aview(row, col));
  //    _cuda_max = _cuda_max.max(Aview(row, col));;
  //  }
  //);

  std::cout << "MIN VAL = " << cuda_min << std::endl;
  std::cout << "MAX VAL = " << cuda_max << std::endl;

   // RAJA::kernel_param<EXEC_SEQ>(
   // // segments
   // RAJA::make_tuple(col_range),//, row_range),
   // // params
   // RAJA::make_tuple(&cuda_min),
   // //RAJA::tuple<double>(0.0),
   // // lambda 1
   // [=] RAJA_HOST_DEVICE (int row, int* cuda_min_) {
   //   if (Aview(row,0) < *cuda_min_) {
   //     *cuda_min_= Aview(row,0);
   //   }
   // }
   //);


  //checkResult<double>(Cview, N);
};
