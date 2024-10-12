#include "RAJA/RAJA.hpp"

#include "memoryManager.hpp"

int main(int RAJA_UNUSED_ARG(argc), char** RAJA_UNUSED_ARG(argv[]))
{

  std::cout << "\n\nRAJA kernel reductions example...\n";

  constexpr int N = 100;
  constexpr int Nsq = N * N;

  using DATA_TYPE = double;
  DATA_TYPE* a = memoryManager::allocate<double>(Nsq);

  using OUTER_LOOP_EXEC = RAJA::omp_parallel_for_exec;
  using REDUCE_POL = RAJA::omp_reduce;

  // Populate arr with values, calculate basic sum solution.
  
  double a_red_sol = 0;
  for (int i=0; i < Nsq; i++){
    a[i] = i * 0.1;
    a_red_sol += a[i];
  }

  using EXEC_POL = 
    RAJA::KernelPolicy<
      RAJA::statement::For<1, OUTER_LOOP_EXEC,
        RAJA::statement::For<0, RAJA::loop_exec,
          RAJA::statement::Lambda<0>
        >
      >
    >;

  //Current Implementation
  {
  std::cout << "\n\n RAJA::ReduceSum example...\n";
  
  RAJA::ReduceSum<REDUCE_POL, DATA_TYPE> work_sum(0);
  DATA_TYPE race_sum(0);

  RAJA::kernel<EXEC_POL>(
      RAJA::make_tuple(
        RAJA::TypedRangeSegment<int>(0, N),
        RAJA::TypedRangeSegment<int>(0, N)
      ),

      [=, &race_sum](int i, int j) {
         work_sum += a[i * N + j]; 
         race_sum += a[i * N + j];
      }
    );

  std::cout << "Seq Solution       : " << a_red_sol << "\n";
  std::cout << "ReduceSum Solution : "<< work_sum.get() << "\n";
  std::cout << "Race Sum Solution  : "<< race_sum << "\n";
  }

  //Param Implementation
  {
  std::cout << "\n\n RAJA::expt::Reduce example...\n";
  
  RAJA::ReduceSum<REDUCE_POL, DATA_TYPE> work_sum(0);
  DATA_TYPE race_sum(0);
  DATA_TYPE expt_sum(0);

  //using EXPT_REDUCE = RAJA::expt::Reduce< RAJA::operators::plus<DATA_TYPE> >;

  RAJA::kernel_param<EXEC_POL>(
      RAJA::make_tuple(
        RAJA::TypedRangeSegment<int>(0, N),
        RAJA::TypedRangeSegment<int>(0, N)
      ),

      RAJA::make_tuple(
        DATA_TYPE(0),
        //EXPT_REDUCE(&expt_sum)
        RAJA::expt::Reduce< RAJA::operators::plus >(&expt_sum)
      ),

      [=, &race_sum](int i, int j
         , DATA_TYPE& r //) {
         , DATA_TYPE& r2 ) {
         //) {
         work_sum += a[i * N + j]; 
         race_sum += a[i * N + j];
         //r += a[i * N + j];
      }
    );

  std::cout << "Seq Solution       : " << a_red_sol << "\n";
  std::cout << "ReduceSum Solution : "<< work_sum.get() << "\n";
  std::cout << "Race Sum Solution  : "<< race_sum << "\n";
  std::cout << "Expt Sun Solution  : "<< expt_sum << "\n";
  }
 
  return EXIT_SUCCESS;
}
