//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include <cstdlib>
#include <iostream>
#include <cstring>
#include <cmath>

#include "memoryManager.hpp"

#include "RAJA/RAJA.hpp"

/*
 *  RAJA Teams Example: 
 *  Matrix-matrix multiplication with shared memory
 */

/*
  Define number of threads in x and y dimensions of a CUDA thread block
*/
#if defined(RAJA_ENABLE_CUDA)
#define CUDA_BLOCK_SIZE 16
#endif

#if defined(__CUDA_ARCH__)
#define TEAM_SHARED __shared__
#define TEAM_SYNC() __syncthreads()
#else
#define TEAM_SHARED
#define TEAM_SYNC() 
#endif

//
// Define dimensionality of matrices.
//
const int DIM = 2;

//
// Define macros to simplify row-col indexing (non-RAJA implementations only)
//
// _matmult_macros_start
#define A(r, c) A[c + N * r]
#define B(r, c) B[c + N * r]
#define C(r, c) C[c + N * r]
// _matmult_macros_end

/*
  Define CUDA matrix multiplication kernel for comparison to RAJA version
*/
#if defined(RAJA_ENABLE_CUDA)
__global__ void matMultKernel(int N, double* C, double* A, double* B)
{
  // Block row and column
  const int by = blockIdx.y;
  const int bx = blockIdx.x;

   // Each thread computes one element of Csub
  // by accumulating results into Cvalue
  double Cvalue(0.0);

  // Thread row and column within local Csub
  const int ty = threadIdx.y; //local row
  const int tx = threadIdx.x; //local column

  const int row = by * CUDA_BLOCK_SIZE + ty;  // Matrix row index
  const int col = bx * CUDA_BLOCK_SIZE + tx;  // Matrix column index

  // Shared memory used to store Asub and Bsub respectively
  __shared__ double As[CUDA_BLOCK_SIZE][CUDA_BLOCK_SIZE];
  __shared__ double Bs[CUDA_BLOCK_SIZE][CUDA_BLOCK_SIZE];

  // Loop over all the sub-matrices of A and B that are
  // required to compute Csub
  // Multiply each pair of sub-matrices together
  // and accumulate the results
  for (int m = 0; m < (N / CUDA_BLOCK_SIZE); ++m) {

    // Load Asub and Bsub from device memory to shared memory
    // Each thread loads one element of each sub-matrix
    As[ty][tx] = A[row*N + m*CUDA_BLOCK_SIZE + tx];
    Bs[ty][tx] = B[(m*CUDA_BLOCK_SIZE + ty)*N + col];

    // Synchronize to make sure the sub-matrices are loaded
    // before starting the computation
    __syncthreads();

    // Multiply Asub and Bsub together
    for (int e = 0; e < CUDA_BLOCK_SIZE; ++e)
      Cvalue += As[ty][e] * Bs[e][tx];

    // Synchronize to make sure that the preceding
    // computation is done before loading two new
    // sub-matrices of A and B in the next iteration
    __syncthreads();
  }

  // Write Csub to device memory
  // Each thread writes one element
  C[col + N*row] = Cvalue;

}
#endif

//
// Functions for checking results
//
template <typename T>
void checkResult(T *C, int N);

template <typename T>
void checkResult(RAJA::View<T, RAJA::Layout<DIM>> Cview, int N);

//
// Functions for printing results
//
template <typename T>
void printResult(T *C, int N);

template <typename T>
void printResult(RAJA::View<T, RAJA::Layout<DIM>> Cview, int N);


namespace RAJA {

  enum ExecPlace {
    HOST,
    DEVICE
  };

  template<ExecPlace EXEC_PLACE, typename POLICY>
  struct LPolicy{
    static constexpr ExecPlace exec_place = EXEC_PLACE;
    using policy_t = POLICY;
  };

  struct Teams{
    int value[3];

    Teams() : value{1,1,1}{}

    Teams(int i) : value{i,1,1}{}

    Teams(int i, int j) : value{i,j,1}{}

    Teams(int i, int j, int k) : value{i,j,k}{}
  };

  struct Threads{
    int value[3];

    Threads() : value{1,1,1}{}

    Threads(int i) : value{i,1,1}{}

    Threads(int i, int j) : value{i,j,1}{}

    Threads(int i, int j, int k) : value{i,j,k}{}
  };

  struct Lanes{
    int value;

    Lanes() : value(0){}

    Lanes(int i) : value(i){}
  };

  class ResourceBase {
  public:
    Teams teams;
    Threads threads;
    Lanes lanes;
  };

  class LaunchContext : public ResourceBase {
    public:
      ExecPlace exec_place;

      LaunchContext(ResourceBase const &base, ExecPlace place) :
        ResourceBase(base),
        exec_place(place)
      {}
  };

  template<ExecPlace EXEC_PLACE>
  class Resources : public ResourceBase {
  public:
    static constexpr ExecPlace exec_place = EXEC_PLACE;

    Resources() : ResourceBase()
    {}

    template<typename ... ARGS>
    explicit Resources(ARGS const &... args) : ResourceBase()
    {
      camp::sink( apply(args)... );
    }

  private:
    RAJA_HOST_DEVICE
    RAJA_INLINE
    Teams apply(Teams const &a){
      return(teams = a);
    }

    RAJA_HOST_DEVICE
    RAJA_INLINE
    Threads apply(Threads const &a){
      return(threads = a);
    }

    RAJA_HOST_DEVICE
    RAJA_INLINE
    Lanes apply(Lanes const &a){
      return(lanes = a);
    }

  };

  template<typename RESOURCE>
  struct LaunchPlaceSwitchboard;

  template<>
  struct LaunchPlaceSwitchboard<Resources<HOST>>{
    template<typename BODY>
    static
    void exec(ExecPlace place, LaunchContext const &ctx, BODY const &body){
      printf("Launching HOST Kernel\n");
      body(ctx);
      printf("Leaving HOST Kernel\n");
    }
  };


  template<typename BODY>
  __launch_bounds__(128, 1)
  __global__ void launch_global_fcn(LaunchContext ctx, BODY body){
    //printf("Entering global function\n");
    body(ctx);
    //printf("Leaving global function\n");
  }

  template<>
  struct LaunchPlaceSwitchboard<Resources<DEVICE>>{
    template<typename BODY>
    static
    void exec(ExecPlace place, LaunchContext const &ctx, BODY const &body){
      //printf("Not implement yet!\n");

      dim3 blocks;
      dim3 threads;

      blocks.x = ctx.teams.value[0];
      blocks.y = ctx.teams.value[1];
      blocks.z = ctx.teams.value[2];

      threads.x = ctx.threads.value[0];
      threads.y = ctx.threads.value[1];
      threads.z = ctx.threads.value[2];

      printf("Launching CUDA Kernel with blocks=%d,%d,%d   thread=%d,%d,%d\n",
          ctx.teams.value[0],
          ctx.teams.value[1],
          ctx.teams.value[2],
          ctx.threads.value[0],
          ctx.threads.value[1],
          ctx.threads.value[2]);

      launch_global_fcn<<<blocks, threads>>>(ctx, body);
      cudaDeviceSynchronize();
      printf("Leaving CUDA Kernel\n");
    }
  };

  template<typename RESOURCE_TUPLE, camp::idx_t I, camp::idx_t IMAX>
  struct LaunchPlaceExtractor {

      template<typename BODY>
      static
      void launch(ExecPlace place, RESOURCE_TUPLE const &resources, BODY const &body){

        using resource_t = camp::at_v<typename RESOURCE_TUPLE::TList, I>;

        if(place == resource_t::exec_place){
          auto const &resource = camp::get<I>(resources);

          LaunchContext ctx(resource, place);

          LaunchPlaceSwitchboard<resource_t>::exec(place, ctx, body);
        }
        else{

          LaunchPlaceExtractor<RESOURCE_TUPLE, I+1, IMAX>::launch(place, resources, body);
        }

      }
  };


  template<typename RESOURCE_TUPLE, camp::idx_t IMAX>
  struct LaunchPlaceExtractor<RESOURCE_TUPLE, IMAX, IMAX> {
      template<typename BODY>
      static
      void launch(ExecPlace place, RESOURCE_TUPLE const &resources, BODY const &body){
        printf("Failed to find resource requirements for execution place %d\n", (int)place);
      }

  };

  template<typename RESOURCES, typename BODY>
  void launch(ExecPlace place, RESOURCES const & resources, BODY const &body){
    LaunchPlaceExtractor<RESOURCES, 0, camp::size<typename RESOURCES::TList>::value>::launch(place, resources, body);
  }



  template<typename POLICY, typename SEGMENT>
  struct LoopExecute;

  template<typename SEGMENT>
  struct LoopExecute<loop_exec, SEGMENT>{

    template<typename BODY>
    static
    RAJA_HOST_DEVICE
    void exec(LaunchContext const &ctx, SEGMENT const &segment, BODY const &body){

      // block stride loop
      int len = segment.end()-segment.begin();
      for(int i = 0;i < len; i++){

        body(*(segment.begin()+i));

      }

    }

  };

  template<typename SEGMENT>
  struct LoopExecute<cuda_thread_x_loop, SEGMENT>{

    template<typename BODY>
    static
    RAJA_DEVICE
    void exec(LaunchContext const &ctx, SEGMENT const &segment, BODY const &body){

      int len = segment.end()-segment.begin();

      for(int i = threadIdx.x;i < len;i += blockDim.x){
        body(*(segment.begin()+i));
      }

    }

  };

  template<typename SEGMENT>
  struct LoopExecute<cuda_thread_y_loop, SEGMENT>{

    template<typename BODY>
    static
    RAJA_DEVICE
    void exec(LaunchContext const &ctx, SEGMENT const &segment, BODY const &body){

      int len = segment.end()-segment.begin();

      for(int i = threadIdx.y;i < len;i += blockDim.y){
        body(*(segment.begin()+i));
      }

    }

  };

  template<typename SEGMENT>
  struct LoopExecute<cuda_block_x_loop, SEGMENT>{

    template<typename BODY>
    static
    RAJA_DEVICE
    void exec(LaunchContext const &ctx, SEGMENT const &segment, BODY const &body){

      int len = segment.end()-segment.begin();

      for(int i = blockIdx.x;i < len;i+= gridDim.x){
        body(*(segment.begin()+i));
      }

    }

  };

  template<typename SEGMENT>
  struct LoopExecute<cuda_block_x_direct, SEGMENT>{

    template<typename BODY>
    static
    RAJA_DEVICE
    void exec(LaunchContext const &ctx, SEGMENT const &segment, BODY const &body){

      int len = segment.end()-segment.begin();
      {
        const int i = blockIdx.x; 
        body(*(segment.begin()+i));
      }

    }

  };

  template<typename SEGMENT>
  struct LoopExecute<cuda_block_y_direct, SEGMENT>{

    template<typename BODY>
    static
    RAJA_DEVICE
    void exec(LaunchContext const &ctx, SEGMENT const &segment, BODY const &body){

      int len = segment.end()-segment.begin();
      {
        const int i = blockIdx.y; 
        body(*(segment.begin()+i));
      }

    }

  };



  template<typename POLICY_LIST, camp::idx_t IDX, camp::idx_t MAX_IDX>
  struct LoopPlaceSwitchboard{
    template<typename SEGMENT, typename BODY>
    static
    RAJA_HOST_DEVICE
    void exec(LaunchContext const &ctx, SEGMENT const &segment, BODY const &body){
      if(camp::at_v<POLICY_LIST, IDX>::exec_place == ctx.exec_place){
        LoopExecute<typename camp::at_v<POLICY_LIST, IDX>::policy_t, SEGMENT>::exec(ctx, segment, body);
      }
      else{
        LoopPlaceSwitchboard<POLICY_LIST, IDX+1, MAX_IDX>::exec(ctx, segment, body);
      }
    }
  };

  template<typename POLICY_LIST, camp::idx_t MAX_IDX>
  struct LoopPlaceSwitchboard<POLICY_LIST, MAX_IDX, MAX_IDX>
  {
    template<typename SEGMENT, typename BODY>
    static
    RAJA_HOST_DEVICE
    void exec(LaunchContext const &ctx, SEGMENT const &segment, BODY const &body){
      printf("whoops!");
    }
  };


  template<typename POLICY_LIST, typename SEGMENT, typename BODY>
  RAJA_HOST_DEVICE
  void loop(LaunchContext const &ctx, SEGMENT const &seg, BODY const &body){


    LoopPlaceSwitchboard<POLICY_LIST, 0, camp::size<POLICY_LIST>::value>::exec(ctx, seg, body);


  }

} // namespace RAJA


  using policy1_HOST = RAJA::LPolicy<RAJA::HOST, RAJA::loop_exec>;
#ifdef RAJA_ENABLE_CUDA
  using policy1x_DEVICE = RAJA::LPolicy<RAJA::DEVICE, RAJA::cuda_block_x_direct >;
  using policy1y_DEVICE = RAJA::LPolicy<RAJA::DEVICE, RAJA::cuda_block_y_direct >;
#else
  using policy1_DEVICE = RAJA::LPolicy<RAJA::DEVICE, RAJA::loop_exec>;
#endif
  using outer0 = camp::list<policy1_HOST, policy1x_DEVICE>;
  using outer1 = camp::list<policy1_HOST, policy1y_DEVICE>;

  using policy2_HOST = RAJA::LPolicy<RAJA::HOST, RAJA::loop_exec>;
#ifdef RAJA_ENABLE_CUDA
  using policy2x_DEVICE = RAJA::LPolicy<RAJA::DEVICE, RAJA::cuda_thread_x_loop >;
  using policy2y_DEVICE = RAJA::LPolicy<RAJA::DEVICE, RAJA::cuda_thread_y_loop >;
#else
  using policy2_DEVICE = RAJA::LPolicy<RAJA::DEVICE, RAJA::loop_exec>;
#endif
  using team0 = camp::list<policy2_HOST, policy2x_DEVICE>;
  using team1 = camp::list<policy2_HOST, policy2y_DEVICE>;


int main(){

  //N is number of blocks in each matrix
  const int NBlocks = 4;
  const int NThreads = CUDA_BLOCK_SIZE;
  const int N = CUDA_BLOCK_SIZE*NBlocks;

//
// Allocate and initialize matrix data.
//
  double *A = memoryManager::allocate<double>(N * N);
  double *B = memoryManager::allocate<double>(N * N);
  double *C = memoryManager::allocate<double>(N * N);

  for (int row = 0; row < N; ++row) {
    for (int col = 0; col < N; ++col) {
      A(row, col) = row;
      B(row, col) = col;
    }
  }

  std::cout << "\n Running RAJA-Teams V2-version of matrix multiplication...\n";

  for(int exec_place = 0;exec_place < 2;++ exec_place)
  {
    RAJA::ExecPlace select_cpu_or_gpu = (RAJA::ExecPlace)exec_place;
    //auto select_cpu_or_gpu = RAJA::HOST;
    //auto select_cpu_or_gpu = RAJA::DEVICE;

    /*
     * launch just starts a "kernel" it's doesn't provide any looping.
     *
     * The first argument determines which policy should be executed,
     *
     * The second argument is the number of teams+threads needed for each of the
     * policies.
     *
     * Third argument is the lambda for the policy.
     *
     *
     * The lambda takes a "resource" object, which has the teams+threads and
     * policy selection information.
     */

    //Set up Teams/Threads

    RAJA::launch(
      select_cpu_or_gpu,
      camp::make_tuple(
        RAJA::Resources<RAJA::HOST>(RAJA::Threads(NBlocks, NBlocks)),
        RAJA::Resources<RAJA::DEVICE>(RAJA::Teams(NBlocks, NBlocks), RAJA::Threads(NThreads, NThreads)) ),
      [=] RAJA_HOST_DEVICE (RAJA::LaunchContext ctx)
    {

      //
      //Loop over teams
      //
      RAJA::loop<outer1>(ctx, RAJA::RangeSegment(0, NBlocks), [&] (int by) {
          RAJA::loop<outer0>(ctx, RAJA::RangeSegment(0, NBlocks), [&] (int bx) {

              
              TEAM_SHARED double As[NThreads][NThreads];
              TEAM_SHARED double Bs[NThreads][NThreads];
              TEAM_SHARED double Cs[NThreads][NThreads];
              
              //Team Parallel loop
              RAJA::loop<team1>(ctx, RAJA::RangeSegment(0, NThreads), [&] (int ty) {
                  RAJA::loop<team0>(ctx, RAJA::RangeSegment(0, NThreads), [&] (int tx) {
                      Cs[ty][tx] = 0.0; 
                    });
                });

              //Slide Across matrix
              for (int m = 0; m < (N / CUDA_BLOCK_SIZE); ++m) {

                RAJA::loop<team1>(ctx, RAJA::RangeSegment(0, NThreads), [&] (int ty) {
                    RAJA::loop<team0>(ctx, RAJA::RangeSegment(0, NThreads), [&] (int tx) {

                        const int row = by * NThreads + ty;  // Matrix row index
                        const int col = bx * NThreads + tx;  // Matrix column index
                        
                        As[ty][tx] = A[row*N + m*NThreads + tx];
                        Bs[ty][tx] = B[(m*NThreads + ty)*N + col];

                      }); 
                  }); 

                TEAM_SYNC();
                
                RAJA::loop<team1>(ctx, RAJA::RangeSegment(0, NThreads), [&] (int ty) {
                    RAJA::loop<team0>(ctx, RAJA::RangeSegment(0, NThreads), [&] (int tx) {
                        
                        for(int e=0; e<NThreads; ++e){
                          Cs[ty][tx] += As[ty][e] * Bs[e][tx];  
                        }

                      });
                  });
                TEAM_SYNC();                
              }//slide across matrix 

              
              RAJA::loop<team1>(ctx, RAJA::RangeSegment(0, NThreads), [&] (int ty) {
                  RAJA::loop<team0>(ctx, RAJA::RangeSegment(0, NThreads), [&] (int tx) {

                      const int row = by * NThreads + ty;  // Matrix row index
                      const int col = bx * NThreads + tx;  // Matrix column index
                      C[col + N*row] = Cs[ty][tx];
                    });
                });

            });       
        });       

    }); // kernel

    checkResult<double>(C, N);
    printf("\n"); 
  }



}


//
// Functions to check result and report P/F.
//
template <typename T>
void checkResult(T* C, int N)
{
  bool match = true;
  for (int row = 0; row < N; ++row) {
    for (int col = 0; col < N; ++col) {
      if ( std::abs( C(row, col) - row * col * N ) > 10e-12 ) {
        match = false;
      }
    }
  }
  if ( match ) {
    std::cout << "\n\t result -- PASS\n";
  } else {
    std::cout << "\n\t result -- FAIL\n";
  }
};

template <typename T>
void checkResult(RAJA::View<T, RAJA::Layout<DIM>> Cview, int N)
{
  bool match = true;
  for (int row = 0; row < N; ++row) {
    for (int col = 0; col < N; ++col) {
      if ( std::abs( Cview(row, col) - row * col * N ) > 10e-12 ) {
        match = false;
      }
    }
  }
  if ( match ) {
    std::cout << "\n\t result -- PASS\n";
  } else {
    std::cout << "\n\t result -- FAIL\n";
  }
};

//
// Functions to print result.
//
template <typename T>
void printResult(T* C, int N)
{
  std::cout << std::endl;
  for (int row = 0; row < N; ++row) {
    for (int col = 0; col < N; ++col) {
      std::cout << "C(" << row << "," << col << ") = "
                << C(row, col) << std::endl;
    }
  }
  std::cout << std::endl;
}

template <typename T>
void printResult(RAJA::View<T, RAJA::Layout<DIM>> Cview, int N)
{
  std::cout << std::endl;
  for (int row = 0; row < N; ++row) {
    for (int col = 0; col < N; ++col) {
      std::cout << "C(" << row << "," << col << ") = "
                << Cview(row, col) << std::endl;
    }
  }
  std::cout << std::endl;
}

