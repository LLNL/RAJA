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
      for(int i = 0;i < len;i ++){

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





int main(){


  using policy1_HOST = RAJA::LPolicy<RAJA::HOST, RAJA::loop_exec>;
#ifdef RAJA_ENABLE_CUDA
  using policy1_DEVICE = RAJA::LPolicy<RAJA::DEVICE, RAJA::cuda_block_x_loop >;
#else
  using policy1_DEVICE = RAJA::LPolicy<RAJA::DEVICE, RAJA::loop_exec>;
#endif
  using policy1 = camp::list<policy1_HOST, policy1_DEVICE>;


  using policy2_HOST = RAJA::LPolicy<RAJA::HOST, RAJA::loop_exec>;
#ifdef RAJA_ENABLE_CUDA
  using policy2_DEVICE = RAJA::LPolicy<RAJA::DEVICE, RAJA::cuda_thread_x_loop >;
#else
  using policy2_DEVICE = RAJA::LPolicy<RAJA::DEVICE, RAJA::loop_exec>;
#endif
  using policy2 = camp::list<policy2_HOST, policy2_DEVICE>;


  for(int exec_place = 0;exec_place < 2;++ exec_place)
  {
    RAJA::ExecPlace select_cpu_or_gpu = (RAJA::ExecPlace)exec_place;
    //auto select_cpu_or_gpu = RAJA::HOST;
    //auto select_cpu_or_gpu = RAJA::DEVICE;

    int N = 5;

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
    RAJA::launch(
      select_cpu_or_gpu,
      camp::make_tuple(
         RAJA::Resources<RAJA::HOST>(RAJA::Threads(N)),
         RAJA::Resources<RAJA::DEVICE>(RAJA::Teams(N), RAJA::Threads(N)) ),
      [=] RAJA_HOST_DEVICE (RAJA::LaunchContext ctx)
    {

      RAJA::loop<policy1>(ctx, RAJA::RangeSegment(0, N), [=] (int i){


        // do a matrix triangular pattern
        RAJA::loop<policy2>(ctx, RAJA::RangeSegment(i, N), [=] (int j){

          printf("i=%d, j=%d\n", i, j);

        }); // loop j


      }); // loop i


    }); // kernel


  }
}
