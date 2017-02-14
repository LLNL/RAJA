#ifndef RAJA_Stream_HXX
#define RAJA_Stream_HXX


//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// copyright (c) 2016, lawrence livermore national security, llc.
//
// produced at the lawrence livermore national laboratory
//
// llnl-code-689114
//
// all rights reserved.
//
// this file is part of raja.
//
// for additional details, please also read raja/license.
//
// redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// * redistributions of source code must retain the above copyright notice,
//   this list of conditions and the disclaimer below.
//
// * redistributions in binary form must reproduce the above copyright notice,
//   this list of conditions and the disclaimer (as noted below) in the
//   documentation and/or other materials provided with the distribution.
//
// * neither the name of the llns/llnl nor the names of its contributors may
//   be used to endorse or promote products derived from this software without
//   specific prior written permission.
//
// this software is provided by the copyright holders and contributors "as is"
// and any express or implied warranties, including, but not limited to, the
// implied warranties of merchantability and fitness for a particular purpose
// are disclaimed. in no event shall lawrence livermore national security,
// llc, the u.s. department of energy or contributors be liable for any
// direct, indirect, incidental, special, exemplary, or consequential
// damages  (including, but not limited to, procurement of substitute goods
// or services; loss of use, data, or profits; or business interruption)
// however caused and on any theory of liability, whether in contract,
// strict liability, or tort (including negligence or otherwise) arising
// in any way out of the use of this software, even if advised of the
// possibility of such damage.
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "RAJA/config.hxx"
#include "RAJA/forall_generic.hxx"
#include "RAJA/forall_context.hxx"

#include "RAJA/int_datatypes.hxx"

#include <tbb/task_group.h>
#ifdef RAJA_ENABLE_CUDA
#include "cuda.h"
#endif //RAJA_ENABLE_CUDA
namespace RAJA
{

class StreamPoolNoGPU {
public:
  template<typename LB>
  void run(int stream_num, LB&& body){

    streams[stream_num].run(body);
  }
  void wait(int stream_num){
    streams[stream_num].wait();
  }
  void operator=(StreamPoolNoGPU const&);
  StreamPoolNoGPU(size_t num_streams){
    streams = new tbb::task_group[num_streams];
  }
private:
  tbb::task_group* streams;  
};


// This function is mainly meant for documentation, so you can know which arguments meant what
constexpr int StreamIndex(int num){
  return num;
}
#ifdef RAJA_ENABLE_CUDA
// TODO: replace with, well, real logic
constexpr bool is_gpu_policy(cuda_exec_base in){
  return true;
}
template<int size>
constexpr bool is_gpu_policy(cuda_exec<size> in){
  return true;
}
constexpr bool is_gpu_policy(omp_parallel_for_exec in){
  return false;
}
constexpr bool is_gpu_policy(seq_exec in){
  return false;
}
//TODO: forgive me Bjarne, for I have sinned
template<typename InnerIter>
struct iterWithStream {
  InnerIter my_iter;
  cudaStream_t stream;
  //std::result_of<my_iter.begin()>::type begin(){
  //  return my_iter.begin();
  //}
  //std::result_of<my_iter.end()>::type end(){
  //  return my_iter.end();
  //}
  auto begin() -> decltype(my_iter.begin()){
    return my_iter.begin();
  }
  auto end() -> decltype(my_iter.end()){
    return my_iter.end();
  }
};
template<typename wrappee>
iterWithStream<wrappee> makeStreamIter(wrappee in, cudaStream_t stream){
    iterWithStream<wrappee> out;
    out.my_iter = in;
    out.my_stream = stream;
    return out;
}
class StreamPool {
public:
 
  template<typename Policy, typename LOOP_BODY, typename... Args>
  typename std::enable_if<std::is_base_of<RAJA::cuda_exec_base,Policy>::value,void>::type safeCPUCall(LOOP_BODY body, Args... args){
    //this function intentionally blank
  }
  template<typename Policy, typename LOOP_BODY, typename... Args>
  typename std::enable_if<!std::is_base_of<RAJA::cuda_exec_base,Policy>::value,void>::type safeCPUCall(LOOP_BODY body, Args... args){
    body(args...);
  }
  template<typename Policy, typename LOOP_BODY, typename... Args>
  typename std::enable_if<std::is_base_of<RAJA::cuda_exec_base,Policy>::value,void>::type safeGPUCall(LOOP_BODY body, Args... args){
    body(args...);
  }
  template<typename Policy, typename LOOP_BODY, typename... Args>
  typename std::enable_if<!std::is_base_of<RAJA::cuda_exec_base,Policy>::value,void>::type safeGPUCall(LOOP_BODY body, Args... args){
    //this function intentionally blank
  }
  template<typename Policy>
  typename std::enable_if<std::is_base_of<RAJA::cuda_exec_base,Policy>::value,cudaStream_t>::type getPolicy(int stream_num){
    return getGPUStream(stream_num);
  }
  template<typename Policy>
  typename std::enable_if<!std::is_base_of<RAJA::cuda_exec_base,Policy>::value,cudaStream_t>::type getPolicy(int stream_num){
    return NULL;
  }
  enum ExecutionSpace{
    CPU,
    GPU
  }; //TODO: can we have multiple GPUs per stream here? Also, more generally, it's not that simple
  void gpu_nonblocking_sync(int stream_num){
     tbb_streams[stream_num].run([=](){
       cudaStreamSynchronize(cuda_streams[stream_num]);
     });
  }
  template<typename EXEC_POLICY_T, typename LB>
  void run(int stream_num, LB body){
    ExecutionSpace last_space = last_execution_space[stream_num];
    ExecutionSpace next_space = is_gpu_policy(EXEC_POLICY_T()) ? ExecutionSpace::GPU : ExecutionSpace::CPU;
    if((last_space == ExecutionSpace::GPU) && (next_space == ExecutionSpace::CPU)){
      gpu_nonblocking_sync(stream_num);
    }
    tbb_streams[stream_num].run(body);
    last_execution_space[stream_num] = next_space;
  }
  template <typename EXEC_POLICY_T, typename... Args>
  RAJA_INLINE typename std::enable_if<!std::is_base_of<cuda_exec_base,EXEC_POLICY_T>::value,void>::type forall(int stream_num, Args... args)
  {
    std::cout<<"Queueing forall on "<<stream_num<<" on CPU\n";
    if(last_execution_space[stream_num] == ExecutionSpace::GPU){
      gpu_nonblocking_sync(stream_num);
    }
    tbb_streams[stream_num].run([=](){
      std::cout<<"Running forall on "<<stream_num<<" on CPU\n";
      RAJA::forall<EXEC_POLICY_T>(args...);
    });
    last_execution_space[stream_num] = ExecutionSpace::CPU;
  }
  template <typename EXEC_POLICY_T, typename... Args>
  RAJA_INLINE typename std::enable_if<std::is_base_of<cuda_exec_base,EXEC_POLICY_T>::value,void>::type forall(int stream_num, Args... args)
  {
    std::cout<<"Queueing forall on "<<stream_num<<" on CPU\n";
    tbb_streams[stream_num].run([=](){
      std::cout<<"Running forall on "<<stream_num<<" on GPU\n";
      RAJA::runtime_context::forall(cuda_stream_exec<256>(getGPUStream(stream_num)),args...);
    });
    last_execution_space[stream_num] = ExecutionSpace::GPU;
  }
  const cudaStream_t getGPUStream(const int stream_num) const{
    return cuda_streams[stream_num];
  }
  template <typename EXEC_POLICY_T, typename... Args>
  RAJA_INLINE void forall_Icount(const int stream_num, Args... args){
    ExecutionSpace last_space = last_execution_space[stream_num];
    ExecutionSpace next_space = is_gpu_policy(EXEC_POLICY_T()) ? ExecutionSpace::GPU : ExecutionSpace::CPU;
    if((last_space == ExecutionSpace::GPU) && (next_space == ExecutionSpace::CPU)){
      gpu_nonblocking_sync(stream_num);
    }
    RAJA::forall_Icount(args...);
    last_execution_space[stream_num] = next_space;
  }
  void wait(int stream_num){
    ExecutionSpace last_space = last_execution_space[stream_num];
    if(last_space == ExecutionSpace::GPU){
      cudaStreamSynchronize(cuda_streams[stream_num]);
    }
    tbb_streams[stream_num].wait();
  }
  StreamPool(size_t num_streams){
    tbb_streams = new tbb::task_group[num_streams];
    last_execution_space = new ExecutionSpace[num_streams];
    cuda_streams = new cudaStream_t[num_streams]; 
    for(int i=0;i<num_streams;i++){
      //tbb_streams[i].run([=](){
        last_execution_space[i] = ExecutionSpace::CPU;
        cudaStreamCreate(&cuda_streams[i]);
      //});
    }
  }
  ~StreamPool(){
#ifndef __CUDA_ARCH__
    cudaDeviceSynchronize();
    std::cout<<"Synchronizing device\n";
#endif
  }
private:
  tbb::task_group* tbb_streams;  
  ExecutionSpace* last_execution_space; 
  cudaStream_t* cuda_streams;
};
#endif //RAJA_ENABLE_CUDA

} // close brace for namespace RAJA

#endif //include guard
