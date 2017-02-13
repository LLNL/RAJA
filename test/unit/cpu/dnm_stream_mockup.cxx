#include <RAJA/RAJA.hxx>
#include <RAJA/Stream.hxx>

int main(){
  constexpr int dataSize   = 10000;
  constexpr int splitPoint =  8000;
  float *cpuA, *cpuB, *cpuC;
  float *gpuA, *gpuB, *gpuC;
  cpuA = (float*)malloc(sizeof(float)*dataSize);
  cpuB = (float*)malloc(sizeof(float)*dataSize);
  cpuC = (float*)malloc(sizeof(float)*dataSize);
  cudaMalloc((void**)&gpuA,sizeof(float)*dataSize);
  cudaMalloc((void**)&gpuB,sizeof(float)*dataSize);
  cudaMalloc((void**)&gpuC,sizeof(float)*dataSize);
  RAJA::StreamPool dog_pool(8);
  dog_pool.forall<RAJA::cuda_exec<256>>(RAJA::StreamIndex(0), 0, splitPoint, [=] __device__(int i){
    gpuA[i] = i * 1.0;
    gpuB[i] = i * 2.0;
    if(i==5){
        printf("%f %f %f\n", gpuA[i], gpuB[i],gpuC[i]);
    }
  });
  dog_pool.forall<RAJA::omp_parallel_for_exec>(RAJA::StreamIndex(1), splitPoint, dataSize, [=] (int i){
    cpuA[i] = i * 1.0;
    cpuB[i] = i * 2.0;
    if(i==(splitPoint+5)){
        printf("%f %f %f\n", cpuA[i], cpuB[i],cpuC[i]);
    }
  });
  dog_pool.forall<RAJA::cuda_exec<256>>(RAJA::StreamIndex(0), 0, splitPoint, [=] __device__(int i){
    gpuC[i] = gpuA[i] * gpuB[i];
    if(i==5){
        printf("%f %f %f\n", gpuA[i], gpuB[i],gpuC[i]);
    }
  });
  dog_pool.forall<RAJA::omp_parallel_for_exec>(RAJA::StreamIndex(1), splitPoint, dataSize, [=] (int i){
    cpuC[i] = cpuA[i] * cpuB[i];
    if(i==(splitPoint+5)){
        printf("%f %f %f\n", cpuA[i], cpuB[i],cpuC[i]);
    }
  });
  dog_pool.run<RAJA::seq_exec>(0,[=](){
    cudaMemcpyAsync(cpuC,gpuC,sizeof(float)*(splitPoint),cudaMemcpyDeviceToHost,dog_pool.getGPUStream(0));
  });
  dog_pool.wait(1);
  dog_pool.run<RAJA::seq_exec>(0,[=](){
    printf("%f, %f, %f, %f, %f, %f\n",cpuC[0],cpuC[1],cpuC[2],cpuC[dataSize-3],cpuC[dataSize-2],cpuC[dataSize-1]);
  });
}


