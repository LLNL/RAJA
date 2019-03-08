#include<RAJA/RAJA.hpp>

#define MAT2D(r,c,size) r*size+c

using namespace RAJA::statement;
using RAJA::seq_exec;

using MatMulPolicy = RAJA::KernelPolicy<
      For<
        0,seq_exec,
        For<
          1, seq_exec,
            For<2, seq_exec,Lambda<0>>
        >
      >
    >;

template<typename Policy, std::size_t... ends, typename Kernel>
[[clang::jit]] void affine_jit_kernel_full(Kernel&& kernel){
  static auto rs =
    camp::make_tuple(
        RAJA::RangeSegment(0,ends)...
    );
  RAJA::kernel<Policy>(
      rs,
      std::forward<Kernel>(kernel)
  );
}

template<std::size_t... ends, typename Kernel>
[[clang::jit]] void affine_jit_kernel(Kernel&& kernel){
  static auto rs =
    camp::make_tuple(
        RAJA::RangeSegment(0,ends)...
    );
  RAJA::kernel<MatMulPolicy>(
      rs,
      std::forward<Kernel>(kernel)
  );
}

template<std::size_t size>
[[clang::jit]] void affine_jit_kernel_simplified(float* out_matrix, float* input_matrix1, float* input_matrix2){
  static auto rs = 
      camp::make_tuple(
        RAJA::RangeSegment(0,size),
        RAJA::RangeSegment(0,size),
        RAJA::RangeSegment(0,size)
      );
  RAJA::kernel<MatMulPolicy>(
      rs,
      [=](const std::size_t i,const std::size_t j,const std::size_t k){
        out_matrix[MAT2D(i,j,size)] += input_matrix1[MAT2D(i,k,size)] * input_matrix2[MAT2D(k,j,size)];
      }  
  );
}

template<typename Policy, typename... Args, typename Kernel>
void affine_kernel(Kernel&& k, Args... ends){
  RAJA::kernel<Policy>(
      camp::make_tuple(
        RAJA::RangeSegment(0,ends)...
      ),
      k
  );
}

#ifndef USE_JIT 
#ifndef NO_JIT
#define USE_JIT
#endif // NO_JIT
#endif // USE_JIT

int main(int argc, char* argv[]){
  std::size_t size = atoi(argv[1]);
  srand(time(nullptr));
  std::cout << "Size is "<<size<<"\n";
  const long repeats = 500000000;
  float* out_matrix = (float*)malloc(sizeof(float)*size*size);
  float* input_matrix1 = (float*)malloc(sizeof(float)*size*size);
  float* input_matrix2 = (float*)malloc(sizeof(float)*size*size);
  for(int r = 0; r < size * size; r++){
    input_matrix1[r] = (1.0f*rand())/RAND_MAX;
    input_matrix2[r] = (1.0f*rand())/RAND_MAX;
  } 
  for(long rep = 0; rep<repeats;rep++){

#ifdef USE_JIT
    affine_jit_kernel_simplified<
       size
     >(
        out_matrix, input_matrix1, input_matrix2
      );
#endif 
    
#ifdef NO_JIT
    affine_kernel<RAJA::KernelPolicy<
      For<
        0,seq_exec,
        For<
          1, seq_exec,
            For<2, seq_exec,Lambda<0>>
        >
      >
    >>(
        [=](const std::size_t i, const std::size_t j, const std::size_t k){
          out_matrix[MAT2D(i,j,size)] += input_matrix1[MAT2D(i,k,size)] * input_matrix2[MAT2D(k,j,size)];
        },
        size,
        size,
        size
      );
#endif
  } // end for loop
}
