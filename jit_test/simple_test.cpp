#include<RAJA/RAJA.hpp>
#include <tuple>

#define MAT2D(r,c,size) r*size+c

using namespace RAJA::statement;
using RAJA::seq_exec;

using MatMulPolicy = RAJA::KernelPolicy<
      For<
        0,seq_exec,
        For<
          1,seq_exec,
          For<
            2, seq_exec,
              For<3, seq_exec,Lambda<0>>
          >
        >
      >
    >;

template<typename Policy, long... ends, typename Kernel>
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

template<typename Policy, typename Kernel, typename... Args>
void affine_jit_kernel_difficult_helper2(Kernel&& kernel, Args... args){
   affine_jit_kernel_full<Policy,*std::end(args)...>(
       std::forward<Kernel>(kernel)
   );
}

template<typename Policy, typename TupleLike, typename Kernel, std::size_t... indices>
void affine_jit_kernel_difficult_helper(TupleLike IndexTuple, Kernel&& kernel, std::index_sequence<indices...>){
  affine_jit_kernel_difficult_helper2<Policy>(
   std::forward<Kernel>(kernel),
   camp::get<indices>(IndexTuple)...
  );
    
}

template<typename Policy, typename TupleLike, typename Kernel>
void affine_jit_kernel_difficult(TupleLike IndexTuple, Kernel&& kernel){
  affine_jit_kernel_difficult_helper<Policy,TupleLike>(
      std::forward<TupleLike>(IndexTuple),
      std::forward<Kernel>(kernel),
      std::make_index_sequence<camp::tuple_size<TupleLike>::value>()
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

template<std::size_t n_matrices, std::size_t size>
[[clang::jit]] void affine_jit_kernel_simplified(float* out_matrix, float* input_matrix1, float* input_matrix2){
  static auto rs = 
      camp::make_tuple(
        RAJA::RangeSegment(0,n_matrices),
        RAJA::RangeSegment(0,size),
        RAJA::RangeSegment(0,size),
        RAJA::RangeSegment(0,size)
      );
  RAJA::kernel<MatMulPolicy>(
      rs,
      [=](const std::size_t matrices, const std::size_t i,const std::size_t j,const std::size_t k){
        (void)matrices;
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
  (void)argc;
  std::size_t size = atoi(argv[1]);
  std::size_t batch_size = atoi(argv[2]);
  srand(time(nullptr));
  std::cout << "Size is "<<size<<", batch size "<< batch_size<<"\n";
  std::size_t repeats = 5000000000;
  if(argc>2){
     repeats = atoi(argv[3]) * 10000000;
  }
  float* out_matrix = (float*)malloc(sizeof(float)*size*size );
  float* input_matrix1 = (float*)malloc(sizeof(float)*size*size);
  float* input_matrix2 = (float*)malloc(sizeof(float)*size*size);
  for(std::size_t r = 0; r < size * size; r++){
    input_matrix1[r] = (1.0f*rand())/RAND_MAX;
    input_matrix2[r] = (1.0f*rand())/RAND_MAX;
  } 
  for(std::size_t rep = 0; rep<(repeats/batch_size);rep++){
#ifdef USE_JIT
    affine_jit_kernel_difficult<MatMulPolicy>(
        camp::make_tuple(
          RAJA::RangeSegment(0,batch_size),
          RAJA::RangeSegment(0,size),
          RAJA::RangeSegment(0,size),
          RAJA::RangeSegment(0,size)
        ),
        [=](const std::size_t matrices, const std::size_t i, const std::size_t j, const std::size_t k){
            (void)matrices;
            out_matrix[MAT2D(i,j,size)] += input_matrix1[MAT2D(i,k,size)] * input_matrix2[MAT2D(k,j,size)];
        }
    );
#endif 
    
#ifdef NO_JIT
    affine_kernel<RAJA::KernelPolicy<
      For<
        0,seq_exec,
      For<
        1,seq_exec,
        For<
          2, seq_exec,
            For<3, seq_exec,Lambda<0>>
        >
      >
    >>>(
        [=](const std::size_t matrices, const std::size_t i, const std::size_t j, const std::size_t k){
          (void)matrices;
          out_matrix[MAT2D(i,j,size)] += input_matrix1[MAT2D(i,k,size)] * input_matrix2[MAT2D(k,j,size)];
        },
        batch_size,
        size,
        size,
        size
      );
#endif
  } // end for loop
}
