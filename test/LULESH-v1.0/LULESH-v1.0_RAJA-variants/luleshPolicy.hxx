// This work was performed under the auspices of the U.S. Department of Energy by
// Lawrence Livermore National Laboratory under Contract DE-AC52-07NA27344.
//

//
//   Tiling modes for different exeuction cases (see luleshPolicy.hxx).
//
enum TilingMode
{
   Canonical,       // canonical element ordering -- single range segment
   Tiled_Index,     // canonical ordering, tiled using unstructured segments
   Tiled_Order,     // elements permuted, tiled using range segments
   Tiled_LockFree,  // tiled ordering, lock-free
   Tiled_LockFreeColor,     // tiled ordering, lock-free, unstructured
   Tiled_LockFreeColorSIMD  // tiled ordering, lock-free, range
};


// Use cases for RAJA execution patterns:

#define LULESH_SEQUENTIAL       1 /* (possible SIMD vectorization applied) */
#define LULESH_CANONICAL        2 /*  OMP forall applied to each for loop */
#define LULESH_TILE_INDEXED     3 /*  OMP Tiles defined by unstructured */
                                  //  Indexset Segment partitioning.
                                  //  One tile per segment.
#define LULESH_TILE_ORDERED     4 /*  OMP The mesh is permuted so a tile */
                                  //  is defined as a contiguous chunk
                                  //  of the iteration space. Tile per thread.
#define LULESH_TILE_TASK        5 /*  OMP Mesh chunked like Canonical, but */
                                  //  now chunks are dependency scheduled,
                                  //  reducing the need for lock constructs
#define LULESH_TILE_COLOR       6 /*  OMP Analogous to Tile_Indexed, but */
                                  //  individual array values are 
                                  //  'checker-boarded' into 'colors' to
                                  //  guarantee indpenedent data access as
                                  //  long as each 'color' of array values
                                  //  completes before executing the next color
#define LULESH_TILE_COLOR_SIMD  7 /*  Colored like USE_CASE 6, but colors */
                                  //  are permuted to be contiguous chunks,
                                  //  like LULESH_TILED_ORDERED
#define LULESH_CILK             8 /*  cilk_for applied to each loop */
#define LULESH_CUDA_CANONICAL   9 /*  CUDA launch applied to each loop */
#define LULESH_CUDA_COLOR_SIMD 10 /*  Technique 7 on GPU to avoid */
                                  //  OMP_FINE_SYNC data movement.

#ifndef USE_CASE
#define USE_CASE   LULESH_TILE_TASK
#endif

// ----------------------------------------------------
#if USE_CASE == LULESH_SEQUENTIAL 

TilingMode const lulesh_tiling_mode = Canonical;

typedef RAJA::seq_segit              Hybrid_Seg_Iter;
typedef RAJA::simd_exec              Segment_Exec;

typedef RAJA::IndexSet::ExecPolicy<Hybrid_Seg_Iter, Segment_Exec> node_exec_policy;
typedef RAJA::IndexSet::ExecPolicy<Hybrid_Seg_Iter, Segment_Exec> elem_exec_policy;
typedef RAJA::IndexSet::ExecPolicy<Hybrid_Seg_Iter, Segment_Exec> mat_exec_policy;
typedef RAJA::IndexSet::ExecPolicy<Hybrid_Seg_Iter, Segment_Exec> symnode_exec_policy;

typedef RAJA::seq_reduce reduce_policy; 

// ----------------------------------------------------
#elif USE_CASE == LULESH_CANONICAL

// Requires OMP_FINE_SYNC when run in parallel
#define OMP_FINE_SYNC 1

// AllocateTouch should definitely be used

TilingMode const lulesh_tiling_mode = Canonical;

typedef RAJA::seq_segit              Hybrid_Seg_Iter;
typedef RAJA::omp_parallel_for_exec  Segment_Exec;

typedef RAJA::IndexSet::ExecPolicy<Hybrid_Seg_Iter, Segment_Exec> node_exec_policy;
typedef RAJA::IndexSet::ExecPolicy<Hybrid_Seg_Iter, Segment_Exec> elem_exec_policy;
typedef RAJA::IndexSet::ExecPolicy<Hybrid_Seg_Iter, Segment_Exec> mat_exec_policy;
typedef RAJA::IndexSet::ExecPolicy<Hybrid_Seg_Iter, Segment_Exec> symnode_exec_policy;

typedef RAJA::omp_reduce reduce_policy;

// ----------------------------------------------------
#elif USE_CASE == LULESH_TILE_INDEXED

// Currently requires OMP_FINE_SYNC when run in parallel
#define OMP_FINE_SYNC 1

// Only use AllocateTouch if tiling is imposed on top of a block decomposition,
// and that block decomposition is the indexset used for the first touch (see CreateMaskedIndexSet)

TilingMode const lulesh_tiling_mode = Tiled_Index;

typedef RAJA::omp_parallel_for_segit  Hybrid_Seg_Iter;
typedef RAJA::simd_exec               Segment_Exec;

typedef RAJA::IndexSet::ExecPolicy<RAJA::seq_segit, RAJA::omp_parallel_for_exec>  node_exec_policy;
typedef RAJA::IndexSet::ExecPolicy<RAJA::omp_parallel_for_segit, RAJA::simd_exec> elem_exec_policy;
typedef RAJA::IndexSet::ExecPolicy<RAJA::omp_parallel_for_segit, RAJA::simd_exec> mat_exec_policy;
typedef RAJA::IndexSet::ExecPolicy<RAJA::seq_segit, RAJA::omp_parallel_for_exec>  symnode_exec_policy;

typedef RAJA::omp_reduce reduce_policy; 

// ----------------------------------------------------
#elif USE_CASE == LULESH_TILE_ORDERED

// Currently requires OMP_FINE_SYNC when run in parallel
#define OMP_FINE_SYNC 1

// AllocateTouch should definitely be used

TilingMode const lulesh_tiling_mode = Tiled_Order;

typedef RAJA::omp_parallel_for_segit  Hybrid_Seg_Iter;
typedef RAJA::simd_exec               Segment_Exec;

typedef RAJA::IndexSet::ExecPolicy<RAJA::seq_segit, RAJA::omp_parallel_for_exec>  node_exec_policy;
typedef RAJA::IndexSet::ExecPolicy<RAJA::omp_parallel_for_segit, RAJA::simd_exec> elem_exec_policy;
typedef RAJA::IndexSet::ExecPolicy<RAJA::omp_parallel_for_segit, RAJA::simd_exec> mat_exec_policy;
typedef RAJA::IndexSet::ExecPolicy<RAJA::seq_segit, RAJA::omp_parallel_for_exec>  symnode_exec_policy;

typedef RAJA::omp_reduce reduce_policy; 

// ----------------------------------------------------
#elif USE_CASE == LULESH_TILE_TASK

// Can be used with or without OMP_FINE_SYNC; without will have less data movement and memory use

// AllocateTouch should definitely be used

// In reality, only the "lock-free" operations need to use the dependence graph embedded in the
// lock-free indexset, and the dependence-graph should likely be deactivated for other operations.

TilingMode const lulesh_tiling_mode = Tiled_LockFree;

typedef RAJA::omp_parallel_for_segit  Hybrid_Seg_Iter;
typedef RAJA::simd_exec               Segment_Exec;

typedef RAJA::IndexSet::ExecPolicy<RAJA::seq_segit, RAJA::omp_parallel_for_exec>  node_exec_policy;
typedef RAJA::IndexSet::ExecPolicy<RAJA::omp_taskgraph_segit, RAJA::simd_exec> elem_exec_policy;
typedef RAJA::IndexSet::ExecPolicy<RAJA::omp_taskgraph_segit, RAJA::simd_exec> mat_exec_policy;
typedef RAJA::IndexSet::ExecPolicy<RAJA::seq_segit, RAJA::omp_parallel_for_exec>  symnode_exec_policy;

typedef RAJA::omp_reduce reduce_policy; 

// ----------------------------------------------------
#elif USE_CASE == LULESH_TILE_COLOR

// Can be used with or without OMP_FINE_SYNC; without will have less data movement and memory use

// AlocateTouch use is very tricky with this lockfree indexset.

TilingMode const lulesh_tiling_mode = Tiled_LockFreeColor;

typedef RAJA::seq_segit              Hybrid_Seg_Iter;
typedef RAJA::omp_parallel_for_exec  Segment_Exec;

typedef RAJA::IndexSet::ExecPolicy<Hybrid_Seg_Iter, RAJA::omp_parallel_for_exec> node_exec_policy;
typedef RAJA::IndexSet::ExecPolicy<Hybrid_Seg_Iter, RAJA::omp_parallel_for_exec> elem_exec_policy;
typedef RAJA::IndexSet::ExecPolicy<Hybrid_Seg_Iter, RAJA::omp_parallel_for_exec> mat_exec_policy;
typedef RAJA::IndexSet::ExecPolicy<Hybrid_Seg_Iter, RAJA::omp_parallel_for_exec> symnode_exec_policy;

typedef RAJA::omp_reduce reduce_policy; 

// ----------------------------------------------------
#elif USE_CASE == LULESH_TILE_COLOR_SIMD

// Can be used with or without OMP_FINE_SYNC; without will have less data movement and memory use

// AlocateTouch use is very tricky with this lockfree indexset.

TilingMode const lulesh_tiling_mode = Tiled_LockFreeColorSIMD;

typedef RAJA::seq_segit              Hybrid_Seg_Iter;
typedef RAJA::omp_parallel_for_exec  Segment_Exec;

typedef RAJA::IndexSet::ExecPolicy<Hybrid_Seg_Iter, Segment_Exec> node_exec_policy;
typedef RAJA::IndexSet::ExecPolicy<Hybrid_Seg_Iter, Segment_Exec> elem_exec_policy;
typedef RAJA::IndexSet::ExecPolicy<Hybrid_Seg_Iter, Segment_Exec> mat_exec_policy;
typedef RAJA::IndexSet::ExecPolicy<Hybrid_Seg_Iter, Segment_Exec> symnode_exec_policy;

typedef RAJA::omp_reduce reduce_policy; 

// ----------------------------------------------------
#elif USE_CASE == LULESH_CILK

// Requires OMP_FINE_SYNC when run in parallel
#define OMP_FINE_SYNC 1

// AllocateTouch should definitely be used

TilingMode const lulesh_tiling_mode = Canonical;

typedef RAJA::cilk_for_segit         Hybrid_Seg_Iter;
typedef RAJA::cilk_for_exec          Segment_Exec;

typedef RAJA::IndexSet::ExecPolicy<Hybrid_Seg_Iter, Segment_Exec> node_exec_policy;
typedef RAJA::IndexSet::ExecPolicy<Hybrid_Seg_Iter, Segment_Exec> elem_exec_policy;
typedef RAJA::IndexSet::ExecPolicy<Hybrid_Seg_Iter, Segment_Exec> mat_exec_policy;
typedef RAJA::IndexSet::ExecPolicy<Hybrid_Seg_Iter, Segment_Exec> symnode_exec_policy;

typedef RAJA::cilk_reduce            reduce_policy ;

// ----------------------------------------------------
#elif USE_CASE == LULESH_CUDA_CANONICAL

// Requires OMP_FINE_SYNC 
#define OMP_FINE_SYNC 1

TilingMode const lulesh_tiling_mode = Canonical;

typedef RAJA::seq_segit         Hybrid_Seg_Iter;

/// Define thread block size for CUDA exec policy
const size_t thread_block_size = 256;
typedef RAJA::cuda_exec<thread_block_size>    Segment_Exec;

typedef RAJA::IndexSet::ExecPolicy<Hybrid_Seg_Iter, Segment_Exec> node_exec_policy;
typedef RAJA::IndexSet::ExecPolicy<Hybrid_Seg_Iter, Segment_Exec> elem_exec_policy;
typedef RAJA::IndexSet::ExecPolicy<Hybrid_Seg_Iter, Segment_Exec> mat_exec_policy;
typedef RAJA::IndexSet::ExecPolicy<Hybrid_Seg_Iter, Segment_Exec> symnode_exec_policy;

typedef RAJA::cuda_reduce<thread_block_size> reduce_policy; 

// ----------------------------------------------------
#elif USE_CASE == LULESH_CUDA_COLOR_SIMD

// Can be used with or without OMP_FINE_SYNC; without will have less data movement and memory use

TilingMode const lulesh_tiling_mode = Tiled_LockFreeColorSIMD;

typedef RAJA::seq_segit         Hybrid_Seg_Iter;

/// Define thread block size for CUDA exec policy
const size_t thread_block_size = 256;
typedef RAJA::cuda_exec<thread_block_size>    Segment_Exec;

typedef RAJA::IndexSet::ExecPolicy<Hybrid_Seg_Iter, Segment_Exec> node_exec_policy;
typedef RAJA::IndexSet::ExecPolicy<Hybrid_Seg_Iter, Segment_Exec> elem_exec_policy;
typedef RAJA::IndexSet::ExecPolicy<Hybrid_Seg_Iter, Segment_Exec> mat_exec_policy;
typedef RAJA::IndexSet::ExecPolicy<Hybrid_Seg_Iter, Segment_Exec> symnode_exec_policy;

typedef RAJA::cuda_reduce<thread_block_size> reduce_policy; 

#else

#error "You must define a use case in luleshPolicy.cxx"

#endif

