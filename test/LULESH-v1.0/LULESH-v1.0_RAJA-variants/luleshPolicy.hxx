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

#define USE_CASE 5

//   1 = Sequential   (with possible SIMD vectorization applied)
//   2 = Canonical    (OMP forall applied to each for loop)
//   3 = Tiled_Index  (OMP Tiles defined by unstructured indexset
//                     iteration space partitioning. One tile per thread)
//   4 = Tiled_order  (OMP The mesh is permuted so a tile is defined by a
//                     contiguous chunk of the iteration space. Tile per thread)
//   5 = Tiled_LockFree (OMP Mesh is chunked like Canonical, but now chunks are
//                       dependency scheduled,removing need for lock constructs)
//   6 = LockFree_Color (OMP Analogous to Tiled_index, but individual array
//                       values are 'checker-boarded' into 'colors' to guarantee
//                       indpenedent data access as long as each 'color' of
//                       array values completes before executing the next color)
//   7 = LockFree_ColorSIMD (Colored like USE_CASE 6, but the colors are then
//                           permuted to be contiguous chunks, like USE_CASE 4)
//   8 = Cilk         (cilk_for applied to each loop)
//   9 = CUDA         (CUDA kernel launch applied to each loop)
//   10 = CUDA        (Use technique 7 on GPU to avoid OMP_HACK data movement)


// ----------------------------------------------------
#if USE_CASE == 1 

TilingMode const lulesh_tiling_mode = Canonical;

typedef RAJA::seq_segit              Hybrid_Seg_Iter;
typedef RAJA::simd_exec              Segment_Exec;

typedef RAJA::IndexSet::ExecPolicy<Hybrid_Seg_Iter, Segment_Exec> node_exec_policy;
typedef RAJA::IndexSet::ExecPolicy<Hybrid_Seg_Iter, Segment_Exec> elem_exec_policy;
typedef RAJA::IndexSet::ExecPolicy<Hybrid_Seg_Iter, Segment_Exec> mat_exec_policy;
typedef RAJA::IndexSet::ExecPolicy<Hybrid_Seg_Iter, Segment_Exec> symnode_exec_policy;

typedef RAJA::seq_reduce reduce_policy; 

// ----------------------------------------------------
#elif USE_CASE == 2

// Requires OMP_HACK when run in parallel
#define OMP_HACK 1

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
#elif USE_CASE == 3

// Currently requires OMP_HACK when run in parallel
#define OMP_HACK 1

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
#elif USE_CASE == 4

// Currently requires OMP_HACK when run in parallel
#define OMP_HACK 1

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
#elif USE_CASE == 5

// Can be used with or without OMP_HACK; without will have less data movement and memory use

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
#elif USE_CASE == 6

// Can be used with or without OMP_HACK; without will have less data movement and memory use

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
#elif USE_CASE == 7

// Can be used with or without OMP_HACK; without will have less data movement and memory use

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
#elif USE_CASE == 8

// Requires OMP_HACK when run in parallel
#define OMP_HACK 1

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
#elif USE_CASE == 9

// Requires OMP_HACK 
#define OMP_HACK 1

TilingMode const lulesh_tiling_mode = Canonical;

typedef RAJA::seq_segit         Hybrid_Seg_Iter;
typedef RAJA::cuda_exec         Segment_Exec;

typedef RAJA::IndexSet::ExecPolicy<Hybrid_Seg_Iter, Segment_Exec> node_exec_policy;
typedef RAJA::IndexSet::ExecPolicy<Hybrid_Seg_Iter, Segment_Exec> elem_exec_policy;
typedef RAJA::IndexSet::ExecPolicy<Hybrid_Seg_Iter, Segment_Exec> mat_exec_policy;
typedef RAJA::IndexSet::ExecPolicy<Hybrid_Seg_Iter, Segment_Exec> symnode_exec_policy;

typedef RAJA::cuda_reduce reduce_policy; 

// ----------------------------------------------------
#elif USE_CASE == 10

// Can be used with or without OMP_HACK; without will have less data movement and memory use

TilingMode const lulesh_tiling_mode = Tiled_LockFreeColorSIMD;

typedef RAJA::seq_segit         Hybrid_Seg_Iter;
typedef RAJA::cuda_exec         Segment_Exec;

typedef RAJA::IndexSet::ExecPolicy<Hybrid_Seg_Iter, Segment_Exec> node_exec_policy;
typedef RAJA::IndexSet::ExecPolicy<Hybrid_Seg_Iter, Segment_Exec> elem_exec_policy;
typedef RAJA::IndexSet::ExecPolicy<Hybrid_Seg_Iter, Segment_Exec> mat_exec_policy;
typedef RAJA::IndexSet::ExecPolicy<Hybrid_Seg_Iter, Segment_Exec> symnode_exec_policy;

typedef RAJA::cuda_reduce reduce_policy; 

#else

#error "You must define a use case in luleshPolicy.cxx"

#endif

