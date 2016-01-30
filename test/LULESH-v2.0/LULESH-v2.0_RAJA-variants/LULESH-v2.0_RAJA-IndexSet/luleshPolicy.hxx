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

#define USE_CASE 2

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
//   10 = CUDA        (technique 7 on GPU to avoid OMP_FINE_SYNC data movement)


// ----------------------------------------------------
#if USE_CASE == 1 

TilingMode const lulesh_tiling_mode = Canonical;

typedef RAJA::seq_segit              Segment_Iter;
typedef RAJA::simd_exec              Segment_Exec;

typedef RAJA::IndexSet::ExecPolicy<Segment_Iter, Segment_Exec> node_exec_policy;
typedef RAJA::IndexSet::ExecPolicy<Segment_Iter, Segment_Exec> elem_exec_policy;
typedef RAJA::IndexSet::ExecPolicy<Segment_Iter, Segment_Exec> mat_exec_policy;
typedef RAJA::IndexSet::ExecPolicy<Segment_Iter, Segment_Exec> min_exec_policy;
typedef RAJA::IndexSet::ExecPolicy<Segment_Iter, Segment_Exec> symnode_exec_policy;

typedef RAJA::seq_reduce reduce_policy; 

// ----------------------------------------------------
#elif USE_CASE == 2

// Requires OMP_FINE_SYNC when run in parallel
#define OMP_FINE_SYNC 1

// AllocateTouch should definitely be used

TilingMode const lulesh_tiling_mode = Canonical;

typedef RAJA::seq_segit              Segment_Iter;
typedef RAJA::omp_parallel_for_exec  Segment_Exec;

typedef RAJA::IndexSet::ExecPolicy<Segment_Iter, Segment_Exec> node_exec_policy;
typedef RAJA::IndexSet::ExecPolicy<Segment_Iter, Segment_Exec> elem_exec_policy;
typedef RAJA::IndexSet::ExecPolicy<Segment_Iter, Segment_Exec> mat_exec_policy;
typedef RAJA::IndexSet::ExecPolicy<Segment_Iter, Segment_Exec> symnode_exec_policy;

typedef RAJA::omp_reduce reduce_policy;

// ----------------------------------------------------
#elif USE_CASE == 9

// Requires OMP_FINE_SYNC 
#define OMP_FINE_SYNC 1

TilingMode const lulesh_tiling_mode = Canonical;

typedef RAJA::seq_segit         Segment_Iter;

/// Define thread block size for CUDA exec policy
const size_t thread_block_size = 256;
typedef RAJA::cuda_exec<thread_block_size>    Segment_Exec;

typedef RAJA::IndexSet::ExecPolicy<Segment_Iter, Segment_Exec> node_exec_policy;
typedef RAJA::IndexSet::ExecPolicy<Segment_Iter, Segment_Exec> elem_exec_policy;
typedef RAJA::IndexSet::ExecPolicy<Segment_Iter, Segment_Exec> mat_exec_policy;
typedef RAJA::IndexSet::ExecPolicy<Segment_Iter, Segment_Exec> symnode_exec_policy;

typedef RAJA::cuda_reduce<thread_block_size> reduce_policy;

#else

#error "You must define a use case in luleshPolicy.cxx"

#endif

