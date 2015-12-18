
#include "RAJA/RAJA.hxx"

//
//   Policies for hybrid segment iteration and segment execution.
//
//   NOTE: Currently, we apply single policy across all loops
//         with same iteration pattern.
//
typedef RAJA::seq_segit              IndexSet_SegIt;
//typedef RAJA::omp_parallel_for_segit IndexSet_SegIt;
//typedef RAJA::cilk_for_segit         IndexSet_SegIt;


//typedef RAJA::seq_exec              SegExec;
//typedef RAJA::simd_exec             SegExec;
typedef RAJA::omp_parallel_for_exec SegExec;
//typedef RAJA::cilk_for_exec         SegExec;

typedef RAJA::IndexSet::ExecPolicy<IndexSet_SegIt, SegExec> node_exec_policy;
typedef RAJA::IndexSet::ExecPolicy<IndexSet_SegIt, SegExec> elem_exec_policy;
typedef RAJA::IndexSet::ExecPolicy<IndexSet_SegIt, SegExec> mat_exec_policy;
//typedef RAJA::IndexSet::ExecPolicy<IndexSet_SegIt, RAJA::seq_exec> mat_exec_policy;
// typedef RAJA::IndexSet::ExecPolicy<IndexSet_SegIt, SegExec> minloc_exec_policy;
typedef RAJA::IndexSet::ExecPolicy<IndexSet_SegIt, SegExec> min_exec_policy;
typedef RAJA::IndexSet::ExecPolicy<IndexSet_SegIt, SegExec> symnode_exec_policy;

//typedef RAJA::seq_reduce              reduce_policy;
typedef RAJA::omp_reduce              reduce_policy;
//typedef RAJA::cilk_reduce              reduce_policy;


#if !defined(LULESH_HEADER)
#include "lulesh_stl.h"
#elif (LULESH_HEADER == 1)
#include "lulesh_ptr.h"
#elif (LULESH_HEADER == 2)
#include "lulesh_raw.h"
#else
#include "lulesh_tuple.h"
#endif
