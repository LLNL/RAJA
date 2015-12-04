
#include "RAJA/RAJA.hxx"
#include "luleshPolicy.hxx"
#include "luleshMemory.hxx"

#define MAX(a, b) ( ((a) > (b)) ? (a) : (b))

#if !defined(LULESH_HEADER)
#include "lulesh_stl.h"
#elif (LULESH_HEADER == 1)
#include "lulesh_ptr.h"
#else
#include "lulesh_tuple.h"
#endif
