
#if !defined(LULESH_HEADER)
#include "lulesh_stl.h"
#elif (LULESH_HEADER == 1)
#include "lulesh_ptr.h"
#elif (LULESH_HEADER == 2)
#include "lulesh_raw.h"
#else
#include "lulesh_tuple.h"
#endif
