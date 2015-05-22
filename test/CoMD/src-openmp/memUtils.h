/// \file
/// Wrappers for memory allocation.

#ifndef _MEMUTILS_H_
#define _MEMUTILS_H_

#include <stdlib.h>
#include <string.h>

#define freeMe(s,element) {if(s->element) comdFree(s->element);  s->element = NULL;}

static void* comdMalloc(size_t iSize)
{
#if 0
   return malloc(iSize);
#else
   void *localPtr ;
   posix_memalign(&localPtr, 64, iSize) ;
   return localPtr ;
#endif
}

static void* comdCalloc(size_t num, size_t iSize)
{
#if 0
   return calloc(num, iSize);
#else
   void *localPtr ;
   posix_memalign(&localPtr, 64, num*iSize) ;
   memset(localPtr, 0, num*iSize) ;
   return localPtr ;
#endif
}

static void* comdRealloc(void* ptr, size_t iSize)
{
   return realloc(ptr, iSize);
}

static void comdFree(void *ptr)
{
   free(ptr);
}
#endif
