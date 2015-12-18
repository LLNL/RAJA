/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   RAJA header file for simple class that can be used to 
 *          time code sections.
 *     
 * \author  Rich Hornung, Center for Applied Scientific Computing, LLNL
 * \author  Jeff Keasler, Applications, Simulations And Quality, LLNL
 *
 ******************************************************************************
 */

#ifndef RAJA_Timer_HXX
#define RAJA_Timer_HXX


#if defined(RAJA_USE_CYCLE)
#include "./cycle.h"
typedef ticks TimeType;

#elif defined(RAJA_USE_CLOCK)
#include <time.h>
typedef clock_t TimeType;

#elif defined(RAJA_USE_GETTIME)
#include <time.h>
typedef timespec TimeType;

#else
#error RAJA_TIMER_TYPE is undefined!

#endif



namespace RAJA {


/*!
 ******************************************************************************
 *
 * \brief  Simple timer class to time code sections.
 *
 ******************************************************************************
 */
class Timer
{
public:
#if defined(RAJA_USE_CYCLE) || defined(RAJA_USE_CLOCK)
   Timer() : telapsed(0) { ; }
#endif
#if defined(RAJA_USE_GETTIME)
   Timer() : telapsed(0), stime_elapsed(0), nstime_elapsed(0) { ; }
#endif

#if defined(RAJA_USE_CYCLE)
   void start() { tstart = getticks(); }
   void stop()  { tstop = getticks();  set_elapsed(); }

   long double elapsed()
      { return static_cast<long double>(telapsed); }
#endif
 
#if defined(RAJA_USE_CLOCK)
   void start() { tstart = clock(); }
   void stop()  { tstop = clock();  set_elapsed(); }

   long double elapsed()
      { return static_cast<long double>(telapsed) / CLOCKS_PER_SEC; }
#endif

#if defined(RAJA_USE_GETTIME)

#if 0
   void start() { clock_gettime(CLOCK_REALTIME, &tstart); }
   void stop()  { clock_gettime(CLOCK_REALTIME, &tstop); set_elapsed(); }
#else
   void start() { clock_gettime(CLOCK_MONOTONIC, &tstart); }
   void stop()  { clock_gettime(CLOCK_MONOTONIC, &tstop); set_elapsed(); }
#endif

   long double elapsed()
      { return (stime_elapsed + nstime_elapsed); }

#endif

private:
   TimeType tstart;
   TimeType tstop;
   long double telapsed;

#if defined(RAJA_USE_CYCLE) || defined(RAJA_USE_CLOCK)
   void set_elapsed() { telapsed += (tstop - tstart); }

#elif defined(RAJA_USE_GETTIME)
   long double stime_elapsed;
   long double nstime_elapsed;

   void set_elapsed() { stime_elapsed += static_cast<long double>(
                                         tstop.tv_sec - tstart.tv_sec); 
                        nstime_elapsed += static_cast<long double>(
                                          tstop.tv_nsec - tstart.tv_nsec ) /
                                          1000000000.0; } 
#endif
   
};


}  // closing brace for RAJA namespace

#endif  // closing endif for header file include guard
