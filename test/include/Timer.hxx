/*
 * Copyright (c) 2016, Lawrence Livermore National Security, LLC.
 *
 * Produced at the Lawrence Livermore National Laboratory.
 *
 * All rights reserved.
 *
 * For release details and restrictions, please see raja/README-license.txt
 */

/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   RAJA header file for simple class that can be used to 
 *          time code sections.
 *     
 ******************************************************************************
 */

#ifndef RAJA_Timer_HXX
#define RAJA_Timer_HXX

#include "RAJA/config.hxx"

#if defined(RAJA_USE_CALIPER)
#include <caliper/Annotation.h>
#endif


namespace RAJA {

#if defined(RAJA_USE_CHRONO)
/*!
 ******************************************************************************
 *
 * \brief  Simple timer class to time code sections.
 *
 ******************************************************************************
 */
#include <chrono>
class ChronoTimer {
#if 0
    using clock = std::chrono::system_clock;
#else
    using clock = std::chrono::steady_clock;
#endif
    using TimeType = clock::time_point;
    using Duration = std::chrono::duration<long double, std::ratio<1>>;
public:
    ChronoTimer() : tstart(clock::now()), tstop(clock::now()), telapsed(0)  {}
    void start() { tstart = clock::now(); }
    void stop()  {
        tstop = clock::now();
        telapsed += std::chrono::duration_cast<std::chrono::seconds>(tstop - tstart);
    }

    Duration::rep elapsed()
    { return telapsed.count(); }

private:
    TimeType tstart;
    TimeType tstop;
    Duration telapsed;
};

using TimerBase = ChronoTimer;

#elif defined(RAJA_USE_GETTIME)
#include <time.h>
typedef timespec TimeType;

/*!
 ******************************************************************************
 *
 * \brief  Simple timer class to time code sections.
 *
 ******************************************************************************
 */
class GettimeTimer
{
public:
   GettimeTimer() : telapsed(0), stime_elapsed(0), nstime_elapsed(0) { ; }
#if 0
   void start() { clock_gettime(CLOCK_REALTIME, &tstart); }
   void stop()  { clock_gettime(CLOCK_REALTIME, &tstop); set_elapsed(); }
#else
   void start() { clock_gettime(CLOCK_MONOTONIC, &tstart); }
   void stop()  { clock_gettime(CLOCK_MONOTONIC, &tstop); set_elapsed(); }
#endif

   long double elapsed()
      { return (stime_elapsed + nstime_elapsed); }

private:
   TimeType tstart;
   TimeType tstop;
   long double telapsed;

   long double stime_elapsed;
   long double nstime_elapsed;

   void set_elapsed() { stime_elapsed += static_cast<long double>(
                                         tstop.tv_sec - tstart.tv_sec);
                        nstime_elapsed += static_cast<long double>(
                                          tstop.tv_nsec - tstart.tv_nsec ) /
                                          1000000000.0; }
};

using TimerBase = GettimeTimer;

#elif defined(RAJA_USE_CYCLE)
#include "./cycle.h"
typedef ticks TimeType;

/*!
 ******************************************************************************
 *
 * \brief  Simple timer class to time code sections.
 *
 ******************************************************************************
 */
class CycleTimer
{
public:
   CycleTimer() : telapsed(0) { ; }
   void start() { tstart = getticks(); }
   void stop()  { tstop = getticks();  set_elapsed(); }

   long double elapsed()
      { return static_cast<long double>(telapsed); }

private:
   TimeType tstart;
   TimeType tstop;
   long double telapsed;

   void set_elapsed() { telapsed += (tstop - tstart); }
};

using TimerBase = CycleTimer;

#elif defined(RAJA_USE_CLOCK)
#include <time.h>
typedef clock_t TimeType;

/*!
 ******************************************************************************
 *
 * \brief  Simple timer class to time code sections.
 *
 ******************************************************************************
 */
class ClockTimer
{
public:
   ClockTimer() : telapsed(0) { ; }

   void start() { tstart = clock(); }
   void stop()  { tstop = clock();  set_elapsed(); }

   long double elapsed()
      { return static_cast<long double>(telapsed) / CLOCKS_PER_SEC; }

private:
   TimeType tstart;
   TimeType tstop;
   long double telapsed;

   void set_elapsed() { telapsed += (tstop - tstart); }

};

using TimerBase = ClockTimer;

#else

#error RAJA_TIMER_TYPE is undefined!

#endif


class Timer : public TimerBase {
    public:
    using TimerBase::start;
    using TimerBase::stop;

#if defined(RAJA_USE_CALIPER)
    void start(const char* name){
        cali::Annotation(name).begin(); 
    }
    void stop(const char* name){
        cali::Annotation(name).end();
    }
#else
    void start(const char*){
        start();
    }
    void stop(const char*){
        stop();
    }
#endif
};

}  // closing brace for RAJA namespace

#endif  // closing endif for header file include guard
