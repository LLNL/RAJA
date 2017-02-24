/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   RAJA header file for simple classes that can be used to
 *          time code sections.
 *
 ******************************************************************************
 */

#ifndef RAJA_Timer_HXX
#define RAJA_Timer_HXX

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016, Lawrence Livermore National Security, LLC.
//
// Produced at the Lawrence Livermore National Laboratory
//
// LLNL-CODE-689114
//
// All rights reserved.
//
// This file is part of RAJA.
//
// For additional details, please also read RAJA/LICENSE.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// * Redistributions of source code must retain the above copyright notice,
//   this list of conditions and the disclaimer below.
//
// * Redistributions in binary form must reproduce the above copyright notice,
//   this list of conditions and the disclaimer (as noted below) in the
//   documentation and/or other materials provided with the distribution.
//
// * Neither the name of the LLNS/LLNL nor the names of its contributors may
//   be used to endorse or promote products derived from this software without
//   specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL LAWRENCE LIVERMORE NATIONAL SECURITY,
// LLC, THE U.S. DEPARTMENT OF ENERGY OR CONTRIBUTORS BE LIABLE FOR ANY
// DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
// DAMAGES  (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
// OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
// HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
// STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING
// IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//


#include "RAJA/config.hxx"

#if defined(RAJA_USE_CALIPER)
#include <caliper/Annotation.h>
#endif

// libstdc++ on BGQ only has gettimeofday for some reason
#if defined(__bgq__) && (!defined(_LIBCPP_VERSION))

#include <sys/time.h>
#include <chrono>

namespace RAJA
{
/*!
 ******************************************************************************
 *
 * \brief  Simple timer class to time code sections.
 *
 ******************************************************************************
 */
class BGQTimer
{
  using TimeType = timeval;
  using Duration = long double;

public:
  BGQTimer() : tstart(), tstop(), telapsed(0) {}
  void start() { gettimeofday(&tstart, 0); }
  void stop()
  {
    gettimeofday(&tstop, 0);
    auto start = std::chrono::seconds(tstart.tv_sec)
                 + std::chrono::microseconds(tstart.tv_usec);
    auto stop = std::chrono::seconds(tstop.tv_sec)
                + std::chrono::microseconds(tstop.tv_usec);
    telapsed += std::chrono::duration<Duration>(stop - start).count();
  }

  Duration elapsed() { return telapsed; }

  void reset() { telapsed = 0; }

private:
  TimeType tstart;
  TimeType tstop;
  Duration telapsed;
};

using TimerBase = BGQTimer;
}

#elif defined(RAJA_USE_CHRONO)

#include <chrono>

namespace RAJA
{
/*!
 ******************************************************************************
 *
 * \brief  Simple timer class to time code sections.
 *
 ******************************************************************************
 */
class ChronoTimer
{
  using clock = std::chrono::steady_clock;
  using TimeType = clock::time_point;
  using Duration = std::chrono::duration<long double>;

public:
  ChronoTimer() : tstart(clock::now()), tstop(clock::now()), telapsed(0) {}
  void start() { tstart = clock::now(); }
  void stop()
  {
    tstop = clock::now();
    telapsed += tstop - tstart;
  }

  Duration::rep elapsed() { return telapsed.count(); }

  void reset() { telapsed = Duration(0); } 

private:
  TimeType tstart;
  TimeType tstop;
  Duration telapsed;
};

using TimerBase = ChronoTimer;
}

#elif defined(RAJA_USE_GETTIME)

#include <time.h>

namespace RAJA
{
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
  void start() { clock_gettime(CLOCK_MONOTONIC, &tstart); }
  void stop()
  {
    clock_gettime(CLOCK_MONOTONIC, &tstop);
    set_elapsed();
  }

  long double elapsed() { return (stime_elapsed + nstime_elapsed); }

  void reset() { stime_elapsed = 0; nstime_elapsed = 0; }

private:
  TimeType tstart;
  TimeType tstop;
  long double telapsed;

  long double stime_elapsed;
  long double nstime_elapsed;

  void set_elapsed()
  {
    stime_elapsed += static_cast<long double>(tstop.tv_sec - tstart.tv_sec);
    nstime_elapsed +=
        static_cast<long double>(tstop.tv_nsec - tstart.tv_nsec) / 1000000000.0;
  }
};

using TimerBase = GettimeTimer;
}  // closing brace for RAJA namespace

#elif defined(RAJA_USE_CYCLE)

#include "./cycle.h"
namespace RAJA
{
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
  void stop()
  {
    tstop = getticks();
    set_elapsed();
  }

  long double elapsed() { return static_cast<long double>(telapsed); }

  void reset() { telapsed = 0; }

private:
  TimeType tstart;
  TimeType tstop;
  long double telapsed;

  void set_elapsed() { telapsed += (tstop - tstart); }
};

using TimerBase = CycleTimer;
}  // closing brace for RAJA namespace

#elif defined(RAJA_USE_CLOCK)

#include <time.h>
namespace RAJA
{
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
  void stop()
  {
    tstop = clock();
    set_elapsed();
  }

  long double elapsed()
  {
    return static_cast<long double>(telapsed) / CLOCKS_PER_SEC;
  }

  void reset() { telapsed = 0; }

private:
  TimeType tstart;
  TimeType tstop;
  long double telapsed;

  void set_elapsed() { telapsed += (tstop - tstart); }
};

using TimerBase = ClockTimer;
}  // closing brace for RAJA namespace

#else

#error RAJA_TIMER_TYPE is undefined!

#endif

namespace RAJA
{

class Timer : public TimerBase
{
public:
  using TimerBase::start;
  using TimerBase::stop;

#if defined(RAJA_USE_CALIPER)
  void start(const char* name) { cali::Annotation(name).begin(); }
  void stop(const char* name) { cali::Annotation(name).end(); }
#else
  void start(const char*) { start(); }
  void stop(const char*) { stop(); }
#endif
};

}  // closing brace for RAJA namespace

#endif  // closing endif for header file include guard
