/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   RAJA header file for simple classes for timing code sections.
 *
 * There are multiple timer classes to deal with platform indiosyncracies.
 *
 ******************************************************************************
 */

#ifndef RAJA_Timer_HPP
#define RAJA_Timer_HPP

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


#include "RAJA/config.hpp"

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
 * \brief  Timer class for BG/Q that uses gettimeofday since full std::chrono
 *         functionality is not supported.
 *
 *         Generates elapsed time in seconds.
 *
 ******************************************************************************
 */
class BGQTimer
{
public:
  using ElapsedType = double;

private:
  using TimeType = timeval;
  using DurationType = std::chrono::duration<ElapsedType>;

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
    telapsed += DurationType(stop - start).count();
  }

  ElapsedType elapsed() { return telapsed; }

  void reset() { telapsed = 0; }

private:
  TimeType tstart;
  TimeType tstop;
  ElapsedType telapsed;
};

using TimerBase = BGQTimer;
}  // closing brace for RAJA namespace


#elif defined(RAJA_USE_CHRONO)

#include <chrono>

namespace RAJA
{
/*!
 ******************************************************************************
 *
 * \brief  Timer class that uses std::chrono.
 *
 *         Generates elapsed time in seconds.
 *
 ******************************************************************************
 */
class ChronoTimer
{
public:
  using ElapsedType = double;

private:
  using ClockType = std::chrono::steady_clock;
  using TimeType = ClockType::time_point;
  using DurationType = std::chrono::duration<ElapsedType>;

public:
  ChronoTimer() : tstart(ClockType::now()), tstop(ClockType::now()), telapsed(0)
  {
  }
  void start() { tstart = ClockType::now(); }
  void stop()
  {
    tstop = ClockType::now();
    telapsed +=
        std::chrono::duration_cast<DurationType>(tstop - tstart).count();
  }

  ElapsedType elapsed() { return telapsed; }

  void reset() { telapsed = 0; }

private:
  TimeType tstart;
  TimeType tstop;
  ElapsedType telapsed;
};

using TimerBase = ChronoTimer;
}  // closing brace for RAJA namespace


#elif defined(RAJA_USE_GETTIME)

#include <time.h>

namespace RAJA
{
/*!
 ******************************************************************************
 *
 * \brief  Timer class that uses std::chrono.
 *
 *         Generates elapsed time in seconds.
 *
 ******************************************************************************
 */
class GettimeTimer
{
public:
  using ElapsedType = double;

private:
  using TimeType = timespec;

public:
  GettimeTimer() : telapsed(0), stime_elapsed(0), nstime_elapsed(0) { ; }
  void start() { clock_gettime(CLOCK_MONOTONIC, &tstart); }
  void stop()
  {
    clock_gettime(CLOCK_MONOTONIC, &tstop);
    set_elapsed();
  }

  ElapsedType elapsed() { return (stime_elapsed + nstime_elapsed); }

  void reset()
  {
    stime_elapsed = 0;
    nstime_elapsed = 0;
  }

private:
  TimeType tstart;
  TimeType tstop;
  ElasedType telapsed;

  ElapsedType stime_elapsed;
  ElapsedType nstime_elapsed;

  void set_elapsed()
  {
    stime_elapsed += static_cast<ElapsedType>(tstop.tv_sec - tstart.tv_sec);
    nstime_elapsed +=
        static_cast<ElapsedType>(tstop.tv_nsec - tstart.tv_nsec) / 1000000000.0;
  }
};

using TimerBase = GettimeTimer;
}  // closing brace for RAJA namespace

#elif defined(RAJA_USE_CLOCK)

#include <time.h>
namespace RAJA
{

/*!
 ******************************************************************************
 *
 * \brief  Timer class that uses clock_t.
 *
 *         Generates elapsed time in seconds.
 *
 ******************************************************************************
 */
class ClockTimer
{
public:
  using ElapsedType = double;

private:
  using TimeType = clock_t;

public:
  ClockTimer() : telapsed(0) { ; }

  void start() { tstart = clock(); }
  void stop()
  {
    tstop = clock();
    set_elapsed();
  }

  ElapsedType elapsed()
  {
    return static_cast<Elapsed Type>(telapsed) / CLOCKS_PER_SEC;
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

#error RAJA_TIMER is undefined!

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
