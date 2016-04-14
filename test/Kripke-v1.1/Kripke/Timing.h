/*
 * NOTICE
 *
 * This work was produced at the Lawrence Livermore National Laboratory (LLNL)
 * under contract no. DE-AC-52-07NA27344 (Contract 44) between the U.S.
 * Department of Energy (DOE) and Lawrence Livermore National Security, LLC
 * (LLNS) for the operation of LLNL. The rights of the Federal Government are
 * reserved under Contract 44.
 *
 * DISCLAIMER
 *
 * This work was prepared as an account of work sponsored by an agency of the
 * United States Government. Neither the United States Government nor Lawrence
 * Livermore National Security, LLC nor any of their employees, makes any
 * warranty, express or implied, or assumes any liability or responsibility
 * for the accuracy, completeness, or usefulness of any information, apparatus,
 * product, or process disclosed, or represents that its use would not infringe
 * privately-owned rights. Reference herein to any specific commercial products,
 * process, or service by trade name, trademark, manufacturer or otherwise does
 * not necessarily constitute or imply its endorsement, recommendation, or
 * favoring by the United States Government or Lawrence Livermore National
 * Security, LLC. The views and opinions of authors expressed herein do not
 * necessarily state or reflect those of the United States Government or
 * Lawrence Livermore National Security, LLC, and shall not be used for
 * advertising or product endorsement purposes.
 *
 * NOTIFICATION OF COMMERCIAL USE
 *
 * Commercialization of this product is prohibited without notifying the
 * Department of Energy (DOE) or Lawrence Livermore National Security.
 */

#ifndef KRIPKE_TIMING_H__
#define KRIPKE_TIMING_H__

#include<Kripke.h>
#include <string>
#include <vector>
#include <map>
#include <stdio.h>
#include <time.h>
#include <sys/time.h>

#ifdef KRIPKE_USE_PAPI
#include<papi.h>
#endif

inline double getTime(void){
  struct timeval tv;
  gettimeofday(&tv, NULL);
  return (double)tv.tv_sec + (double)tv.tv_usec/1000000.0;
}


struct Timer {
  Timer() :
    started(false),
    start_time(0.0),
    total_time(0.0),
    count(0)
  {}

  bool started;
  double start_time;
  double total_time;
  size_t count;
#ifdef KRIPKE_USE_PAPI
  std::vector<long long> papi_start_values;
  std::vector<size_t> papi_total;
#endif
};

class Timing {
  public:
    ~Timing();

    void start(std::string const &name);
    void stop(std::string const &name);

    void stopAll(void);
    void clear(void);

    void print(void) const;
    double getTotal(std::string const &name) const;

    void setPapiEvents(std::vector<std::string> names);

  private:
    typedef std::map<std::string, Timer> TimerMap;
    TimerMap timers;
#ifdef KRIPKE_USE_PAPI
  std::vector<std::string> papi_names;
  std::vector<int> papi_event;
  int papi_set;
#endif
};


#include<stdio.h>

// Aides timing a block of code, with automatic timer stopping
class BlockTimer {
  public:
  inline BlockTimer(Timing &timer_obj, std::string const &timer_name) :
      timer(timer_obj),
      name(timer_name)
  {
      timer.start(name);
  }
  inline ~BlockTimer(){
    timer.stop(name);
  }

  private:
      Timing &timer;
      std::string name;
};

#define BLOCK_TIMER(TIMER, NAME) BlockTimer BLK_TIMER_##NAME(TIMER, #NAME);


#endif
