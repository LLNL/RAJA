/*
 * Copyright (c) 2016, Lawrence Livermore National Security, LLC.
 *
 * Produced at the Lawrence Livermore National Laboratory.
 *
 * All rights reserved.
 *
 * For release details and restrictions, please see raja/README-license.txt
 */

//
// Main program illustrating simple RAJA index set creation
// and execution and methods.
//

#include <time.h>
#include <cmath>
#include <cfloat>
#include <cstdlib>

#include <iostream>
#include <iomanip>
#include <string>
#include <random>
#include <vector>

#include "RAJA/RAJA.hxx"
#include "RAJA/internal/defines.hxx"

#include "Compare.hxx"


#define TEST_VEC_LEN 1024 * 32

using namespace RAJA;
using namespace std;

//
// Global variables for counting tests executed/passed.
//
unsigned s_ntests_run_total = 0;
unsigned s_ntests_passed_total = 0;

unsigned s_ntests_run = 0;
unsigned s_ntests_passed = 0;

//
// global values to test results against
//
Real_type seq_min = 1.0;
Real_type seq_max =-1.0;
Real_type seq_sum = 0.0;
Real_type seq_pos_sum = 0.0;
Real_type seq_neg_sum = 0.0;
Real_type seq_prod = 1.0;
int seq_pos_cnt = 0;
int seq_neg_cnt = 0;
int seq_and = 0xffffffff;
int seq_or = 0x0;
int seq_xor = 0x0f0f0f0f;

//=========================================================================
//=========================================================================
//
// Methods that define and run various RAJA reduction tests
//
//=========================================================================
//=========================================================================

///////////////////////////////////////////////////////////////////////////
//
// Method that defines and runs basic RAJA min max atomic tests
// based on execution policy template
//
///////////////////////////////////////////////////////////////////////////
template <typename POLICY_T, typename ATOMIC_POLICY_T, typename T>
void runBasicMinMaxAtomicTest(const string& policy,
                              const T* in_array,
                              Index_type alen)
{
  cout << "\n Test MIN MAX atomics for " << policy << "\n";

  s_ntests_run++;
  s_ntests_run_total++;

  RAJA::atomic<ATOMIC_POLICY_T, T> atm_min(T(1));
  RAJA::atomic<ATOMIC_POLICY_T, T> atm_max(T(1));

  forall<POLICY_T>(0, alen, [=] (int i) {
    atm_min.fetch_min(in_array[i]);
    atm_max.fetch_max(in_array[i]);
  });

  if (   !equal(T(atm_min), seq_min)
      || !equal(T(atm_max), seq_max) ) {
    cout << "\n TEST FAILURE:"
         << endl;
    cout << setprecision(20) << "\tatm_min = " << static_cast<T>(atm_min)
         << " (" << seq_min << ") " << endl;
    cout << setprecision(20) << "\tatm_max = " << static_cast<T>(atm_max)
         << " (" << seq_max << ") " << endl;
  } else {
    s_ntests_passed++;
    s_ntests_passed_total++;
  }
}

///////////////////////////////////////////////////////////////////////////
//
// Run RAJA min max atomic tests with available RAJA execution policies....
//
///////////////////////////////////////////////////////////////////////////
template <typename T>
void runMinMaxAtomicTests(const T* in_array,
                          Index_type alen)
{
  cout << "\n\n   BEGIN RAJA::forall MIN MAX ATOMIC tests...." << endl;

  // initialize test counters for this test set
  s_ntests_run = 0;
  s_ntests_passed = 0;

  runBasicMinMaxAtomicTest<seq_exec,
                           cpu_nonatomic>(
      "seq_exec", in_array, alen);

  runBasicMinMaxAtomicTest<simd_exec,
                           cpu_nonatomic>(
      "simd_exec", in_array, alen);

#ifdef RAJA_ENABLE_OPENMP
  runBasicMinMaxAtomicTest<omp_parallel_for_exec,
                           cpu_atomic>(
      "omp_parallel_for_exec",
      in_array,
      alen);
#endif

#ifdef RAJA_ENABLE_CILK
  runBasicMinMaxAtomicTest<cilk_for_exec,
                           cpu_atomic>(
      "cilk_for_exec", in_array, alen);
#endif

  cout << "\n tests passed / test run: " << s_ntests_passed << " / "
       << s_ntests_run << endl;

  cout << "\n   END RAJA::forall MIN MAX ATOMIC tests... " << endl;
}

///////////////////////////////////////////////////////////////////////////
//
// Method that defines and runs basic RAJA add sub atomic tests
// based on execution policy template
//
///////////////////////////////////////////////////////////////////////////
template <typename POLICY_T, typename ATOMIC_POLICY_T, typename T>
void runBasicAddSubAtomicTest(const string& policy,
                              const T* in_array,
                              Index_type alen)
{
  cout << "\n Test ADD SUB atomics for " << policy << "\n";

  s_ntests_run++;
  s_ntests_run_total++;
    
  RAJA::atomic<ATOMIC_POLICY_T, T> atm_fad_sum(T(0));
  RAJA::atomic<ATOMIC_POLICY_T, T> atm_fsb_sum(T(0));
  RAJA::atomic<ATOMIC_POLICY_T, T> atm_ple_sum(T(0));
  RAJA::atomic<ATOMIC_POLICY_T, T> atm_mie_sum(T(0));
  RAJA::atomic<ATOMIC_POLICY_T, T> atm_pos_sum(T(0));
  RAJA::atomic<ATOMIC_POLICY_T, T> atm_neg_sum(T(0));

  forall<POLICY_T>(0, alen, [=] (int i) {
    atm_fad_sum.fetch_add(in_array[i]);
    atm_fsb_sum.fetch_sub(in_array[i]);
    atm_ple_sum += in_array[i];
    atm_mie_sum -= in_array[i];

    if(in_array[i] < 0.0) {
      atm_neg_sum += in_array[i];
    }
    else {
      atm_pos_sum += in_array[i];
    }
  });

  if (   !equal(T(atm_fad_sum), seq_sum)
      || !equal(T(atm_fsb_sum), -seq_sum)
      || !equal(T(atm_ple_sum), seq_sum)
      || !equal(T(atm_mie_sum), -seq_sum)
      || !equal(T(atm_pos_sum), seq_pos_sum)
      || !equal(T(atm_neg_sum), seq_neg_sum) ) {
    cout << "\n TEST FAILURE:"
         << endl;
    cout << setprecision(20) << "\tatm_fetch_add_sum = " << static_cast<T>(atm_fad_sum)
         << " (" << seq_sum << ") " << endl;
    cout << setprecision(20) << "\tatm_fetch_sub_sum = " << static_cast<T>(atm_fsb_sum)
         << " (" << -seq_sum << ") " << endl;
    cout << setprecision(20) << "\tatm_+=_sum = " << static_cast<T>(atm_ple_sum)
         << " (" << seq_sum << ") " << endl;
    cout << setprecision(20) << "\tatm_-=_sum = " << static_cast<T>(atm_mie_sum)
         << " (" << -seq_sum << ") " << endl;
    cout << setprecision(20) << "\tatm_pos_sum = " << static_cast<T>(atm_pos_sum)
         << " (" << seq_pos_sum << ") " << endl;
    cout << setprecision(20) << "\tatm_neg_sum = " << static_cast<T>(atm_neg_sum)
         << " (" << seq_neg_sum << ") " << endl;
  } else {
    s_ntests_passed++;
    s_ntests_passed_total++;
  }
}

///////////////////////////////////////////////////////////////////////////
//
// Run RAJA add sub atomic tests with available RAJA execution policies....
//
///////////////////////////////////////////////////////////////////////////
template <typename T>
void runAddSubAtomicTests(const T* in_array,
                          Index_type alen)
{
  cout << "\n\n   BEGIN RAJA::forall ADD SUB ATOMIC tests...." << endl;

  // initialize test counters for this test set
  s_ntests_run = 0;
  s_ntests_passed = 0;

  runBasicAddSubAtomicTest<seq_exec,
                           cpu_nonatomic>(
      "seq_exec", in_array, alen);

  runBasicAddSubAtomicTest<simd_exec,
                           cpu_nonatomic>(
      "simd_exec", in_array, alen);

#ifdef RAJA_ENABLE_OPENMP
  runBasicAddSubAtomicTest<omp_parallel_for_exec,
                           cpu_atomic>(
      "omp_parallel_for_exec",
      in_array,
      alen);
#endif

#ifdef RAJA_ENABLE_CILK
  runBasicAddSubAtomicTest<cilk_for_exec,
                           cpu_atomic>(
      "cilk_for_exec", in_array, alen);
#endif

  cout << "\n tests passed / test run: " << s_ntests_passed << " / "
       << s_ntests_run << endl;

  cout << "\n   END RAJA::forall ADD SUB ATOMIC tests... " << endl;
}

///////////////////////////////////////////////////////////////////////////
//
// Method that defines and runs basic RAJA and or xor atomic tests
// based on execution policy template
//
///////////////////////////////////////////////////////////////////////////
template <typename POLICY_T, typename ATOMIC_POLICY_T, typename T>
void runBasicAndOrXorAtomicTest(const string& policy,
                              const T* in_array,
                              Index_type alen)
{
  cout << "\n Test AND OR XOR atomics for " << policy << "\n";

  s_ntests_run++;
  s_ntests_run_total++;
  
  union reinterp_u {
    unsigned long long int ull;
    T t;
  };

  reinterp_u and_init; and_init.ull = 0xffffffffffffffffULL;
  reinterp_u or_init;  or_init.ull  = 0x0000000000000000ULL;
  reinterp_u xor_init; xor_init.ull = 0x0f0f0f0f0f0f0f0fULL;

  RAJA::atomic<ATOMIC_POLICY_T, T> atm_fan(and_init.t);
  RAJA::atomic<ATOMIC_POLICY_T, T> atm_for(or_init.t);
  RAJA::atomic<ATOMIC_POLICY_T, T> atm_fxr(xor_init.t);
  RAJA::atomic<ATOMIC_POLICY_T, T> atm_ane(and_init.t);
  RAJA::atomic<ATOMIC_POLICY_T, T> atm_ore(or_init.t);
  RAJA::atomic<ATOMIC_POLICY_T, T> atm_xre(xor_init.t);

  forall<POLICY_T>(0, alen, [=] (int i) {
    atm_fan.fetch_and(in_array[i]);
    atm_for.fetch_or(in_array[i]);
    atm_fxr.fetch_xor(in_array[i]);
    atm_ane &= in_array[i];
    atm_ore |= in_array[i];
    atm_xre ^= in_array[i];
  });

  if (   !equal(T(atm_fan), seq_and)
      || !equal(T(atm_for), seq_or) 
      || !equal(T(atm_fxr), seq_xor)
      || !equal(T(atm_ane), seq_and)
      || !equal(T(atm_ore), seq_or) 
      || !equal(T(atm_xre), seq_xor) ) {
    cout << "\n TEST FAILURE:"
         << endl;
    cout << setprecision(20) << "\tatm_fetch_and = " << static_cast<T>(atm_fan)
         << " (" << seq_and << ") " << endl;
    cout << setprecision(20) << "\tatm_fetch_or = " << static_cast<T>(atm_for)
         << " (" << seq_or << ") " << endl;
    cout << setprecision(20) << "\tatm_fetch_xor = " << static_cast<T>(atm_fxr)
         << " (" << seq_xor << ") " << endl;
    cout << setprecision(20) << "\tatm_&= = " << static_cast<T>(atm_ane)
         << " (" << seq_and << ") " << endl;
    cout << setprecision(20) << "\tatm_|= = " << static_cast<T>(atm_ore)
         << " (" << seq_or << ") " << endl;
    cout << setprecision(20) << "\tatm_^= = " << static_cast<T>(atm_xre)
         << " (" << seq_xor << ") " << endl;
  } else {
    s_ntests_passed++;
    s_ntests_passed_total++;
  }
}

///////////////////////////////////////////////////////////////////////////
//
// Run RAJA and or xor atomic tests with available RAJA execution policies....
//
///////////////////////////////////////////////////////////////////////////
template <typename T>
void runAndOrXorAtomicTests(const T* in_array,
                          Index_type alen)
{
  cout << "\n\n   BEGIN RAJA::forall AND OR XOR ATOMIC tests...." << endl;

  // initialize test counters for this test set
  s_ntests_run = 0;
  s_ntests_passed = 0;

  runBasicAndOrXorAtomicTest<seq_exec,
                           cpu_nonatomic>(
      "seq_exec", in_array, alen);

  runBasicAndOrXorAtomicTest<simd_exec,
                           cpu_nonatomic>(
      "simd_exec", in_array, alen);

#ifdef RAJA_ENABLE_OPENMP
  runBasicAndOrXorAtomicTest<omp_parallel_for_exec,
                           cpu_atomic>(
      "omp_parallel_for_exec",
      in_array,
      alen);
#endif

#ifdef RAJA_ENABLE_CILK
  runBasicAndOrXorAtomicTest<cilk_for_exec,
                           cpu_atomic>(
      "cilk_for_exec", in_array, alen);
#endif

  cout << "\n tests passed / test run: " << s_ntests_passed << " / "
       << s_ntests_run << endl;

  cout << "\n   END RAJA::forall AND OR XOR ATOMIC tests... " << endl;
}

///////////////////////////////////////////////////////////////////////////
//
// Method that defines and runs basic RAJA inc dec atomic tests
// based on execution policy template
//
///////////////////////////////////////////////////////////////////////////
template <typename POLICY_T, typename ATOMIC_POLICY_T, typename T>
void runBasicIncDecAtomicTest(const string& policy,
                              const T* in_array,
                              Index_type alen)
{
  cout << "\n Test INC DEC atomics for " << policy << "\n";

  s_ntests_run++;
  s_ntests_run_total++;
    
  RAJA::atomic<ATOMIC_POLICY_T, int> atm_pos_post_inc(0);
  RAJA::atomic<ATOMIC_POLICY_T, int> atm_pos_pre_inc(0);
  RAJA::atomic<ATOMIC_POLICY_T, int> atm_pos_post_dec(0);
  RAJA::atomic<ATOMIC_POLICY_T, int> atm_pos_pre_dec(0);
  RAJA::atomic<ATOMIC_POLICY_T, int> atm_neg_post_inc(0);
  RAJA::atomic<ATOMIC_POLICY_T, int> atm_neg_pre_inc(0);
  RAJA::atomic<ATOMIC_POLICY_T, int> atm_neg_post_dec(0);
  RAJA::atomic<ATOMIC_POLICY_T, int> atm_neg_pre_dec(0);

  forall<POLICY_T>(0, alen, [=] (int i) {
    if(in_array[i] < 0.0) {
      atm_neg_post_inc++;
      ++atm_neg_pre_inc;
      atm_neg_post_dec--;
      --atm_neg_pre_dec;
    }
    else {
      atm_pos_post_inc++;
      ++atm_pos_pre_inc;
      atm_pos_post_dec--;
      --atm_pos_pre_dec;
    }
  });

  if (   !equal(int(atm_pos_post_inc), seq_pos_cnt)
      || !equal(int(atm_pos_pre_inc),  seq_pos_cnt) 
      || !equal(int(atm_pos_post_dec),-seq_pos_cnt)
      || !equal(int(atm_pos_pre_dec), -seq_pos_cnt)
      || !equal(int(atm_neg_post_inc), seq_neg_cnt)
      || !equal(int(atm_neg_pre_inc),  seq_neg_cnt) 
      || !equal(int(atm_neg_post_dec),-seq_neg_cnt)
      || !equal(int(atm_neg_pre_dec), -seq_neg_cnt) ) {
    cout << "\n TEST FAILURE:"
         << endl;
    cout << setprecision(20) << "\tatm_pos_a++ = " << static_cast<int>(atm_pos_post_inc)
         << " (" << seq_pos_cnt << ") " << endl;
    cout << setprecision(20) << "\tatm_pos_++a = " << static_cast<int>(atm_pos_pre_inc)
         << " (" << seq_pos_cnt << ") " << endl;
    cout << setprecision(20) << "\tatm_pos_a++ = " << static_cast<int>(atm_pos_post_dec)
         << " (" << -seq_pos_cnt << ") " << endl;
    cout << setprecision(20) << "\tatm_pos_++a = " << static_cast<int>(atm_pos_pre_dec)
         << " (" << -seq_pos_cnt << ") " << endl;
    cout << setprecision(20) << "\tatm_neg_a++ = " << static_cast<int>(atm_neg_post_inc)
         << " (" << seq_neg_cnt << ") " << endl;
    cout << setprecision(20) << "\tatm_neg_++a = " << static_cast<int>(atm_neg_pre_inc)
         << " (" << seq_neg_cnt << ") " << endl;
    cout << setprecision(20) << "\tatm_neg_a++ = " << static_cast<int>(atm_neg_post_dec)
         << " (" << -seq_neg_cnt << ") " << endl;
    cout << setprecision(20) << "\tatm_neg_++a = " << static_cast<int>(atm_neg_pre_dec)
         << " (" << -seq_neg_cnt << ") " << endl;
  } else {
    s_ntests_passed++;
    s_ntests_passed_total++;
  }
}

///////////////////////////////////////////////////////////////////////////
//
// Run RAJA inc dec atomic tests with available RAJA execution policies....
//
///////////////////////////////////////////////////////////////////////////
template <typename T>
void runIncDecAtomicTests(const T* in_array,
                          Index_type alen)
{
  cout << "\n\n   BEGIN RAJA::forall INC DEC ATOMIC tests...." << endl;

  // initialize test counters for this test set
  s_ntests_run = 0;
  s_ntests_passed = 0;

  runBasicIncDecAtomicTest<seq_exec,
                           cpu_nonatomic>(
      "seq_exec", in_array, alen);

  runBasicIncDecAtomicTest<simd_exec,
                           cpu_nonatomic>(
      "simd_exec", in_array, alen);

#ifdef RAJA_ENABLE_OPENMP
  runBasicIncDecAtomicTest<omp_parallel_for_exec,
                           cpu_atomic>(
      "omp_parallel_for_exec",
      in_array,
      alen);
#endif

#ifdef RAJA_ENABLE_CILK
  runBasicIncDecAtomicTest<cilk_for_exec,
                           cpu_atomic>(
      "cilk_for_exec", in_array, alen);
#endif

  cout << "\n tests passed / test run: " << s_ntests_passed << " / "
       << s_ntests_run << endl;

  cout << "\n   END RAJA::forall INC DEC ATOMIC tests... " << endl;
}

///////////////////////////////////////////////////////////////////////////
//
// Method that defines and runs basic RAJA compexch atomic tests
// based on execution policy template
//
///////////////////////////////////////////////////////////////////////////
template <typename POLICY_T, typename ATOMIC_POLICY_T, typename T>
void runBasicCompExchAtomicTest(const string& policy,
                              const T* in_array,
                              Index_type alen)
{
  cout << "\n Test COMPEXCH atomics for " << policy << "\n";

  s_ntests_run++;
  s_ntests_run_total++;

  RAJA::atomic<ATOMIC_POLICY_T, T> atm_cew_prod(1.0);
  RAJA::atomic<ATOMIC_POLICY_T, T> atm_ces_prod(1.0);

  forall<POLICY_T>(0, alen, [=] (int i) {

    T expect = atm_cew_prod.load();
    while (!atm_cew_prod.compare_exchange_weak(expect, expect * in_array[i]));

    expect = (T)atm_ces_prod;
    while (!atm_ces_prod.compare_exchange_strong(expect, expect * in_array[i]));
  });

  if (   !equal(T(atm_cew_prod), seq_prod)
      || !equal(T(atm_ces_prod), seq_prod) ) {
    cout << "\n TEST FAILURE:"
         << endl;
    cout << setprecision(20) << "\tatm_compare_exchange_weak_prod = " << static_cast<T>(atm_cew_prod)
         << " (" << seq_prod << ") " << endl;
    cout << setprecision(20) << "\tatm_compare_exchange_strong_prod = " << static_cast<T>(atm_ces_prod)
         << " (" << seq_prod << ") " << endl;
  } else {
    s_ntests_passed++;
    s_ntests_passed_total++;
  }
}

///////////////////////////////////////////////////////////////////////////
//
// Run RAJA compexch atomic tests with available RAJA execution policies....
//
///////////////////////////////////////////////////////////////////////////
template <typename T>
void runCompExchAtomicTests(const T* in_array,
                          Index_type alen)
{
  cout << "\n\n   BEGIN RAJA::forall COMPEXCH ATOMIC tests...." << endl;

  // initialize test counters for this test set
  s_ntests_run = 0;
  s_ntests_passed = 0;

  runBasicCompExchAtomicTest<seq_exec,
                           cpu_nonatomic>(
      "seq_exec", in_array, alen);

  runBasicCompExchAtomicTest<simd_exec,
                           cpu_nonatomic>(
      "simd_exec", in_array, alen);

#ifdef RAJA_ENABLE_OPENMP
  runBasicCompExchAtomicTest<omp_parallel_for_exec,
                           cpu_atomic>(
      "omp_parallel_for_exec",
      in_array,
      alen);
#endif

#ifdef RAJA_ENABLE_CILK
  runBasicCompExchAtomicTest<cilk_for_exec,
                           cpu_atomic>(
      "cilk_for_exec", in_array, alen);
#endif

  cout << "\n tests passed / test run: " << s_ntests_passed << " / "
       << s_ntests_run << endl;

  cout << "\n   END RAJA::forall COMPEXCH ATOMIC tests... " << endl;
}

///////////////////////////////////////////////////////////////////////////
//
// Method that defines and runs basic RAJA load exch atomic tests
// based on execution policy template
//
///////////////////////////////////////////////////////////////////////////
template <typename POLICY_T, typename ATOMIC_POLICY_T, typename T>
void runBasicLoadExchAtomicTest(const string& policy,
                              const T*,
                              Index_type)
{
  cout << "\n Test LOADEXCH atomics for " << policy << "\n";

  s_ntests_run++;
  s_ntests_run_total++;

  const unsigned long long int n = 5317;

  RAJA::atomic<ATOMIC_POLICY_T, unsigned long long int> atm_exch_val(0);
  RAJA::atomic<ATOMIC_POLICY_T, unsigned long long int> atm_sum(0);
  RAJA::atomic<ATOMIC_POLICY_T, int> atm_load_val_chk(0);
  RAJA::atomic<ATOMIC_POLICY_T, int> atm_exch_prev_chk(0);
  RAJA::atomic<ATOMIC_POLICY_T, int> atm_load_later_chk(0);

  forall<POLICY_T>(0, (int)n, [=] (int i) {
      unsigned long long int load_first   = atm_exch_val.load();
      unsigned long long int exch_prev  = atm_exch_val.exchange(i + 1);
      unsigned long long int load_later   = atm_exch_val;

      atm_sum += exch_prev;

      if (load_first <= n) atm_load_val_chk++;
      if (exch_prev <= n) atm_exch_prev_chk++;
      if (load_later > 0 && load_later <= n) atm_load_later_chk++;
  });

  // note that one value in [1, n] will be missing
  const unsigned long long int sum = n*(n+1)/2;

  unsigned long long int diff = sum - atm_sum;

  if (   int(atm_load_val_chk) != n
      || int(atm_exch_prev_chk) != n
      || int(atm_load_later_chk) != n
      || !(diff >= 1 && diff <= n) ) {
    cout << "\n TEST FAILURE:"
         << endl;
    cout << setprecision(20) << "\tatm_load_val_chk = " << static_cast<int>(atm_load_val_chk)
         << " (" << n << ") " << endl;
    cout << setprecision(20) << "\tatm_exch_prev_chk = " << static_cast<int>(atm_exch_prev_chk)
         << " (" << n << ") " << endl;
    cout << setprecision(20) << "\tatm_load_later_chk = " << static_cast<int>(atm_load_later_chk)
         << " (" << n << ") " << endl;
    cout << setprecision(20) << "\tatm_exch_prev_chk = " << static_cast<unsigned long long int>(diff)
         << " ( [1, " << n << "] ) " << endl;
  } else {
    s_ntests_passed++;
    s_ntests_passed_total++;
  }
}

///////////////////////////////////////////////////////////////////////////
//
// Run RAJA loadexch atomic tests with available RAJA execution policies....
//
///////////////////////////////////////////////////////////////////////////
template <typename T>
void runLoadExchAtomicTests(const T* in_array,
                          Index_type alen)
{
  cout << "\n\n   BEGIN RAJA::forall LOADEXCH ATOMIC tests...." << endl;

  // initialize test counters for this test set
  s_ntests_run = 0;
  s_ntests_passed = 0;

  runBasicLoadExchAtomicTest<seq_exec,
                           cpu_nonatomic>(
      "seq_exec", in_array, alen);

  runBasicLoadExchAtomicTest<simd_exec,
                           cpu_nonatomic>(
      "simd_exec", in_array, alen);

#ifdef RAJA_ENABLE_OPENMP
  runBasicLoadExchAtomicTest<omp_parallel_for_exec,
                           cpu_atomic>(
      "omp_parallel_for_exec",
      in_array,
      alen);
#endif

#ifdef RAJA_ENABLE_CILK
  runBasicLoadExchAtomicTest<cilk_for_exec,
                           cpu_atomic>(
      "cilk_for_exec", in_array, alen);
#endif

  cout << "\n tests passed / test run: " << s_ntests_passed << " / "
       << s_ntests_run << endl;

  cout << "\n   END RAJA::forall LOADEXCH ATOMIC tests... " << endl;
}

///////////////////////////////////////////////////////////////////////////
//
// Method that defines and runs basic RAJA load store atomic tests
// based on execution policy template
//
///////////////////////////////////////////////////////////////////////////
template <typename POLICY_T, typename ATOMIC_POLICY_T, typename T>
void runBasicLoadStoreAtomicTest(const string& policy,
                              const T*,
                              Index_type)
{
  cout << "\n Test LOADSTORE atomics for " << policy << "\n";

  s_ntests_run++;
  s_ntests_run_total++;

  const unsigned long long int n = 8367;

  RAJA::atomic<ATOMIC_POLICY_T, unsigned long long int> atm_store_val0(n);
  RAJA::atomic<ATOMIC_POLICY_T, unsigned long long int> atm_store_val1(n);
  RAJA::atomic<ATOMIC_POLICY_T, unsigned long long int> atm_store_chk0(0);
  RAJA::atomic<ATOMIC_POLICY_T, unsigned long long int> atm_store_chk1(0);

  forall<POLICY_T>(0, (int)n, [=] (int i) {

      int store_i0 = int(atm_store_val0 = i);
      atm_store_val1.store(i);
      int store_i1 = atm_store_val1;

      if (store_i0 == i) atm_store_chk0++;
      if (store_i1 >= 0 && store_i1 < n) atm_store_chk1++;
  });

  if (   ((unsigned long long int)atm_store_chk0) != n
      || ((unsigned long long int)atm_store_chk1) != n ) {
    cout << "\n TEST FAILURE:"
         << endl;
    cout << setprecision(20) << "\tatm_load_chk = " << static_cast<unsigned long long int>(atm_store_chk0)
         << " (" << n << ") " << endl;
    cout << setprecision(20) << "\tatm_exch_prev_chk = " << static_cast<unsigned long long int>(atm_store_chk1)
         << " (" << n << ") " << endl;
  } else {
    s_ntests_passed++;
    s_ntests_passed_total++;
  }
}

///////////////////////////////////////////////////////////////////////////
//
// Run RAJA load store atomic tests with available RAJA execution policies....
//
///////////////////////////////////////////////////////////////////////////
template <typename T>
void runLoadStoreAtomicTests(const T* in_array,
                          Index_type alen)
{
  cout << "\n\n   BEGIN RAJA::forall LOAD STORE ATOMIC tests...." << endl;

  // initialize test counters for this test set
  s_ntests_run = 0;
  s_ntests_passed = 0;

  runBasicLoadStoreAtomicTest<seq_exec,
                           cpu_nonatomic>(
      "seq_exec", in_array, alen);

  runBasicLoadStoreAtomicTest<simd_exec,
                           cpu_nonatomic>(
      "simd_exec", in_array, alen);

#ifdef RAJA_ENABLE_OPENMP
  runBasicLoadStoreAtomicTest<omp_parallel_for_exec,
                           cpu_atomic>(
      "omp_parallel_for_exec",
      in_array,
      alen);
#endif

#ifdef RAJA_ENABLE_CILK
  runBasicLoadStoreAtomicTest<cilk_for_exec,
                           cpu_atomic>(
      "cilk_for_exec", in_array, alen);
#endif

  cout << "\n tests passed / test run: " << s_ntests_passed << " / "
       << s_ntests_run << endl;

  cout << "\n   END RAJA::forall LOAD STORE ATOMIC tests... " << endl;
}

///////////////////////////////////////////////////////////////////////////
//
// Method that defines and runs basic RAJA counting atomic tests
// based on execution policy template
//
///////////////////////////////////////////////////////////////////////////
template <typename POLICY_T, typename ATOMIC_POLICY_T, typename T>
void runBasicCountingAtomicTest(const string& policy,
                              const T* in_array,
                              Index_type alen)
{
  cout << "\n Test COUNTING atomics for " << policy << "\n";

  s_ntests_run++;
  s_ntests_run_total++;

  int* ivalue = new int[alen]();
  int* ivalue_check = new int[alen]();

  RAJA::atomic<ATOMIC_POLICY_T, int> atm_cnt_up(0);
  RAJA::atomic<ATOMIC_POLICY_T, int> atm_cnt_down(alen - 1);

  forall<POLICY_T>(0, alen, [=] (int i) {

    if (in_array[i] < 0.0) {

      ivalue[atm_cnt_down--] = i;

    } else {

      ivalue[atm_cnt_up++] = i;

    }

  });

  for(int i = 0; i < alen; i++) {
    ivalue_check[i] = 0;
  }

  int first_wrong = -1;

  for(int i = 0; i < alen && first_wrong == -1; i++) {
    if (ivalue[i] >= 0 && ivalue[i] < alen) {
      ivalue_check[ivalue[i]] += 1;
    } else {
      first_wrong = i;
    }
  }

  for(int i = 0; i < alen && first_wrong == -1; i++) {
    if (ivalue_check[i] != 1) {
      first_wrong = i;
    } else if (i < seq_pos_cnt && in_array[ivalue[i]] < 0.0) {
      first_wrong = i;
    } else if (i >= seq_pos_cnt && in_array[ivalue[i]] >= 0.0) {
      first_wrong = i;
    }
  }

  if (   ((int)atm_cnt_down) != (alen - seq_neg_cnt - 1)
      || ((int)atm_cnt_up) != seq_pos_cnt
      || (first_wrong != -1) ) {
    cout << "\n TEST FAILURE:"
         << endl;
    cout << setprecision(20) << "\tatm_count_down = " << static_cast<int>(atm_cnt_down)
         << " (" << (alen - seq_neg_cnt - 1) << ") " << endl;
    cout << setprecision(20) << "\tatm_count_up = " << static_cast<int>(atm_cnt_up)
         << " (" << seq_pos_cnt << ") " << endl;
    cout << setprecision(20) << "\tsomething wrong at i = " << first_wrong
         << " ivalue[i] = " << ivalue[first_wrong]
         << " ivalue_check[i] = " << ivalue_check[first_wrong]
         << " in_array[ivalue[i]] = " << ((ivalue[first_wrong] >= 0 && ivalue[first_wrong] < alen) ? in_array[ivalue[first_wrong]] : 0.0)
         << endl;
  } else {
    s_ntests_passed++;
    s_ntests_passed_total++;
  }
  delete[] ivalue;
  delete[] ivalue_check;
}

///////////////////////////////////////////////////////////////////////////
//
// Run RAJA counting atomic tests with available RAJA execution policies....
//
///////////////////////////////////////////////////////////////////////////
template <typename T>
void runCountingAtomicTests(const T* in_array,
                          Index_type alen)
{
  cout << "\n\n   BEGIN RAJA::forall COUNTING ATOMIC tests...." << endl;

  // initialize test counters for this test set
  s_ntests_run = 0;
  s_ntests_passed = 0;

  runBasicCountingAtomicTest<seq_exec,
                           cpu_nonatomic>(
      "seq_exec", in_array, alen);

  runBasicCountingAtomicTest<simd_exec,
                           cpu_nonatomic>(
      "simd_exec", in_array, alen);

#ifdef RAJA_ENABLE_OPENMP
  runBasicCountingAtomicTest<omp_parallel_for_exec,
                           cpu_atomic>(
      "omp_parallel_for_exec",
      in_array,
      alen);
#endif

#ifdef RAJA_ENABLE_CILK
  runBasicCountingAtomicTest<cilk_for_exec,
                           cpu_atomic>(
      "cilk_for_exec", in_array, alen);
#endif

  cout << "\n tests passed / test run: " << s_ntests_passed << " / "
       << s_ntests_run << endl;

  cout << "\n   END RAJA::forall COUNTING ATOMIC tests... " << endl;
}

///////////////////////////////////////////////////////////////////////////
//
// Main Program.
//
///////////////////////////////////////////////////////////////////////////

int main(int RAJA_UNUSED_ARG(argc), char** RAJA_UNUSED_ARG(argv))
{
  cout << "\n Begin RAJA Atomic tests!!! " << endl;



  ///////////////////////////////////////////////////////////////////////////
  //
  // Run RAJA::forall atomic tests...
  //
  ///////////////////////////////////////////////////////////////////////////

  const Index_type array_length = TEST_VEC_LEN;

  //
  // Allocate and initialize managed data arrays
  //

  Real_ptr rand_dvalue;

  Index_type iinit_val = -1;
  Index_type* ivalue;
  Index_type* ivalue_check;
  Index_type* rand_ivalue;

  int err_val = 0; 
  err_val += posix_memalign((void**)&rand_dvalue,
                            DATA_ALIGN,
                            array_length * sizeof(Real_type));

  err_val += posix_memalign((void**)&ivalue,
                            DATA_ALIGN,
                            array_length * sizeof(Index_type));

  err_val += posix_memalign((void**)&ivalue_check,
                            DATA_ALIGN,
                            array_length * sizeof(Index_type));

  err_val += posix_memalign((void**)&rand_ivalue,
                            DATA_ALIGN,
                            array_length * sizeof(Index_type));
  RAJA_UNUSED_VAR(err_val);


  //
  // initialize arrays and set global check variables
  //
  seq_min = 1.0;
  seq_max =-1.0;
  seq_sum = 0.0;
  seq_pos_sum = 0.0;
  seq_neg_sum = 0.0;
  seq_prod = 1.0;
  seq_pos_cnt = 0;
  seq_neg_cnt = 0;
  seq_and = 0xffffffff;
  seq_or = 0x0;
  seq_xor = 0x0f0f0f0f;

  for (int i = 0; i < TEST_VEC_LEN; ++i) {
    // create distribution equally distributed in (-2, -1], (-1, -0.5], [0.5, 1), [1, 2)
    Real_type tmp = drand48();
    if (tmp < 0.5) tmp = 2.0 * tmp + 1.0;
    if (drand48() < 0.5) tmp = -tmp;
    rand_dvalue[i] = tmp;

    seq_sum += rand_dvalue[i];
    seq_prod *= rand_dvalue[i];
    if (rand_dvalue[i] < seq_min) {
      seq_min = rand_dvalue[i];
    }
    if (rand_dvalue[i] > seq_max) {
      seq_max = rand_dvalue[i];
    }
    if(rand_dvalue[i] < 0.0) {
      seq_neg_cnt++;
      seq_neg_sum += rand_dvalue[i];
    }
    else {
      seq_pos_cnt++;
      seq_pos_sum += rand_dvalue[i];
    }

    ivalue[i] = iinit_val;
    ivalue_check[i] = iinit_val;
    rand_ivalue[i] = i << 3;
    seq_and &= rand_ivalue[i];
    seq_or |= rand_ivalue[i];
    seq_xor ^= rand_ivalue[i];
  }

  //
  // Run actual tests.
  //
  runMinMaxAtomicTests(rand_dvalue, TEST_VEC_LEN);

  runAddSubAtomicTests(rand_dvalue, TEST_VEC_LEN);

  runAndOrXorAtomicTests(rand_ivalue, TEST_VEC_LEN);

  runIncDecAtomicTests(rand_dvalue, TEST_VEC_LEN);

  runCompExchAtomicTests(rand_dvalue, TEST_VEC_LEN);

  runLoadExchAtomicTests((void*)0, 0);

  runLoadStoreAtomicTests((void*)0, 0);

  runCountingAtomicTests(rand_dvalue, TEST_VEC_LEN);
  

  ///
  /// Print total number of tests passed/run.
  ///
  cout << "\n All Tests : # passed / # run = " << s_ntests_passed_total << " / "
       << s_ntests_run_total << endl;

  //
  // Clean up....
  //
  free(rand_dvalue);
  free(ivalue);
  free(ivalue_check);
  free(rand_ivalue);

  cout << "\n DONE!!! " << endl;

  if (s_ntests_passed_total == s_ntests_run_total) {
    return 0;
  } else {
    return 1;
  }
}
