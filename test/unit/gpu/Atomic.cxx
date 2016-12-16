/*
 * Copyright (c) 2016, Lawrence Livermore National Security, LLC.
 *
 * Produced at the Lawrence Livermore National Laboratory.
 *
 * All rights reserved.
 *
 * For release details and restrictions, please see raja/README-license.txt
 */

#include <math.h>
#include <cfloat>
#include <cstdio>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>

#include "RAJA/RAJA.hxx"

#include "Compare.hxx"

#define TEST_VEC_LEN 1024 * 24

using namespace RAJA;
using namespace std;

//
// Global variables for counting tests executed/passed.
//
unsigned s_ntests_run = 0;
unsigned s_ntests_passed = 0;

int main(int argc, char *argv[])
{
  cout << "\n Begin RAJA GPU Atomic tests!!! " << endl;

  const int test_repeat = 10;

  //
  // Allocate and initialize managed data arrays
  //

  double *rand_dvalue;

  int iinit_val = -1;
  int *ivalue;
  int* ivalue_check;
  int *rand_ivalue;

  cudaMallocManaged((void **)&rand_dvalue,
                    sizeof(double) * TEST_VEC_LEN,
                    cudaMemAttachGlobal);

  cudaMallocManaged((void **)&ivalue,
                    sizeof(int) * TEST_VEC_LEN,
                    cudaMemAttachGlobal);
  ivalue_check = new int[TEST_VEC_LEN];
  for (int i = 0; i < TEST_VEC_LEN; ++i) {
    ivalue_check[i] = iinit_val;
    ivalue[i] = iinit_val;
  }

  cudaMallocManaged((void **)&rand_ivalue,
                    sizeof(int) * TEST_VEC_LEN,
                    cudaMemAttachGlobal);

  ///
  /// Define thread block size for CUDA exec policy
  ///
  const size_t block_size = 256;

  ////////////////////////////////////////////////////////////////////////////
  // Run 3 different sum reduction tests in a loop
  ////////////////////////////////////////////////////////////////////////////

  for (int tcount = 0; tcount < test_repeat; ++tcount) {
    cout << "\t tcount = " << tcount << endl;

    double seq_min = 1.0;
    double seq_max =-1.0;
    double seq_sum = 0.0;
    double seq_pos_sum = 0.0;
    double seq_neg_sum = 0.0;
    double seq_prod = 1.0;
    int seq_pos_cnt = 0;
    int seq_neg_cnt = 0;
    int seq_and = 0xffffffff;
    int seq_or = 0x0;
    int seq_xor = 0x00ff00ff;

    for (int i = 0; i < TEST_VEC_LEN; ++i) {
      // create distribution with product that doesn't diverge too often
      double tmp = drand48();
      tmp = 1.103050709 * tmp + 0.5;
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
      rand_ivalue[i] = i << 3;
      seq_and &= rand_ivalue[i];
      seq_or |= rand_ivalue[i];
      seq_xor ^= rand_ivalue[i];
    }

    //
    // test 1 runs atomics using fetch_min, fetch_max.
    //
    {  // begin test 1
      s_ntests_run++;

      RAJA::atomic<cuda_atomic, double> atm_min(1.0);
      RAJA::atomic<cuda_atomic, double> atm_max(1.0);

      forall<cuda_exec<block_size> >(0, TEST_VEC_LEN, [=] __device__(int i) {
        atm_min.fetch_min(rand_dvalue[i]);
        atm_max.fetch_max(rand_dvalue[i]);
      });

      if (   !equal(double(atm_min), seq_min)
          || !equal(double(atm_max), seq_max) ) {
        cout << "\n TEST 1 FAILURE: tcount = " << tcount
             << endl;
        cout << setprecision(20) << "\tatm_min = " << static_cast<double>(atm_min)
             << " (" << seq_min << ") " << endl;
        cout << setprecision(20) << "\tatm_max = " << static_cast<double>(atm_max)
             << " (" << seq_max << ") " << endl;
      } else {
        s_ntests_passed++;
      }
    } // end test 1

    ///////////////////////////////////////////////////////////////////////////
    ///////////////////////////////////////////////////////////////////////////

    //
    // test 2 runs atomics using fetch_add, fetch_sub, a+=, a-=.
    //
    {  // begin test 2
      s_ntests_run++;
    
      RAJA::atomic<cuda_atomic, double> atm_fad_sum(0.0);
      RAJA::atomic<cuda_atomic, double> atm_fsb_sum(0.0);
      RAJA::atomic<cuda_atomic, double> atm_ple_sum(0.0);
      RAJA::atomic<cuda_atomic, double> atm_mie_sum(0.0);
      RAJA::atomic<cuda_atomic, double> atm_pos_sum(0.0);
      RAJA::atomic<cuda_atomic, double> atm_neg_sum(0.0);

      forall<cuda_exec<block_size> >(0, TEST_VEC_LEN, [=] __device__(int i) {
        atm_fad_sum.fetch_add(rand_dvalue[i]);
        atm_fsb_sum.fetch_sub(rand_dvalue[i]);
        atm_ple_sum += rand_dvalue[i];
        atm_mie_sum -= rand_dvalue[i];

        if(rand_dvalue[i] < 0.0) {
          atm_neg_sum += rand_dvalue[i];
        }
        else {
          atm_pos_sum += rand_dvalue[i];
        }
      });

      if (   !equal(double(atm_fad_sum), seq_sum)
          || !equal(double(atm_fsb_sum), -seq_sum)
          || !equal(double(atm_ple_sum), seq_sum)
          || !equal(double(atm_mie_sum), -seq_sum)
          || !equal(double(atm_pos_sum), seq_pos_sum)
          || !equal(double(atm_neg_sum), seq_neg_sum) ) {
        cout << "\n TEST 2 FAILURE: tcount = " << tcount
             << endl;
        cout << setprecision(20) << "\tatm_fetch_add_sum = " << static_cast<double>(atm_fad_sum)
             << " (" << seq_sum << ") " << endl;
        cout << setprecision(20) << "\tatm_fetch_sub_sum = " << static_cast<double>(atm_fsb_sum)
             << " (" << -seq_sum << ") " << endl;
        cout << setprecision(20) << "\tatm_+=_sum = " << static_cast<double>(atm_ple_sum)
             << " (" << seq_sum << ") " << endl;
        cout << setprecision(20) << "\tatm_-=_sum = " << static_cast<double>(atm_mie_sum)
             << " (" << -seq_sum << ") " << endl;
        cout << setprecision(20) << "\tatm_pos_sum = " << static_cast<double>(atm_pos_sum)
             << " (" << seq_pos_sum << ") " << endl;
        cout << setprecision(20) << "\tatm_neg_sum = " << static_cast<double>(atm_neg_sum)
             << " (" << seq_neg_sum << ") " << endl;
      } else {
        s_ntests_passed++;
      }
    } // end test 2

    ///////////////////////////////////////////////////////////////////////////
    ///////////////////////////////////////////////////////////////////////////

    //
    // test 3 runs atomics using fetch_and, fetch_or, fetch_xor, a&=, a|=, a^=.
    //
    {  // begin test 3
      s_ntests_run++;
    
      RAJA::atomic<cuda_atomic, int> atm_fan(0xffffffff);
      RAJA::atomic<cuda_atomic, int> atm_for(0x0);
      RAJA::atomic<cuda_atomic, int> atm_fxr(0x00ff00ff);
      RAJA::atomic<cuda_atomic, int> atm_ane(0xffffffff);
      RAJA::atomic<cuda_atomic, int> atm_ore(0x0);
      RAJA::atomic<cuda_atomic, int> atm_xre(0x00ff00ff);

      forall<cuda_exec<block_size> >(0, TEST_VEC_LEN, [=] __device__(int i) {
        atm_fan.fetch_and(rand_ivalue[i]);
        atm_for.fetch_or(rand_ivalue[i]);
        atm_fxr.fetch_xor(rand_ivalue[i]);
        atm_ane &= rand_ivalue[i];
        atm_ore |= rand_ivalue[i];
        atm_xre ^= rand_ivalue[i];
      });

      if (   !equal(int(atm_fan), seq_and)
          || !equal(int(atm_for), seq_or) 
          || !equal(int(atm_fxr), seq_xor)
          || !equal(int(atm_ane), seq_and)
          || !equal(int(atm_ore), seq_or) 
          || !equal(int(atm_xre), seq_xor) ) {
        cout << "\n TEST 3 FAILURE: tcount = " << tcount
             << endl;
        cout << setprecision(20) << "\tatm_fetch_and = " << static_cast<int>(atm_fan)
             << " (" << seq_and << ") " << endl;
        cout << setprecision(20) << "\tatm_fetch_or = " << static_cast<int>(atm_for)
             << " (" << seq_or << ") " << endl;
        cout << setprecision(20) << "\tatm_fetch_xor = " << static_cast<int>(atm_fxr)
             << " (" << seq_xor << ") " << endl;
        cout << setprecision(20) << "\tatm_&= = " << static_cast<int>(atm_ane)
             << " (" << seq_and << ") " << endl;
        cout << setprecision(20) << "\tatm_|= = " << static_cast<int>(atm_ore)
             << " (" << seq_or << ") " << endl;
        cout << setprecision(20) << "\tatm_^= = " << static_cast<int>(atm_xre)
             << " (" << seq_xor << ") " << endl;
      } else {
        s_ntests_passed++;
      }
    } // end test 3

    ///////////////////////////////////////////////////////////////////////////
    ///////////////////////////////////////////////////////////////////////////

    //
    // test 4 runs atomics using a++, ++a, a--, --a.
    //
    {  // begin test 4
      s_ntests_run++;
    
      RAJA::atomic<cuda_atomic, int> atm_pos_post_inc(0);
      RAJA::atomic<cuda_atomic, int> atm_pos_pre_inc(0);
      RAJA::atomic<cuda_atomic, int> atm_pos_post_dec(0);
      RAJA::atomic<cuda_atomic, int> atm_pos_pre_dec(0);
      RAJA::atomic<cuda_atomic, int> atm_neg_post_inc(0);
      RAJA::atomic<cuda_atomic, int> atm_neg_pre_inc(0);
      RAJA::atomic<cuda_atomic, int> atm_neg_post_dec(0);
      RAJA::atomic<cuda_atomic, int> atm_neg_pre_dec(0);

      forall<cuda_exec<block_size> >(0, TEST_VEC_LEN, [=] __device__(int i) {
        if(rand_dvalue[i] < 0.0) {
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
        cout << "\n TEST 4 FAILURE: tcount = " << tcount
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
      }
    } // end test 4

    ///////////////////////////////////////////////////////////////////////////
    ///////////////////////////////////////////////////////////////////////////

    //
    // test 5 runs atomics using (T)a, load, compare_exchange_weak, compare_exchange_strong.
    // implements *= using compare_exchange.
    //
    {  // begin test 5
      s_ntests_run++;
    
      RAJA::atomic<cuda_atomic, double> atm_cew_prod(1.0);
      RAJA::atomic<cuda_atomic, double> atm_ces_prod(1.0);

      forall<cuda_exec<block_size> >(0, TEST_VEC_LEN, [=] __device__(int i) {

        double expect = atm_cew_prod.load();
        while (!atm_cew_prod.compare_exchange_weak(expect, expect * rand_dvalue[i]));

        expect = (double)atm_ces_prod;
        while (!atm_ces_prod.compare_exchange_strong(expect, expect * rand_dvalue[i]));
      });

      if (   !equal(double(atm_cew_prod), seq_prod)
          || !equal(double(atm_ces_prod), seq_prod) ) {
        cout << "\n TEST 5 FAILURE: tcount = " << tcount
             << endl;
        cout << setprecision(20) << "\tatm_compare_exchange_weak_prod = " << static_cast<double>(atm_cew_prod)
             << " (" << seq_prod << ") " << endl;
        cout << setprecision(20) << "\tatm_compare_exchange_strong_prod = " << static_cast<double>(atm_ces_prod)
             << " (" << seq_prod << ") " << endl;
      } else {
        s_ntests_passed++;
      }
    } // end test 5

    ///////////////////////////////////////////////////////////////////////////
    ///////////////////////////////////////////////////////////////////////////

    //
    // test 6 runs atomics using load, exchange.
    //
    {  // begin test 6
      s_ntests_run++;
    
      const unsigned long long int n = 5317;

      RAJA::atomic<cuda_atomic, unsigned long long int> atm_exch_val(0);
      RAJA::atomic<cuda_atomic, unsigned long long int> atm_sum(0);
      RAJA::atomic<cuda_atomic, int> atm_load_val_chk(0);
      RAJA::atomic<cuda_atomic, int> atm_exch_prev_chk(0);
      RAJA::atomic<cuda_atomic, int> atm_load_later_chk(0);

      forall<cuda_exec<block_size> >(0, (int)n, [=] __device__(int i) {
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
        cout << "\n TEST 6 FAILURE: tcount = " << tcount
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
      }
    } // end test 6

    ///////////////////////////////////////////////////////////////////////////
    ///////////////////////////////////////////////////////////////////////////

    //
    // test 7 runs atomics using a=(T), store.
    //
    {  // begin test 7
      s_ntests_run++;
    
      const unsigned long long int n = 8367;

      RAJA::atomic<cuda_atomic, unsigned long long int> atm_store_val0(n);
      RAJA::atomic<cuda_atomic, unsigned long long int> atm_store_val1(n);
      RAJA::atomic<cuda_atomic, unsigned long long int> atm_store_chk0(0);
      RAJA::atomic<cuda_atomic, unsigned long long int> atm_store_chk1(0);

      forall<cuda_exec<block_size> >(0, (int)n, [=] __device__(int i) {

          int store_i0 = int(atm_store_val0 = i);
          atm_store_val1.store(i);
          int store_i1 = atm_store_val1;

          if (store_i0 == i) atm_store_chk0++;
          if (store_i1 >= 0 && store_i1 < n) atm_store_chk1++;
      });

      if (   ((unsigned long long int)atm_store_chk0) != n
          || ((unsigned long long int)atm_store_chk1) != n ) {
        cout << "\n TEST 7 FAILURE: tcount = " << tcount
             << endl;
        cout << setprecision(20) << "\tatm_load_chk = " << static_cast<unsigned long long int>(atm_store_chk0)
             << " (" << n << ") " << endl;
        cout << setprecision(20) << "\tatm_exch_prev_chk = " << static_cast<unsigned long long int>(atm_store_chk1)
             << " (" << n << ") " << endl;
      } else {
        s_ntests_passed++;
      }
    } // end test 7

    ///////////////////////////////////////////////////////////////////////////
    ///////////////////////////////////////////////////////////////////////////

    //
    // test 8 use atomics to add values to a list.
    //
    {  // begin test 8
      s_ntests_run++;

      RAJA::atomic<cuda_atomic, int> atm_cnt_up(0);
      RAJA::atomic<cuda_atomic, int> atm_cnt_down(TEST_VEC_LEN - 1);

      forall<cuda_exec<block_size> >(0, TEST_VEC_LEN, [=] __device__(int i) {

        if (rand_dvalue[i] < 0.0) {

          ivalue[atm_cnt_down--] = i;

        } else {

          ivalue[atm_cnt_up++] = i;

        }

      });

      for(int i = 0; i < TEST_VEC_LEN; i++) {
        ivalue_check[i] = 0;
      }

      int first_wrong = -1;

      for(int i = 0; i < TEST_VEC_LEN && first_wrong == -1; i++) {
        if (ivalue[i] >= 0 && ivalue[i] < TEST_VEC_LEN) {
          ivalue_check[ivalue[i]] += 1;
        } else {
          first_wrong = i;
        }
      }

      for(int i = 0; i < TEST_VEC_LEN && first_wrong == -1; i++) {
        if (ivalue_check[i] != 1) {
          first_wrong = i;
        } else if (i < seq_pos_cnt && rand_dvalue[ivalue[i]] < 0.0) {
          first_wrong = i;
        } else if (i >= seq_pos_cnt && rand_dvalue[ivalue[i]] >= 0.0) {
          first_wrong = i;
        }
      }

      if (   ((int)atm_cnt_down) != (TEST_VEC_LEN - seq_neg_cnt - 1)
          || ((int)atm_cnt_up) != seq_pos_cnt
          || (first_wrong != -1) ) {
        cout << "\n TEST 7 FAILURE: tcount = " << tcount
             << endl;
        cout << setprecision(20) << "\tatm_count_down = " << static_cast<int>(atm_cnt_down)
             << " (" << (TEST_VEC_LEN - seq_neg_cnt - 1) << ") " << endl;
        cout << setprecision(20) << "\tatm_count_up = " << static_cast<int>(atm_cnt_up)
             << " (" << seq_pos_cnt << ") " << endl;
        cout << setprecision(20) << "\tsomething wrong at i = " << first_wrong
             << " ivalue[i] = " << ivalue[first_wrong]
             << " ivalue_check[i] = " << ivalue_check[first_wrong]
             << " rand_dvalue[ivalue[i]] = " << ((ivalue[first_wrong] >= 0 && ivalue[first_wrong] < TEST_VEC_LEN) ? rand_dvalue[ivalue[first_wrong]] : 0.0)
             << endl;
      } else {
        s_ntests_passed++;
      }
    } // end test 8

    ///////////////////////////////////////////////////////////////////////////
    ///////////////////////////////////////////////////////////////////////////

  }  // end test repeat loop

  ///
  /// Print total number of tests passed/run.
  ///
  cout << "\n Tests Passed / Tests Run = " << s_ntests_passed << " / "
       << s_ntests_run << endl;

  cudaFree(rand_dvalue);
  cudaFree(ivalue);
  cudaFree(rand_ivalue);
  delete[] ivalue_check;

  return !(s_ntests_passed == s_ntests_run);
}
