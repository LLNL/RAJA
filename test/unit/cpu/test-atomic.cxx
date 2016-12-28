/*
 * Copyright (c) 2016, Lawrence Livermore National Security, LLC.
 *
 * Produced at the Lawrence Livermore National Laboratory.
 *
 * All rights reserved.
 *
 * For release details and restrictions, please see raja/README-license.txt
 */

#include <time.h>
#include <cmath>
#include <cfloat>
#include <cstdlib>

#include <iostream>
#include <iomanip>
#include <string>
#include <random>
#include <vector>

#include <gtest/gtest.h>

#include "RAJA/RAJA.hxx"
#include "RAJA/internal/defines.hxx"

#include "Compare.hxx"


template < typename T >
struct policy_traits {
};

template < >
struct policy_traits < RAJA::seq_exec > {
  using type = RAJA::cpu_nonatomic;
};

template < >
struct policy_traits < RAJA::simd_exec > {
  using type = RAJA::cpu_nonatomic;
};

#ifdef RAJA_ENABLE_OPENMP
template < >
struct policy_traits < RAJA::omp_parallel_for_exec > {
  using type = RAJA::cpu_atomic;
};
#endif

#ifdef RAJA_ENABLE_CILK
template < >
struct policy_traits < RAJA::cilk_for_exec > {
  using type = RAJA::cpu_atomic;
};
#endif

const RAJA::Index_type array_length = 1024 * 32;

template <typename T>
class AtomicTest : public ::testing::Test
{
protected:
  virtual void SetUp()
  {
    this->rand_dvalue = (RAJA::Real_ptr) RAJA::allocate_aligned(RAJA::DATA_ALIGN,
                   array_length * sizeof(RAJA::Real_type));
    this->rand_ivalue = (RAJA::Index_type*) RAJA::allocate_aligned(RAJA::DATA_ALIGN,
                   array_length * sizeof(RAJA::Index_type));

    this->seq_min = 1.0;
    this->seq_max =-1.0;
    this->seq_sum = 0.0;
    this->seq_pos_sum = 0.0;
    this->seq_neg_sum = 0.0;
    this->seq_prod = 1.0;
    this->seq_pos_cnt = 0;
    this->seq_neg_cnt = 0;
    this->seq_and = 0xffffffff;
    this->seq_or = 0x0;
    this->seq_xor = 0x0f0f0f0f;

    for (int i = 0; i < array_length; ++i) {

      RAJA::Real_type tmp = drand48();
      if (tmp < 0.5) tmp = 2.0 * tmp + 1.0;
      if (drand48() < 0.5) tmp = -tmp;
      this->rand_dvalue[i] = tmp;

      this->seq_sum += this->rand_dvalue[i];
      this->seq_prod *= this->rand_dvalue[i];
      if (this->rand_dvalue[i] < this->seq_min) {
        this->seq_min = this->rand_dvalue[i];
      }
      if (this->rand_dvalue[i] > this->seq_max) {
        this->seq_max = this->rand_dvalue[i];
      }
      if(this->rand_dvalue[i] < 0.0) {
        this->seq_neg_cnt++;
        this->seq_neg_sum += this->rand_dvalue[i];
      }
      else {
        this->seq_pos_cnt++;
        this->seq_pos_sum += this->rand_dvalue[i];
      }

      this->rand_ivalue[i] = i << 3;
      this->seq_and &= this->rand_ivalue[i];
      this->seq_or |= this->rand_ivalue[i];
      this->seq_xor ^= this->rand_ivalue[i];
    }
  }

  virtual void TearDown()
  {
    RAJA::free_aligned(this->rand_dvalue);
    RAJA::free_aligned(this->rand_ivalue);
  }

  RAJA::Real_ptr rand_dvalue;
  RAJA::Index_type* rand_ivalue;

  RAJA::Real_type seq_min;
  RAJA::Real_type seq_max;
  RAJA::Real_type seq_sum;
  RAJA::Real_type seq_pos_sum;
  RAJA::Real_type seq_neg_sum;
  RAJA::Real_type seq_prod;
  int seq_pos_cnt;
  int seq_neg_cnt;
  int seq_and;
  int seq_or;
  int seq_xor;
};

TYPED_TEST_CASE_P(AtomicTest);


///////////////////////////////////////////////////////////////////////////
//
// Method that defines and runs basic RAJA min max atomic tests
//
///////////////////////////////////////////////////////////////////////////
TYPED_TEST_P(AtomicTest, MinMax)
{

  using POLICY_T = TypeParam;
  using ATOMIC_POLICY_T = typename policy_traits<POLICY_T>::type;
  using T = RAJA::Real_type;
  const T* in_array = this->rand_dvalue;
  RAJA::Index_type alen = array_length;

  RAJA::atomic<ATOMIC_POLICY_T, T> atm_min(T(1));
  RAJA::atomic<ATOMIC_POLICY_T, T> atm_max(T(1));

  RAJA::forall<TypeParam>(0, alen, [=] (int i) {
    atm_min.fetch_min(in_array[i]);
    atm_max.fetch_max(in_array[i]);
  });

  EXPECT_TRUE( RAJA::equal(T(atm_min), this->seq_min) );
  EXPECT_TRUE( RAJA::equal(T(atm_max), this->seq_max) );
  
}

///////////////////////////////////////////////////////////////////////////
//
// Method that defines and runs basic RAJA add sub atomic tests
//
///////////////////////////////////////////////////////////////////////////
TYPED_TEST_P(AtomicTest, AddSub)
{

  using POLICY_T = TypeParam;
  using ATOMIC_POLICY_T = typename policy_traits<POLICY_T>::type;
  using T = RAJA::Real_type;
  const T* in_array = this->rand_dvalue;
  RAJA::Index_type alen = array_length;
    
  RAJA::atomic<ATOMIC_POLICY_T, T> atm_fad_sum(T(0));
  RAJA::atomic<ATOMIC_POLICY_T, T> atm_fsb_sum(T(0));
  RAJA::atomic<ATOMIC_POLICY_T, T> atm_ple_sum(T(0));
  RAJA::atomic<ATOMIC_POLICY_T, T> atm_mie_sum(T(0));
  RAJA::atomic<ATOMIC_POLICY_T, T> atm_pos_sum(T(0));
  RAJA::atomic<ATOMIC_POLICY_T, T> atm_neg_sum(T(0));

  RAJA::forall<POLICY_T>(0, alen, [=] (int i) {
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

  EXPECT_TRUE( RAJA::equal(T(atm_fad_sum), this->seq_sum) );
  EXPECT_TRUE( RAJA::equal(T(atm_fsb_sum), -this->seq_sum) );
  EXPECT_TRUE( RAJA::equal(T(atm_ple_sum), this->seq_sum) );
  EXPECT_TRUE( RAJA::equal(T(atm_mie_sum), -this->seq_sum) );
  EXPECT_TRUE( RAJA::equal(T(atm_pos_sum), this->seq_pos_sum) );
  EXPECT_TRUE( RAJA::equal(T(atm_neg_sum), this->seq_neg_sum) );

}

///////////////////////////////////////////////////////////////////////////
//
// Method that defines and runs basic RAJA and or xor atomic tests
//
///////////////////////////////////////////////////////////////////////////
TYPED_TEST_P(AtomicTest, AndOrXor)
{

  using POLICY_T = TypeParam;
  using ATOMIC_POLICY_T = typename policy_traits<POLICY_T>::type;
  using T = RAJA::Index_type;
  const T* in_array = this->rand_ivalue;
  RAJA::Index_type alen = array_length;
  
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

  RAJA::forall<POLICY_T>(0, alen, [=] (int i) {
    atm_fan.fetch_and(in_array[i]);
    atm_for.fetch_or(in_array[i]);
    atm_fxr.fetch_xor(in_array[i]);
    atm_ane &= in_array[i];
    atm_ore |= in_array[i];
    atm_xre ^= in_array[i];
  });

  EXPECT_TRUE( RAJA::equal(T(atm_fan), this->seq_and) );
  EXPECT_TRUE( RAJA::equal(T(atm_for), this->seq_or) );
  EXPECT_TRUE( RAJA::equal(T(atm_fxr), this->seq_xor) );
  EXPECT_TRUE( RAJA::equal(T(atm_ane), this->seq_and) );
  EXPECT_TRUE( RAJA::equal(T(atm_ore), this->seq_or) );
  EXPECT_TRUE( RAJA::equal(T(atm_xre), this->seq_xor) );
}

///////////////////////////////////////////////////////////////////////////
//
// Method that defines and runs basic RAJA inc dec atomic tests
//
///////////////////////////////////////////////////////////////////////////
TYPED_TEST_P(AtomicTest, IncDec)
{

  using POLICY_T = TypeParam;
  using ATOMIC_POLICY_T = typename policy_traits<POLICY_T>::type;
  using T = RAJA::Real_type;
  const T* in_array = this->rand_dvalue;
  RAJA::Index_type alen = array_length;
    
  RAJA::atomic<ATOMIC_POLICY_T, int> atm_pos_post_inc(0);
  RAJA::atomic<ATOMIC_POLICY_T, int> atm_pos_pre_inc(0);
  RAJA::atomic<ATOMIC_POLICY_T, int> atm_pos_post_dec(0);
  RAJA::atomic<ATOMIC_POLICY_T, int> atm_pos_pre_dec(0);
  RAJA::atomic<ATOMIC_POLICY_T, int> atm_neg_post_inc(0);
  RAJA::atomic<ATOMIC_POLICY_T, int> atm_neg_pre_inc(0);
  RAJA::atomic<ATOMIC_POLICY_T, int> atm_neg_post_dec(0);
  RAJA::atomic<ATOMIC_POLICY_T, int> atm_neg_pre_dec(0);

  RAJA::forall<POLICY_T>(0, alen, [=] (int i) {
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

  EXPECT_TRUE( RAJA::equal(int(atm_pos_post_inc), this->seq_pos_cnt) );
  EXPECT_TRUE( RAJA::equal(int(atm_pos_pre_inc),  this->seq_pos_cnt) );
  EXPECT_TRUE( RAJA::equal(int(atm_pos_post_dec),-this->seq_pos_cnt) );
  EXPECT_TRUE( RAJA::equal(int(atm_pos_pre_dec), -this->seq_pos_cnt) );
  EXPECT_TRUE( RAJA::equal(int(atm_neg_post_inc), this->seq_neg_cnt) );
  EXPECT_TRUE( RAJA::equal(int(atm_neg_pre_inc),  this->seq_neg_cnt) );
  EXPECT_TRUE( RAJA::equal(int(atm_neg_post_dec),-this->seq_neg_cnt) );
  EXPECT_TRUE( RAJA::equal(int(atm_neg_pre_dec), -this->seq_neg_cnt) );
}

///////////////////////////////////////////////////////////////////////////
//
// Method that defines and runs basic RAJA compexch atomic tests
//
///////////////////////////////////////////////////////////////////////////
TYPED_TEST_P(AtomicTest, CompExch)
{

  using POLICY_T = TypeParam;
  using ATOMIC_POLICY_T = typename policy_traits<POLICY_T>::type;
  using T = RAJA::Real_type;
  const T* in_array = this->rand_dvalue;
  RAJA::Index_type alen = array_length;

  RAJA::atomic<ATOMIC_POLICY_T, T> atm_cew_prod(1.0);
  RAJA::atomic<ATOMIC_POLICY_T, T> atm_ces_prod(1.0);

  RAJA::forall<POLICY_T>(0, alen, [=] (int i) {

    T expect = atm_cew_prod.load();
    while (!atm_cew_prod.compare_exchange_weak(expect, expect * in_array[i]));

    expect = (T)atm_ces_prod;
    while (!atm_ces_prod.compare_exchange_strong(expect, expect * in_array[i]));
  });

  EXPECT_TRUE( RAJA::equal(T(atm_cew_prod), this->seq_prod) );
  EXPECT_TRUE( RAJA::equal(T(atm_ces_prod), this->seq_prod) );
}

///////////////////////////////////////////////////////////////////////////
//
// Method that defines and runs basic RAJA load exch atomic tests
//
///////////////////////////////////////////////////////////////////////////
TYPED_TEST_P(AtomicTest, LoadExch)
{

  using POLICY_T = TypeParam;
  using ATOMIC_POLICY_T = typename policy_traits<POLICY_T>::type;
  using T = RAJA::Index_type;

  const unsigned long long int n = 5317;

  RAJA::atomic<ATOMIC_POLICY_T, unsigned long long int> atm_exch_val(0);
  RAJA::atomic<ATOMIC_POLICY_T, unsigned long long int> atm_sum(0);
  RAJA::atomic<ATOMIC_POLICY_T, int> atm_load_val_chk(0);
  RAJA::atomic<ATOMIC_POLICY_T, int> atm_exch_prev_chk(0);
  RAJA::atomic<ATOMIC_POLICY_T, int> atm_load_later_chk(0);

  RAJA::forall<POLICY_T>(0, (int)n, [=] (int i) {
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

  EXPECT_EQ( int(atm_load_val_chk), n );
  EXPECT_EQ( int(atm_exch_prev_chk), n );
  EXPECT_EQ( int(atm_load_later_chk), n );
  EXPECT_TRUE( diff >= 1 && diff <= n );
}

///////////////////////////////////////////////////////////////////////////
//
// Method that defines and runs basic RAJA load store atomic tests
//
///////////////////////////////////////////////////////////////////////////
TYPED_TEST_P(AtomicTest, LoadStore)
{

  using POLICY_T = TypeParam;
  using ATOMIC_POLICY_T = typename policy_traits<POLICY_T>::type;
  using T = RAJA::Index_type;

  const unsigned long long int n = 8367;

  RAJA::atomic<ATOMIC_POLICY_T, unsigned long long int> atm_store_val0(n);
  RAJA::atomic<ATOMIC_POLICY_T, unsigned long long int> atm_store_val1(n);
  RAJA::atomic<ATOMIC_POLICY_T, unsigned long long int> atm_store_chk0(0);
  RAJA::atomic<ATOMIC_POLICY_T, unsigned long long int> atm_store_chk1(0);

  RAJA::forall<POLICY_T>(0, (int)n, [=] (int i) {

      int store_i0 = int(atm_store_val0 = i);
      atm_store_val1.store(i);
      int store_i1 = atm_store_val1;

      if (store_i0 == i) atm_store_chk0++;
      if (store_i1 >= 0 && store_i1 < n) atm_store_chk1++;
  });

  EXPECT_EQ( (unsigned long long int)atm_store_chk0, n );
  EXPECT_EQ( (unsigned long long int)atm_store_chk1, n );
}

///////////////////////////////////////////////////////////////////////////
//
// Method that defines and runs basic RAJA counting atomic tests
//
///////////////////////////////////////////////////////////////////////////
TYPED_TEST_P(AtomicTest, Counting)
{

  using POLICY_T = TypeParam;
  using ATOMIC_POLICY_T = typename policy_traits<POLICY_T>::type;
  using T = RAJA::Real_type;
  const T* in_array = this->rand_dvalue;
  RAJA::Index_type alen = array_length;

  RAJA::Index_type* ivalue = (RAJA::Index_type*) RAJA::allocate_aligned(RAJA::DATA_ALIGN,
                 alen * sizeof(RAJA::Index_type));
  RAJA::Index_type* ivalue_check = (RAJA::Index_type*) RAJA::allocate_aligned(RAJA::DATA_ALIGN,
                 alen * sizeof(RAJA::Index_type));

  for(int i = 0; i < alen; i++) {
    ivalue[i] = 0;
    ivalue_check[i] = 0;
  }

  RAJA::atomic<ATOMIC_POLICY_T, int> atm_cnt_up(0);
  RAJA::atomic<ATOMIC_POLICY_T, int> atm_cnt_down(alen - 1);

  RAJA::forall<POLICY_T>(0, alen, [=] (int i) {

    if (in_array[i] < 0.0) {

      ivalue[atm_cnt_down--] = i;

    } else {

      ivalue[atm_cnt_up++] = i;

    }

  });

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
    } else if (i < this->seq_pos_cnt && in_array[ivalue[i]] < 0.0) {
      first_wrong = i;
    } else if (i >= this->seq_pos_cnt && in_array[ivalue[i]] >= 0.0) {
      first_wrong = i;
    }
  }

  EXPECT_EQ( (int)atm_cnt_down, alen - this->seq_neg_cnt - 1 );
  EXPECT_EQ( (int)atm_cnt_up, this->seq_pos_cnt );
  EXPECT_EQ( first_wrong, -1 ) ;

  RAJA::free_aligned(ivalue);
  RAJA::free_aligned(ivalue_check);
}

REGISTER_TYPED_TEST_CASE_P(AtomicTest, MinMax, AddSub, AndOrXor, IncDec, CompExch, LoadExch, LoadStore, Counting);


typedef ::testing::
    Types<RAJA::seq_exec,
          RAJA::simd_exec>
        SequentialTypes;

INSTANTIATE_TYPED_TEST_CASE_P(Sequential, AtomicTest, SequentialTypes);


#if defined(RAJA_ENABLE_OPENMP)
typedef ::testing::
    Types<RAJA::omp_parallel_for_exec>
        OpenMPTypes;

INSTANTIATE_TYPED_TEST_CASE_P(OpenMP, AtomicTest, OpenMPTypes);
#endif

#if defined(RAJA_ENABLE_CILK)
typedef ::testing::
    Types<RAJA::cilk_for_exec>
        CilkTypes;

INSTANTIATE_TYPED_TEST_CASE_P(Cilk, AtomicTest, CilkTypes);
#endif
