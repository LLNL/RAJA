/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   Header file containing RAJA reduction templates for OpenMP
 *          OpenMP execution.
 *
 *          These methods should work on any platform that supports OpenMP.
 *
 ******************************************************************************
 */

#ifndef RAJA_target_reduce_omp_HXX
#define RAJA_target_reduce_omp_HXX

#include "RAJA/config.hpp"

#if defined(RAJA_ENABLE_TARGET_OPENMP)

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

#include "RAJA/util/types.hpp"

#include "RAJA/pattern/reduce.hpp"

#include "RAJA/policy/openmp/policy.hpp"

#include "RAJA/internal/MemUtils_CPU.hpp"

#if defined(_OPENMP)
#include <omp.h>
#endif

namespace RAJA
{

template <typename T>
class ReduceMin<omp_target_reduce, T>
{
  using my_type = ReduceMin<omp_target_reduce, T>;

public:
  //
  // Constructor takes default value (default ctor is disabled).
  //
  explicit ReduceMin(T init_val):
    parent(NULL), val(init_val), num_teams(16), num_threads(512)
  {

        hostid = omp_get_initial_device();
        devid  = omp_get_default_device();
        is_mapped = false;
        dev_val = (T *)omp_target_alloc( num_teams*sizeof(T), devid );
        if( dev_val == NULL ){
                printf("Unable to allocate space on device\n" );
                exit(1);
        }
        
        host_val = new T [ num_teams ];
        
        for(int i=0; i<num_teams; ++i ) host_val[i] = init_val;
        
        omp_target_memcpy( (void *)dev_val, (void *)host_val, num_teams*sizeof(T), 0, 0, devid, hostid );
  }

  //
  // Copy ctor.
  //
  ReduceMin(const ReduceMin<omp_target_reduce, T>& other):
    parent(other.parent ? other.parent : &other),
    val(other.val), dev_val(other.dev_val), hostid(other.hostid), devid(other.devid), 
    num_teams(other.num_teams), num_threads(other.num_threads), is_mapped(other.is_mapped)
  {
  }

  //
  // Destruction folds value into parent object.
  //
  ~ReduceMin<omp_target_reduce, T>()
  {
   if( !omp_is_initial_device() )
    {
#pragma omp critical
      {
      	int tid = omp_get_team_num();
        dev_val[tid] = RAJA_MIN(dev_val[tid], ptr2this->val);
      }
    }
  }

  //
  // Operator that returns reduced min value.
  //
  operator T()
  {
    if( !is_mapped ){
        omp_target_memcpy( (void*)host_val, (void*)dev_val , num_teams*sizeof(T), 0, 0, hostid, devid );
        
        for(int i=0; i<num_teams; ++i )
        	val = RAJA_MIN( val, host_val[i] );
        
        omp_target_free( (void*)dev_val, devid );
        delete [] host_val;
        is_mapped = true;
    }
    return val;
  }

  //
  // Method that returns reduced min value.
  //
  T get() { return operator T(); }

  //
  // Method that updates min value for current object, assumes each thread
  // has its own copy of the object.
  //
  const ReduceMin<omp_target_reduce, T>& min(T rhs) const
  {
    ptr2this->val = RAJA_MIN(ptr2this->val, rhs);
    return *this;
  }

  ReduceMin<omp_target_reduce, T>& min(T rhs) {
    val = RAJA_MIN(val, rhs);
    return *this;
  }

private:
  //
  // Default ctor is declared private and not implemented.
  //
  ReduceMin<omp_target_reduce, T>();

  const my_type * parent;
  my_type * ptr2this = this;
  T *host_val, *dev_val;
  bool is_mapped;
  int hostid, devid, num_teams, num_threads;
  mutable T val;
};

/*!
 ******************************************************************************
 *
 * \brief  Min-loc reducer class template for use in OpenMP execution.
 *
 *         For usage example, see reducers.hxx.
 *
 ******************************************************************************
 */
template <typename T>
class ReduceMinLoc<omp_target_reduce, T>
{
  using my_type = ReduceMinLoc<omp_target_reduce, T>;

public:
  //
  // Constructor takes default value (default ctor is disabled).
  //
  explicit ReduceMinLoc(T init_val, Index_type init_loc):
    parent(NULL), val(init_val), idx(init_loc), num_teams(16), num_threads(512)
  {
        hostid = omp_get_initial_device();
        devid  = omp_get_default_device();
        is_mapped = false;
        
        dev_val = (T *)omp_target_alloc( num_teams*sizeof(T), devid );
        if( dev_val == NULL ){
                printf("Unable to allocate space on device\n" );
                exit(1);
        }
        dev_idx = (Index_type *)omp_target_alloc( num_teams*sizeof(Index_type), devid );
        if( dev_idx == NULL ){
                printf("Unable to allocate space on device\n" );
                exit(1);
        }        
        
        host_val = new T [num_teams];
        host_idx = new Index_type [num_teams];
        
        for(int i=0; i<num_teams; ++i )
        {
        	host_val[i] = init_val;
        	host_idx[i] = init_loc;
        }
        
        omp_target_memcpy( (void *)dev_val, (void *)host_val, num_teams*sizeof(T), 0, 0, devid, hostid );
        omp_target_memcpy( (void *)dev_idx, (void *)host_idx, num_teams*sizeof(Index_type), 0, 0, devid, hostid );
 }

  //
  // Copy ctor.
  //
  ReduceMinLoc(const ReduceMinLoc<omp_target_reduce, T>& other):
    parent(other.parent ? other.parent : &other),
    val(other.val), dev_val(other.dev_val),
    idx(other.idx), dev_idx(other.dev_idx),
    hostid(other.hostid), devid(other.devid), is_mapped(other.is_mapped), 
    num_teams(other.num_teams), num_threads(other.num_threads)
  {
  }

  //
  // Destruction releases the shared memory block chunk for reduction id
  // and id itself for others to use.
  //
  ~ReduceMinLoc<omp_target_reduce, T>()
  {
   if( !omp_is_initial_device() )
    {
#pragma omp critical
      {
      	int tid = omp_get_team_num();
        if( ptr2this->val < dev_val[tid] )
        {
            dev_val[tid] = ptr2this->val;
            dev_idx[tid] = ptr2this->idx;
        }        
      }
    }
  }

  //
  // Operator that returns reduced min value.
  //
  operator T()
  {
    if( !is_mapped ){
        omp_target_memcpy( (void*)host_val, (void*)dev_val , num_teams*sizeof(T), 0, 0, hostid, devid );
        omp_target_free( (void*)dev_val, devid );
        omp_target_memcpy( (void*)host_idx, (void*)dev_idx , num_teams*sizeof(Index_type), 0, 0, hostid, devid );
        omp_target_free( (void*)dev_idx, devid );
        
        for(int i=0; i<num_teams; ++i)
        {
        	if( host_val[i] < val )
        	{
        		val = host_val[i];
        		idx = host_idx[i];
        	}
        }
        
        delete [] host_idx;
        delete [] host_val;
        is_mapped = true;
    }
    return val;
  }

  //
  // Method that returns reduced min value.
  //
  T get() { return operator T(); }

  //
  // Method that returns index corresponding to reduced min value.
  //
  Index_type getLoc()
  {
    if( !is_mapped )
    {
       T val = get();
    }
    return idx;
  }

  //
  // Method that updates min and index values for current thread.
  //
  const ReduceMinLoc<omp_target_reduce, T>& minloc(T rhs, Index_type rhs_idx) const
  {
    if (rhs < ptr2this->val) {
      ptr2this->val = rhs;
      ptr2this->idx = rhs_idx;
    }
    return *this;
  }

  ReduceMinLoc<omp_target_reduce, T>& minloc(T rhs, Index_type rhs_idx)
  {
    if (rhs < val) {
      val = rhs;
      idx = rhs_idx;
    }
    return *this;
  }

private:
  //
  // Default ctor is declared private and not implemented.
  //
  ReduceMinLoc<omp_target_reduce, T>();

  const my_type * parent;

  mutable T val;
  mutable Index_type idx;

  my_type * ptr2this = this;
  T *host_val, *dev_val;
  Index_type *host_idx, *dev_idx;
  bool is_mapped;
  int hostid, devid, num_teams, num_threads;
  
};

/*!
 ******************************************************************************
 *
 * \brief  Max reducer class template for use in OpenMP execution.
 *
 *         For usage example, see reducers.hxx.
 *
 ******************************************************************************
 */
template <typename T>
class ReduceMax<omp_target_reduce, T>
{
  using my_type = ReduceMax<omp_target_reduce, T>;

public:
  //
  // Constructor takes default value (default ctor is disabled).
  //
  explicit ReduceMax(T init_val):
    parent(NULL), val(init_val), num_teams(16), num_threads(512)
  {
        hostid = omp_get_initial_device();
        devid  = omp_get_default_device();
        is_mapped = false;
        
        host_val = new T [num_teams];
        
        for(int i=0; i<num_teams; ++i ) host_val[i] = init_val;
        
        dev_val = (T *)omp_target_alloc( num_teams*sizeof(T), devid );
        if( dev_val == NULL ){
                printf("Unable to allocate space on device\n" );
                exit(1);
        }
        omp_target_memcpy( (void *)dev_val, (void *)host_val, num_teams*sizeof(T), 0, 0, devid, hostid );
  }

  //
  // Copy ctor.
  //
  ReduceMax(const ReduceMax<omp_target_reduce, T>& other) :
    parent(other.parent ? other.parent : &other),
    val(other.val), dev_val(other.dev_val), hostid(other.hostid), devid(other.devid), 
    num_teams(other.num_teams), num_threads(other.num_threads), is_mapped(other.is_mapped)
  {
  }

  //
  // Destruction releases the shared memory block chunk for reduction id
  // and id itself for others to use.
  //
  ~ReduceMax<omp_target_reduce, T>()
  {
   if( !omp_is_initial_device()  )
    {
#pragma omp critical
      {
      	int tid = omp_get_team_num();
        dev_val[tid] = RAJA_MAX(dev_val[tid], ptr2this->val);
      }
    }
  }

  //
  // Operator that returns reduced max value.
  //
  operator T()
  {
    if( !is_mapped ){
        omp_target_memcpy( (void*)host_val, (void*)dev_val , num_teams*sizeof(T), 0, 0, hostid, devid );
        for(int i=0; i<num_teams; ++i )
        	val = RAJA_MAX(val, host_val[i]);
        omp_target_free( (void*)dev_val, devid );
        delete [] host_val;
        is_mapped = true;
    }
    return val;
  }

  //
  // Method that returns reduced max value.
  //
  T get() { return operator T(); }

  //
  // Method that updates max value for current thread.
  //
  const ReduceMax<omp_target_reduce, T>& max(T rhs) const
  {
    ptr2this->val = RAJA_MAX(ptr2this->val, rhs);
    return *this;
  }

  ReduceMax<omp_target_reduce, T>& max(T rhs)
  {
    val = RAJA_MAX(val, rhs);
    return *this;
  }

private:
  //
  // Default ctor is declared private and not implemented.
  //
  ReduceMax<omp_target_reduce, T>();

  const my_type * parent;
  my_type * ptr2this = this;
  T *host_val, *dev_val;
  bool is_mapped;
  int hostid, devid, num_teams, num_threads;
  mutable T val;};

/*!
 ******************************************************************************
 *
 * \brief  Max-loc reducer class template for use in OpenMP execution.
 *
 *         For usage example, see reducers.hxx.
 *
 ******************************************************************************
 */
template <typename T>
class ReduceMaxLoc<omp_target_reduce, T>
{
  using my_type = ReduceMaxLoc<omp_target_reduce, T>;

public:
  //
  // Constructor takes default value (default ctor is disabled).
  //
  explicit ReduceMaxLoc(T init_val, Index_type init_loc):
    parent(NULL), val(init_val), idx(init_loc), num_teams(16), num_threads(512)
  {
        hostid = omp_get_initial_device();
        devid  = omp_get_default_device();
        is_mapped = false;
        
        dev_val = (T *)omp_target_alloc( num_teams*sizeof(T), devid );
        if( dev_val == NULL ){
                printf("Unable to allocate space on device\n" );
                exit(1);
        }
        dev_idx = (Index_type *)omp_target_alloc( num_teams*sizeof(Index_type), devid );
        if( dev_idx == NULL ){
                printf("Unable to allocate space on device\n" );
                exit(1);
        }
        
        host_val = new T [num_teams];
        host_idx = new Index_type [num_teams];
        
        for(int i=0; i<num_teams; ++i)
        {
        	host_val[i] = init_val;
        	host_idx[i]= init_loc;
        }
        
        omp_target_memcpy( (void *)dev_val, (void *)host_val, num_teams*sizeof(T), 0, 0, devid, hostid );
        omp_target_memcpy( (void *)dev_idx, (void *)host_idx, num_teams*sizeof(Index_type), 0, 0, devid, hostid );
  }

  //
  // Copy ctor.
  //
  ReduceMaxLoc(const ReduceMaxLoc<omp_target_reduce, T>& other):
    parent(other.parent ? other.parent : &other),
    val(other.val), dev_val(other.dev_val),
    idx(other.idx), dev_idx(other.dev_idx),
    hostid(other.hostid), devid(other.devid), is_mapped(other.is_mapped),
    num_teams(other.num_teams), num_threads(other.num_threads)
  {
  }

  //
  // Destruction releases the shared memory block chunk for reduction id
  // and id itself for others to use.
  //
  ~ReduceMaxLoc<omp_target_reduce, T>()
  {
   if( !omp_is_initial_device() )
    {
#pragma omp critical
      {
      	int tid = omp_get_team_num();
        if( ptr2this->val > dev_val[tid] )
        {
            dev_val[tid] = ptr2this->val;
            dev_idx[tid] = ptr2this->idx;
        }        
      }
    }
  }

  //
  // Operator that returns reduced max value.
  //
  operator T()
  {
    if( !is_mapped ){
        omp_target_memcpy( (void*)host_val, (void*)dev_val , num_teams*sizeof(T), 0, 0, hostid, devid );
        omp_target_free( (void*)dev_val, devid );
        omp_target_memcpy( (void*)host_idx, (void*)dev_idx , num_teams*sizeof(Index_type), 0, 0, hostid, devid );
        omp_target_free( (void*)dev_idx, devid );
        for(int i=0; i<num_teams; ++i)
        {
        	if( host_val[i] > val )
        	{
        		val = host_val[i];
        		idx = host_idx[i];
        	}
        }
        delete [] host_val;
        delete [] host_idx;
        is_mapped = true;
    }
    return val;
  }

  //
  // Method that returns reduced max value.
  //
  T get() { return operator T(); }

  //
  // Method that returns index corresponding to reduced max value.
  //
  Index_type getLoc()
  {
    if( !is_mapped )
    {
       T val = get();
    }
    return idx;
  }

  //
  // Method that updates max and index values for current thread.
  //
  const ReduceMaxLoc<omp_target_reduce, T>& maxloc(T rhs, Index_type rhs_idx) const
  {
    if (rhs > val) {
      ptr2this->val = rhs;
      ptr2this->idx = rhs_idx;
    }
    return *this;
  }

  ReduceMaxLoc<omp_target_reduce, T>& maxloc(T rhs, Index_type rhs_idx)
  {
    if (rhs > val) {
      val = rhs;
      idx = rhs_idx;
    }
    return *this;
  }

private:
  //
  // Default ctor is declared private and not implemented.
  //
  ReduceMaxLoc<omp_target_reduce, T>();

  const my_type * parent;

  mutable T val;
  mutable Index_type idx;
  
  my_type * ptr2this = this;
  T *host_val, *dev_val;
  Index_type *host_idx, *dev_idx;
  bool is_mapped;
  int hostid, devid, num_teams, num_threads;

};

/*
 ******************************************************************************
 *
 * \brief  Sum reducer class template for use in OpenMP execution.
 *
 *         For usage example, see reducers.hxx.
 *
 ******************************************************************************
 */
 
template <typename T>
class ReduceSum<omp_target_reduce, T>
{
  using my_type = ReduceSum<omp_target_reduce, T>;

public:
  //
  // Constructor takes default value (default ctor is disabled).
  //
  explicit ReduceSum(T init_val, T initializer = 0)
    : parent(NULL), val(init_val), dev_val(NULL), custom_init(initializer),
    hostid(0), devid(0), num_teams(16), num_threads(512), is_mapped(false)
  {
  	int flag;
	hostid = omp_get_initial_device();
	devid  = omp_get_default_device();

	char *p = std::getenv( "RAJA_OMP_NUM_TEAMS" );
	if( p != NULL )
        	num_teams = std::atoi( p );
	else
		printf("RAJA_OMP_NUM_TEAMS not set. Default value: %d\n", num_teams );
	
        dev_val = (T *)omp_target_alloc( num_teams*sizeof(T), devid );
	if( dev_val == NULL ){
		printf("Unable to allocate space on device\n" );
		exit(1);
	} 
		
	
	host_val = new T [num_teams];
	
	for(int i=0; i<num_teams; i++) host_val[i] = init_val;
	
	flag = omp_target_memcpy( (void *)dev_val, (void *)host_val, 
	                    num_teams*sizeof(T), 0, 0, devid, hostid );
    if( flag != 0 ){
    	printf("Unable to copy memory from host to device\n" );
    	exit(1);
    }
 
  }

  //
  // Copy ctor.
  //
  ReduceSum(const ReduceSum<omp_target_reduce, T>& other) :
    parent(other.parent ? other.parent : &other), 
    val(other.val), dev_val(other.dev_val),
    num_teams(other.num_teams), num_threads(other.num_threads),
    is_mapped(other.is_mapped), hostid(other.hostid), devid(other.devid)
  {
		
  }

  //
  // Destruction releases the shared memory block chunk for reduction id
  // and id itself for others to use.
  //
  ~ReduceSum<omp_target_reduce, T>()
  {
   if( !omp_is_initial_device() )
    { 
        #pragma omp critical
	{
          int tid = omp_get_team_num();
	  dev_val[tid] += ptr2this->val;
	}  

    } 
  } 
  // 
  // Operator that returns reduced sum value.
  //
  operator T()
  {
    if( !is_mapped ){
    	int flag = omp_target_memcpy( (void*)host_val, (void*)dev_val , 
				       num_teams*sizeof(T), 0, 0, hostid, devid );
        if( flag != 0 ){
    		printf("Unable to copy memory from device to host\n" );
    		exit(1);
    	}

        for(int i=0; i<num_teams; ++i)
        {
        	val += host_val[i];
        }

        omp_target_free( (void*)dev_val, devid );
	delete [] host_val;
        is_mapped = true;
   }
 
    return val;
  }

  //
  // Method that returns sum value.
  //
  T get() { return operator T(); }

  //
  // += operator that adds value to sum for current thread.
  //
  const ReduceSum<omp_target_reduce, T>& operator+=(T rhs) const 
  {
  ptr2this->val += rhs;    
  return *this;
  }

  ReduceSum<omp_target_reduce, T>& operator+=(T rhs)
  {
    this->val += rhs;
    return *this;
  }
  
private:
  //
  // Default ctor is declared private and not implemented.
  //
  ReduceSum<omp_target_reduce, T>();
  const my_type * parent;
  my_type * ptr2this = this;
  mutable T val;
  T *host_val, *dev_val;
  T custom_init;
  int hostid, devid, num_teams, num_threads;
  bool is_mapped;
};

}  // closing brace for RAJA namespace

#endif  // closing endif for RAJA_ENABLE_TARGET_OPENMP guard

#endif  // closing endif for header file include guard
