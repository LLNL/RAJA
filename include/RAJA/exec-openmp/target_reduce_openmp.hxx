/*!
 ******************************************************************************
 *
 * \brief  Min reducer class template for use in OpenMP execution.
 *
 *         For usage example, see reducers.hxx.
 *
 ******************************************************************************
 */

#pragma omp declare target
#include <omp.h>

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
    parent(NULL), val(init_val)
  {
        hostid = omp_get_initial_device();
        devid  = omp_get_default_device();
        is_mapped = false;
        dev_val = (T *)omp_target_alloc( sizeof(T), devid );
        if( dev_val == NULL ){
                printf("Unable to allocate space on device\n" );
                exit(1);
        }
        omp_target_memcpy( (void *)dev_val, (void *)&val, sizeof(T), 0, 0, devid, hostid );
  }

  //
  // Copy ctor.
  //
  ReduceMin(const ReduceMin<omp_target_reduce, T>& other):
    parent(other.parent ? other.parent : &other),
    val(other.val), dev_val(other.dev_val), hostid(other.hostid), devid(other.devid), is_mapped(other.is_mapped)
  {
  }

  //
  // Destruction folds value into parent object.
  //
  ~ReduceMin<omp_target_reduce, T>()
  {
   if( !omp_is_initial_device() && omp_in_parallel() )
    {
#pragma omp critical
      {
        *dev_val = RAJA_MIN(*dev_val, val);
      }
    }
  }

  //
  // Operator that returns reduced min value.
  //
  operator T()
  {
    if( !is_mapped ){
        omp_target_memcpy( (void*)&val, (void*)dev_val , sizeof(T), 0, 0, hostid, devid );
        omp_target_free( (void*)dev_val, devid );
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
    ptrToThis->val = RAJA_MIN(ptrToThis->val, rhs);
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
  my_type * ptrToThis = this;
  mutable T * dev_val;
  bool is_mapped;
  int hostid, devid;
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
    parent(NULL), val(init_val), idx(init_loc)
  {
        hostid = omp_get_initial_device();
        devid  = omp_get_default_device();
        is_mapped = false;
        dev_val = (T *)omp_target_alloc( sizeof(T), devid );
        if( dev_val == NULL ){
                printf("Unable to allocate space on device\n" );
                exit(1);
        }
        omp_target_memcpy( (void *)dev_val, (void *)&val, sizeof(T), 0, 0, devid, hostid );
        dev_idx = (Index_type *)omp_target_alloc( sizeof(Index_type), devid );
        if( dev_idx == NULL ){
                printf("Unable to allocate space on device\n" );
                exit(1);
        }
        omp_target_memcpy( (void *)dev_idx, (void *)&idx, sizeof(Index_type), 0, 0, devid, hostid );
 }

  //
  // Copy ctor.
  //
  ReduceMinLoc(const ReduceMinLoc<omp_target_reduce, T>& other):
    parent(other.parent ? other.parent : &other),
    val(other.val), dev_val(other.dev_val),
    idx(other.idx), dev_idx(other.dev_idx),
    hostid(other.hostid), devid(other.devid), is_mapped(other.is_mapped)
  {
  }

  //
  // Destruction releases the shared memory block chunk for reduction id
  // and id itself for others to use.
  //
  ~ReduceMinLoc<omp_target_reduce, T>()
  {
   if( !omp_is_initial_device() && omp_in_parallel() )
    {
#pragma omp critical
      {
        if( val < *dev_val )
        {
            *dev_val = val;
            *dev_idx = idx;
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
        omp_target_memcpy( (void*)&val, (void*)dev_val , sizeof(T), 0, 0, hostid, devid );
        omp_target_free( (void*)dev_val, devid );
        omp_target_memcpy( (void*)&idx, (void*)dev_idx , sizeof(Index_type), 0, 0, hostid, devid );
        omp_target_free( (void*)dev_idx, devid );
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
    if (rhs < val) {
      ptr2This->val = rhs;
      ptr2This->idx = rhs_idx;
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

  my_type * ptr2This = this;
  mutable T * dev_val;
  mutable Index_type *dev_idx;
  bool is_mapped;
  int hostid, devid;
  
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
    parent(NULL), val(init_val)
  {
        hostid = omp_get_initial_device();
        devid  = omp_get_default_device();
        is_mapped = false;
        dev_val = (T *)omp_target_alloc( sizeof(T), devid );
        if( dev_val == NULL ){
                printf("Unable to allocate space on device\n" );
                exit(1);
        }
        omp_target_memcpy( (void *)dev_val, (void *)&val, sizeof(T), 0, 0, devid, hostid );
  }

  //
  // Copy ctor.
  //
  ReduceMax(const ReduceMax<omp_target_reduce, T>& other) :
    parent(other.parent ? other.parent : &other),
    val(other.val), dev_val(other.dev_val), hostid(other.hostid), devid(other.devid), is_mapped(other.is_mapped)
  {
  }

  //
  // Destruction releases the shared memory block chunk for reduction id
  // and id itself for others to use.
  //
  ~ReduceMax<omp_target_reduce, T>()
  {
   if( !omp_is_initial_device() && omp_in_parallel() )
    {
#pragma omp critical
      {
        *dev_val = RAJA_MAX(*dev_val, val);
      }
    }
  }

  //
  // Operator that returns reduced max value.
  //
  operator T()
  {
    if( !is_mapped ){
        omp_target_memcpy( (void*)&val, (void*)dev_val , sizeof(T), 0, 0, hostid, devid );
        omp_target_free( (void*)dev_val, devid );
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
    ptrToThis->val = RAJA_MAX(ptrToThis->val, rhs);
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
  my_type * ptrToThis = this;
  mutable T * dev_val;
  bool is_mapped;
  int hostid, devid;
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
    parent(NULL), val(init_val), idx(init_loc)
  {
        hostid = omp_get_initial_device();
        devid  = omp_get_default_device();
        is_mapped = false;
        dev_val = (T *)omp_target_alloc( sizeof(T), devid );
        if( dev_val == NULL ){
                printf("Unable to allocate space on device\n" );
                exit(1);
        }
        omp_target_memcpy( (void *)dev_val, (void *)&val, sizeof(T), 0, 0, devid, hostid );
        dev_idx = (Index_type *)omp_target_alloc( sizeof(Index_type), devid );
        if( dev_idx == NULL ){
                printf("Unable to allocate space on device\n" );
                exit(1);
        }
        omp_target_memcpy( (void *)dev_idx, (void *)&idx, sizeof(Index_type), 0, 0, devid, hostid );
  }

  //
  // Copy ctor.
  //
  ReduceMaxLoc(const ReduceMaxLoc<omp_target_reduce, T>& other):
    parent(other.parent ? other.parent : &other),
    val(other.val), dev_val(other.dev_val),
    idx(other.idx), dev_idx(other.dev_idx),
    hostid(other.hostid), devid(other.devid), is_mapped(other.is_mapped)
  {
  }

  //
  // Destruction releases the shared memory block chunk for reduction id
  // and id itself for others to use.
  //
  ~ReduceMaxLoc<omp_target_reduce, T>()
  {
   if( !omp_is_initial_device() && omp_in_parallel() )
    {
#pragma omp critical
      {
        if( val > *dev_val )
        {
            *dev_val = val;
            *dev_idx = idx;
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
        omp_target_memcpy( (void*)&val, (void*)dev_val , sizeof(T), 0, 0, hostid, devid );
        omp_target_free( (void*)dev_val, devid );
        omp_target_memcpy( (void*)&idx, (void*)dev_idx , sizeof(Index_type), 0, 0, hostid, devid );
        omp_target_free( (void*)dev_idx, devid );
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
      ptr2This->val = rhs;
      ptr2This->idx = rhs_idx;
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
  
  my_type * ptr2This = this;
  mutable T * dev_val;
  mutable Index_type *dev_idx;
  bool is_mapped;
  int hostid, devid;

};

/*!
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
    : parent(NULL), val(init_val), custom_init(initializer)
  {
	hostid = omp_get_initial_device();
	devid  = omp_get_default_device();
	is_mapped = false;
        dev_val = (T *)omp_target_alloc( sizeof(T), devid );
	if( dev_val == NULL ){
		printf("Unable to allocate space on device\n" );
		exit(1);
	}
	omp_target_memcpy( (void *)dev_val, (void *)&val, sizeof(T), 0, 0, devid, hostid );
	
	
  }

  //
  // Copy ctor.
  //
  ReduceSum(const ReduceSum<omp_target_reduce, T>& other) :
    parent(other.parent ? other.parent : &other), 
    val(other.custom_init),
    dev_val(other.dev_val),
    custom_init(other.custom_init),
    is_mapped(other.is_mapped), hostid(other.hostid), devid(other.devid)
  {
  }

  //
  // Destruction releases the shared memory block chunk for reduction id
  // and id itself for others to use.
  //
  ~ReduceSum<omp_target_reduce, T>()
  {

   if( !omp_is_initial_device() && omp_in_parallel() )
    { 
#pragma omp critical
	{
	  *dev_val += val;
	}  

    }
  }  
  // 
  // Operator that returns reduced sum value.
  //
  operator T()
  {
    if( !is_mapped ){
        omp_target_memcpy( (void*)&val, (void*)dev_val , sizeof(T), 0, 0, hostid, devid );
        omp_target_free( (void*)dev_val, devid );
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
//  this->val += rhs;
  ptr2This->val += rhs;
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
  my_type * ptr2This = this;
  mutable T val;
  mutable T * dev_val;
  T custom_init;
  int hostid, devid;
  bool is_mapped;
};


} // end RAJA namespace

#pragma omp end declare target
