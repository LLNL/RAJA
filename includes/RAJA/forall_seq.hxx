/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   Header file containing RAJA index set iteration template 
 *          methods for sequential execution. 
 *
 *          These methods should work on any platform.
 *
 * \author  Rich Hornung, Center for Applied Scientific Computing, LLNL
 * \author  Jeff Keasler, Applications, Simulations And Quality, LLNL
 *
 ******************************************************************************
 */

#ifndef RAJA_forall_seq_HXX
#define RAJA_forall_seq_HXX

#include "config.hxx"

#include "int_datatypes.hxx"

#include "execpolicy.hxx"
#include "reducers.hxx"

#include "fault_tolerance.hxx"

#include "MemUtils.hxx"

#include<string>
#include<iostream> 


namespace RAJA {

//
//////////////////////////////////////////////////////////////////////
//
// Reduction classes.
//
//////////////////////////////////////////////////////////////////////
//

/*!
 ******************************************************************************
 *
 * \brief  Min reducer class template for use in sequential reduction.
 *
 * \verbatim
 *         Fill this in...
 * \endverbatim
 *
 ******************************************************************************
 */
template <typename T>
class ReduceMin<seq_reduce, T> 
{
public:
   //
   // Constructor takes default value (default ctor is disabled).
   //
   explicit ReduceMin(T init_val) 
   {
      m_is_copy = false;

      m_reduced_val = init_val;

      m_myID = getCPUReductionId();
//    std::cout << "ReduceMin id = " << m_myID << std::endl;
     
      m_blockdata = getCPUReductionMemBlock(m_myID);  

      m_blockdata[0] = init_val; 
   }

   //
   // Copy ctor.
   //
   ReduceMin( const ReduceMin<seq_reduce, T>& other ) 
   {
      *this = other;
      m_is_copy = true;
   }

   //
   // Destructor.
   //
   ~ReduceMin<seq_reduce, T>() 
   {
      if (!m_is_copy) {
         releaseCPUReductionId(m_myID);
         // free any data owned by reduction object 
      }
   }

   //
   // Operator to retrieve min value (before object is destroyed).
   //
   operator T()
   {
      m_reduced_val = RAJA_MIN(m_reduced_val, static_cast<T>(m_blockdata[0]));

      return m_reduced_val;
   }

   //
   // Min function that sets object min to minimum of current value and arg.
   //
   ReduceMin<seq_reduce, T> min(T val) const 
   {
      m_blockdata[0] = RAJA_MIN(static_cast<T>(m_blockdata[0]), val);
      return *this ;
   }

private:
   //
   // Default ctor is declared private and not implemented.
   //
   ReduceMin<seq_reduce, T>();

   bool m_is_copy;
   int m_myID;

   T m_reduced_val;

   CPUReductionBlockDataType* m_blockdata;
} ;

/*!
 ******************************************************************************
 *
 * \brief  Min-loc reducer class template for use in sequential reduction.
 *
 * \verbatim
 *         Fill this in...
 * \endverbatim
 *
 ******************************************************************************
 */
template <typename T>
class ReduceMinLoc<seq_reduce, T> 
{
public:
   //
   // Constructor takes default value (default ctor is disabled).
   //
   explicit ReduceMinLoc(T init_val, Index_type init_loc) 
   {
      m_is_copy = false;

      m_reduced_val = init_val;

      m_myID = getCPUReductionId();
//    std::cout << "ReduceMinLoc id = " << m_myID << std::endl;
     
      m_blockdata = getCPUReductionMemBlock(m_myID);  
      m_blockdata[0] = init_val; 

      m_idxdata = getCPUReductionLocBlock(m_myID);  
      m_idxdata[0] = init_loc; 
   }

   //
   // Copy ctor.
   //
   ReduceMinLoc( const ReduceMinLoc<seq_reduce, T>& other ) 
   {
      *this = other;
      m_is_copy = true;
   }

   //
   // Destructor.
   //
   ~ReduceMinLoc<seq_reduce, T>() 
   {
      if (!m_is_copy) {
         releaseCPUReductionId(m_myID);
         // free any data owned by reduction object 
      }
   }

   //
   // Operator to retrieve min value (before object is destroyed).
   //
   operator T()
   {
      if ( static_cast<T>(m_blockdata[0]) <= m_reduced_val ) {
         m_reduced_val = m_blockdata[0];
         m_reduced_idx = m_idxdata[0];
      }
      return m_reduced_val;
   }

   //
   // Operator to retrieve min value (before object is destroyed).
   //
   Index_type getMinLoc()
   {
      if ( static_cast<T>(m_blockdata[0]) <= m_reduced_val ) {
         m_reduced_val = m_blockdata[0];
         m_reduced_idx = m_idxdata[0];
      }
      return m_reduced_idx;
   }

   //
   // Min-loc function that sets object min to minimum of current value 
   // and value arg and updates location index accordingly.
   //
   ReduceMinLoc<seq_reduce, T> minloc(T val, Index_type idx) const 
   {
      if ( val <= static_cast<T>(m_blockdata[0]) ) {
         m_blockdata[0] = val;
         m_idxdata[0] = idx;
      }
      return *this ;
   }

private:
   //
   // Default ctor is declared private and not implemented.
   //
   ReduceMinLoc<seq_reduce, T>();

   bool m_is_copy;
   int m_myID;

   T m_reduced_val;
   Index_type m_reduced_idx;

   CPUReductionBlockDataType* m_blockdata;
   Index_type* m_idxdata;
} ;

/*!
 ******************************************************************************
 *
 * \brief  Max reducer class template for use in sequential reduction.
 *
 * \verbatim
 *         Fill this in...
 * \endverbatim
 *
 ******************************************************************************
 */
template <typename T>
class ReduceMax<seq_reduce, T> 
{
public:
   //
   // Constructor takes default value (default ctor is disabled).
   //
   explicit ReduceMax(T init_val) 
   {
      m_is_copy = false;

      m_reduced_val = init_val;

      m_myID = getCPUReductionId();
//    std::cout << "ReduceMax id = " << m_myID << std::endl;
     
      m_blockdata = getCPUReductionMemBlock(m_myID);  

      m_blockdata[0] = init_val; 
   }

   //
   // Copy ctor.
   //
   ReduceMax( const ReduceMax<seq_reduce, T>& other ) 
   {
      *this = other;
      m_is_copy = true;
   }

   //
   // Destructor.
   //
   ~ReduceMax<seq_reduce, T>() 
   {
      if (!m_is_copy) {
         releaseCPUReductionId(m_myID);
         // free any data owned by reduction object 
      }
   }

   //
   // Operator to retrieve max value (before object is destroyed).
   //
   operator T()
   {
      m_reduced_val = RAJA_MAX(m_reduced_val, static_cast<T>(m_blockdata[0]));

      return m_reduced_val;
   }

   //
   // Max function that sets object max to maximum of current value and arg.
   //
   ReduceMax<seq_reduce, T> max(T val) const 
   {
      m_blockdata[0] = RAJA_MAX(static_cast<T>(m_blockdata[0]), val);
      return *this ;
   }

private:
   //
   // Default ctor is declared private and not implemented.
   //
   ReduceMax<seq_reduce, T>();

   bool m_is_copy;
   int m_myID;

   T m_reduced_val;

   CPUReductionBlockDataType* m_blockdata;
} ;

/*!
 ******************************************************************************
 *
 * \brief  Max-loc reducer class template for use in sequential reduction.
 *
 * \verbatim
 *         Fill this in...
 * \endverbatim
 *
 ******************************************************************************
 */
template <typename T>
class ReduceMaxLoc<seq_reduce, T> 
{
public:
   //
   // Constructor takes default value (default ctor is disabled).
   //
   explicit ReduceMaxLoc(T init_val, Index_type init_loc) 
   {
      m_is_copy = false;

      m_reduced_val = init_val;

      m_myID = getCPUReductionId();
//    std::cout << "ReduceMinLoc id = " << m_myID << std::endl;
     
      m_blockdata = getCPUReductionMemBlock(m_myID);  
      m_blockdata[0] = init_val; 

      m_idxdata = getCPUReductionLocBlock(m_myID);  
      m_idxdata[0] = init_loc; 
   }

   //
   // Copy ctor.
   //
   ReduceMaxLoc( const ReduceMaxLoc<seq_reduce, T>& other ) 
   {
      *this = other;
      m_is_copy = true;
   }

   //
   // Destructor.
   //
   ~ReduceMaxLoc<seq_reduce, T>() 
   {
      if (!m_is_copy) {
         releaseCPUReductionId(m_myID);
         // free any data owned by reduction object 
      }
   }

   //
   // Operator to retrieve max value (before object is destroyed).
   //
   operator T()
   {
      if ( static_cast<T>(m_blockdata[0]) >= m_reduced_val ) {
         m_reduced_val = m_blockdata[0];
         m_reduced_idx = m_idxdata[0];
      }
      return m_reduced_val;
   }

   //
   // Operator to retrieve max value (before object is destroyed).
   //
   Index_type getMaxLoc()
   {
      if ( static_cast<T>(m_blockdata[0]) >= m_reduced_val ) {
         m_reduced_val = m_blockdata[0];
         m_reduced_idx = m_idxdata[0];
      }
      return m_reduced_idx;
   }

   //
   // Max-loc function that sets object max to maximum of current value 
   // and value arg and updates location index accordingly.
   //
   ReduceMaxLoc<seq_reduce, T> maxloc(T val, Index_type idx) const 
   {
      if ( val >= static_cast<T>(m_blockdata[0]) ) {
         m_blockdata[0] = val;
         m_idxdata[0] = idx;
      }
      return *this ;
   }

private:
   //
   // Default ctor is declared private and not implemented.
   //
   ReduceMaxLoc<seq_reduce, T>();

   bool m_is_copy;
   int m_myID;

   T m_reduced_val;
   Index_type m_reduced_idx;

   CPUReductionBlockDataType* m_blockdata;
   Index_type* m_idxdata;
} ;

/*!
 ******************************************************************************
 *
 * \brief  Max reducer class template for use in sequential reduction.
 *
 * \verbatim
 *         Fill this in...
 * \endverbatim
 *
/*!
 ******************************************************************************
 *
 * \brief  Sum reducer class template for use in sequential reduction.
 *
 * \verbatim
 *         Fill this in...
 * \endverbatim
 *
 ******************************************************************************
 */
template <typename T>
class ReduceSum<seq_reduce, T> 
{
public:
   //
   // Constructor takes default value (default ctor is disabled).
   //
   explicit ReduceSum(T init_val)
   {
      m_is_copy = false;

      m_init_val = init_val;
      m_reduced_val = static_cast<T>(0);

      m_myID = getCPUReductionId();

      m_blockdata = getCPUReductionMemBlock(m_myID);

      m_blockdata[0] = 0;
   }

   //
   // Copy ctor.
   //
   ReduceSum( const ReduceSum<seq_reduce, T>& other )
   {
      *this = other;
      m_is_copy = true;
   }

   //
   // Destructor.
   //
   ~ReduceSum<seq_reduce, T>() 
   {
      if (!m_is_copy) {
         releaseCPUReductionId(m_myID);
         // free any data owned by reduction object
      }
   }

   //
   // Operator to retrieve sum value (before object is destroyed).
   //
   operator T()
   {
      m_reduced_val = m_init_val + static_cast<T>(m_blockdata[0]);

      return m_reduced_val;
   }

   //
   // += operator that performs accumulation into object min val.
   //
   ReduceSum<seq_reduce, T> operator+=(T val) const 
   {
      m_blockdata[0] += val;
      return *this ;
   }

private:
   //
   // Default ctor is declared private and not implemented.
   //
   ReduceSum<seq_reduce, T>();

   bool m_is_copy;
   int m_myID;

   T m_init_val;
   T m_reduced_val;

   CPUReductionBlockDataType* m_blockdata;
} ;



//
//////////////////////////////////////////////////////////////////////
//
// Function templates that iterate over index ranges.
//
//////////////////////////////////////////////////////////////////////
//

/*!
 ******************************************************************************
 *
 * \brief  Sequential iteration over index range.
 *
 ******************************************************************************
 */
template <typename LOOP_BODY>
RAJA_INLINE
void forall(seq_exec,
            Index_type begin, Index_type end, 
            LOOP_BODY loop_body)
{

   RAJA_FT_BEGIN ;

#pragma novector
   for ( Index_type ii = begin ; ii < end ; ++ii ) {
      loop_body( ii );
   }

   RAJA_FT_END ;
}

/*!
 ******************************************************************************
 *
 * \brief  Sequential iteration over index range with index count.
 *
 *         NOTE: lambda loop body requires two args (icount, index).
 *
 ******************************************************************************
 */
template <typename LOOP_BODY>
RAJA_INLINE
void forall_Icount(seq_exec,
                   Index_type begin, Index_type end,
                   Index_type icount,
                   LOOP_BODY loop_body)
{
   const Index_type loop_end = end - begin;

   RAJA_FT_BEGIN ;

#pragma novector
   for ( Index_type ii = 0 ; ii < loop_end ; ++ii ) {
      loop_body( ii+icount, ii+begin );
   }

   RAJA_FT_END ;
}

/*!
 ******************************************************************************
 *
 * \brief  Sequential min reduction over index range.
 *
 ******************************************************************************
 */
template <typename T,
          typename LOOP_BODY>
RAJA_INLINE
void forall_min(seq_exec,
                Index_type begin, Index_type end,
                T* min,
                LOOP_BODY loop_body)
{
   RAJA_FT_BEGIN ;

#pragma novector
   for ( Index_type ii = begin ; ii < end ; ++ii ) {
      loop_body( ii, min );
   }

   RAJA_FT_END ;
}

/*!
 ******************************************************************************
 *
 * \brief  Sequential minloc reduction over index range.
 *
 ******************************************************************************
 */
template <typename T, 
          typename LOOP_BODY>
RAJA_INLINE
void forall_minloc(seq_exec,
                   Index_type begin, Index_type end, 
                   T* min, Index_type* loc,
                   LOOP_BODY loop_body)
{
   RAJA_FT_BEGIN ;

#pragma novector
   for ( Index_type ii = begin ; ii < end ; ++ii ) {
      loop_body( ii, min, loc );
   }

   RAJA_FT_END ;
}

/*!
 ******************************************************************************
 *
 * \brief  Sequential max reduction over index range.
 *
 ******************************************************************************
 */
template <typename T,
          typename LOOP_BODY>
RAJA_INLINE
void forall_max(seq_exec,
                Index_type begin, Index_type end,
                T* max,
                LOOP_BODY loop_body)
{
   RAJA_FT_BEGIN ;

#pragma novector
   for ( Index_type ii = begin ; ii < end ; ++ii ) {
      loop_body( ii, max );
   }

   RAJA_FT_END ;
}

/*!
 ******************************************************************************
 *
 * \brief  Sequential maxloc reduction over index range.
 *
 ******************************************************************************
 */
template <typename T,
          typename LOOP_BODY>
RAJA_INLINE
void forall_maxloc(seq_exec,
                   Index_type begin, Index_type end,
                   T* max, Index_type* loc,
                   LOOP_BODY loop_body)
{

   RAJA_FT_BEGIN ;

#pragma novector
   for ( Index_type ii = begin ; ii < end ; ++ii ) {
      loop_body( ii, max, loc );
   }

   RAJA_FT_END ;
}

/*!
 ******************************************************************************
 *
 * \brief  Sequential sum reduction over index range.
 *
 ******************************************************************************
 */
template <typename T,
          typename LOOP_BODY>
RAJA_INLINE
void forall_sum(seq_exec,
                Index_type begin, Index_type end,
                T* sum,
                LOOP_BODY loop_body)
{

   RAJA_FT_BEGIN ;

#pragma novector
   for ( Index_type ii = begin ; ii < end ; ++ii ) {
      loop_body( ii, sum );
   }

   RAJA_FT_END ;
}


//
//////////////////////////////////////////////////////////////////////
//
// Function templates that iterate over range segments. 
//
//////////////////////////////////////////////////////////////////////
//

/*!
 ******************************************************************************
 *
 * \brief  Sequential iteration over range segment object.
 *
 ******************************************************************************
 */
template <typename LOOP_BODY>
RAJA_INLINE
void forall(seq_exec,
            const RangeSegment& iseg,
            LOOP_BODY loop_body)
{
   const Index_type begin = iseg.getBegin();
   const Index_type end   = iseg.getEnd();

   RAJA_FT_BEGIN ;

#pragma novector
   for ( Index_type ii = begin ; ii < end ; ++ii ) {
      loop_body( ii );
   }

   RAJA_FT_END ;
}

/*!
 ******************************************************************************
 *
 * \brief  Sequential iteration over range segment object with index count.
 *
 *         NOTE: lambda loop body requires two args (icount, index).
 *
 ******************************************************************************
 */
template <typename LOOP_BODY>
RAJA_INLINE
void forall_Icount(seq_exec,
                   const RangeSegment& iseg,
                   Index_type icount,
                   LOOP_BODY loop_body)
{
   const Index_type begin = iseg.getBegin();
   const Index_type loop_end = iseg.getEnd() - iseg.getBegin();

   RAJA_FT_BEGIN ;

#pragma novector
   for ( Index_type ii = 0 ; ii < loop_end ; ++ii ) {
      loop_body( ii+icount, ii+begin );
   }

   RAJA_FT_END ;
}

/*!
 ******************************************************************************
 *
 * \brief  Sequential min reduction over range segment object.
 *
 ******************************************************************************
 */
template <typename T,
          typename LOOP_BODY>
RAJA_INLINE
void forall_min(seq_exec,
                const RangeSegment& iseg,
                T* min, Index_type* loc,
                LOOP_BODY loop_body)
{
   const Index_type begin = iseg.getBegin();
   const Index_type end   = iseg.getEnd();

   RAJA_FT_BEGIN ;

#pragma novector
   for ( Index_type ii = begin ; ii < end ; ++ii ) {
      loop_body( ii, min );
   }

   RAJA_FT_END ;
}

/*!
 ******************************************************************************
 *
 * \brief  Sequential minloc reduction over range segment object.
 *
 ******************************************************************************
 */
template <typename T,
          typename LOOP_BODY>
RAJA_INLINE
void forall_minloc(seq_exec,
                   const RangeSegment& iseg,
                   T* min, Index_type* loc,
                   LOOP_BODY loop_body)
{
   const Index_type begin = iseg.getBegin();
   const Index_type end   = iseg.getEnd();

   RAJA_FT_BEGIN ;

#pragma novector
   for ( Index_type ii = begin ; ii < end ; ++ii ) {
      loop_body( ii, min, loc );
   }

   RAJA_FT_END ;
}

/*!
 ******************************************************************************
 *
 * \brief  Sequential max reduction over range segment object.
 *
 ******************************************************************************
 */
template <typename T,
          typename LOOP_BODY>
RAJA_INLINE
void forall_max(seq_exec,
                const RangeSegment& iseg,
                T* max, Index_type* loc,
                LOOP_BODY loop_body)
{
   const Index_type begin = iseg.getBegin();
   const Index_type end   = iseg.getEnd();

   RAJA_FT_BEGIN ;

#pragma novector
   for ( Index_type ii = begin ; ii < end ; ++ii ) {
      loop_body( ii, max );
   }

   RAJA_FT_END ;
}

/*!
 ******************************************************************************
 *
 * \brief  Sequential maxloc reduction over range segment object.
 *
 ******************************************************************************
 */
template <typename T,
          typename LOOP_BODY>
RAJA_INLINE
void forall_maxloc(seq_exec,
                   const RangeSegment& iseg,
                   T* max, Index_type* loc,
                   LOOP_BODY loop_body)
{
   const Index_type begin = iseg.getBegin();
   const Index_type end   = iseg.getEnd();

   RAJA_FT_BEGIN ;

#pragma novector
   for ( Index_type ii = begin ; ii < end ; ++ii ) {
      loop_body( ii, max, loc );
   }

   RAJA_FT_END ;
}

/*!
 ******************************************************************************
 *
 * \brief  Sequential sum reduction over range segment object.
 *
 ******************************************************************************
 */
template <typename T,
          typename LOOP_BODY>
RAJA_INLINE
void forall_sum(seq_exec,
                const RangeSegment& iseg,
                T* sum,
                LOOP_BODY loop_body)
{
   const Index_type begin = iseg.getBegin();
   const Index_type end   = iseg.getEnd();

   RAJA_FT_BEGIN ;

#pragma novector
   for ( Index_type ii = begin ; ii < end ; ++ii ) {
      loop_body( ii, sum );
   }

   RAJA_FT_END ;
}


//
//////////////////////////////////////////////////////////////////////
//
// Function templates that iterate over index ranges with stride.
//
//////////////////////////////////////////////////////////////////////
//

/*!
 ******************************************************************************
 *
 * \brief  Sequential iteration over index range with stride.
 *
 ******************************************************************************
 */
template <typename LOOP_BODY>
RAJA_INLINE
void forall(seq_exec,
            Index_type begin, Index_type end,
            Index_type stride,
            LOOP_BODY loop_body)
{  

   RAJA_FT_BEGIN ;

#pragma novector
   for ( Index_type ii = begin ; ii < end ; ii += stride ) {
      loop_body( ii );
   }

   RAJA_FT_END ;
}

/*!
 ******************************************************************************
 *
 * \brief  Sequential iteration over index range with stride,
 *         including index count.
 *
 *         NOTE: lambda loop body requires two args (icount, index).
 *
 ******************************************************************************
 */
template <typename LOOP_BODY>
RAJA_INLINE
void forall_Icount(seq_exec,
                   Index_type begin, Index_type end,
                   Index_type stride,
                   Index_type icount,
                   LOOP_BODY loop_body)
{
   const Index_type loop_end = (end-begin)/stride;

   RAJA_FT_BEGIN ;

#pragma novector
   for ( Index_type ii = 0 ; ii < loop_end ; ++ii ) {
      loop_body( ii+icount, begin + ii*stride );
   }

   RAJA_FT_END ;
}

/*!
 ******************************************************************************
 *
 * \brief  Sequential min reduction over index range with stride.
 *
 ******************************************************************************
 */
template <typename T,
          typename LOOP_BODY>
RAJA_INLINE
void forall_min(seq_exec,
                Index_type begin, Index_type end,
                Index_type stride,
                T* min,
                LOOP_BODY loop_body)
{
   RAJA_FT_BEGIN ;

#pragma novector
   for ( Index_type ii = begin ; ii < end ; ii += stride ) {
      loop_body( ii, min );
   }

   RAJA_FT_END ;
}

/*!
 ******************************************************************************
 *
 * \brief  Sequential minloc reduction over index range with stride.
 *
 ******************************************************************************
 */
template <typename T,
          typename LOOP_BODY>
RAJA_INLINE
void forall_minloc(seq_exec,
                   Index_type begin, Index_type end,
                   Index_type stride,
                   T* min, Index_type* loc,
                   LOOP_BODY loop_body)
{
   RAJA_FT_BEGIN ;

#pragma novector
   for ( Index_type ii = begin ; ii < end ; ii += stride ) {
      loop_body( ii, min, loc );
   }

   RAJA_FT_END ;
}

/*!
 ******************************************************************************
 *
 * \brief  Sequential max reduction over index range with stride.
 *
 ******************************************************************************
 */
template <typename T,
          typename LOOP_BODY>
RAJA_INLINE
void forall_max(seq_exec,
                Index_type begin, Index_type end,
                Index_type stride,
                T* max,
                LOOP_BODY loop_body)
{
   RAJA_FT_BEGIN ;

#pragma novector
   for ( Index_type ii = begin ; ii < end ; ii += stride ) {
      loop_body( ii, max );
   }

   RAJA_FT_END ;
}

/*!
 ******************************************************************************
 *
 * \brief  Sequential maxloc reduction over index range with stride.
 *
 ******************************************************************************
 */
template <typename T,
          typename LOOP_BODY>
RAJA_INLINE
void forall_maxloc(seq_exec,
                   Index_type begin, Index_type end,
                   Index_type stride,
                   T* max, Index_type* loc,
                   LOOP_BODY loop_body)
{
   RAJA_FT_BEGIN ;

#pragma novector
   for ( Index_type ii = begin ; ii < end ; ii += stride ) {
      loop_body( ii, max, loc );
   }

   RAJA_FT_END ;
}

/*!
 ******************************************************************************
 *
 * \brief  Sequential sum reduction over index range with stride.
 *
 ******************************************************************************
 */
template <typename T,
          typename LOOP_BODY>
RAJA_INLINE
void forall_sum(seq_exec,
                Index_type begin, Index_type end,
                Index_type stride,
                T* sum, 
                LOOP_BODY loop_body)
{
   RAJA_FT_BEGIN ;

#pragma novector
   for ( Index_type ii = begin ; ii < end ; ii += stride ) {
      loop_body( ii, sum );
   }

   RAJA_FT_END ;
}


//
//////////////////////////////////////////////////////////////////////
//
// Function templates that iterate over range-stride segment objects.
//
//////////////////////////////////////////////////////////////////////
//

/*!
 ******************************************************************************
 *
 * \brief  Sequential iteration over range-stride segment object.
 *
 ******************************************************************************
 */
template <typename LOOP_BODY>
RAJA_INLINE
void forall(seq_exec,
            const RangeStrideSegment& iseg,
            LOOP_BODY loop_body)
{
   const Index_type begin  = iseg.getBegin();
   const Index_type end    = iseg.getEnd();
   const Index_type stride = iseg.getStride();

   RAJA_FT_BEGIN ;

#pragma novector
   for ( Index_type ii = begin ; ii < end ; ii += stride ) {
      loop_body( ii );
   }

   RAJA_FT_END ;
}

/*!
 ******************************************************************************
 *
 * \brief  Sequential iteration over range-stride segment object 
 *         with index count,
 *
 *         NOTE: lambda loop body requires two args (icount, index).
 *
 ******************************************************************************
 */
template <typename LOOP_BODY>
RAJA_INLINE
void forall_Icount(seq_exec,
                   const RangeStrideSegment& iseg,
                   Index_type icount,
                   LOOP_BODY loop_body)
{
   const Index_type begin    = iseg.getBegin();
   const Index_type stride   = iseg.getStride();
   const Index_type loop_end = iseg.getLength();

   RAJA_FT_BEGIN ;

#pragma novector
   for ( Index_type ii = 0 ; ii < loop_end ; ++ii ) {
      loop_body( ii+icount, begin + ii*stride );
   }

   RAJA_FT_END ;
}

/*!
 ******************************************************************************
 *
 * \brief  Sequential min reduction over range-stride segment object
 *         with index count.
 * 
 ******************************************************************************
 */
template <typename T,
          typename LOOP_BODY>
RAJA_INLINE
void forall_min(seq_exec,
                const RangeStrideSegment& iseg,
                T* min,
                LOOP_BODY loop_body)
{
   const Index_type begin  = iseg.getBegin();
   const Index_type end    = iseg.getEnd();
   const Index_type stride = iseg.getStride();

   RAJA_FT_BEGIN ;

#pragma novector
   for ( Index_type ii = begin ; ii < end ; ii += stride ) {
      loop_body( ii, min );
   }

   RAJA_FT_END ;
}

/*!
 ******************************************************************************
 *
 * \brief  Sequential minloc reduction over range-stride segment object
 *         with index count.
 * 
 ******************************************************************************
 */
template <typename T,
          typename LOOP_BODY>
RAJA_INLINE
void forall_minloc(seq_exec,
                   const RangeStrideSegment& iseg,
                   T* min, Index_type* loc,
                   LOOP_BODY loop_body)
{
   const Index_type begin  = iseg.getBegin();
   const Index_type end    = iseg.getEnd();
   const Index_type stride = iseg.getStride();

   RAJA_FT_BEGIN ;

#pragma novector
   for ( Index_type ii = begin ; ii < end ; ii += stride ) {
      loop_body( ii, min, loc );
   }

   RAJA_FT_END ;
}

/*!
 ******************************************************************************
 *
 * \brief  Sequential max reduction over range-stride segment object.
 *
 ******************************************************************************
 */
template <typename T,
          typename LOOP_BODY>
RAJA_INLINE
void forall_max(seq_exec,
                const RangeStrideSegment& iseg,
                T* max,
                LOOP_BODY loop_body)
{
   const Index_type begin  = iseg.getBegin();
   const Index_type end    = iseg.getEnd();
   const Index_type stride = iseg.getStride();

   RAJA_FT_BEGIN ;

#pragma novector
   for ( Index_type ii = begin ; ii < end ; ii += stride ) {
      loop_body( ii, max );
   }

   RAJA_FT_END ;
}

/*!
 ******************************************************************************
 *
 * \brief  Sequential maxloc reduction over range-stride segment object.
 *
 ******************************************************************************
 */
template <typename T,
          typename LOOP_BODY>
RAJA_INLINE
void forall_maxloc(seq_exec,
                   const RangeStrideSegment& iseg,
                   T* max, Index_type* loc,
                   LOOP_BODY loop_body)
{
   const Index_type begin  = iseg.getBegin();
   const Index_type end    = iseg.getEnd();
   const Index_type stride = iseg.getStride();

   RAJA_FT_BEGIN ;

#pragma novector
   for ( Index_type ii = begin ; ii < end ; ii += stride ) {
      loop_body( ii, max, loc );
   }

   RAJA_FT_END ;
}

/*!
 ******************************************************************************
 *
 * \brief  Sequential sum reduction over range-stride segment object.
 *
 ******************************************************************************
 */
template <typename T,
          typename LOOP_BODY>
RAJA_INLINE
void forall_sum(seq_exec,
                const RangeStrideSegment& iseg,
                T* sum,
                LOOP_BODY loop_body)
{
   const Index_type begin  = iseg.getBegin();
   const Index_type end    = iseg.getEnd();
   const Index_type stride = iseg.getStride();

   RAJA_FT_BEGIN ;

#pragma novector
   for ( Index_type ii = begin ; ii < end ; ii += stride ) {
      loop_body( ii, sum );
   }

   RAJA_FT_END ;
}


//
//////////////////////////////////////////////////////////////////////
//
// Function templates that iterate over indirection arrays.
//
//////////////////////////////////////////////////////////////////////
//

/*!
 ******************************************************************************
 *
 * \brief  Sequential iteration over indices in indirection array.
 *
 ******************************************************************************
 */
template <typename LOOP_BODY>
RAJA_INLINE
void forall(seq_exec,
            const Index_type* __restrict__ idx, Index_type len,
            LOOP_BODY loop_body)
{
   RAJA_FT_BEGIN ;

#pragma novector
   for ( Index_type k = 0 ; k < len ; ++k ) {
      loop_body( idx[k] );
   }

   RAJA_FT_END ;
}

/*!
 ******************************************************************************
 *
 * \brief  Sequential iteration over indices in indirection array 
 *         with index count.
 *
 *         NOTE: lambda loop body requires two args (icount, index).
 *
 ******************************************************************************
 */
template <typename LOOP_BODY>
RAJA_INLINE
void forall_Icount(seq_exec,
                   const Index_type* __restrict__ idx, Index_type len,
                   Index_type icount,
                   LOOP_BODY loop_body)
{
   RAJA_FT_BEGIN ;

#pragma novector
   for ( Index_type k = 0 ; k < len ; ++k ) {
      loop_body( k+icount, idx[k] );
   }

   RAJA_FT_END ;
}

/*!
 ******************************************************************************
 *
 * \brief  Sequential min reduction over indices in indirection array.
 *
 ******************************************************************************
 */
template <typename T,
          typename LOOP_BODY>
RAJA_INLINE
void forall_min(seq_exec,
                const Index_type* __restrict__ idx, Index_type len,
                T* min,
                LOOP_BODY loop_body)
{
   RAJA_FT_BEGIN ;

#pragma novector
   for ( Index_type k = 0 ; k < len ; ++k ) {
      loop_body( idx[k], min );
   }

   RAJA_FT_END ;
}

/*!
 ******************************************************************************
 *
 * \brief  Sequential minloc reduction over indices in indirection array.
 *
 ******************************************************************************
 */
template <typename T,
          typename LOOP_BODY>
RAJA_INLINE
void forall_minloc(seq_exec,
                   const Index_type* __restrict__ idx, Index_type len,
                   T* min, Index_type* loc,
                   LOOP_BODY loop_body)
{
   RAJA_FT_BEGIN ;

#pragma novector
   for ( Index_type k = 0 ; k < len ; ++k ) {
      loop_body( idx[k], min, loc );
   }

   RAJA_FT_END ;
}

/*!
 ******************************************************************************
 *
 * \brief  Sequential max reduction over indices in indirection array.
 *
 ******************************************************************************
 */
template <typename T,
          typename LOOP_BODY>
RAJA_INLINE
void forall_max(seq_exec,
                const Index_type* __restrict__ idx, Index_type len,
                T* max,
                LOOP_BODY loop_body)
{
   RAJA_FT_BEGIN ;

#pragma novector
   for ( Index_type k = 0 ; k < len ; ++k ) {
      loop_body( idx[k], max );
   }

   RAJA_FT_END ;
}

/*!
 ******************************************************************************
 *
 * \brief  Sequential maxloc reduction over indices in indirection array.
 *
 ******************************************************************************
 */
template <typename T,
          typename LOOP_BODY>
RAJA_INLINE
void forall_maxloc(seq_exec,
                   const Index_type* __restrict__ idx, Index_type len,
                   T* max, Index_type* loc,
                   LOOP_BODY loop_body)
{
   RAJA_FT_BEGIN ;

#pragma novector
   for ( Index_type k = 0 ; k < len ; ++k ) {
      loop_body( idx[k], max, loc );
   }

   RAJA_FT_END ;
}

/*!
 ******************************************************************************
 *
 * \brief  Sequential sum reduction over indices in indirection array.
 *
 ******************************************************************************
 */
template <typename T,
          typename LOOP_BODY>
RAJA_INLINE
void forall_sum(seq_exec,
                const Index_type* __restrict__ idx, Index_type len,
                T* sum,
                LOOP_BODY loop_body)
{
   RAJA_FT_BEGIN ;

#pragma novector
   for ( Index_type k = 0 ; k < len ; ++k ) {
      loop_body( idx[k], sum );
   }

   RAJA_FT_END ;
}


//
//////////////////////////////////////////////////////////////////////
//
// Function templates that iterate over list segment objects.
//
//////////////////////////////////////////////////////////////////////
//

/*!
 ******************************************************************************
 *
 * \brief  Sequential iteration over list segment object.
 *
 ******************************************************************************
 */
template <typename LOOP_BODY>
RAJA_INLINE
void forall(seq_exec,
            const ListSegment& iseg,
            LOOP_BODY loop_body)
{
   const Index_type* __restrict__ idx = iseg.getIndex();
   const Index_type len = iseg.getLength();

   RAJA_FT_BEGIN ;

#pragma novector
   for ( Index_type k = 0 ; k < len ; ++k ) {
      loop_body( idx[k] );
   }

   RAJA_FT_END ;
}

/*!
 ******************************************************************************
 *
 * \brief  Sequential iteration over list segment object with index count.
 *
 *         NOTE: lambda loop body requires two args (icount, index).
 *
 ******************************************************************************
 */
template <typename LOOP_BODY>
RAJA_INLINE
void forall_Icount(seq_exec,
                   const ListSegment& iseg,
                   Index_type icount, 
                   LOOP_BODY loop_body)
{
   const Index_type* __restrict__ idx = iseg.getIndex();
   const Index_type len = iseg.getLength();

   RAJA_FT_BEGIN ;

#pragma novector
   for ( Index_type k = 0 ; k < len ; ++k ) {
      loop_body( k+icount, idx[k] );
   }

   RAJA_FT_END ;
}

/*!
 ******************************************************************************
 *
 * \brief  Sequential min reduction over list segment object.
 *
 ******************************************************************************
 */
template <typename T,
          typename LOOP_BODY>
RAJA_INLINE
void forall_min(seq_exec,
                const ListSegment& iseg,
                T* min,
                LOOP_BODY loop_body)
{
   const Index_type* __restrict__ idx = iseg.getIndex();
   const Index_type len = iseg.getLength();

   RAJA_FT_BEGIN ;

#pragma novector
   for ( Index_type k = 0 ; k < len ; ++k ) {
      loop_body( idx[k], min );
   }

   RAJA_FT_END ;
}

/*!
 ******************************************************************************
 *
 * \brief  Sequential minloc reduction over list segment object.
 *
 ******************************************************************************
 */
template <typename T,
          typename LOOP_BODY>
RAJA_INLINE
void forall_minloc(seq_exec,
                   const ListSegment& iseg,
                   T* min, Index_type* loc,
                   LOOP_BODY loop_body)
{
   const Index_type* __restrict__ idx = iseg.getIndex();
   const Index_type len = iseg.getLength();

   RAJA_FT_BEGIN ;

#pragma novector
   for ( Index_type k = 0 ; k < len ; ++k ) {
      loop_body( idx[k], min, loc );
   }

   RAJA_FT_END ;
}

/*!
 ******************************************************************************
 *
 * \brief  Sequential max reduction over list segment object.
 *
 ******************************************************************************
 */
template <typename T,
          typename LOOP_BODY>
RAJA_INLINE
void forall_max(seq_exec,
                const ListSegment& iseg,
                T* max,
                LOOP_BODY loop_body)
{
   const Index_type* __restrict__ idx = iseg.getIndex();
   const Index_type len = iseg.getLength();

   RAJA_FT_BEGIN ;

#pragma novector
   for ( Index_type k = 0 ; k < len ; ++k ) {
      loop_body( idx[k], max );
   }

   RAJA_FT_END ;
}

/*!
 ******************************************************************************
 *
 * \brief  Sequential maxloc reduction over list segment object.
 *
 ******************************************************************************
 */
template <typename T,
          typename LOOP_BODY>
RAJA_INLINE
void forall_maxloc(seq_exec,
                   const ListSegment& iseg,
                   T* max, Index_type* loc,
                   LOOP_BODY loop_body)
{
   const Index_type* __restrict__ idx = iseg.getIndex();
   const Index_type len = iseg.getLength();

   RAJA_FT_BEGIN ;

#pragma novector
   for ( Index_type k = 0 ; k < len ; ++k ) {
      loop_body( idx[k], max, loc );
   }

   RAJA_FT_END ;
}

/*!
 ******************************************************************************
 *
 * \brief  Sequential sum reduction over list segment object.
 *
 ******************************************************************************
 */
template <typename T,
          typename LOOP_BODY>
RAJA_INLINE
void forall_sum(seq_exec,
                const ListSegment& iseg,
                T* sum,
                LOOP_BODY loop_body)
{
   const Index_type* __restrict__ idx = iseg.getIndex();
   const Index_type len = iseg.getLength();

   RAJA_FT_BEGIN ;

#pragma novector
   for ( Index_type k = 0 ; k < len ; ++k ) {
      loop_body( idx[k], sum );
   }

   RAJA_FT_END ;
}

//
//////////////////////////////////////////////////////////////////////
//
// The following function templates iterate over index set
// segments sequentially.  Segment execution is defined by segment
// execution policy template parameter.
//
//////////////////////////////////////////////////////////////////////
//

/*!
 ******************************************************************************
 *
 * \brief  Sequential iteration over segments of index set and
 *         use execution policy template parameter to execute segments.
 *
 ******************************************************************************
 */
template <typename SEG_EXEC_POLICY_T,
          typename LOOP_BODY>
RAJA_INLINE
void forall( IndexSet::ExecPolicy<seq_segit, SEG_EXEC_POLICY_T>,
             const IndexSet& iset, 
             LOOP_BODY loop_body )
{
   const int num_seg = iset.getNumSegments();
   for ( int isi = 0; isi < num_seg; ++isi ) {

      const BaseSegment* iseg = iset.getSegment(isi);
      SegmentType segtype = iseg->getType();

      switch ( segtype ) {

         case _RangeSeg_ : {
            const RangeSegment* tseg =
               static_cast<const RangeSegment*>(iseg);
            forall(
               SEG_EXEC_POLICY_T(),
               tseg->getBegin(), tseg->getEnd(),
               loop_body
            );
            break;
         }

#if 0  // RDH RETHINK
         case _RangeStrideSeg_ : {
            const RangeStrideSegment* tseg =
               static_cast<const RangeStrideSegment*>(iseg);
            forall(
               SEG_EXEC_POLICY_T(),
               tseg->getBegin(), tseg->getEnd(), tseg->getStride(),
               loop_body
            );
            break;
         }
#endif

         case _ListSeg_ : {
            const ListSegment* tseg =
               static_cast<const ListSegment*>(iseg);
            forall(
               SEG_EXEC_POLICY_T(),
               tseg->getIndex(), tseg->getLength(),
               loop_body
            );
            break;
         }

         default : {
         }

      }  // switch on segment type

   } // iterate over segments of index set
}

/*!
 ******************************************************************************
 *
 * \brief  Sequential iteration over segments of index set and
 *         use execution policy template parameter to execute segments.
 *
 *         This method passes index count to segment iteration.
 *
 *         NOTE: lambda loop body requires two args (icount, index).
 *
 ******************************************************************************
 */
template <typename SEG_EXEC_POLICY_T,
          typename LOOP_BODY>
RAJA_INLINE
void forall_Icount( IndexSet::ExecPolicy<seq_segit, SEG_EXEC_POLICY_T>,
                    const IndexSet& iset, 
                    LOOP_BODY loop_body )
{
   const int num_seg = iset.getNumSegments();
   for ( int isi = 0; isi < num_seg; ++isi ) {

      const IndexSetSegInfo* seg_info = iset.getSegmentInfo(isi);

      const BaseSegment* iseg = seg_info->getSegment();
      SegmentType segtype = iseg->getType();

      Index_type icount = seg_info->getIcount();

      switch ( segtype ) {

         case _RangeSeg_ : {
            const RangeSegment* tseg =
               static_cast<const RangeSegment*>(iseg);
            forall_Icount(
               SEG_EXEC_POLICY_T(),
               tseg->getBegin(), tseg->getEnd(),
               icount,
               loop_body
            );
            break;
         }

#if 0  // RDH RETHINK
         case _RangeStrideSeg_ : {
            const RangeStrideSegment* tseg =
               static_cast<const RangeStrideSegment*>(iseg);
            forall_Icount(
               SEG_EXEC_POLICY_T(),
               tseg->getBegin(), tseg->getEnd(), tseg->getStride(),
               icount,
               loop_body
            );
            break;
         }
#endif

         case _ListSeg_ : {
            const ListSegment* tseg =
               static_cast<const ListSegment*>(iseg);
            forall_Icount(
               SEG_EXEC_POLICY_T(),
               tseg->getIndex(), tseg->getLength(),
               icount,
               loop_body
            );
            break;
         }

         default : {
         }

      }  // switch on segment type

   } // iterate over segments of index set
}


/*!
 ******************************************************************************
 *
 * \brief  min reduction that iterates over index set segments
 *         sequentially and uses execution policy template parameter to 
 *         execute segments.
 *
 ******************************************************************************
 */
template <typename SEG_EXEC_POLICY_T,
          typename T,
          typename LOOP_BODY>
RAJA_INLINE
void forall_min( IndexSet::ExecPolicy<seq_segit, SEG_EXEC_POLICY_T>,
                 const IndexSet& iset,
                 T* min,
                 LOOP_BODY loop_body)
{
   const int num_seg = iset.getNumSegments();
   for ( int isi = 0; isi < num_seg; ++isi ) {

      const BaseSegment* iseg = iset.getSegment(isi);
      SegmentType segtype = iseg->getType();

      switch ( segtype ) {

         case _RangeSeg_ : {
            const RangeSegment* tseg =
               static_cast<const RangeSegment*>(iseg);
            forall_min(
               SEG_EXEC_POLICY_T(),
               tseg->getBegin(), tseg->getEnd(),
               min,
               loop_body
            );
            break;
         }

#if 0  // RDH RETHINK
         case _RangeStrideSeg_ : {
            const RangeStrideSegment* tseg =
               static_cast<const RangeStrideSegment*>(iseg);
            forall_min(
               SEG_EXEC_POLICY_T(),
               tseg->getBegin(), tseg->getEnd(), tseg->getStride(),
               min,
               loop_body
            );
            break;
         }
#endif

         case _ListSeg_ : {
            const ListSegment* tseg =
               static_cast<const ListSegment*>(iseg);
            forall_min(
               SEG_EXEC_POLICY_T(),
               tseg->getIndex(), tseg->getLength(),
               min,
               loop_body
            );
            break;
         }

         default : {
         }

      }  // switch on segment type

   } // iterate over segments of index set

}

/*!
 ******************************************************************************
 *
 * \brief  minloc reduction that iterates over index set segments
 *         sequentially and uses execution policy template parameter to 
 *         execute segments.
 *
 ******************************************************************************
 */
template <typename SEG_EXEC_POLICY_T,
          typename T,
          typename LOOP_BODY>
RAJA_INLINE
void forall_minloc( IndexSet::ExecPolicy<seq_segit, SEG_EXEC_POLICY_T>,
                    const IndexSet& iset,
                    T* min, Index_type *loc,
                    LOOP_BODY loop_body)
{
   const int num_seg = iset.getNumSegments();
   for ( int isi = 0; isi < num_seg; ++isi ) {

      const BaseSegment* iseg = iset.getSegment(isi);
      SegmentType segtype = iseg->getType();

      switch ( segtype ) {

         case _RangeSeg_ : {
            const RangeSegment* tseg =
               static_cast<const RangeSegment*>(iseg);
            forall_minloc(
               SEG_EXEC_POLICY_T(),
               tseg->getBegin(), tseg->getEnd(),
               min, loc,
               loop_body
            );
            break;
         }

#if 0  // RDH RETHINK
         case _RangeStrideSeg_ : {
            const RangeStrideSegment* tseg =
               static_cast<const RangeStrideSegment*>(iseg);
            forall_minloc(
               SEG_EXEC_POLICY_T(),
               tseg->getBegin(), tseg->getEnd(), tseg->getStride(),
               min, loc,
               loop_body
            );
            break;
         }
#endif

         case _ListSeg_ : {
            const ListSegment* tseg =
               static_cast<const ListSegment*>(iseg);
            forall_minloc(
               SEG_EXEC_POLICY_T(),
               tseg->getIndex(), tseg->getLength(),
               min, loc,
               loop_body
            );
            break;
         }

         default : {
         }

      }  // switch on segment type

   } // iterate over segments of index set

}

/*!
 ******************************************************************************
 *
 * \brief  maxloc reduction that iterates over index set segments
 *         sequentially and uses execution policy template parameter to 
 *         execute segments.
 *
 ******************************************************************************
 */
template <typename SEG_EXEC_POLICY_T,
          typename T,
          typename LOOP_BODY>
RAJA_INLINE
void forall_maxloc( IndexSet::ExecPolicy<seq_segit, SEG_EXEC_POLICY_T>,
                    const IndexSet& iset,
                    T* max, Index_type *loc,
                    LOOP_BODY loop_body)
{
   const int num_seg = iset.getNumSegments();
   for ( int isi = 0; isi < num_seg; ++isi ) {

      const BaseSegment* iseg = iset.getSegment(isi);
      SegmentType segtype = iseg->getType();

      switch ( segtype ) {

         case _RangeSeg_ : {
            const RangeSegment* tseg =
               static_cast<const RangeSegment*>(iseg);
            forall_maxloc(
               SEG_EXEC_POLICY_T(),
               tseg->getBegin(), tseg->getEnd(),
               max, loc,
               loop_body
            );
            break;
         }

#if 0  // RDH RETHINK
         case _RangeStrideSeg_ : {
            const RangeStrideSegment* tseg =
               static_cast<const RangeStrideSegment*>(iseg);
            forall_maxloc(
               SEG_EXEC_POLICY_T(),
               tseg->getBegin(), tseg->getEnd(), tseg->getStride(),
               max, loc,
               loop_body
            );
            break;
         }
#endif

         case _ListSeg_ : {
            const ListSegment* tseg =
               static_cast<const ListSegment*>(iseg);
            forall_maxloc(
               SEG_EXEC_POLICY_T(),
               tseg->getIndex(), tseg->getLength(),
               max, loc,
               loop_body
            );
            break;
         }

         default : {
         }

      }  // switch on segment type

   } // iterate over segments of index set

}

/*!
 ******************************************************************************
 *
 * \brief  sum reduction that iterates over index set segments
 *         sequentially and uses execution policy template parameter to 
 *         execute segments.
 *
 ******************************************************************************
 */
template <typename SEG_EXEC_POLICY_T,
          typename T,
          typename LOOP_BODY>
RAJA_INLINE
void forall_sum( IndexSet::ExecPolicy<seq_segit, SEG_EXEC_POLICY_T>,
                 const IndexSet& iset,
                 T* sum,
                 LOOP_BODY loop_body)
{
   const int num_seg = iset.getNumSegments();
   for ( int isi = 0; isi < num_seg; ++isi ) {

      const BaseSegment* iseg = iset.getSegment(isi);
      SegmentType segtype = iseg->getType();

      switch ( segtype ) {

         case _RangeSeg_ : {
            const RangeSegment* tseg =
               static_cast<const RangeSegment*>(iseg);
            forall_sum(
               SEG_EXEC_POLICY_T(),
               tseg->getBegin(), tseg->getEnd(),
               sum,
               loop_body
            );
            break;
         }

#if 0  // RDH RETHINK
         case _RangeStrideSeg_ : {
            const RangeStrideSegment* tseg =
               static_cast<const RangeStrideSegment*>(iseg);
            forall_sum(
               SEG_EXEC_POLICY_T(),
               tseg->getBegin(), tseg->getEnd(), tseg->getStride(),
               sum,
               loop_body
            );
            break;
         }
#endif

         case _ListSeg_ : {
            const ListSegment* tseg =
               static_cast<const ListSegment*>(iseg);
            forall_sum(
               SEG_EXEC_POLICY_T(),
               tseg->getIndex(), tseg->getLength(),
               sum,
               loop_body
            );
            break;
         }

         default : {
         }

      }  // switch on segment type

   } // iterate over segments of index set

}


/*!
 ******************************************************************************
 *
 * \brief  Special segment iteration using sequential segment iteration loop 
 *         (no dependency graph used or needed). Individual segment execution 
 *         is defined in loop body.
 *
 *         NOTE: IndexSet must contain only RangeSegments.
 *
 ******************************************************************************
 */
template <typename LOOP_BODY>
RAJA_INLINE
void forall_segments(seq_segit,
                     const IndexSet& iset,
                     LOOP_BODY loop_body)
{
   IndexSet& ncis = (*const_cast<IndexSet *>(&iset)) ;
   const int num_seg = ncis.getNumSegments();

   /* Create a temporary IndexSet with one Segment */
   IndexSet is_tmp;
   is_tmp.push_back( RangeSegment(0, 0) ) ; // create a dummy range segment

   RangeSegment* segTmp = static_cast<RangeSegment*>(is_tmp.getSegment(0));

   for ( int isi = 0; isi < num_seg; ++isi ) {

      RangeSegment* isetSeg = 
         static_cast<RangeSegment*>(ncis.getSegment(isi));

      segTmp->setBegin(isetSeg->getBegin()) ;
      segTmp->setEnd(isetSeg->getEnd()) ;
      segTmp->setPrivate(isetSeg->getPrivate()) ;

      loop_body(&is_tmp) ;

   } // loop over index set segments
}


}  // closing brace for RAJA namespace


#endif  // closing endif for header file include guard
