//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef __TEST_FORALL_FUNCTORS_HPP__
#define __TEST_FORALL_FUNCTORS_HPP__

////////////////////////////////////////////////////////////
// Functors that provide loop bodies for range segment tests
////////////////////////////////////////////////////////////
template <typename T>
class RangeSegmentTestFunctor
{
public:
   RangeSegmentTestFunctor(T* wa, T beg)
   : working_array(wa), rbegin(beg) { ; }

   RAJA_HOST_DEVICE
   void operator() (T idx)
   {
     working_array[idx - rbegin] = idx; 
   }

   T* working_array;
   T rbegin;
};

template <typename T, typename VT>
class RangeSegmentViewTestFunctor
{
public:
   RangeSegmentViewTestFunctor(VT& v, T beg)
   : work_view(v), rbegin(beg) { ; }

   RAJA_HOST_DEVICE
   void operator() (T idx)
   {
     work_view( idx - rbegin ) = idx;
   }

   VT work_view;
   T rbegin;
};

template <typename T, typename VT>
class RangeSegmentOffsetViewTestFunctor
{
public:
   RangeSegmentOffsetViewTestFunctor(VT& v)
   : work_view(v) { ; }

   RAJA_HOST_DEVICE
   void operator() (T idx)
   {
     work_view( idx ) = idx;
   }

   VT work_view;
};

template <typename T, typename VT>
class RangeSegment2DViewTestFunctor
{
public:
   RangeSegment2DViewTestFunctor(VT& v, T n)
   : work_view(v), N(n) { ; }

   RAJA_HOST_DEVICE
   void operator() (T idx)
   {
     const T row = idx / N;
     const T col = idx % N;
     work_view(row, col) = row * N + col;
   }

   VT work_view;
   T N;
};

template <typename T, typename VT>
class RangeSegment2DOffsetViewTestFunctor
{
public:
   RangeSegment2DOffsetViewTestFunctor(VT& v, T n)
   : work_view(v), N(n) { ; }

   RAJA_HOST_DEVICE
   void operator() (T idx)
   {
     const T row = idx / N;
     const T col = idx % N;
     work_view(row, col) = idx;
   }

   VT work_view;
   T N;
};


////////////////////////////////////////////////////////////////////
// Functors that provide loop bodies for range-stride segment tests
////////////////////////////////////////////////////////////////////
template <typename T>
class RangeStrideSegmentTestFunctor
{
public:
   RangeStrideSegmentTestFunctor(T* wa, T fir, T str)
   : working_array(wa), first(fir), stride(str) { ; }

   RAJA_HOST_DEVICE
   void operator() (T idx)
   {
     working_array[ (idx-first)/stride ] = idx;
   }

   T* working_array;
   T first;
   T stride;
};

template <typename T, typename VT>
class RangeStrideSegmentViewTestFunctor
{
public:
   RangeStrideSegmentViewTestFunctor(VT& v, T fir, T str)
   : work_view(v), first(fir), stride(str) { ; }

   RAJA_HOST_DEVICE
   void operator() (T idx)
   {
     work_view( (idx-first)/stride ) = idx;
   }

   VT work_view;
   T first;
   T stride;
};

 
////////////////////////////////////////////////////////////
// Functors that provide loop bodies for list segment tests
////////////////////////////////////////////////////////////
template <typename T>
class ListSegmentTestFunctor
{
public:
   ListSegmentTestFunctor(T* wa)
   : working_array(wa) { ; }

   RAJA_HOST_DEVICE
   void operator() (T idx)
   {
     working_array[idx] = idx;
   }

   T* working_array;
};

template <typename T, typename VT>
class ListSegmentViewTestFunctor
{
public:
   ListSegmentViewTestFunctor(VT& v)
   : work_view(v) { ; }

   RAJA_HOST_DEVICE
   void operator() (T idx)
   {
     work_view( idx ) = idx;
   }

   VT work_view;
};


////////////////////////////////////////////////////////////
// Functors that provide loop bodies for indexset tests
////////////////////////////////////////////////////////////

template <typename T>
using IndexSetTestFunctor = ListSegmentTestFunctor<T>;

template <typename T, typename VT>
using IndexSetViewTestFunctor = ListSegmentViewTestFunctor<T, VT>;


#endif  // __TEST_FORALL_UTILS_HPP__
