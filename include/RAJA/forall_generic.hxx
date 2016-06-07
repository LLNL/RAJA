/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   Header file containing RAJA index set and segment iteration 
 *          template methods that take an execution policy as a template 
 *          parameter.
 *
 *          The templates for segments support the following usage pattern:
 *
 *             forall<exec_policy>( index set, loop body );
 *
 *          which is equivalent to:
 *
 *             forall( exec_policy(), index set, loop body );
 *
 *          The former is slightly more concise. Here, the execution policy 
 *          type is associated with a tag struct defined in the exec_poilicy 
 *          hearder file. Usage of the forall_Icount() is similar. 
 * 
 *          The forall() and forall_Icount() methods that take an index set
 *          take an execution policy of the form:
 *
 *          IndexSet::ExecPolicy< seg_it_policy, seg_exec_policy >
 *
 *          Here, the first template parameter determines the scheme for
 *          iteratiing over the index set segments and the second determines
 *          how each segment is executed.
 *
 *          The forall() templates accept a loop body argument that takes 
 *          a single Index_type argument identifying the index of a loop
 *          iteration. The forall_Icount() templates accept a loop body that
 *          takes two Index_type arguments. The first is the number of the 
 *          iteration in the indes set or segment, the second if the actual 
 *          index of the loop iteration.
 *
 *
 *          IMPORTANT: Use of any of these methods requires a specialization
 *                     for the given index set type and execution policy.
 *
 ******************************************************************************
 */

#ifndef RAJA_forall_generic_HXX
#define RAJA_forall_generic_HXX

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
// For additional details, please also read raja/README-license.txt.
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

#include "RAJA/int_datatypes.hxx"
#include "RAJA/Iterators.hxx"
#include "RAJA/fault_tolerance.hxx"
#include "RAJA/PolicyBase.hxx"

#include <type_traits>
#include <iterator>


namespace RAJA {

//
//////////////////////////////////////////////////////////////////////
//
// Iteration over generic iterators
//
//////////////////////////////////////////////////////////////////////
//

template<typename Iterator,
         typename Body>
struct IcountWrapper {
    using vtype = typename std::iterator_traits<Iterator>::value_type;
    constexpr IcountWrapper(Body && body) : m_body(body) {}
    RAJA_HOST_DEVICE RAJA_INLINE void operator()(vtype&& i) const {
        m_body(i.first, *i.second);
    }
    private:
        Body m_body;
};
template<typename Iterator,
         typename Body>
constexpr auto make_icount_wrapper(Body &&loop_body)
    -> IcountWrapper<Iterator, decltype(loop_body)> {
    return IcountWrapper<Iterator, decltype(loop_body)>(loop_body);
}

template<typename Iterable>
struct IcountIterableWrapper : public Iterable {
   using iterator = Iterators::Enumerater<typename Iterable::iterator>;

   IcountIterableWrapper() = delete;
   constexpr IcountIterableWrapper(Iterable&& iter, std::ptrdiff_t val = 0, std::ptrdiff_t offset = 0) :
       Iterable(iter), m_val(val), m_off(offset) {}

   ///
   /// Get an iterator to the end.
   ///
   constexpr iterator end() const {
       return iterator(Iterable::end(), m_val + size(), m_off);
   }

   ///
   /// Get an iterator to the beginning.
   ///
   constexpr iterator begin() const {
       return iterator(Iterable::begin(), m_val, m_off);
   }

   ///
   /// Return the number of elements in the range.
   ///
   constexpr Index_type size() const {
       return std::distance(Iterable::begin(), Iterable::end());
   }
    private:
   std::ptrdiff_t m_val;
   std::ptrdiff_t m_off;
};

template<typename Iterable>
constexpr auto make_icount_iterable_wrapper(Iterable&& iter, std::ptrdiff_t val=0, std::ptrdiff_t offset=0)
    -> IcountIterableWrapper<Iterable> {
    return IcountIterableWrapper<Iterable>(std::forward<Iterable>(iter), val, offset);
}


/*!
 ******************************************************************************
 *
 * \brief  Generic iteration over random access iterators.
 *
 ******************************************************************************
 */
template <typename Policy,
          typename Iterable,
          typename LOOP_BODY,
          typename std::enable_if<
                           Iterators::OffersRAI<Iterable>::value
                        && (!std::is_base_of<IndexSet, Iterable>::value)>::type * = nullptr
          >
RAJA_INLINE
void forall(Policy &&p,
            Iterable&& iter,
            LOOP_BODY loop_body)
{
   RAJA_FT_BEGIN ;

   p(std::forward<Iterable>(iter), std::forward<LOOP_BODY>(loop_body));

   RAJA_FT_END ;
}


/*!
 ******************************************************************************
 *
 * \brief Generic dispatch over iterators with icount.
 *
 ******************************************************************************
 */
template <typename EXEC_POLICY_T,
          typename Iterator,
          typename LOOP_BODY>
RAJA_INLINE
void forall_icount(
            Iterator&& begin,
            Iterator&& end,
            Index_type icount,
            LOOP_BODY loop_body)
{
   using category = typename std::iterator_traits<Iterator>::iterator_category;
   static_assert(std::is_base_of<std::random_access_iterator_tag, category>::value,
                 "Iterators passed to RAJA must be Random Access or Contiguous iterators");

   using IterT = Iterators::Enumerater<Iterator>;

   forall(EXEC_POLICY_T(),
          IterT(begin, 0, icount),
          IterT(end, std::distance(begin, end), icount),
          make_icount_wrapper<IterT>(loop_body));
}

/*!
 ******************************************************************************
 *
 * \brief Generic dispatch over iterators.
 *
 ******************************************************************************
 */
template <typename EXEC_POLICY_T,
          typename Iterator,
          typename LOOP_BODY>
RAJA_INLINE
void forall(Iterator&& begin,
            Iterator&& end,
            LOOP_BODY loop_body)
{
   using category = typename std::iterator_traits<Iterator>::iterator_category;
   static_assert(std::is_base_of<std::random_access_iterator_tag, category>::value,
                 "Iterators passed to RAJA must be Random Access or Contiguous iterators");


   forall(EXEC_POLICY_T(),
          begin, end,
          loop_body );
}

/*!
 ******************************************************************************
 *
 * \brief Generic dispatch over containers with icount
 *
 ******************************************************************************
 */
template <typename EXEC_POLICY_T,
          typename Container,
          typename LOOP_BODY,
          typename std::enable_if<
                           Iterators::OffersRAI<Container>::value
                        && (!std::is_base_of<IndexSet, Container>::value)>::type * = nullptr
          >
RAJA_INLINE
void forall_Icount(Container& c,
            Index_type icount,
            LOOP_BODY loop_body)
{
   auto begin = std::begin(c);
   auto end = std::end(c);
   using Iterator = decltype(std::begin(c));
   using category = typename std::iterator_traits<Iterator>::iterator_category;
   static_assert(std::is_base_of<std::random_access_iterator_tag, category>::value,
                 "Iterators passed to RAJA must be Random Access or Contiguous iterators");

   auto wrapped = make_icount_iterable_wrapper(std::forward<Container>(c), 0, icount);
   forall(EXEC_POLICY_T(),
          wrapped,
          make_icount_wrapper<typename decltype(wrapped)::iterator>(loop_body));
}

/*!
 ******************************************************************************
 *
 * \brief Generic dispatch over containers
 *
 ******************************************************************************
 */
template <typename EXEC_POLICY_T,
          typename Container,
          typename LOOP_BODY,
          typename std::enable_if<
                           Iterators::OffersRAI<Container>::value
                        && (!std::is_base_of<IndexSet, Container>::value)>::type * = nullptr
          >
RAJA_INLINE
void forall(Container&& c,
            LOOP_BODY loop_body)
{
   auto begin = std::begin(c);
   auto end = std::end(c);
   using category = typename std::iterator_traits<decltype(std::begin(c))>::iterator_category;
   static_assert(std::is_base_of<std::random_access_iterator_tag, category>::value,
                 "Iterators passed to RAJA must be Random Access or Contiguous iterators");

   // printf("running container\n");

   forall(EXEC_POLICY_T(),
          std::forward<Container>(c),
          loop_body );
}

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
 * \brief Generic iteration over index range.
 *
 ******************************************************************************
 */
template <typename EXEC_POLICY_T,
          typename LOOP_BODY>
RAJA_INLINE
void forall(Index_type begin, Index_type end,
            LOOP_BODY loop_body)
{
   forall<EXEC_POLICY_T>(
           RangeSegment(begin, end),
           loop_body );
}

/*!
 ******************************************************************************
 *
 * \brief Generic iteration over index range with index count.
 *
 *        NOTE: lambda loop body requires two args (icount, index).
 *
 ******************************************************************************
 */
template <typename EXEC_POLICY_T,
          typename LOOP_BODY>
RAJA_INLINE
void forall_Icount(Index_type begin, Index_type end,
                   Index_type icount,
                   LOOP_BODY loop_body)
{
   auto wrapped = make_icount_iterable_wrapper(RangeSegment(begin, end), 0, icount);
   forall(EXEC_POLICY_T(),
          wrapped,
          make_icount_wrapper<typename decltype(wrapped)::iterator>(loop_body));
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
 * \brief Generic iteration over index range with stride.
 *
 ******************************************************************************
 */
template <typename EXEC_POLICY_T,
          typename LOOP_BODY>
RAJA_INLINE
void forall(Index_type begin, Index_type end, 
            Index_type stride,
            LOOP_BODY loop_body)
{
   forall( EXEC_POLICY_T(),
           RangeStrideSegment(begin, end, stride),
           loop_body );
}

/*!
 ******************************************************************************
 *
 * \brief Generic iteration over index range with stride with index count.
 *
 *        NOTE: lambda loop body requires two args (icount, index).
 *
 ******************************************************************************
 */
template <typename EXEC_POLICY_T,
          typename LOOP_BODY>
RAJA_INLINE
void forall_Icount(Index_type begin, Index_type end,
                   Index_type stride,
                   Index_type icount, 
                   LOOP_BODY loop_body)
{
   forall_Icount( EXEC_POLICY_T(),
                  make_icount_iterable_wrapper(RangeStrideSegment(begin, end, stride), 0, icount),
                  icount,
                  loop_body );
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
 * \brief  Generic iteration over indices in indirection array.
 *
 ******************************************************************************
 */
template <typename EXEC_POLICY_T,
          typename LOOP_BODY>
RAJA_INLINE
void forall(const Index_type* idx, Index_type len,
            LOOP_BODY loop_body)
{
   // turn into an iterator
   forall<EXEC_POLICY_T>(
           ListSegment(idx, len),
           loop_body );
}

/*!
 ******************************************************************************
 *
 * \brief  Generic iteration over indices in indirection array with index count.
 *
 *         NOTE: lambda loop body requires two args (icount, index).
 *
 ******************************************************************************
 */
template <typename EXEC_POLICY_T,
          typename LOOP_BODY>
RAJA_INLINE
void forall_Icount(const Index_type* idx, Index_type len,
                   Index_type icount,
                   LOOP_BODY loop_body)
{
   // turn into an iterator
   forall_Icount<EXEC_POLICY_T>(
           make_icount_iterable_wrapper(ListSegment(idx, len), 0, icount),
           icount,
           loop_body );
}


//
//////////////////////////////////////////////////////////////////////
//
// Function templates that iterate over index sets or segments 
// according to execution policy template parameter.
//
//////////////////////////////////////////////////////////////////////
//

/*!
 ******************************************************************************
 *
 * \brief Generic iteration over arbitrary index set or segment.
 *
 ******************************************************************************
 */
template <typename EXEC_POLICY_T,
          typename INDEXSET_T, 
          typename LOOP_BODY,
          typename std::enable_if<std::is_base_of<IndexSet, INDEXSET_T>::value>::type * = nullptr
          >
RAJA_INLINE
void forall(const INDEXSET_T& iset, LOOP_BODY loop_body)
{
   forall(EXEC_POLICY_T(),
          iset, loop_body);
}

/*!
 ******************************************************************************
 *
 * \brief Generic iteration over arbitrary index set or segment with index 
 *        count.
 *
 *        NOTE: lambda loop body requires two args (icount, index). 
 *
 ******************************************************************************
 */
template <typename EXEC_POLICY_T,
          typename INDEXSET_T, 
          typename LOOP_BODY,
          typename std::enable_if<std::is_base_of<IndexSet, INDEXSET_T>::value>::type * = nullptr
          >
RAJA_INLINE
void forall_Icount(const INDEXSET_T& iset, LOOP_BODY loop_body)
{
   forall_Icount(EXEC_POLICY_T(),
                 iset, loop_body);
}

/*!
 ******************************************************************************
 *
 * \brief Generic iteration over arbitrary index set or segment with 
 *        starting index count.
 *
 *        NOTE: lambda loop body requires two args (icount, index).
 *
 ******************************************************************************
 */
template <typename EXEC_POLICY_T,
          typename INDEXSET_T,
          typename LOOP_BODY,
          typename std::enable_if<std::is_base_of<IndexSet, INDEXSET_T>::value>::type * = nullptr
          >
RAJA_INLINE
void forall_Icount(const INDEXSET_T& iset, 
                   Index_type icount,
                   LOOP_BODY loop_body)
{
   forall_Icount(EXEC_POLICY_T(),
                 iset, 
                 icount,
                 loop_body);
}

/*!
 ******************************************************************************
 *
 * \brief Generic dependency-graph segment iteration over index set segments.
 *
 ******************************************************************************
 */
template <typename EXEC_POLICY_T,
          typename LOOP_BODY>
RAJA_INLINE
void forall_segments(const IndexSet& iset,
                     LOOP_BODY loop_body)
{
   forall_segments(EXEC_POLICY_T(),
                   iset,
                   loop_body);
}


}  // closing brace for RAJA namespace

#endif  // closing endif for header file include guard
