/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   RAJA header file defining index set classes.
 *
 ******************************************************************************
 */

#ifndef RAJA_IndexSet_HPP
#define RAJA_IndexSet_HPP

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

#include "RAJA/policy/PolicyBase.hpp"

#include "RAJA/internal/Iterators.hpp"

#include "RAJA/index/ListSegment.hpp"
#include "RAJA/index/RangeSegment.hpp"

#include "RAJA/internal/RAJAVec.hpp"

#include "RAJA/util/concepts.hpp"

#include <iosfwd>

namespace RAJA
{

enum PushEnd { PUSH_FRONT, PUSH_BACK };
enum PushCopy { PUSH_COPY, PUSH_NOCOPY };


///
/// Class representing index set execution policy.
///
/// The first template parameter describes the policy for iterating
/// over segments.  The second describes the policy for executing
/// each segment.
///
template <typename SEG_ITER_POLICY_T, typename SEG_EXEC_POLICY_T>
struct ExecPolicy
    : public RAJA::make_policy_pattern_t<SEG_EXEC_POLICY_T::policy,
                                         RAJA::Pattern::forall> {
  typedef SEG_ITER_POLICY_T seg_it;
  typedef SEG_EXEC_POLICY_T seg_exec;
};

template <typename... TALL>
class StaticIndexSet;


/*!
 ******************************************************************************
 *
 * \brief  Class representing an index set which is a collection
 *         of segment objects.
 *
 ******************************************************************************
 */
template <typename T0, typename... TREST>
class StaticIndexSet<T0, TREST...> : public StaticIndexSet<TREST...>
{
private:
  using PARENT = StaticIndexSet<TREST...>;
  const int T0_TypeId = sizeof...(TREST);

public:
  //@{
  //!  @name Constructor and destructor methods

  ///
  /// Construct empty index set
  ///
  RAJA_INLINE
  constexpr StaticIndexSet() : PARENT() {}

  ///
  /// Copy-constructor for index set
  ///
  RAJA_INLINE
  StaticIndexSet(StaticIndexSet<T0, TREST...> const &c)
      : PARENT((PARENT const &)c)
  {
    size_t num = c.data.size();

    // Copy all segments of type T0
    data.resize(num);
    for (size_t i = 0; i < num; ++i) {
      // construct a copy of the segment in c
      // data[i] = new T0(*c.data[i]); // this would be an actual copy - copy
      // the ref only
      data[i] = c.data[i];
    }

    // mark all as not owned by us
    owner.resize(num, 0);
  }

  ///
  /// Copy-assignment operator for index set
  ///
  StaticIndexSet<T0, TREST...> &operator=(
      const StaticIndexSet<T0, TREST...> &rhs)
  {
    if (&rhs != this) {
      StaticIndexSet<T0, TREST...> copy(rhs);
      this->swap(copy);
    }
    return *this;
  }

  ///
  /// Destroy index set including all index set segments.
  ///
  RAJA_INLINE
  ~StaticIndexSet()
  {
    size_t num_seg = data.size();
    for (size_t i = 0; i < num_seg; ++i) {

      // Only free segment of we allocated it
      if (owner[i]) {
        delete data[i];
      }
    }
  }


  ///
  /// Swap function for copy-and-swap idiom.
  ///
  void swap(StaticIndexSet<T0, TREST...> &other)
  {

    // Swap parents data
    PARENT::swap((PARENT &)other);

    // Swap our data
    using std::swap;
    swap(data, other.data);
    swap(owner, other.owner);
  }

  //@}

  //@{
  //!  @name Segment insertion and accessor methods

  ///
  /// Equality operator for given segment index.
  ///
  /// This is used to implement the == and != operators
  ///
  template <typename P0, typename... PREST>
  RAJA_INLINE bool compareSegmentById(
      size_t segid,
      const StaticIndexSet<P0, PREST...> &other) const
  {
    // drill down our types until we have the right type
    if (getSegmentTypes()[segid] == T0_TypeId) {

      // Check that other's segid is of type T0
      if (!other.template checkSegmentType<T0>(segid)) {
        return false;
      }

      // Compare to others segid
      Index_type offset = getSegmentOffsets()[segid];
      return *data[offset] == other.template getSegment<T0>(segid);
    } else {
      // peel off T0
      return PARENT::compareSegmentById(segid, other);
    }
  }


  template <typename P0>
  RAJA_INLINE bool checkSegmentType(size_t segid) const
  {
    if (getSegmentTypes()[segid] == T0_TypeId) {
      return std::is_same<T0, P0>::value;
    }
    return PARENT::template checkSegmentType<P0>(segid);
  }

  ///
  /// Return segment
  ///
  /// Notes: No error-checking on segment index.
  ///
  template <typename P0>
  RAJA_INLINE P0 &getSegment(size_t segid)
  {
    if (getSegmentTypes()[segid] == T0_TypeId) {
      Index_type offset = getSegmentOffsets()[segid];
      return *(P0 *)data[offset];
    }
    return PARENT::template getSegment<P0>(segid);
  }

  template <typename P0>
  RAJA_INLINE P0 const &getSegment(size_t segid) const
  {
    if (getSegmentTypes()[segid] == T0_TypeId) {
      Index_type offset = getSegmentOffsets()[segid];
      return *(P0 const *)data[offset];
    }
    return PARENT::template getSegment<P0>(segid);
  }

  ///
  /// Returns the number of types this IndexSet can store.
  ///
  RAJA_INLINE
  constexpr size_t getNumTypes(void) const { return 1 + PARENT::getNumTypes(); }

  /*
   * IMPORTANT: Some methods to add a segment to an index set
   *            make a copy of the segment object passed in. Others do not.
   *
   *            The no-copy method names indicate the choice.
   *            The copy/no-copy methods are further distinguished
   *            by taking a const reference (copy) or non-const
   *            pointer (no-copy).
   *
   *            Each method returns true if segment is added successfully;
   *            false otherwise.
   */

  ///
  /// Append an IndexSet of another type to this one.
  ///
  /// This requires the set of types supported by this IndexSet to be a
  /// superset of the one copied in.
  ///
  /// Order is preserved, and all segments are appended to the end of this one
  ///

private:
  template <typename... CALL>
  RAJA_INLINE bool push_into(StaticIndexSet<CALL...> &c,
                             PushEnd pend = PUSH_BACK,
                             PushCopy pcopy = PUSH_COPY)
  {
    size_t num = getNumSegments();

    if (pend == PUSH_BACK) {
      for (size_t i = 0; i < num; ++i) {
        segment_push_into(i, c, pend, pcopy);
      }
    } else {
      // Reverse push_front iteration so we preserve segment ordering
      for (int i = num - 1; i >= 0; --i) {
        segment_push_into(i, c, pend, pcopy);
      }
    }
    return true;
  }

public:
  template <typename... CALL>
  RAJA_INLINE bool segment_push_into(size_t segid,
                                     StaticIndexSet<CALL...> &c,
                                     PushEnd pend = PUSH_BACK,
                                     PushCopy pcopy = PUSH_COPY)
  {
    if (getSegmentTypes()[segid] == T0_TypeId) {
      Index_type offset = getSegmentOffsets()[segid];

      if (pcopy == PUSH_COPY) {
        if (pend == PUSH_BACK) {
          c.push_back(*data[offset]);
        } else {
          c.push_front(*data[offset]);
        }
      } else {
        if (pend == PUSH_BACK) {
          c.push_back_nocopy(data[offset]);
        } else {
          c.push_front_nocopy(data[offset]);
        }
      }

    } else {
      PARENT::segment_push_into(segid, c, pend, pcopy);
    }
    return true;
  }


  ///
  /// Add segment to back end of index set without making a copy.
  ///
  template <typename Tnew>
  RAJA_INLINE bool push_back_nocopy(Tnew *val)
  {
    return push_internal(val, PUSH_BACK, PUSH_NOCOPY);
  }

  ///
  /// Add segment to front end of index set without making a copy.
  ///
  template <typename Tnew>
  RAJA_INLINE bool push_front_nocopy(Tnew *val)
  {
    return push_internal(val, PUSH_FRONT, PUSH_NOCOPY);
  }

  ///
  /// Add copy of segment to back end of index set.
  ///
  template <typename Tnew>
  RAJA_INLINE bool push_back(Tnew const &val)
  {
    return push_internal(new Tnew(val), PUSH_BACK, PUSH_COPY);
  }

  ///
  /// Add copy of segment to front end of index set.
  ///
  template <typename Tnew>
  RAJA_INLINE bool push_front(Tnew const &val)
  {
    return push_internal(new Tnew(val), PUSH_FRONT, PUSH_COPY);
  }

  ///
  /// Return total length of index set; i.e., sum of lengths
  /// of all segments.
  ///
  RAJA_INLINE
  size_t getLength(void) const
  {
    size_t total = PARENT::getLength();
    size_t num = data.size();
    for (size_t i = 0; i < num; ++i) {
      total += data[i]->size();
    }
    return total;
  }

  ///
  /// Return total number of segments in index set.
  ///
  RAJA_INLINE
  constexpr size_t getNumSegments(void) const
  {
    return data.size() + PARENT::getNumSegments();
  }


  ///
  /// Calls the operator "body" with the segment stored at segid.
  ///
  /// This requires that "body" be templated, as the segment will be passed
  /// in as a properly typed object.
  ///
  /// The "args..." are passed-thru to the body as arguments AFTER the segment.
  ///
  template <typename BODY, typename... ARGS>
  RAJA_INLINE void segmentCall(size_t segid, BODY body, ARGS... args) const
  {
    if (getSegmentTypes()[segid] == T0_TypeId) {
      Index_type offset = getSegmentOffsets()[segid];
      body(*data[offset], args...);
    } else {
      PARENT::segmentCall(segid, body, args...);
    }
  }

protected:
  ///
  /// Internal logic to add a new segment.
  ///
  template <typename Tnew>
  RAJA_INLINE bool push_internal(Tnew *val,
                                 PushEnd pend = PUSH_BACK,
                                 PushCopy pcopy = PUSH_COPY)
  {
    static_assert(sizeof...(TREST) > 0, "Invalid type for this IndexSet");
    PARENT::push_internal(val, pend, pcopy);
    return true;
  }

  RAJA_INLINE
  bool push_internal(T0 *val,
                     PushEnd pend = PUSH_BACK,
                     PushCopy pcopy = PUSH_COPY)
  {
    data.push_back(val);
    owner.push_back(pcopy == PUSH_COPY);

    // Determine if we push at the front or back of the segment list
    if (pend == PUSH_BACK) {
      // Store the segment type
      getSegmentTypes().push_back(T0_TypeId);

      // Store the segment offset in data[]
      getSegmentOffsets().push_back(data.size() - 1);

      // Store the segment icount
      size_t icount = val->size();
      getSegmentIcounts().push_back(getTotalLength());
      increaseTotalLength(icount);
    } else {
      // Store the segment type
      getSegmentTypes().push_front(T0_TypeId);

      // Store the segment offset in data[]
      getSegmentOffsets().push_front(data.size() - 1);

      // Store the segment icount
      getSegmentIcounts().push_front(0);
      size_t icount = val->size();
      for (size_t i = 1; i < getSegmentIcounts().size(); ++i) {
        getSegmentIcounts()[i] += icount;
      }
      increaseTotalLength(icount);
    }
    return true;
  }

  ///
  /// Returns the number of indices (the total icount of segments
  ///
  RAJA_INLINE
  Index_type &getTotalLength(void) { return PARENT::getTotalLength(); }

  RAJA_INLINE
  void setTotalLength(int n) { return PARENT::setTotalLength(n); }

  RAJA_INLINE
  void increaseTotalLength(int n) { return PARENT::increaseTotalLength(n); }

  //@{
  //!  @name IndexSet iterator methods
public:
  using iterator = Iterators::numeric_iterator<Index_type>;

  ///
  /// Get an iterator to the end.
  ///
  iterator end() const { return iterator(getNumSegments()); }

  ///
  /// Get an iterator to the beginning.
  ///
  iterator begin() const { return iterator(0); }

  ///
  /// Return the number of elements in the range.
  ///
  Index_type size() const { return getNumSegments(); }
  //@}

  //@{
  //!  @name IndexSet segment subsetting methods (slices ranges)

  ///
  /// Return a new IndexSet object that contains the subset of
  /// segments in this IndexSet with ids in the interval [begin, end).
  ///
  /// This IndexSet will not change and the created "slice" into it
  /// will not own any of its segments.
  ///
  StaticIndexSet<T0, TREST...> *createSlice(int begin, int end)
  {
    StaticIndexSet<T0, TREST...> *retVal = new StaticIndexSet<T0, TREST...>();

    int numSeg = getNumSegments();
    int minSeg = ((begin >= 0) ? begin : 0);
    int maxSeg = ((end < numSeg) ? end : numSeg);

    for (int i = minSeg; i < maxSeg; ++i) {
      segment_push_into(i, *retVal, PUSH_BACK, PUSH_NOCOPY);
    }
    return retVal;
  }

  ///
  /// Return a new IndexSet object that contains the subset of
  /// segments in this IndexSet with ids in the given int array.
  ///
  /// This IndexSet will not change and the created "slice" into it
  /// will not own any of its segments.
  ///
  StaticIndexSet<T0, TREST...> *createSlice(const int *segIds, int len)
  {
    StaticIndexSet<T0, TREST...> *retVal = new StaticIndexSet<T0, TREST...>();

    int numSeg = getNumSegments();
    for (int i = 0; i < len; ++i) {
      if (segIds[i] >= 0 && segIds[i] < numSeg) {
        segment_push_into(segIds[i], *retVal, PUSH_BACK, PUSH_NOCOPY);
      }
    }
    return retVal;
  }

  ///
  /// Return a new IndexSet object that contains the subset of
  /// segments in this IndexSet with ids in the argument object.
  ///
  /// This IndexSet will not change and the created "slice" into it
  /// will not own any of its segments.
  ///
  /// The object must provide methods begin(), end(), and its
  /// iterator type must de-reference to an integral value.
  ///
  template <typename T>
  StaticIndexSet<T0, TREST...> *createSlice(const T &segIds)
  {
    StaticIndexSet<T0, TREST...> *retVal = new StaticIndexSet<T0, TREST...>();

    int numSeg = getNumSegments();
    for (auto it = segIds.begin(); it != segIds.end(); ++it) {
      if (*it >= 0 && *it < numSeg) {
        segment_push_into(*it, *retVal, PUSH_BACK, PUSH_NOCOPY);
      }
    }
    return retVal;
  }
  //@}


  //@{
  /// Set [begin, end) interval of segment ids identified by
  /// given interval id.
  ///
  /// For example, this method can be used to assign an interval of
  /// segments to be processed by a given thread.
  ///
  void setSegmentInterval(size_t interval_id, int begin, int end);

  ///
  /// Get lower bound or upper bound of [begin, end) interval of segment
  /// ids identified by given interval id.
  ///
  /// Notes: No error-checking on interval id.
  ///
  int getSegmentIntervalBegin(size_t interval_id) const
  {
    return m_seg_interval_begin[interval_id];
  }
  ///
  int getSegmentIntervalEnd(size_t interval_id) const
  {
    return m_seg_interval_end[interval_id];
  }

  //@}


  //@{
  //!  @name Private data get methods
protected:
  ///
  /// Returns the mapping of  segment_index -> segment_type
  ///
  RAJA_INLINE
  RAJA::RAJAVec<Index_type> &getSegmentTypes(void)
  {
    return PARENT::getSegmentTypes();
  }

  RAJA_INLINE
  RAJA::RAJAVec<Index_type> const &getSegmentTypes(void) const
  {
    return PARENT::getSegmentTypes();
  }

  ///
  /// Returns the mapping of  segment_index -> segment_offset
  ///
  RAJA_INLINE
  RAJA::RAJAVec<Index_type> &getSegmentOffsets(void)
  {
    return PARENT::getSegmentOffsets();
  }

  RAJA_INLINE
  RAJA::RAJAVec<Index_type> const &getSegmentOffsets(void) const
  {
    return PARENT::getSegmentOffsets();
  }

  ///
  /// Returns the icount of segments
  ///
  RAJA_INLINE
  RAJA::RAJAVec<Index_type> &getSegmentIcounts(void)
  {
    return PARENT::getSegmentIcounts();
  }

  RAJA_INLINE
  RAJA::RAJAVec<Index_type> const &getSegmentIcounts(void) const
  {
    return PARENT::getSegmentIcounts();
  }
  //@}

  //@{
  //!  @name Index set equality/inequality check methods
public:
  ///
  /// Equality operator returns true if all segments are equal; else false.
  ///
  /// Note: method does not check equality of anything other than segment
  ///       types and indices; e.g., dependency info not checked.
  ///
  template <typename P0, typename... PREST>
  RAJA_INLINE bool operator==(const StaticIndexSet<P0, PREST...> &other) const
  {
    size_t num_seg = getNumSegments();
    if (num_seg == other.getNumSegments()) {

      for (size_t segid = 0; segid < num_seg; ++segid) {

        if (!compareSegmentById(segid, other)) {
          return false;
        }
      }
      return true;
    }
    return false;
  }

  ///
  /// Inequality operator returns true if any segment is not equal, else false.
  ///
  template <typename P0, typename... PREST>
  RAJA_INLINE bool operator!=(const StaticIndexSet<P0, PREST...> &other) const
  {
    return (!(*this == other));
  }

  //@}

  ///
  /// Print index set data, including segments, to given output stream.
  ///
  RAJA_INLINE
  void printSegment(size_t segid, std::ostream &os) const
  {
    if (getSegmentTypes()[segid] == T0_TypeId) {
      Index_type offset = getSegmentOffsets()[segid];
      data[offset]->print(os);
      if (owner[offset]) {
        os << "(1) ";
      } else {
        os << "(0) ";
      }
    } else {
      PARENT::printSegment(segid, os);
    }
  }

  void print(std::ostream &os) const
  {
    size_t n = getNumSegments();

    os << "\nBASIC INDEX SET : "
       << " length = " << getLength() << std::endl
       << "      num segments = " << n << std::endl;

    os << "Segment Types:   ";
    for (size_t i = 0; i < n; ++i) {
      os << " " << getSegmentTypes()[i];
    }
    os << std::endl;

    os << "Segment Offsets: ";
    for (size_t i = 0; i < n; ++i) {
      os << " " << getSegmentOffsets()[i];
    }
    os << std::endl;

    os << "Segment Icounts: ";
    for (size_t i = 0; i < n; ++i) {
      os << " " << getSegmentIcounts()[i];
    }
    os << std::endl;

    for (size_t i = 0; i < n; ++i) {
      printSegment(i, os);
    }  // end iterate over segments

    os << "END IndexSet::print()" << std::endl;

  }  // end print

private:
  ///
  /// Collection of IndexSet data objects of type T0 and whether the IndexSet
  /// owns them
  ///
  RAJA::RAJAVec<T0 *> data;
  RAJA::RAJAVec<Index_type> owner;

  ///
  /// Vectors holding user defined segment intervals; each is [begin, end).
  ///
  RAJAVec<Index_type> m_seg_interval_begin;
  RAJAVec<Index_type> m_seg_interval_end;
};


template <>
class StaticIndexSet<>
{
public:
  /*!
   * \brief Default ctor produces empty IndexSet.
   */
  RAJA_INLINE
  StaticIndexSet() : m_len(0) {}


  /*!
   * \brief Dtor cleans up segements that we own.
   */
  RAJA_INLINE
  ~StaticIndexSet() {}


  /*!
   * \brief Copy-constructor.
   */
  RAJA_INLINE
  StaticIndexSet(StaticIndexSet const &c)
  {
    segment_types = c.segment_types;
    segment_offsets = c.segment_offsets;
    segment_icounts = c.segment_icounts;
    m_len = c.m_len;
  }

  /*!
   * \brief Copy-assignment operator.
   */
  /*
    IndexSet<>& operator=(const IndexSet<>& rhs){

    }*/

  ///
  /// Swap function for copy-and-swap idiom (deep copy).
  ///
  void swap(StaticIndexSet &other)
  {
    using std::swap;
    swap(segment_types, other.segment_types);
    swap(segment_offsets, other.segment_offsets);
    swap(segment_icounts, other.segment_icounts);
    swap(m_len, other.m_len);
  }

protected:
  RAJA_INLINE
  // constexpr
  static size_t getNumTypes(void) /*const*/ { return 0; }

  template <typename T>
  RAJA_INLINE constexpr bool isValidSegmentType(T const &) const
  {
    // Segment type wasn't found
    return false;
  }

  RAJA_INLINE  // constexpr
      static int
      getNumSegments(void) /*const*/
  {
    return 0;
  }

  RAJA_INLINE  // constexpr
      static size_t
      getLength(void) /*const*/
  {
    return 0;
  }

  RAJA_INLINE
  void printSegment(size_t, std::ostream &os) const
  {
    os << "UNKNOWN" << std::endl;
  }

  template <typename BODY, typename... ARGS>
  RAJA_INLINE void segmentCall(size_t, BODY, ARGS...) const
  {
  }

  RAJA_INLINE
  RAJA::RAJAVec<Index_type> &getSegmentTypes(void) { return segment_types; }

  RAJA_INLINE
  RAJA::RAJAVec<Index_type> const &getSegmentTypes(void) const
  {
    return segment_types;
  }

  RAJA_INLINE
  RAJA::RAJAVec<Index_type> &getSegmentOffsets(void) { return segment_offsets; }

  RAJA_INLINE
  RAJA::RAJAVec<Index_type> const &getSegmentOffsets(void) const
  {
    return segment_offsets;
  }

  RAJA_INLINE
  RAJA::RAJAVec<Index_type> &getSegmentIcounts(void) { return segment_icounts; }

  RAJA_INLINE
  RAJA::RAJAVec<Index_type> const &getSegmentIcounts(void) const
  {
    return segment_icounts;
  }

  RAJA_INLINE
  Index_type &getTotalLength(void) { return m_len; }

  RAJA_INLINE
  void setTotalLength(int n) { m_len = n; }

  RAJA_INLINE
  void increaseTotalLength(int n) { m_len += n; }

  template <typename P0, typename... PREST>
  RAJA_INLINE bool compareSegmentById(
      size_t,
      const StaticIndexSet<P0, PREST...> &) const
  {
    return false;
  }

  template <typename P0>
  RAJA_INLINE bool checkSegmentType(size_t) const
  {
    return false;
  }

  template <typename P0>
  RAJA_INLINE P0 &getSegment(size_t)
  {
    // cause a segfault
    P0 *x = 0;
    return *x;
  }

  template <typename P0>
  RAJA_INLINE P0 const &getSegment(size_t) const
  {
    // cause a segfault
    P0 const *x = 0;
    return *x;
  }


  template <typename... CALL>
  RAJA_INLINE void push_into(StaticIndexSet<CALL...> &, PushEnd, PushCopy) const
  {
  }

  template <typename... CALL>
  RAJA_INLINE void segment_push_into(size_t,
                                     StaticIndexSet<CALL...> &,
                                     PushEnd,
                                     PushCopy) const
  {
  }


  template <typename Tnew>
  RAJA_INLINE void push(Tnew const &, PushEnd, PushCopy)
  {
  }

public:
  using iterator = Iterators::numeric_iterator<Index_type>;

  RAJA_INLINE
  int getStartingIcount(int segid) { return segment_icounts[segid]; }

  RAJA_INLINE
  int getStartingIcount(int segid) const { return segment_icounts[segid]; }

  ///
  /// Get an iterator to the end.
  ///
  iterator end() const { return iterator(getNumSegments()); }

  ///
  /// Get an iterator to the beginning.
  ///
  iterator begin() const { return iterator(0); }

  ///
  /// Return the number of elements in the range.
  ///
  Index_type size() const { return getNumSegments(); }

private:
  // Vector of segment types:    seg_index -> seg_type
  RAJA::RAJAVec<Index_type> segment_types;

  // offsets into each segment vector:    seg_index -> seg_offset
  // used as segment_data[seg_type][seg_offset]
  RAJA::RAJAVec<Index_type> segment_offsets;

  // the icount of each segment
  RAJA::RAJAVec<Index_type> segment_icounts;

  ///
  /// Total length of all IndexSet segments.
  ///
  Index_type m_len;
};

using IndexSet = StaticIndexSet<RAJA::RangeSegment,
                                RAJA::ListSegment,
                                RAJA::RangeStrideSegment>;

namespace type_traits
{

template <typename T>
struct is_index_set
    : SpecializationOf<RAJA::StaticIndexSet, typename std::decay<T>::type> {
};

template <typename T>
struct is_indexset_policy
    : SpecializationOf<RAJA::ExecPolicy, typename std::decay<T>::type> {
};
}

}  // closing brace for RAJA namespace

#endif  // closing endif for header file include guard
