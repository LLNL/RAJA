/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   RAJA header file defining segment base class.
 *
 ******************************************************************************
 */

#ifndef RAJA_BaseSegment_HPP
#define RAJA_BaseSegment_HPP

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
#include "RAJA/util/types.hpp"

namespace RAJA
{

/*!
 ******************************************************************************
 *
 * \brief  Base class for all segment classes.
 *
 ******************************************************************************
 */
class BaseSegment
{
public:
  ///
  /// Ctor for base segment type.
  ///
  explicit BaseSegment(SegmentType type) : m_type(type), m_private(0) { ; }

  /*
   * Using compiler-generated copy ctor, copy assignment.
   */

  ///
  /// Virtual dtor.
  ///
  virtual ~BaseSegment() { ; }

  ///
  /// Get index count associated with start of segment.
  ///
  SegmentType getType() const { return m_type; }

  ///
  /// Retrieve pointer to private data. Must be cast to proper type by user.
  ///
  void* getPrivate() const { return m_private; }

  ///
  /// Set pointer to private data. Can be used to associate any data
  /// to segment.
  ///
  /// NOTE: Caller retains ownership of data object.
  ///
  void setPrivate(void* ptr) { m_private = ptr; }

  //
  // Pure virtual methods that must be provided by concrete segment classes.
  //

  ///
  /// Get segment length (i.e., number of indices in segment).
  ///
  virtual Index_type getLength() const = 0;

  ///
  /// Return enum value indicating whether segment owns the data rapresenting
  /// its indices.
  ///
  virtual IndexOwnership getIndexOwnership() const = 0;

  ///
  /// Pure virtual equality operator returns true if segments are equal;
  /// else false.
  ///
  virtual bool operator==(const BaseSegment& other) const = 0;

  ///
  /// Pure virtual inequality operator returns true if segments are not
  /// equal, else false.
  ///
  virtual bool operator!=(const BaseSegment& other) const = 0;

private:
  ///
  /// The default ctor is not implemented.
  ///
  BaseSegment();

  ///
  /// Enum value indicating segment type.
  ///
  SegmentType m_type;

  ///
  /// Pointer that can be used to hold arbitrary data associated with segment.
  ///
  void* m_private;
};

}  // closing brace for RAJA namespace

#endif  // closing endif for header file include guard
