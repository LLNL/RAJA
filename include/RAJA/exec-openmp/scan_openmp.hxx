/*!
******************************************************************************
*
* \file
*
* \brief   Header file providing RAJA scan declarations.
*
******************************************************************************
*/

#ifndef RAJA_scan_openmp_HXX
#define RAJA_scan_openmp_HXX

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

#include "RAJA/scan.hxx"

#include <tuple>
#include <utility>

namespace RAJA
{
namespace internal
{
namespace scan
{
namespace inplace
{

struct upsweep <omp_parallel_for_exec> {

	template <typename Iter,
						typename BinFn,
						typename Value>
	static RAJA_INLINE
	void inclusive (Iter begin, Iter end, size_t n, BinFn f, const Value v) {
		// TODO
		Value agg { v };
		for (Iter i { begin }; i != end; ++i) {
			*i = agg = f (*i, agg);
		}
	}

	template <typename Iter,
						typename BinFn,
						typename Value>
	static RAJA_INLINE
	void exclusive (Iter begin, Iter end, size_t n, BinFn f, const Value v) {
		// TODO
		Value agg { v };
		for (Iter i { begin }; i != end - 1; ++i) {
			std::tie (*i, agg) = std::make_tuple (agg, f (*i, agg));
		}
	}

};

struct downsweep <omp_parallel_for_exec> {

	template <typename Iter,
						typename BinFn,
						typename Value>
	static RAJA_INLINE
	void inclusive (Iter begin, Iter end, size_t n, BinFn f, const Value v) {
		// TODO
	}

	template <typename Iter,
						typename BinFn,
						typename Value>
	static RAJA_INLINE
	void exclusive (Iter begin, Iter end, size_t n, BinFn f, const Value v) {
		// TODO
	}

};

} // namespace inplace

struct upsweep <omp_parallel_for_exec> {

	template <typename Iter,
						typename OutIter,
						typename BinFn,
						typename Value>
	static RAJA_INLINE
	void inclusive (const Iter begin,
									const Iter end,
									OutIter out,
									size_t n,
									BinFn f,
									const Value v) {
		// TODO
		Value agg { v };
		OutIter o { out };
		for (Iter i { begin }; i != end; ++i) {
			*o++ = agg = f (*i, agg);
		}
	}

	template <typename Iter,
						typename OutIter,
						typename BinFn,
						typename Value>
	static RAJA_INLINE
	void exclusive (const Iter begin,
									const Iter end,
									OutIter out,
									size_t n,
									BinFn f,
									const Value v) {
		// TODO
		Value agg { v };
		OutIter o { out };
		*o++ = v;
		for (Iter i { begin }; i != end - 1; ++i, ++o) {
			*o = agg = f (*i, agg);
		}
	}

};

struct downsweep <omp_parallel_for_exec, Iter, OutIter> {

	template <typename Iter,
						typename OutIter,
						typename BinFn,
						typename Value>
	static RAJA_INLINE
	void inclusive (const Iter begin,
									const Iter end,
									OutIter out,
									size_t n,
									BinFn f,
									const Value v) {
		// do nothing
	}

	template <typename Iter,
						typename OutIter,
						typename BinFn,
						typename Value>
	static RAJA_INLINE
	void exclusive (const Iter begin,
									const Iter end,
									OutIter out,
									size_t n,
									BinFn f,
									const Value v) {
		// do nothing
	}

};

} // namespace scan
} // namespace internal
} // namespace RAJA

#endif
