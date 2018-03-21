/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   RAJA header file containing tiling policies and mechanics
 *          for forallN templates.
 *
 ******************************************************************************
 */

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-18, Lawrence Livermore National Security, LLC.
//
// Produced at the Lawrence Livermore National Laboratory
//
// LLNL-CODE-689114
//
// All rights reserved.
//
// This file is part of RAJA.
//
// For details about use and distribution, please read RAJA/LICENSE.
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef RAJA_forallN_tile_HPP
#define RAJA_forallN_tile_HPP

#include <type_traits>

#include "RAJA/config.hpp"

#include "RAJA/util/types.hpp"

namespace RAJA
{


// Policy for no tiling
struct tile_none {
};

// Policy to tile by given block size
template <int TileSize>
struct tile_fixed {
};

// Struct used to create a list of tiling policies
template <typename... PLIST>
struct TileList {
  static constexpr const size_t num_loops = sizeof...(PLIST);
};

// Tiling Policy
struct ForallN_Tile_Tag {
};
template <typename TILE_LIST, typename NEXT = Execute>
struct Tile {
  // Identify this policy
  typedef ForallN_Tile_Tag PolicyTag;

  // A TileList<> that contains a list of Tile policies (tile_none, etc.)
  typedef TILE_LIST TilePolicy;

  // The next nested-loop execution policy
  typedef NEXT NextPolicy;
};

/******************************************************************
 *  Tiling mechanics
 ******************************************************************/

// Forward declaration so the apply_tile's can recurse into peel_tile
template <typename BODY, int TIDX, typename... POLICY_REST, typename TilePolicy>
RAJA_INLINE void forallN_peel_tile(TilePolicy,
                                   BODY body,
                                   POLICY_REST const &... prest);

/*!
 * \brief Applys the tile_none policy
 */
template <typename BODY,
          typename TilePolicy,
          int TIDX,
          typename POLICY_INIT,
          typename... POLICY_REST>
RAJA_INLINE void forallN_apply_tile(tile_none,
                                    BODY body,
                                    POLICY_INIT const &pi,
                                    POLICY_REST const &... prest)
{
  // printf("TIDX=%d: policy=tile_none\n", (int)TIDX);

  // Pass thru, so just bind the index set
  typedef ForallN_BindFirstArg_HostDevice<BODY, POLICY_INIT> BOUND;
  BOUND new_body(body, pi);

  // Recurse to the next policy
  forallN_peel_tile<BOUND, TIDX + 1>(TilePolicy{}, new_body, prest...);
}

template <typename T, typename U>
constexpr auto const_min(T t, U u) ->
    typename std::remove_reference<decltype((t < u) ? t : u)>::type
{
  return (t < u) ? t : u;
}

/*!
 * \brief Applys the tile_fixed<N> policy
 */
template <typename BODY,
          typename TilePolicy,
          int TIDX,
          int TileSize,
          typename POLICY_INIT,
          typename... POLICY_REST>
RAJA_INLINE void forallN_apply_tile(tile_fixed<TileSize>,
                                    BODY body,
                                    POLICY_INIT const &pi,
                                    POLICY_REST const &... prest)
{
  // printf("TIDX=%d: policy=tile_fixed<%d>\n", TIDX, TileSize);

  using BOUND = ForallN_BindFirstArg_HostDevice<BODY, POLICY_INIT>;

  // tile loop
  auto end = *pi.end();
  for (auto i0 = *pi.begin(); i0 < end; i0 += TileSize) {
    auto i1 = const_min(i0 + TileSize, end);
    POLICY_INIT pi_tile(make_range(i0, i1));
    // Pass thru, so just bind the index set
    BOUND new_body(body, pi_tile);
    // Recurse to the next policy
    forallN_peel_tile<BOUND, TIDX + 1>(TilePolicy{}, new_body, prest...);
  }
}

/*!
 * \brief Functor that wraps calling the next nested-loop execution policy.
 *
 * This is passed into the recursive tiling function forallN_peel_tile.
 */
template <typename NextPolicy, typename BODY, typename... ARGS>
struct ForallN_NextPolicyWrapper {
  BODY body;

  explicit ForallN_NextPolicyWrapper(BODY b) : body(b) {}

  RAJA_INLINE
  void operator()(ARGS const &... args) const
  {
    typedef typename NextPolicy::PolicyTag NextPolicyTag;
    forallN_policy<NextPolicy>(NextPolicyTag(), body, args...);
  }
};

/*!
 * \brief Tiling policy peeling function (termination case)
 *
 * This just executes the built-up tiling function passed in by outer
 * forallN_peel_tile calls.
 */
template <typename BODY, int TIDX, typename TilePolicy>
RAJA_INLINE void forallN_peel_tile(TilePolicy, BODY body)
{
  // Termination case just calls the tiling function that was constructed
  body();
}

/*!
 * \brief Tiling policy peeling function, recursively peels off tiling
 * policies and applys them.
 *
 * This peels off the policy, and hands it off to the policy-overloaded
 * forallN_apply_tile function... which in turn recursively calls this function
 */
template <typename BODY,
          int TIDX,
          typename POLICY_INIT,
          typename... POLICY_REST,
          typename... TilePolicies>
RAJA_INLINE void forallN_peel_tile(TileList<TilePolicies...>,
                                   BODY body,
                                   POLICY_INIT const &pi,
                                   POLICY_REST const &... prest)
{
  using TilePolicy = TileList<TilePolicies...>;

  // Extract the tiling policy for loop nest TIDX
  using TP = typename VarOps::get_type_at<TIDX, TilePolicies...>::type;

  // Apply this index's policy, then recurse to remaining tile policies
  forallN_apply_tile<BODY, TilePolicy, TIDX>(TP(), body, pi, prest...);
}

/******************************************************************
 *  forallN_policy(), tiling execution
 ******************************************************************/

/*!
 * \brief Tiling policy front-end function.
 */
template <typename POLICY, typename BODY, typename... PARGS>
RAJA_INLINE void forallN_policy(ForallN_Tile_Tag, BODY body, PARGS... pargs)
{
  typedef typename POLICY::NextPolicy NextPolicy;

  // Extract the list of tiling policies from the policy
  using TilePolicy = typename POLICY::TilePolicy;

  // Apply the tiling policies one-by-one with a peeling approach
  typedef ForallN_NextPolicyWrapper<NextPolicy, BODY, PARGS...> WRAPPER;
  WRAPPER wrapper(body);
  forallN_peel_tile<WRAPPER, 0>(TilePolicy{}, wrapper, pargs...);
}

}  // namespace RAJA

#endif
