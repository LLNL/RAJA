#ifndef NEW_REDUCE_SIMD_REDUCE_HPP
#define NEW_REDUCE_SIMD_REDUCE_HPP

#include "RAJA/pattern/params/reducer.hpp"

namespace RAJA
{
namespace expt
{
namespace detail
{

// Init
template<typename EXEC_POL, typename OP, typename T, typename VOp>
camp::concepts::enable_if<std::is_same<EXEC_POL, RAJA::simd_exec>> param_init(
    EXEC_POL const&,
    Reducer<OP, T, VOp>& red)
{
  red.m_valop.val = OP::identity();
}

// Combine
template<typename EXEC_POL, typename OP, typename T, typename VOp>
camp::concepts::enable_if<std::is_same<EXEC_POL, RAJA::simd_exec>> param_combine(
    EXEC_POL const&,
    Reducer<OP, T, VOp>& out,
    const Reducer<OP, T, VOp>& in)
{
  out.m_valop.val = OP {}(out.m_valop.val, in.m_valop.val);
}

// Resolve
template<typename EXEC_POL, typename OP, typename T, typename VOp>
camp::concepts::enable_if<std::is_same<EXEC_POL, RAJA::simd_exec>> param_resolve(
    EXEC_POL const&,
    Reducer<OP, T, VOp>& red)
{
  red.combineTarget(red.m_valop.val);
}

}  //  namespace detail
}  //  namespace expt
}  //  namespace RAJA

#endif  //  NEW_REDUCE_SIMD_REDUCE_HPP
