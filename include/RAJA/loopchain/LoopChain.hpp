

#ifndef RAJA_LoopChain_HPP
#define RAJA_LoopChain_HPP

#include "RAJA/config.hpp"

#include "RAJA/loopchain/Utils.hpp"
#include "RAJA/loopchain/SymExec.hpp"
#include "RAJA/loopchain/KernelWrapper.hpp"
#include "RAJA/loopchain/Chain.hpp"



#include "RAJA/loopchain/Transformations.hpp"
//#include "RAJA/loopchain/ISLAnalysis.hpp"
#include "RAJA/loopchain/KernelConversion.hpp"

#include "RAJA/loopchain/transformations/Shift.hpp"
#include "RAJA/loopchain/transformations/Fuse.hpp"
#include "RAJA/loopchain/transformations/ShiftAndFuse.hpp"
#include "RAJA/loopchain/transformations/OverlappedTile.hpp"

//#include "RAJA/pattern/kernel/TiledLambda.hpp"
#include "RAJA/pattern/kernel/OverlappedTile.hpp"

namespace RAJA
{

template <typename...Knls>
auto overlapped_tile_fuse(Knls...knls);

template <typename...Knls>
auto chain(Knls...knls);

} //namespace RAJA

#endif
