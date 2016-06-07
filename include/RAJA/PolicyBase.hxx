#ifndef RAJA_POLICYBASE_HXX
#define RAJA_POLICYBASE_HXX

#include "RAJA/int_datatypes.hxx"
#include "RAJA/RangeSegment.hxx"

namespace RAJA {

struct PolicyBase {
    // NOTE: this overrides the non-icount variants, to override icount-types
    // it must take an IcountIterableWrapper<RangeSegment>
    template<typename Func>
    inline void operator()(const RangeSegment& iter, Func &&loop_body) const {
        auto end = iter.getEnd();
        for ( auto ii = iter.getBegin() ; ii < end ; ++ii ) {
            loop_body( ii );
        }
    }

    template<typename Iterable,
             typename Func>
    inline void operator()(Iterable &&iter, Func &&loop_body) const {
        auto end = std::end(iter);
        for ( auto ii = std::begin(iter) ; ii < end ; ++ii ) {
            loop_body( *ii );
        }
    }
};

}

#endif /* RAJA_POLICYBASE_HXX */
