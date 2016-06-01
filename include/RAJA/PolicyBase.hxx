#ifndef RAJA_POLICYBASE_HXX
#define RAJA_POLICYBASE_HXX

#include "RAJA/int_datatypes.hxx"

namespace RAJA {

struct PolicyBase {
    template<typename IndexT = Index_type,
             typename Func>
    void range(IndexT begin, IndexT end, Func &&f) const {
        // printf("base...\n");
        for ( auto ii = begin ; ii < end ; ++ii ) {
            loop_body( ii );
        }
    }

    template<typename Iterator,
             typename Func>
    void iterator(Iterator &&begin, Iterator &&end, Func &&loop_body) const {
        printf("base2...\n");
        for ( auto &ii = begin ; ii < end ; ++ii ) {
            loop_body( *ii );
        }
    }
};

}

#endif /* RAJA_POLICYBASE_HXX */
