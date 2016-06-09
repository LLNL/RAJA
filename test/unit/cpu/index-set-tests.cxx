#include "gtest/gtest.h"

class IndexSetTest :
  public testing::Test {
    protected:
      virtual void SetUp() {
         for (unsigned ibuild = 0; ibuild < NumBuildMethods; ++ibuild) {
            last_indx = max( last_indx, 
               buildIndexSet( index_sets, static_cast<IndexSetBuildMethod>(ibuild) ) );
      }
      
      IndexSet index_sets[NUM_BUILD_METHODS];


};

TEST(IndexSet, IndexSetConstructors) {
   IndexSet index[NumBuildMethods];
   for (unsigned ibuild = 0; ibuild < NumBuildMethods; ++ibuild) {
      last_indx = max( last_indx, 
         buildIndexSet( index, static_cast<IndexSetBuildMethod>(ibuild) ) );

}
