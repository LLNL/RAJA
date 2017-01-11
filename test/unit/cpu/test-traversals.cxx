#include "gtest/gtest.h"

#include "RAJA/RAJA.hxx"
#include "RAJA/MemUtils_CPU.hxx"

#include "buildIndexSet.hxx"

template <typename T>
class TraversalTest : public ::testing::Test
{
protected:
  virtual void SetUp()
  {
    array_length =
        buildIndexSet(index_sets_, static_cast<IndexSetBuildMethod>(0));
    array_length += 1;

    RAJA::getIndices(is_indices, index_sets_[0]);

    test_array = (RAJA::Real_ptr) RAJA::allocate_aligned(RAJA::DATA_ALIGN,
                   array_length * sizeof(RAJA::Real_type));
    ref_array = (RAJA::Real_ptr) RAJA::allocate_aligned(RAJA::DATA_ALIGN,
                   array_length * sizeof(RAJA::Real_type));
    in_array = (RAJA::Real_ptr) RAJA::allocate_aligned(RAJA::DATA_ALIGN,
                   array_length * sizeof(RAJA::Real_type));

    for (RAJA::Index_type i = 0; i < array_length; ++i) {
      in_array[i] = RAJA::Real_type(rand() % 65536);
    }

    for (RAJA::Index_type i = 0; i < array_length; ++i) {
      test_array[i] = 0.0;
      ref_array[i] = 0.0;
    }

    for (size_t i = 0; i < is_indices.size(); ++i) {
      ref_array[is_indices[i]] =
          in_array[is_indices[i]] * in_array[is_indices[i]];
    }
  }

  virtual void TearDown()
  {
    RAJA::free_aligned(test_array);
    RAJA::free_aligned(ref_array);
    RAJA::free_aligned(in_array);
  }

  RAJA::IndexSet index_sets_[1];
  RAJA::Real_ptr test_array;
  RAJA::Real_ptr ref_array;
  RAJA::Real_ptr in_array;
  RAJA::Index_type array_length;
  std::vector<RAJA::Index_type> is_indices;
};

TYPED_TEST_CASE_P(TraversalTest);

TYPED_TEST_P(TraversalTest, BasicForall)
{
  RAJA::forall<TypeParam>(this->index_sets_[0], [=](RAJA::Index_type idx) {
    this->test_array[idx] = this->in_array[idx] * this->in_array[idx];
  });

  for (int i = 0; i < this->array_length; ++i) {
    EXPECT_EQ(this->ref_array[i], this->test_array[i])
        << "Arrays differ at index " << i;
  }
}

REGISTER_TYPED_TEST_CASE_P(TraversalTest, BasicForall);


typedef ::testing::
    Types<RAJA::IndexSet::ExecPolicy<RAJA::seq_segit, RAJA::seq_exec>,
          RAJA::IndexSet::ExecPolicy<RAJA::seq_segit, RAJA::simd_exec> >
        SequentialTypes;

INSTANTIATE_TYPED_TEST_CASE_P(Sequential, TraversalTest, SequentialTypes);


#if defined(RAJA_ENABLE_OPENMP)
typedef ::testing::
    Types<RAJA::IndexSet::ExecPolicy<RAJA::seq_segit,
                                     RAJA::omp_parallel_for_exec>,
          RAJA::IndexSet::ExecPolicy<RAJA::omp_parallel_for_segit,
                                     RAJA::seq_exec>,
          RAJA::IndexSet::ExecPolicy<RAJA::omp_parallel_for_segit,
                                     RAJA::simd_exec> >
        OpenMPTypes;

INSTANTIATE_TYPED_TEST_CASE_P(OpenMP, TraversalTest, OpenMPTypes);
#endif

#if defined(RAJA_ENABLE_CILK)
typedef ::testing::
    Types<RAJA::IndexSet::ExecPolicy<RAJA::seq_segit, RAJA::cilk_for_exec>,
          RAJA::IndexSet::ExecPolicy<RAJA::cilk_for_segit, RAJA::seq_exec>,
          RAJA::IndexSet::ExecPolicy<RAJA::cilk_for_segit, RAJA::simd_exec> >
        CilkTypes;

INSTANTIATE_TYPED_TEST_CASE_P(Cilk, TraversalTest, CilkTypes);
#endif
