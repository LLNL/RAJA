RAJA v0.6.0 Release Notes
=========================

This release contains two major changes, a variety of bug fixes and feature
enhancements, and expanded user documentation and accompanying example codes.

Major changes include:

  * RAJA::forallN methods are marked deprecated. They will be removed in 
    the 0.7.0 release.
  * RAJA::forall methods that take loop bounds rather than segments  (e.g.,
    RAJA::forall(beg, end, ...) are marked deprecated. They will be removed 
    in the 0.7.0 release.
  * RAJA::nested has been replaced with RAJA::kernel. The RAJA::kernel interface
    is much more flexible and full featured. Going forward, it will be the 
    supported interface for nested loops and more complex kernels in RAJA. 
  * This release contains new documentation and example codes for the 
    RAJA::kernel interface. The documentation described key features and
    summarizes available 'statement' types. However, it remains a 
    work-in-progress and expanded documentation with more examples will be 
    available in future releases.
  * Documentation of other RAJA features have been expanded and improved in
    this release along with additional example codes.

Other notable changes include:

  * New or improved features: 
      * RAJA CUDA reductions now work with host/device lambdas 
      * List segments now work with RAJA::kernel loops.
      * New and expanded collection of build files for LC and ALCF machines.
        Hopefully, these will be helpful to folks getting started.

  * Performance improvements: 
      * Some RAJA::View use cases
      * Unnecessary operations removed in min/max atomics
    
  * Bug fixes: 
      * Issues in View with OffsetLayout fixed.
      * Construction of a const View from a non-const View now works
      * CUDA kernel no longer launched in RAJA::kernel loops when iteration 
        space has size zero


RAJA v0.5.3 Release Notes
=========================

This is a bugfix release that fixes bugs in the IndexSetBuilder methods. These
methods now work correctly with the strongly-typed IndexSet.


RAJA v0.5.2 Release Notes
=========================

This release fixes some small bugs, including compiler warnings issued for
deprecated features, type narrowing, and the slice method for the
RangeStrideSegment class.

It also adds a new CMake variable, RAJA_LOADED, that is used to determine
whether RAJA's CMakeLists file has already been processed. This is useful when
including RAJA as part of another CMake project.


RAJA v0.5.1 Release Notes
=========================

This release contains fixes for compiler warnings with newer GCC and Clang
compilers, and allows strongly-typed indices to work with RangeStrideSegment.

Additionally, the index type for all segments in an IndexSet needs to be the
same. This requirement is enforced with a static_assert.


RAJA v0.5.0 Release Notes
=========================

This release contains a variety of bug fixes, removes nvcc compiler
warnings, addition of unit tests to expand coverage, and a variety of 
other code cleanup and improvements. The most notable changes in this 
version include:

  * New RAJA User Guide and Tutorial along with a set of example codes
    that illustrate basic usage of RAJA features and which accompany
    the tutorial. The examples are in the ``RAJA/examples`` directory.
    The user guide is available online here:
    [RAJA User Guide and Tutorial](http://raja.readthedocs.io/en/master/).

  * ``RAJA::IndexSet`` is now deprecated. You may still use it until it is
    removed in a future release -- you will see a notification message at
    compile time that it is deprecated.

    Index set functionality will now be available via ``RAJA::TypedIndexSet``
    where you specify all segment types as template parameters when you 
    declare an instance of it. This change allows us to: remove all virtual
    methods from the index set, be able to use index set objects to CUDA
    GPU kernels and all of their functionality, and support any arbitrary
    segment type even user-defined. Please see User Guide for details.

    Segment dependencies are being developed for the typed index set and 
    will be available in a future release.

  * ``RAJA::nested::forall`` changes:

    * Addition of CUDA and OpenMP collapse policies for nested loops.
      OpenMP collapse will do what the OpenMP collapse clause does. 
      CUDA collapse will collapse a loop nest into a single CUDA kernel based 
      on how nested policies specify how the loop levels should be distributed 
      over blocks and threads.

    * Added new policy ``RAJA::cuda_loop_exec`` to enable inner loops to run
      sequentially inside a CUDA kernel with ``RAJA::nested::forall``.

    * Fixed ``RAJA::nested::forall`` so it now works with RAJA's CUDA Reducer 
      types.

    * Removed ``TypedFor`` policies. For type safety of nested loop iteration
      variables, it makes more sense to use ``TypedRangeSegment`` since the
      variables are associated with the loop kernel and not the execution 
      policy, which may be applied to multiple loops with different variables.

  * Fixed OpenMP scans to calculate chunks of work based on actual number of
    threads the OpenMP runtime makes available.

  * Enhancements and fixes to RAJA/CHAI interoperability.

  * Added aliases for several ``camp`` types in the RAJA namespace; e.g.,
    ``camp::make_tuple`` can now be accessed as ``RAJA::make_tuple``. This 
    change makes the RAJA API more consistent and clear.
