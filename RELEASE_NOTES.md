
[comment]: # (#################################################################)
[comment]: # (Copyright 2016-19, Lawrence Livermore National Security, LLC.)

[comment]: # (Produced at the Lawrence Livermore National Laboratory)

[comment]: # (LLNL-CODE-689114)

[comment]: # (All rights reserved.)

[comment]: # (This file is part of RAJA.)

[comment]: # (For details about use and distribution, please read RAJA/LICENSE.)
[comment]: # (#################################################################)

RAJA v0.7.0 Release Notes
=========================

This release contains several major changes, new features, a variety of bug 
fixes, and expanded user documentation and accompanying example codes. For
more information and details about any of the changes listed below, please 
consult the RAJA documentation for the 0.7.0 release which is linked to 
our Github project.

Major changes include:

  * RAJA::forallN methods were marked deprecated in the 0.6.0 release. They 
    have been removed. All applications that contain nested loops and 
    have been using forallN methods should convert them to use the RAJA::kernel
    interface.
  * RAJA::forall methods that take explicit loop bounds rather than segments  
    (e.g., RAJA::forall(beg, end, ...) were marked deprecated in the 0.6.0
    release. They have been removed. Hopefully, this will result in faster
    compile times due to simpler template resolution. Users who have been 
    passing loop bounds directly to forall methods should convert those
    cases to use RAJA segments instead.
  * CUDA execution policies for use in RAJA::kernel policies have been 
    significantly reworked and redefined. The new set of policies are
    much more flexible and provide improved run time performance.
  * New, improved support for loop tiling algorithms and support for 
    CPU cache blocking, CUDA GPU thread local data and shared memory is 
    available. This includes RAJA::kernel policy statement types to make tile
    numbers and local tile indices available in user kernels (TileTCount and
    ForICount statement types), and a new RAJA::LocalArray type with various 
    CPU and GPU memory policies. Due to these new features, RAJA 'shmem window'
    statements have been removed.
  * This release contains expanded documentation and example codes for the
    RAJA::kernel interface, including loop tiling algorithms and support for
    CPU cache blocking, CUDA GPU thread local data and shared memory.

Other notable changes include:

  * Features:
    * Initial support for OpenMP target execution policies with RAJA::kernel
      added.
    * The RAJA::AtomicRef interface is now consistent with the 
      C++20 std::atomic_ref interface.
    * Atomic compare-exchange operations added.
    * CUDA reduce policies no longer require a thread-block size parameter.
    * New features considered preliminary with no significant documentation or
      examples available yet:
        * RAJA::statement::Reduce type for use in RAJA::kernel execution 
          policies. This enables the ability to perform reductions and access
          reduced values inside user kernels.
        * Warp-level execution policies added for CUDA.

  * Performance improvements:
    * Better use of inline directives to improve likelihood of SIMD 
      instruction generation with the Intel compiler.

  * Bug fixes:
    * Several CHAI integration issues resolved.
    * Resolve issue with alignx directive when using XL compiler as host
      compiler with CUDA.
    * Fix issue associated with how XL compiler interprets OpenMP region
      definition.
    * Various tweaks to camp implementation to improve robustness.

  * Build changes/improvements:
    * The minimum required version of CMake has changed to 3.8 for all
      programming model back-ends, except CUDA. The minimum CMake version
      for CUDA support is 3.9.
    * Improved support for clang-cuda compiler. Some features still do not
      work with that compiler.
    * Update NVIDIA cub module to version 1.8.0.
    * Enable use of 'BLT_SOURCE_DIR' CMake variable to help prevent conflicts
      with BLT versions in RAJA and other libraries used in applications.

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
