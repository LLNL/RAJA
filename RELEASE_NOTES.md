
[comment]: # (#################################################################)
[comment]: # (Copyright 2016-21, Lawrence Livermore National Security, LLC)
[comment]: # (and RAJA project contributors. See the RAJA/COPYRIGHT file)
[comment]: # (for details.)
[comment]: # 
[comment]: # (# SPDX-License-Identifier: BSD-3-Clause)
[comment]: # (#################################################################)

Version vxx.yy.zz -- Release date 20yy-mm-dd
============================================

Version v0.13.0 -- Release date 2020-10-30
============================================

This release contains new features, bug fixes, and build improvements. Please 
see the RAJA user guide for more information about items in this release.

Notable changes include:

  * New features:
      * Execution policies for the RAJA HIP back-end and examples have been
        added to the RAJA User Guide and Tutorial.
      * Strongly-typed indices now work with Multiview.

  * Build changes/improvements:
      * Update BLT to latest develop branch (SHA-1: cbe99c93d)
      * Added option to enable/disable runtime plugin loading. This is now 
        off by default. Previously, it was always enabled and there was no
        way to disable it.

  * Bug fixes/improvements:
      * Issues have been addressed so that the OpenMP target back-end is
        now working properly for all supported features. This has been
        verified with multiple clang compilers, including clang 10, and the
        XL compiler.
      * Various data structures have been made trivially copyable to 
        ensure they are mapped properly to device memory with OpenMP 
        target execution.
      * Numerous improvements and fixes (formatting, typos, etc.) in User Guide.

Version v0.12.1 -- Release date 2020-09-09
============================================

This release contains fixes for errors when using a CUDA build with a
non-CUDA compiler and compiler warnings, plus some other bug fixes related
to OpenMP target compilation.

Version v0.12.0 -- Release date 2020-09-03
============================================

This release contains new features, notable changes, and bug fixes. Please
see the RAJA user guide for more information about items in this release.

Notable changes include:

  * Notable repository change:
      * The 'master' branch in the RAJA git repo has been renamed to 'main'.

  * New features:
      * New RAJA "work group" capability added. This allows multiple GPU
        kernels to be fused into one kernel launch, greatly reducing the
        run time overhead of launching CUDA kernels.
      * Added support for dynamic plug-ins in RAJA, which enable the use of
        things like Kokkos Performance Profiline Tools to be used with RAJA
        (https://github.com/kokkos/kokkos-tools)
      * Added ability to pass a resource object to RAJA::forall methods to
        enable asynchronous execution for CUDA and HIP back-ends.
      * Added "Multi-view" that works like a regular view, except that it
        can wrap multiple arrays so their accesses can share index arithmetic.
      * Multiple sort algorithms added. This provides portable parallel sort 
        operations, which are basic parallel algorithm building blocks.
      * Introduced RAJA "Teams" concept as an experimental feature. This
        enables hierarchical parallelism and additional nested loop patterns
        beyond what RAJA::kernel supports. Please note that this is very much
        a work-in-progress and is not yet documented in the user guide.
      * Added initial support for dynamic loop tiling.
      * New OpenMP execution policies added to support static, dynamic, and 
        guided scheduling.
      * Added support for const iterators to be used with RAJA scans.
      * Support for bitwise and and or reductions have been added.
      * The RAJA::kernel interface has been expanded to allow only segment 
        index arguments used in a lambda to be passed to the lambda. In 
        previous versions of RAJA, every lambda invoked in a kernel had to 
        accept an index argument for every segment in the segment tuple passed 
        to RAJA::kernel execution templates, even if not all segment indices 
        were used in a lambda. This release still allows that usage pattern.
        The new capability requires an additional template parameter to be 
        passed to the RAJA::statement::Lambda type, which identify the segment 
        indices that will be passed and in which order.
     
  * API Changes:
      * The RAJA 'VarOps' namespace has been removed. All entities previously
        in that namespace are now in the 'RAJA' namespace.
      * RAJA span is now public for users to access and has been made more like
        std::span.
      * RAJA::statement::tile_fixed has been moved to RAJA::tile_fixed
        (namespace change).
      * RAJA::statement::{Segs, Offsets, Params, ValuesT} have been moved to
        RAJA::{Segs, Offsets, Params, ValuesT} (namespace change).
      * RAJA ListSegment constructors have been expanded to accept a camp
        Resource object. This enables run time specification of the memory
        space where the data for list segment indices will live. In earlier
        RAJA versions, the space in which list segment index data lived was a 
        compile-time choice based on whether CUDA or HIP was enabled and the 
        data resided in unified memory for either case. This is still supported
        in this release, but is marked as a DEPRECATED FEATURE. In the next RAJA
        release, ListSegment construction will require a camp Resource object.
        When compiling RAJA with your application, you will see deprecation
        warnings if you are using the deprecated ListSegment constructor. 
      * A reset method was added to OpenMP target offload reduction classes
        so they contain the same functionality as reductions for all other 
        back-ends.

  * Build changes/improvements:
      * The BLT, camp, CUB, and rocPRIM submodules have all been updated to 
        more recent versions. Please note that RAJA now requires rocm version 
        3.5 or newer to use the HIP back-end.
      * Build for clang9 on macosx has been fixed.
      * Build for Intel19 on Windows has been fixed.
      * Host/device annotations have been added to reduction operations to
        eliminate compiler warnings for certain use cases.
      * Several warnings generated by the MSVC compiler have been eliminated.
      * A couple of PGI compiler warnings have been removed.
      * CMake improvements to make it is easier to use an external camp or 
        CUB library with RAJA. 
      * Note that the RAJA tests are undergoing a substantial overhaul. Users,
        who chose to build and run RAJA tests, should know that many tests
        are now being generated in the build space directory structure which
        mimics the RAJA source directory structure. As a result, only some
        test executables appear in the top-level 'test' subdirectory of the 
        build directory; others can be found in lower-level directories. The
        reason for this change is to reduce test build times for certain 
        compilers.

  * Bug fixes:
      * An issue with SIMD privatization with the Intel compiler, required
        to generate correct code, has been fixed.
      * An issue with the atomicExchange() operation for the RAJA HIP back-end
        has been fixed.
      * A type issue in the RAJA::kernel implementation involving RAJA span
        usage has been fixed.
      * Checks for iterator ranges and container sizes have been added to
        RAJA scans, which fixes an issue when users attempted to run a 
        scan over a range of size zero.
      * Several type errors in the Layout.hpp header file have been fixed.
      * Several fixes have been made in the Layout and Static Layout types.
      * Several fixes have been made to the OpenMP target offload back-end
        to address host-device memory issues.
      * A variety of RAJA User Guide issues have been addressed, as well as
        issues in RAJA example codes.

Version v0.11.0 -- Release date 2020-01-29
==========================================

This release contains new features, several notable changes, and some bug fixes.

Notable changes include:

  * New features:
      * HIP compiler back-end added to support AMD GPUs. Usage is essentially
        the same as for CUDA. Note that this feature is considered a
        work-in-progress and not yet production ready. It is undocumented,
        but noted here, for friendly users who would like to try it out. 
      * Updated version of camp third-party library, which includes variety
        of portability fixes. Most users should not need to concern 
        themselves with the details of camp.
      * Added new tutorial material and exercises.
      * Documentation improvements.

  * API Changes:
      * None.
 
  * Build changes/improvements:
      * RAJA version number is now accessible as #define macro variable 
        constants so that users who need to parameterize their code to support 
        multiple RAJA versions can do this more easily. See the file 
        RAJA/include/RAJA/config.hpp for details. RAJA version numbers 
        are also experted as CMake variables.
      * Added support to link to external camp library. By default, the camp
        git submodule will be used. If you prefer to use a different version
        of camp, set the RAJA CMake variable 'EXTERNAL_CAMP_SOURCE_DIR' to
        the location of the desired camp directory.
      * BLT submodule (CMake-based build system) has been updated to latest
        BLT release (v0.3.0). The release contains a new version of GoogleTest,
        which required us to modify our use of gtest macros and our own 
        testing macros. For the most part, this change should be invisible to 
        users. However, the new GoogleTest does not work with CUDA versions 
        9.1.x or earlier. Therefore, if you compile RAJA with CUDA enabled and 
        also wish to enable RAJA tests, you must use CUDA 9.2.x or newer.

  * Bug fixes:
      * Fixed various issues to make internal implementations more robust,
        resolved issues with non fully-qualified types in some places, 
        and work arounds for some compiler issues.


Version v0.10.0 -- Release date 2019-10-31
==========================================

This release contains new features, several notable changes, and some bug fixes.

Notable changes include:

  * New features:
      * Added CUDA block direct execution policies, which can be used to map
        loop iterations directly to CUDA thread block. These are analogous to
        the pre-existing thread direct policies. The new block direct policies
        can provide better performance for kernels than the block loop policies
        when load balancing may be an issue. Please see the RAJA User Guide for 
        a description of all available RAJA execution policies.
      * Added a plugin registry feature that will allow plugins to be linked
        into RAJA that can act before and after kernel launches. One benefit
        of this is that RAJA no longer has an explicit CHAI dependency if RAJA 
        is used with CHAI. Future benefits will include integration with other
        tools for performance analysis, etc.
      * Added a shift method to RAJA::View, which allows one to create a new
        view object from an existing one that is shifted in index space from 
        the original. Please see the RAJA User Guide for details.
      * Added support for RAJA::TypedView and RAJA::TypedOffsetLayout, so that 
        the index type can be specified as a template parameter.
      * Added helper functions to convert a RAJA::Layout object to a 
        RAJA::OffsetLayout object and RAJA::TypedLayout to 
        RAJA::TypedOffsetLayout. Please see the RAJA User Guide for details.
      * Added a bounds checking option to RAJA Layout types as a debugging
        feature. This is a compile-time option that will report user errors
        when given View or Layout indices are out-of-bounds. See View/Layout
        section in the RAJA User Guide for instructions on enabling this and 
        how this feature works. 
      * We've added a RAJA Template Project on GitHub, which shows how to
        use RAJA in an application, either as a Git submodule or as an
        externally installed library that you link your application against.
        It is available here: https://github.com/LLNL/RAJA-project-template.
        It is also linked to the main RAJA project page on GitHub. 
      * Various user documentation improvements.

  * API Change.
      * The type alias RAJA::IndexSet that was marked deprecated previously
        has been removed. Now, all index set usage must use the type 
        RAJA::TypedIndexSet and specify all segment types (as template 
        parameters) that the index set may potentially hold.

  * Bug fixes:
      * Fix for issue in OpenMP target offload back-end that previously caused
        some RAJA Performance Suite kernels to seg fault when built with the
        XL compiler.
      * Removed an internal RAJA class constructor to prevent users to do
        potentially incorrect, and very difficult to hunt down, things in 
        their code that are technically not supported in RAJA, such as
        inserting RAJA::statement::CudaSyncThreads() in arbitrary places 
        inside a lambda expression.

  * Build changes/improvements:
      * RAJA now enforces a minimum CUDA compute capability of sm_35. Users
        can use the CMake variable 'CUDA_ARCH' to specify this. If not 
        specified, the value of sm_35 will be used and an informational 
        message will be emitted indicating this. If a user attempts to set
        the value lower than sm_35, CMake will error out and a message will
        be emitted indicating why this happened.
      * Transition to using camp as a submodule after its open source release
        (https://github.com/llnl/camp). 
      * Made minimum required CMake version 3.9.
      * Update BLT build system submodule to newer version 
        (SHA-1 hash: 96419df).
      * Cleaned up compiler warnings in OpenMP target back-end implementation.


Version v0.9.0 -- Release date 2019-07-25
=========================================

This release contains feature enhancements, one breaking change, and some 
bug fixes. 

  * Breaking change
    * The atomic namespace in RAJA has been removed. Now, use atomic operations
      as RAJA::atomicAdd(), not RAJA::atomic::atomicAdd(), for example. This
      was done to make atomic usage consistent with other RAJA features, such
      as reductions, scans, etc.

Other notable changes include:

  * Features
    * The lambda statement interface has been extended in the RAJA kernel API.
      Earlier, when multiple lambda expressions were used in a kernel, they
      were required to all have the same arguments, although not all 
      arguments had to be used in each lambda expression. Now, lambda 
      arguments may be specified in the RAJA::statement::Lambda type so 
      that each lambda expression need only take the arguments it uses.
      However, the previous usage pattern will continue to be supported.
      To support the new interface, new statement types have been introduced 
      to indicate iteration space variables (Segs), local variable/array 
      parameters (Params), and index offsets (Offsets). The offsets can be used
      with a For statement as a replacement for the ForICount statement. The
      new API features are described in the RAJA User Guide.
    * Minloc and maxloc reductions now support a tuple of index values. So 
      now if you have a nested loop kernel with i, j, k loops, you can get
      the 'loc' value out as an i, j, k triple.

  * Bug Fixes:
    * Small change to make RAJA Views work properly with OpenMP target kernels.
    * Changes to fix OpenMP target back-end for XL compilers.
    * Fix build issue with older versions of GNU compiler.
    * Fixes to resolve issues associated with corner cases in choosing 
      improper number of threads per block or number of thread blocks for
      CUDA execution policies.

  * Build changes/improvements:
    * A few minor portability improvements


Version v0.8.0 -- Release date 2019-03-28
=========================================

This release contains one major change and some minor improvements to 
compilation and performance.

Major changes include:

  * Build system updated to use the latest version of BLT (or close to it). 
    Depending on how one builds RAJA, this could require changes to how 
    information is passed to CMake. Content has been added to the relevant 
    sections of the RAJA User Guide which describes how this is done.

Other notable changes include:

  * Features (These are not yet documented and should be considered 
    experimental. There will be documentation and usage examples in the
    next RAJA release.)
    * New thread, warp, and bitmask policies for CUDA. These are not
      yet documented and should be considered experimental.
    * Added AtomicLocalArray type which returns data elements wrapped
      in an AtomicRef object.

  * Bug Fixes:
    * Fixed issue in RangeStrideSegment iteration.
    * Fix 'align hint' macro to eliminate compile warning when XL compiler
      is used with nvcc.
    * Fix issues associated with CUDA architecture level (i.e., sm_*) set
      too low and generated compiler warning/errors. Caveats for RAJA features
      (mostly atomic operations) available at different CUDA architecture 
      levels added to User Guide.

  * Performance Improvements:
    * Some performance improvements in RAJA::kernel usage with CUDA back-end.


Version v0.7.0 -- Release date 2019-02-07
=========================================

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


Version v0.6.0 -- Release date 2018-07-27
=========================================

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


Version v0.5.3 -- Release date 2018-01-31
=========================================

This is a bugfix release that fixes bugs in the IndexSetBuilder methods. These
methods now work correctly with the strongly-typed IndexSet.


Version v0.5.2 -- Release date 2018-01-30
=========================================

This release fixes some small bugs, including compiler warnings issued for
deprecated features, type narrowing, and the slice method for the
RangeStrideSegment class.

It also adds a new CMake variable, RAJA_LOADED, that is used to determine
whether RAJA's CMakeLists file has already been processed. This is useful when
including RAJA as part of another CMake project.


Version v0.5.1 -- Release date 2018-01-17
=========================================

This release contains fixes for compiler warnings with newer GCC and Clang
compilers, and allows strongly-typed indices to work with RangeStrideSegment.

Additionally, the index type for all segments in an IndexSet needs to be the
same. This requirement is enforced with a static_assert.


Version v0.5.0 -- Release date 2018-01-11
=========================================

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


Version v0.4.1 -- Release date 2017-10-11
=========================================

This release contains a bugfix for warnings when using the -Wpedantic flag.


Version v0.4.0 -- Release date 2017-10-11
=========================================

This release contains minor fixes for issues in the previous v0.3.1 release, 
plus some improvements to documentation, reduction performance, improved 
portability across a growing set of compilers and environments (e.g., Windows),
namespace refactoring to avoid cyclic dependencies and leverage argument-
dependent lookup, etc. In addition, the RAJA backend for Intel TBB is now 
off by default, whereas previously it was on by default.

A few major changes are included in this version:

  * Changes to the way RAJA is configured and built. We are now using the 
    BLT build system which is a Git submodule of RAJA. In addition to 
    requiring the '--recursive' option to be passed to 'git clone', this 
    introduces the following major change: RAJA_ENABLE_XXX options passed to 
    CMake are now just ENABLE_XXX.

  * A new API and implementation for nested-loop RAJA constructs has been 
    added. It is still a work in progress, but users are welcome to try it 
    out and provide feedback. Eventually, RAJA::nested::forall will replace 
    RAJA::forallN.


Version v0.3.1 -- Release date 2017-09-21
=========================================

This release contains some new RAJA features, plus a bunch of internal changes 
including more tests, conversion of nearly all unit tests to use Google Test, 
improved testing coverage, and compilation portability improvements (e.g., 
Intel, nvcc, msvc). Also, the prefix for all RAJA source files has been changed
from 'cxx'to 'cpp' for consistency with the header file prefix conversion in 
the last release. The source file prefix change should not require users to 
change anything.

New features included in this release:

  * Execution policy modifications and additions: 
     
      * seq_exec is now strictly sequential (no SIMD, etc.) 
      * simd_exec will force SIMD vectorization 
      * loop_exec (new policy) will allow compiler to optimize however it can, 
        including SIMD. 

    So, loop_exec is really what our previous simd_exec policy was before, 
    and 'no vector' pragmas have been added to all sequential implementations. 

    NOTE: SIMD changes are still being evaluated with different compilers on 
    different platforms. More information will be provided as we learn more.

  * Added support for atomic operations (min, max, inc, dec, and, or, xor, 
    exchange, and CAS) for all programming model backends. These appear in the 
    RAJA::atomic namespace. 

  * Support added for Intel Threading Building Blocks backend (considered 
    experimental at this point). 

  * Added macros that will be used to mark features for future deprecation 
    (please watch for this as we will be deprecating some features in the 
    next release).

  * Added support for C++17 if CMake knows about it.

  * Remove limit on number of ordered OpenMP reductions that can be used in 
    a kernel.

  * Remove compile-time error from memutils, add portable aligned allocator.

  * Improved ListSegment implementation.

  * RAJA::Index_type is now ptrdiff_t instead of int.

Notable bug fixes included in this release:

  * Fixed strided_numeric_iterator to apply stride sign in comparison.

  * Bug in RangeStrideSegment when using CUDA is fixed.

  * Fixed reducer logic for openmp_ordered policy.


Version v0.3.0 -- Release date 2017-07-13
=========================================

This release contains breaking changes and is not backward compatible with 
prior versions. The largest change is a re-organization of header files, 
and the switch to .hpp as a file extension for all headers.

New features included in this release:

  * Re-organization of header files.

  * Renaming of file extensions.

  * Rudimentary OpenMP 4.5 support.

  * CHAI support.


Version v0.2.5 -- Release date 2017-03-28
=========================================

This release includes some small fixes, as well as an initial re-organization 
of the RAJA header files as we move towards a more flexible usage model.


Version v0.2.4 -- Release date 2017-02-22
=========================================

This release includes the following changes:

 * Initial support of clang-cuda compiler.

 * New, faster OpenMP reductions.

N.B. The default OpenMP reductions are no longer performed in a ordered 
     fashion, so results may not be reproducible. The old reductions are 
     still available with the policy RAJA::omp_reduce_ordered.


Version v0.2.3 -- Release date 2016-12-15
=========================================

Hotfix to update the URLs used for fetching clang during Travis builds.


Version v0.2.2 -- Release date 2016-12-14
=========================================

Bugfix release that address an error when launching forall cuda kernels 
with a 0-length range.


Version v0.2.1 -- Release date 2016-12-07
=========================================

This release contains fixes for compiler warnings and removes the usage of 
the custom FindCUDA CMake package.


Version v0.2.0 -- Release date 2016-12-02
=========================================

Includes internal changes for performance and code maintenance.


Version v0.1.0 -- Release date 2016-06-22
=========================================

Initial release.
