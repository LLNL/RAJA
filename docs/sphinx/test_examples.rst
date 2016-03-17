
================
Examples
================

The RAJA code comes with several simple test codes and proxy-app examples
that illustrate various usage scenarios. When RAJA is compiled, these 
executables are built in their default configurations. They can be run by
invoking the executables in the build space 'raja/test' subdirectories whose 
names are descriptive of their contents. These codes can be modified to 
experiment with them and run using various options that we describe later 
in this section. **Note that the source code is modified in the 'raja/test'
subdirectories in the actual source code, not in the build space.**

--------------
Basic tests
--------------

The directory 'raja/test/unit-tests' has two subdirectories, each of which
contains files that run various traversal and reduction operations for RAJA 
IndexSets and Segments. All RAJA "forall" template and execution poilicy 
options that are available for a given compiler are included. Running these
tests is a good sanity check that the code is built correctly and works. The
two subdirectories are:

  * CPUtests. It contains codes to run traversl and reduction tests for 
    sequential, OpenMP, and CilkPlus (if available) execution policies.

  * GPUtests. It contains codes to run traversl and reduction tests for 
    GPU CUDA execution policies. Note that these tests use Unified Memory 
    to simplify host-device memory transfers.


-----------------------
Example applications
-----------------------

The directory 'raja/test' has subdirectories containing examples of RAJA 
used in proxy apps, such as `LULESH <https://codesign.llnl.gov/lulesh.php>`_ (versions 1.0 and 2.0), `Kripke <https://codesign.llnl.gov/kripke.php>`_, and 
`CoMD <https://github.com/exmatex/CoMD>`_. Reference versions of these 
applications, that can be downloaded by clicking the links above, are also
included so that it is easy to compare them to the RAJA variants, both in
terms of source code differences and runtimes. Here is a brief explanation 
of the RAJA variant of each of these proxy apps:

  * **LULESH 1.0.** The RAJA version of this code is parameterized to 
    illustrate ten different execution patterns that can be enabled using
    RAJA. These patterns include sequential execution, six variants
    using OpenMP (using RAJA IndexSets for tiling, permuting elements, 
    coloring, dependency scheduling, etc.), CilkPlus, and two GPU variants
    that use CUDA. Some of these require specific initialization code 
    for data and RAJA IndexSets. However, all variations are enabled by 
    RAJA using the same algorithmic source code and switching the RAJA
    execution policies. To try different variants, simply change the 
    definition of the macro constant ::

      #define USE_CASE

    in the header file called 'luleshPolicy.hxx'. There you will find a listing
    of available options and the RAJA policies used for each.

  * **LULESH 2.0.** This example contains three different RAJA implementations:
    "basic" (uses only RAJA forall traversals that take begin-end args or 
    arrays of indirection indices), "IndexSet" (uses RAJA IndexSets similarly
    to LULESH 1.0), and "MICfriendly" (uses RAJA IndexSets to permute data
    and loop iteration ordering which can be beneficial in a manycore 
    environment. The "IndexSet" variant is parameterized to run with 
    different execution patterns, similar to the RAJA version of LULESH 1.0.
    However, only three variants are enabled for LULESH 2.0 (we got tired...).
    As in the case of LULESH 1.0, change the definition of the 'USE_CASE'
    macro constant in the 'luleshPolicy.hxx' header file. 

  * **Kripke.** Fill this in...

  * **CoMD.** Fill this in...

