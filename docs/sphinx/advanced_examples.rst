.. ##
.. ## Copyright (c) 2016, Lawrence Livermore National Security, LLC.
.. ##
.. ## Produced at the Lawrence Livermore National Laboratory.
.. ##
.. ## All rights reserved.
.. ##
.. ## For release details and restrictions, please see raja/README-license.txt
.. ##


-----------------------
RAJA Proxy Applications
-----------------------

.. warning:: Under construction! 
             We will break out the RAJA proxy apps (LULESH, Kripke, eventually
             CoMD, etc.) into a separate repo. We can point to it here. The
             text below is from the original docs. I kept it so we can
             use bits that we want to keep.
 

The directory 'raja/test' has subdirectories containing examples of RAJA 
used in the proxy apps LULESH (versions 1.0 and 2.0), Kripke, and CoMD.
Reference versions of these applications are also included so that it is 
easy to compare them to the RAJA variants, in terms of source code 
differences and runtimes. Here is a brief explanation of the contents of 
the directories containing the proxy app examples.

  * **LULESH 1.0.** 

    The directory 'LULESH-v1.0_baseline' contains the reference version of 
    LULESH 1.0 that is available at 
    `LULESH <https://codesign.llnl.gov/lulesh.php>`_

    The directory 'LULESH-v1.0_RAJA-variants' contains three RAJA variants of 
    LULESH 1.0: serial-only, a highly-parametrized version that can be run 
    using various execution patterns, and a version that shows a RAJA-based 
    transient fault recovery capability. The names of the files indicate which
    version is which. The parametrized version of this code  
    illustrates ten different execution patterns that can be enabled using
    RAJA. These patterns include sequential execution, six variants
    using OpenMP (using RAJA IndexSets for tiling, permuting elements, 
    coloring, dependency scheduling, etc.), and two GPU variants
    that use CUDA. Some of these require specific initialization code 
    for data and RAJA IndexSets. However, all variations use the same 
    algorithm source code; different variations use different RAJA
    execution policies. To try different variants, simply change the 
    definition of the macro constant ::

      #define USE_CASE ....

    in the header file called 'luleshPolicy.hxx'. There you will find a listing
    of available options and the RAJA policies used for each.

  * **LULESH 2.0.** 

    The directory 'LULESH-v2.0_baseline' contains the reference version of
    LULESH 2.0 that is available at 
    `LULESH <https://codesign.llnl.gov/lulesh.php>`_

    The directory 'LULESH-v2.0_RAJA-basic' contains a basic translation to 
    RAJA that uses only RAJA forall traversals that take begin-end arguments or 
    arrays of indirection indices.

    The directory 'LULESH-v2.0_RAJA-IndexSet' contains a version that uses 
    RAJA IndexSets similarly to the parallel RAJA variant of LULESH 1.0.
    Similar to the RAJA version of LULESH 1.0, this variant is parametrized 
    to run with different execution patterns. However, only three variants
    are available (we got tired...). As in the case of LULESH 1.0, change 
    the definition of the 'USE_CASE' macro constant in the 'luleshPolicy.hxx' 
    header file to change the variant.

    The directory 'LULESH-v2.0_RAJA-MICfriendly' contains a version that
    uses RAJA IndexSets to permute data and loop iteration ordering in ways 
    that can be beneficial in a manycore environment. 

  * **Kripke.** 

    The directory 'Kripke-v1.1-baseline' contains the reference version of 
    Kripke v1.1 that uses basic OpenMP threading and zone-sequential sweep
    traversals.  It is available at
    `Kripke <https://codesign.llnl.gov/kripke.php>`_
    
    The directory 'Kripke-v1.1-RAJA' contains the RAJA version of Kripke v1.1 
    that uses the nested-loop RAJA forallN traversals and 
    IndexSets to perform hyperplane sweep traversals.

    Currently there are issues building with the Intel compiler which are being
    investigated.  As a result, building the RAJA version of Kripke with icpc
    will diable all but the DGZ data layouts, and disables complex execution
    policies.  Once a resolution to these issues have been found, this code will
    be updated.

