.. ##
.. ## Copyright (c) 2016-17, Lawrence Livermore National Security, LLC.
.. ##
.. ## Produced at the Lawrence Livermore National Laboratory
.. ##
.. ## LLNL-CODE-689114
.. ##
.. ## All rights reserved.
.. ##
.. ## This file is part of RAJA.
.. ##
.. ## For details about use and distribution, please read RAJA/LICENSE.
.. ##

.. _nestedreorder-label:

---------------------------------
Nested Loops and Loop Interchange
---------------------------------

Key RAJA features shown in this example:

  * ``RAJA::RangeSegment`` iteration space construct
  * ``RAJA::nested::forall`` loop iteration templates 
  * RAJA nested loop execution policies
  * Nested loop re-ordering (i.e., loop interchange)
  * RAJA strongly-types indices.

In the example :ref:`matrixmultiply-label`, we introduced RAJA nested loop
constructs and briefly described how this works.
This example shows how to re-order the levels in a loop nest by re-ordering 
nested policy arguments and introduces strongly-typed index variables that
can help users write correct nested loop RAJA code. The example does not 
perform any actual computation; each kernel simply prints out the loop 
indices in the order that the iteration spaces are traversed. Thus, only 
sequential execution policies are used.

Before we begin with the triply-nested loop example, we note important 
features applied here that show the main differences between nested-loop 
RAJA and the ``RAJA::forall`` loop construct we have been using in the
examples to this point:

  * A range and lambda index argument are required for each level in a
    loop nest. In this example, we focus on triply-nested loops.

  * A range for each loop nest level is specified in a RAJA tuple object.
    The order of ranges in the tuple must match the order of arguments to the
    lambda for this to be correct, in general. RAJA provides strongly-typed
    indices to help with this which we discuss here.

  * An execution policy is required for each level in the loop nest. These
    are specified in the ``RAJA::nested::Policy`` type.

  * The loop nest ordering is specified in the nested execution policy --
    the first 'For' policy is the outermost loop, the second 'For' policy
    is the loop nested inside the outermost loop, and so on.

We define three named loop index types for the loop index variables used in
the triply-nested loop examples. 

.. literalinclude:: ../../../../examples/ex5-nested-loop-reorder.cpp
                    :lines: 40-42

We also define three `typed` range segments which bind the ranges to the
index variable types:

.. literalinclude:: ../../../../examples/ex5-nested-loop-reorder.cpp
                    :lines: 53-55

When these features are used as they are in the example, the compiler will 
generate error messages if the lambda index argument ordering
and types do not match the typed range index ordering.

First, we present a complete example, and then describe its key elements:

.. literalinclude:: ../../../../examples/ex5-nested-loop-reorder.cpp
                    :lines: 62-71

The ``RAJA::nested::forall`` template takes three arguments: a nested policy,
a tuple of ranges, one for each level in the loop nest, and the lambda
loop body. The lambda has an index argument for each level in the
loop nest.

.. note :: The number and order of lambda arguments must match the number and
           order of ranges in the tuple for this to be correct.

The execution policy for each level in the loop nest is defined by the
``RAJA::nested::For`` arguments in the nested policy type. The integer that 
appears as the first argument to each 'For' template corresponds to the index 
of a range in the tuple and also to the associated lambda index argument; 
i.e., '0' is for 'i', '1' is for 'j', and '2' is for 'k'. The reason that 
the integer arguments are needed is that the levels in the loop nest can be 
permuted by reordering the policy arguments and the kernel remains the same.

**In particular, the ordering of the 'For' statements determines the loop nest 
ordering.** 

In this case, the 'k' index corresponds to the outermost loop (slowest index), 
the 'j' index corresponds to the middle loop, and the 'i' index is for the 
innermost loop (fastest index). Thus, the loop generates the following output::

 (I, J, K)
 ---------
 (0, 1, 2)
 (1, 1, 2)
 (0, 2, 2)
 (1, 2, 2)
 (0, 1, 3)
 (1, 1, 3)
 (0, 2, 3)
 (1, 2, 3)

Next we permute the loop nest ordering so that the 'j' loop is the outermost, 
the 'i' loop is in the middle, and the 'k' loop is the innermost with the 
following form:

.. literalinclude:: ../../../../examples/ex5-nested-loop-reorder.cpp
                    :lines: 79-88

This generates the output::

 (I, J, K)
 ---------
 (0, 1, 2) 
 (0, 1, 3) 
 (1, 1, 2) 
 (1, 1, 3) 
 (0, 2, 2) 
 (0, 2, 3) 
 (1, 2, 2) 
 (1, 2, 3) 

Finally, we permute the loops again so that the 'i' loop is the outermost, 
the 'k' loop is in the middle, and the 'j' loop is the innermost with the 
following form:

.. literalinclude:: ../../../../examples/ex5-nested-loop-reorder.cpp
                    :lines: 96-105

This generates the output::

 (I, J, K)
 ---------
 (0, 1, 2) 
 (0, 2, 2) 
 (0, 1, 3) 
 (0, 2, 3) 
 (1, 1, 2) 
 (1, 2, 2) 
 (1, 1, 3) 
 (1, 2, 3)

Hopefully, it is clear how this works. If it seems clear as mud, the typed
indices and range segments can help by indicating errors at compile-time.

For example, this version of the loop will not compile (execution policy
is the same as in the previous version): 

.. literalinclude:: ../../../../examples/ex5-nested-loop-reorder.cpp
                    :lines: 114-118

Hopefully, if you carefully compare the range ordering in the tuple to the
lambda argument types, you will see what's wrong.

Do you see the issue?

The file ``RAJA/examples/ex5-nested-loop-reorder.CPU`` contains the complete 
working example code.
