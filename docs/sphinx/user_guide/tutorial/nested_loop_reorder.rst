.. ##
.. ## Copyright (c) 2016-22, Lawrence Livermore National Security, LLC
.. ## and RAJA project contributors. See the RAJA/LICENSE file
.. ## for details.
.. ##
.. ## SPDX-License-Identifier: (BSD-3-Clause)
.. ##

.. _kernelnestedreorder-label:

---------------------------------
Nested Loop Interchange
---------------------------------

Key RAJA features shown in this example:

  * ``RAJA::kernel`` loop iteration templates 
  * RAJA nested loop execution policies
  * Nested loop reordering (i.e., loop interchange)
  * RAJA strongly-types indices

In :ref:`loop_elements-kernel-label`, we introduced the basic mechanics in
RAJA for representing nested loops. In :ref:`matrixmultiply-label`, we 
presented a complete example using RAJA nested loop features. The following 
example shows the nested loop interchange process in more detail. 
Specifically, we describe how to reorder nested policy arguments and introduce
strongly-typed index variables that can help users write correct nested loop 
code with RAJA. The example does not perform any actual computation; each 
kernel simply prints out the loop indices in the order that the iteration 
spaces are traversed. Thus, only sequential execution policies are used. 
However, the mechanics work the same way for other RAJA execution policies.

Before we dive into the example, we note important features applied here that 
represent the main differences between nested-loop RAJA and the 
``RAJA::forall`` loop construct for simple (i.e., non-nested) loops: 

  * An index space (e.g., range segment) and lambda index argument are 
    required for each level in a loop nest. This example contains
    triply-nested loops, so there will be three ranges and three index 
    arguments.

  * The index spaces for the nested loop levels are specified in a RAJA tuple 
    object. The order of spaces in the tuple must match the order of index 
    arguments to the lambda for this to be correct, in general. RAJA provides 
    strongly-typed indices to help with this, which we show here.

  * An execution policy is required for each level in a loop nest. These
    are specified as nested statements in the ``RAJA::KernelPolicy`` type.

  * The loop nest ordering is specified in the nested kernel policy --
    the first ``statement::For`` type identifies the outermost loop, the 
    second ``statement::For`` type identifies the loop nested inside the 
    outermost loop, and so on.

We begin by defining three named **strongly-typed** variables for the loop 
index variables.

.. literalinclude:: ../../../../examples/tut_nested-loop-reorder.cpp
   :start-after: _nestedreorder_idxtypes_start
   :end-before: _nestedreorder_idxtypes_end
   :language: C++

We also define three **typed** range segments which bind the ranges to the
index variable types via template specialization:

.. literalinclude:: ../../../../examples/tut_nested-loop-reorder.cpp
   :start-after: _nestedreorder_ranges_start
   :end-before: _nestedreorder_ranges_end
   :language: C++

When these features are used as in this example, the compiler will 
generate error messages if the lambda expression index argument ordering
and types do not match the index ordering in the tuple.

We present a complete example, and then describe its key elements:

.. literalinclude:: ../../../../examples/tut_nested-loop-reorder.cpp
   :start-after: _nestedreorder_kji_start
   :end-before: _nestedreorder_kji_end
   :language: C++

Here, the ``RAJA::kernel`` execution template takes two arguments: a tuple of 
ranges, one for each of the three levels in the loop nest, and the lambda 
expression loop body. Note that the lambda has an index argument for each 
range and that their order and types match.

The execution policy for the loop nest is specified in the 
``RAJA::KernelPolicy`` type. Each level in the loop nest is identified by a
``statement::For`` type, which identifies the iteration space and
execution policy for the level. Here, each level uses a 
sequential execution policy. This is for 
illustration purposes; if you run the example code, you will see the loop
index triple printed in the exact order in which the kernel executes.
The integer that appears as the first template argument to each 
``statement::For`` type corresponds to the index of a range in the tuple 
and also to the associated lambda index argument; i.e., '0' is for 'i', 
'1' is for 'j', and '2' is for 'k'. 

Here, the 'k' index corresponds to the outermost loop (slowest index), 
the 'j' index corresponds to the middle loop, and the 'i' index is for the 
innermost loop (fastest index). In other words, if written using C-style 
for-loops, the loop would appear as::

  for (int k = 2; k< 4; ++k) {
    for (int j = 1; j < 3; ++j) { 
      for (int i = 0; i < 2; ++i) { 
        // print loop index triple...
      }
    }
  }

The integer argument to each ``statement::For`` type is needed so 
that the levels in the loop nest can be reordered by changing the policy 
while the kernel remains the same. Next, we permute the loop nest ordering 
so that the 'j' loop is the outermost, the 'i' loop is in the middle, and 
the 'k' loop is the innermost with the following policy:

.. literalinclude:: ../../../../examples/tut_nested-loop-reorder.cpp
   :start-after: _nestedreorder_jik_start
   :end-before: _nestedreorder_jik_end
   :language: C++

Note that we have simply reordered the nesting of the ``RAJA::statement::For``
types. This is analogous to reordering 'for' statements in traditional C-style
nested loops. Here, the analogous C-style loop nest would appear as::

  for (int j = 1; j < 3; ++j) {
    for (int i = 0; i < 2; ++i) {
      for (int k = 2; k< 4; ++k) {
        // print loop index triple...
      }
    }
  }

Finally, for completeness, we permute the loops again so that the 'i' loop 
is the outermost, the 'k' loop is in the middle, and the 'j' loop is the 
innermost with the following policy:

.. literalinclude:: ../../../../examples/tut_nested-loop-reorder.cpp
   :start-after: _nestedreorder_ikj_start
   :end-before: _nestedreorder_ikj_end
   :language: C++

The analogous C-style loop nest would appear as::

  for (int i = 0; j < 2; ++i) {
    for (int k = 2; k< 4; ++k) {
      for (int j = 1; j < 3; ++j) {
        // print loop index triple...
      }
    }
  }

Hopefully, it should be clear how this works at this point. If not,
the typed indices and typed range segments can help by enabling the 
compiler to let you know when something is not correct.

For example, this version of the loop will generate a compilation error
(note that the kernel execution policy is the same as in the previous example): 

.. literalinclude:: ../../../../examples/tut_nested-loop-reorder.cpp
   :start-after: _nestedreorder_typemismatch_start
   :end-before: _nestedreorder_typemismatch_end
   :language: C++

If you carefully compare the range ordering in the tuple to the
lambda argument types, you will see what's wrong.

Do you see the problem?

The file ``RAJA/examples/tut_nested-loop-reorder.cpp`` contains the complete 
working example code.
