.. _tutorial::

=====================
Overview
=====================
As RAJA is built on the C++ lambda, we provide a brief overview on lambda functions.
Lambda's were introduced to allow for the construction of
in place functions. A lambda has the ability to "capture" variables from a local context and use within the
function.  A lambda expression takes the following form 

.. code-block:: cpp

   [capture list] (parameter list) {function body}  

The capture list corresponds to variables within the scope, while the parameter list corresponds to values which will be used within the function. By default, a lambda will capture by copying variables in the capture list. Capturing by reference may be accomplished by using the & symbol; for example 

.. code-block:: cpp

   int x;
   int y = 100;
   [&x, &y](){x=y;]

will generate a lambda which will assign the value of y to x. 
By setting the capture list as ``[=]`` or ``[&]`` all variables within scope will be captured by copy or reference respectively.

Building from the C++ lambda, RAJA introduces two types of templated methods, namely
``RAJA::forall`` and ``RAJA::forallN``. The ``RAJA::forall`` method is an abstraction
of the standard C++ loop which is templated on an execution policy, takes an iteration space, and a lambda capturing the loop body.

.. code-block:: cpp
                
  RAJA::forall<exec_policy>(iter_space I, [=] (index_type i)) {
    //body
  });

Similarly, the ``RAJA::ForallN`` loop is an abstraction of nested ``for`` loops. The ``RAJA::ForallN`` loop is
templated on up to N execution policies, expects an iteration space, and index for each execution policy.
  
.. code-block:: cpp
                
  RAJA::forallN<
    RAJA::NestedPolicy<exec_policy1, .... , exec_policyN> >(
      iter_space I1,..., iter_space IN, [=](index_type i1,..., index_type iN) {
         //body
  });

In summary, using one of the RAJA templated methods requires the developer to supply the following
1. Capture type [=] or [&]
3. exec_policy - Specifying how the traversal occurs
4. iter_space  - An iteration space for the RAJA loop (any random access container is expected)
5. index_type  - Index for RAJA loops
6. lambda      - capturing the body of the loop

The remainder of the tutorial demonstrates the utility of RAJA by drawing from commonly used
computing patterns.



--------
Examples
--------

.. toctree::
   :maxdepth: 2

   tutorial/addVectors.rst
   tutorial/gaussSeidel.rst
   tutorial/jacobi.rst
   tutorial/matrixMultiply.rst
   tutorial/wave.rst
