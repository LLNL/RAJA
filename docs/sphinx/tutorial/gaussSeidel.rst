.. _gaussSeidel::

======================
Red-Black Gauss-Seidel
======================

In this example we revisit the boundary value problem solved by the Jacobi method and apply a Red-Black Gauss-Sidel scheme.
The classic Gauss-Sidel method is inherrently a sequential algorithm; however, parallism may be exposed by coloring the interior nodes
in the following manner

.. image:: figures/redblackGS.png
   :scale: 10 %
   :align: center

By applying such a coloring we may iterate over the different colors sequentially while updating each color in parallel. We can encapsulate the sequential/parallel transversal using by using a ``RAJA::NestedPolicy``.

.. literalinclude:: ../../../examples/example-gauss-seidel.cpp
                    :lines: 240-241

Furthermore, we may construct a ``RAJA::StaticIndexSet`` which encapsulates the indices for the red and black nodes as two seperate list segments (for more info see ____ ). The following block of code illustrates the construction of the ``RAJA::StaticIndexSet``
           
.. literalinclude:: ../../../examples/example-gauss-seidel.cpp
                    :lines: 188-220
                            
A full working version ``example-gauss-seidel.cpp`` may be found in the example folder.
