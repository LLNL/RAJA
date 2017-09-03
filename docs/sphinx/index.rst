.. ##
.. ## Copyright (c) 2016, Lawrence Livermore National Security, LLC.
.. ##
.. ## Produced at the Lawrence Livermore National Laboratory.
.. ##
.. ## All rights reserved.
.. ##
.. ## For details and restrictions, please read the README-license.txt file.
.. ##


****
RAJA
****

RAJA is a collection of C++ software abstractions, being developed at Lawrence
Livermore National Laboratory (LLNL), that enable architecture portability for
HPC applications. The overarching goals of RAJA are to make existing
(production) applications *portable with minimal disruption*, and to provide a
model for new applications so that they are portable from inception.

RAJA uses standard C++11 -- C++ is the predominant programming language in
which many LLNL codes are written. RAJA is rooted in a perspective based on 
substantial experience working on production mesh-based multiphysics 
applications at LLNL. Another goal of RAJA is to enable application developers 
to adapt RAJA concepts and specialize them for different code implementation 
patterns and C++ usage, since data structures and algorithms vary widely 
across applications.

If you are new to RAJA, check out our guide to :doc:`getting_started`.

See the :doc:`features` for detailed documentation for each feature.

If you are interested in keeping up with RAJA development and communicating
with developers and users, please join our `Google Group
<https://groups.google.com/forum/#!forum/raja-users>`_.

If you have questions, find a bug, or have ideas about expanding the
functionality or applicability of RAJA and are interested in contributing
to its development, please do not hesitate to contact us. We are always
interested in improving RAJA and exploring new ways to use it.

.. toctree::
   :maxdepth: 2
   :caption: Basics

   getting_started
   tutorial
   features

.. toctree::
   :maxdepth: 2
   :caption: Reference

   config_build
   plugins

.. toctree::
   :maxdepth: 2
   :caption: Contributing

   contributing
   raja_license

   future
