.. developer_guide:

===============
Developer Guide
===============

Generating RAJA host-config files
===================================

.. note::
  This is optional if you are on LC machines, since some host-config files have already been generated (at least for Quartz and Lassen) and can be found in the ``host-configs`` repository directory.

RAJA only directly depends on CMake. However, this mechanism will generate a cmake configuration file that reproduces the configuration `Spack <https://github.com/spack/spack>` would have generated in the same context. It contains all the information necessary to build RAJA with the described toolchain.

In particular, the host config file will setup:
* flags corresponding with the target required (Release, Debug).
* compilers path, and other toolkits (cuda if required), etc.

This provides an easy way to build RAJA based on `Spack <https://github.com/spack/spack>` and encapsulated in `Uberenv <https://github.com/LLNL/uberenv>`_.

Uberenv role
------------

Uberenv helps by doing the following:

* Pulls a blessed version of Spack locally
* If you are on a known operating system (like TOSS3), we have defined compilers and system packages so you don't have to rebuild the world (CMake typically in RAJA).
* Overrides RAJA Spack packages with the local one if it exists. (see ``scripts/uberenv/packages``).
* Covers both dependencies and project build in one command.

Uberenv will create a directory ``uberenv_libs`` containing a Spack instance with the required RAJA dependencies installed. It then generates a host-config file (``<config_dependent_name>.cmake``) at the root of RAJA repository.

Using Uberenv to generate the host-config file
----------------------------------------------

.. code-block:: bash

  $ python scripts/uberenv/uberenv.py

.. note::
  On LC machines, it is good practice to do the build step in parallel on a compute node. Here is an example command: ``srun -ppdebug -N1 --exclusive python scripts/uberenv/uberenv.py``

Unless otherwise specified Spack will default to a compiler. It is recommended to specify which compiler to use: add the compiler spec to the ``--spec`` Uberenv command line option.

On blessed systems, compiler specs can be found in the Spack compiler files in our repository: ``scripts/uberenv/spack_configs/<System type>/compilers.yaml``.

Some examples uberenv options:

* ``--spec=%clang@9.0.0``
* ``--spec=%clang@8.0.1+cuda``
* ``--prefix=<Path to uberenv build directory (defaults to ./uberenv_libs)>``

Building dependencies can take a long time. If you already have a spack instance you would like to reuse (in supplement of the local one managed by Uberenv), you can do so changing the uberenv command as follow:

.. code-block:: bash

   $ python scripts/uberenv/uberenv.py --upstream=</path/to/my/spack>/opt/spack

Using host-config files to build RAJA
-------------------------------------

When a host-config file exists for the desired machine and toolchain, it can easily be used in the CMake build process:

If I need to build RAJA with _clang_ and _cuda_ on _lassen_, I can see there is already a host-config file named `lassen-blueos_3_ppc64le_ib_p9-clang@8.0.1-cuda.cmake`. To use it (on lassen):

.. code-block:: bash

  $ mkdir build && cd build
  $ cmake -C ../host-configs/lassen-blueos_3_ppc64le_ib_p9-clang@8.0.1-cuda.cmake ..
  $ cmake --build -j .
  $ ctest --output-on-failure -T test

.. note::
  This will build the default configuration. Not all parameters are embedded into the host-config file. For example, producing shared/static libraries, using OppenMP, enabling tests, is to be configured on command line.
