.. ##
.. ## Copyright (c) 2016-20, Lawrence Livermore National Security, LLC
.. ## and other RAJA project contributors. See the RAJA/COPYRIGHT file
.. ## for details.
.. ##
.. ## SPDX-License-Identifier: (BSD-3-Clause)
.. ##

.. _plugins-label:

========
Plugins
========

------------
About RAJA Plugins
------------

RAJA supports user-made plugins that may be loaded either at the time of compilation or during runtime. These two methods are not mutually exclusive, as plugins loaded statically can be run alongside plugins that are loaded dynamically.

------------
Using RAJA Plugins
------------

^^^^^^^^^^^^^^^^^
Static vs Dynamic Loading
^^^^^^^^^^^^^^^^^

**Static loading** is done at compile time and requires recompilation in order to add, remove, or change a plugin. This is arguably the easier method to implement, requiring only simple file linking to make work. However, recompilation may get tedious and resource-heavy when working with many plugins or on large projects. In these cases, it may be better to load plugins dynamically, requiring no recompilation of the project most of the time.

**Dynamic loading** is done at runtime and only requires the recompilation or moving of plugin files in order to add, remove, or change a plugin. This will likely require more work to set up, but in the long run may save time and resources. RAJA will look at the environment variable ``RAJA_PLUGINS`` for a path to a plugin or plugin directory, and automatically load them at runtime. This means that a plugin can be added to a project as easily as making a shared object file and setting ``RAJA_PLUGINS`` to the appropriate path.

^^^^^^^^^^^
Quick Start Guide
^^^^^^^^^^^

**Static Plugins**

1. Build RAJA normally.

2. Either use an ``#include`` statement within the code or compiler flags to load your plugin file with your project at compile time. A brief example of this would be something like ``g++ project.cpp plugin.cpp -lRAJA -fopenmp -ldl -o project``.

3. When you run your project, your plugin should work!

**Dynamic Plugins**

1. Build RAJA normally.

2. Compile your plugin to be shared object files with a .so extension. A brief example of this wouldbe something like ``g++ plugin.cpp -lRAJA -fopenmp -fPIC -shared -o plugin.so``.

3. Set the environment variable ``RAJA_PLUGINS`` to be the path of your .so file. This can either be the path to its directory or to the shared object file itself. If the path is to a directory, it will attempt to load all .so files in that directory.

4. When you run your project, your plugins should work!

^^^^^^^^^^^
Interfacing with Plugins
^^^^^^^^^^^

The RAJA Plugin API allows for limited interfacing between a project and a 
plugin. There are a couple of functions that allow for this to take place, 
``init_plugins`` and ``finalize_plugins``. These will call the corresponding 
``init`` and ``finalize`` functions, respectively, of *every* currently loaded 
plugin. It's worth noting that plugins don't require either an init or finalize
function by default.

* ``RAJA::util::init_plugins();`` - Will call the ``init`` function of every 
  currently loaded plugin.

* ``RAJA::util::init_plugins("path/to/plugins");`` - Does the same as the above
  call to ``init_plugins``, but will also dynamically load plugins located at 
  the path specified.

* ``RAJA::util::finalize_plugins();`` - Will call the ``finalize`` function of 
  every currently loaded plugin. 


------------
Creating Plugins For RAJA
------------

Plugins are classes derived from the ``RAJA::util::PluginStrategy`` base class
and implement the required functions for the API. An example implementation 
can be found at the bottom of this page.

^^^^^^^^^^^
Functions
^^^^^^^^^^^

The ``preLaunch`` and ``postLaunch`` functions are automatically called by 
RAJA before and after executing a kernel that uses ``RAJA::forall`` or 
``RAJA::kernel`` methods.

* ``void init(const PluginOptions& p) override {}`` - runs on all plugins when 
  a user calls ``init_plugins``

* ``void preCapture(const PluginContext& p) override {}`` - is called before 
  lambda capture in ``RAJA::forall`` or ``RAJA::kernel``.

* ``void postCapture(const PluginContext& p) override {}`` - is called after 
  lambda capture in ``RAJA::forall`` or ``RAJA::kernel``.

* ``void preLaunch(const PluginContext& p) override {}`` - is called before 
  ``RAJA::forall`` or ``RAJA::kernel`` runs a kernel.

* ``void postLaunch(const PluginContext& p) override {}`` - is called after 
  ``RAJA::forall`` or ``RAJA::kernel`` runs a kernel.

* ``void finalize() override {}`` - Runs on all plugins when a user calls 
  ``finalize_plugins``. This will also unload all currently loaded plugins.

``init`` and ``finalize`` are never called by RAJA by default and are only 
called when a user calls ``RAJA::util::init_plugins()`` or 
``RAJA::util::finalize_plugin()``, respectively.

^^^^^^^^^^^^^^^^^
Static Loading
^^^^^^^^^^^^^^^^^

If a plugin is to be loaded into a project at compile time, adding the 
following method call will add the plugin to the RAJA PluginRegistry and will 
be loaded every time the compiled executable is run. This requires the plugin 
to be loaded with either an ``#include`` statement within a project or by 
compiler commands::

  static RAJA::util::PluginRegistry::add<PluginName> P("Name", "Description");


^^^^^^^^^^^^^^^^^
Dynamic Loading
^^^^^^^^^^^^^^^^^

If a plugin is to be dynamically loaded to a project at run time, the RAJA 
Plugin API requires a few conditions to be met. The following must be true 
about the plugin, not necessarily of the project using it.

1. **The plugin must have the following factory function.** This will return 
   a pointer to an instance of your plugin. Thanks to the ``extern "C"``, a 
   project will be able to search for "getPlugin" within the dynamically 
   loaded plugin correctly::

     extern "C" RAJA::util::PluginStrategy *getPlugin ()
     {
       return new MyPluginName;
     }
  

2. **The plugin must be compiled to be a shared object with a .so extension.** 
   A simple example containing required flags would be: ``g++ plugin.cpp -lRAJA -fopenmp -fPIC -shared -o plugin.so``. 

   At the moment, RAJA will only attempt to load files with .so extensions. 
   It's worth noting why these flags (or their equivalents) are important. 

     * ``-lRAJA -fopenmp`` are standard flags for compiling the RAJA library. 

     * ``-fPIC`` tells the compiler to produce *position independent code*, 
       which prevents conflicts in the address space of the executable. 

     * ``-shared`` will let the compiler know that you want the resulting 
       object file to be shared, removing the need for a *main* as well as 
       giving dynamically loaded executables access to functions flagged 
       with ``extern "C"``.

3. **The** ``RAJA_PLUGINS`` **environment variable has been set**, or a user 
   has made a call to ``RAJA::util::init_plugins("path");`` with a path 
   specified to either a directory or a .so file. It's worth noting that these 
   are not mutually exclusive. RAJA will look for plugins based on the the 
   environment variable on program startup and new plugins may be loaded after 
   that by calling ``init_plugins``.


^^^^^^^^^^^^^^^^^
Example Implementation
^^^^^^^^^^^^^^^^^

The following is an example plugin that simply will print out the number of 
times a kernel has been launched and has the ability to be loaded either 
statically or dynamically.

.. literalinclude:: ../../../../examples/plugin/counter-plugin.cpp
   :start-after: _plugin_example_start
   :end-before: _plugin_example_end
   :language: C++
