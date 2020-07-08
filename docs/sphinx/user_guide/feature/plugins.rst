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
Static vs Dynamic Linking
^^^^^^^^^^^^^^^^^

**Static linking** is done at compile time and requires recompilation in order to add, remove, or change a plugin. This is arguably the easier method to implement, requiring simple file linking to make work. However, recompilation may get tedious and resource-heavy when working with many plugins or on large projects. In these cases, it may be a better idea to load plugins dynamically, requiring no recompilation of the project most of the time.

**Dynamic linking** is done at runtime and only requires the recompilation or moving of plugin files in order to add, remove, or change a plugin. This will likely require more work to set up, but in the long run may save time and resources. RAJA will look at the environment variable ``RAJA_PLUGINS`` for a path to a plugin or plugin directory, and automatically load them at runtime. This means that a plugin can be added to a project as easily as making a shared object file and setting ``RAJA_PLUGINS`` to the appropriate path.

^^^^^^^^^^^
Quick Start Guide
^^^^^^^^^^^

**Static**

1. Build RAJA normally.

2. Either use an ``#include`` statement within the code or compiler flags to link your plugin file with your project at compile time. A brief example of this would be something like ``g++ project.cpp plugin.cpp -lRAJA -fopenmp -ldl -o project``.

3. When you run your project, your plugin should work!

**Dynamic**

1. Build RAJA normally.

2. Compile your plugin to be shared object files with a .so extension. A brief example of this wouldbe something like ``g++ plugin.cpp -lRAJA -fopenmp -fPIC -shared -o plugin.so``.

3. Set the environment variable ``RAJA_PLUGINS`` to be the path of your .so file. This can either be the path to its directory or to the shared object file itself. If the path is to a directory, it will attempt to load all .so files in that directory.

4. When you run your project, your plugins should work!

^^^^^^^^^^^
Interfacing with Plugins
^^^^^^^^^^^
The RAJA Plugin API allows for limited interfacing between a project and a plugin. There are, however a couple functions that allow for this to take place. ``init_plugins`` and ``finalize_plugins``. Using one of these will call the corresponding ``init`` or ``finalize`` function inside of *every* currently loaded plugin. It's worth noting that plugins don't require either an init or finalize function by default.

* ``RAJA::util::init_plugins();`` - Will call the ``init`` function of every currently loaded plugin.

* ``RAJA::util::init_plugins("path/to/plugins");`` - Does the same as the above call to init_plugins, but will also dynamically load plugins located at the path specified.

* ``RAJA::util::finalize_plugins();`` - Will call the ``finalize`` function of every currently loaded plugin. 


------------
Creating Plugins For RAJA
------------

Plugins take advantage of *polymorphism*, using ``RAJA::util::PluginStrategy`` as the parent and implementing the required functions for the API. An example implementation can be found at the bottom of this page.

^^^^^^^^^^^
Required Functions
^^^^^^^^^^^
The preLaunch and postLaunch functions are automatically called by RAJA before and after loop execution. This applies to RAJA's kernel and forall implementations.

* ``void preLaunch(PluginContext& p) {}`` - Will occur before kernel/forall execution.

* ``void postLaunch(PluginContext& p) {}`` - Will occur after kernel/forall execution.

* ``At least one method of loading the plugin, either statically or dynamically.``

^^^^^^^^^^^
Optional Functions
^^^^^^^^^^^
The init and finalize functions have standard implementations and thus are not needed in a user-made plugin. Init and finalize are never run by RAJA by default and are only run when the user makes a call to RAJA::util::init_plugin() or RAJA::util::finalize_plugin() respectively.

* ``void init(PluginOptions p) {}`` - runs on all plugins when the user makes a call to ``init_plugins``

* ``void finalize() {}`` - runs on all plugins when the user makes a call to ``finalize_plugins``

^^^^^^^^^^^^^^^^^
Static Loading
^^^^^^^^^^^^^^^^^
If the plugin is to be linked to a project at compile time, adding the following one-liner will add the plugin to the RAJA PluginRegistry and will be loaded every time the compiled executable is run. This requires the plugin to be linked either in an ``#include`` statement within the project, or linked via compiler commands.
::

  static RAJA::util::PluginRegistry::add<PluginName> P("Name", "Description");


^^^^^^^^^^^^^^^^^
Dynamic Loading
^^^^^^^^^^^^^^^^^
If the plugin is to be dynamically linked to a project during runtime, the RAJA Plugin API requires a few conditions to be met. The following must be true about the plugin, not necessarily of the project using it.

1. **The plugin must have following factory function.** This will return a pointer to an instance of your plugin, and thanks to the ``extern "C"``, a dynamically linked project will be able to access this function as well as the instance it returns.
::

  extern "C" RAJA::util::PluginStrategy *getPlugin ()
  {
    return new MyPluginName;
  }
  

2. **The plugin must be compiled to be a shared object with a .so extension.** A simple example containing required flags would be: ``g++ plugin.cpp -lRAJA -fopenmp -fPIC -shared -o plugin.so``. At the moment, RAJA will only attempt to load files with .so extensions. It's worth noting why these flags (or their equivalents) are important. ``-lRAJA -fopenmp`` are the standard flags for compiling the RAJA library. For the purposes of dynamic linking, ``-fPIC`` tells the compiler to produce *position independent code*, which is needed to prevent conflicts in the address space of the executable. ``-shared`` will let the compiler know that you want the resulting object file to be shared, removing the need for a *main* as well as giving dynamically linked executables access to functions flagged with ``extern "C"``.

3. **The** ``RAJA_PLUGINS`` **environment variable has been set**, or the user has made a call to ``RAJA::util::init_plugins("path");`` with a path specified to either a directory or a .so file. It's worth noting that these are not mutually exclusive, RAJA will look for plugins from the environment variable on program startup and new plugins may be loaded after that using ``init_plugins``.


^^^^^^^^^^^^^^^^^
Example Implementation
^^^^^^^^^^^^^^^^^

The following is an example plugin that simply will print out the number of times a kernel has been launched and has the ability to be loaded either statically or dynamically.

.. literalinclude:: ../../../../examples/plugin/counter-plugin.cpp
   :start-after: _plugin_example_start
   :end-before: _plugin_example_end
   :language: C++
