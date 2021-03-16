# Copyright 2013-2020 Lawrence Livermore National Security, LLC and other
# Spack Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (Apache-2.0 OR MIT)


from spack import *

import socket
import os

from os import environ as env
from os.path import join as pjoin

def cmake_cache_entry(name, value, comment=""):
    """Generate a string for a cmake cache variable"""

    return 'set(%s "%s" CACHE PATH "%s")\n\n' % (name,value,comment)


def cmake_cache_string(name, string, comment=""):
    """Generate a string for a cmake cache variable"""

    return 'set(%s "%s" CACHE STRING "%s")\n\n' % (name,string,comment)


def cmake_cache_option(name, boolean_value, comment=""):
    """Generate a string for a cmake configuration option"""

    value = "ON" if boolean_value else "OFF"
    return 'set(%s %s CACHE BOOL "%s")\n\n' % (name,value,comment)


def get_spec_path(spec, package_name, path_replacements = {}, use_bin = False) :
    """Extracts the prefix path for the given spack package
       path_replacements is a dictionary with string replacements for the path.
    """

    if not use_bin:
        path = spec[package_name].prefix
    else:
        path = spec[package_name].prefix.bin

    path = os.path.realpath(path)

    for key in path_replacements:
        path = path.replace(key, path_replacements[key])

    return path


class Chai(CMakePackage, CudaPackage):
    """
    Copy-hiding array interface for data migration between memory spaces
    """

    homepage = "https://github.com/LLNL/CHAI"
    git      = "https://github.com/LLNL/CHAI.git"

    version('develop', branch='develop', submodules='True')
    version('main', branch='main', submodules='True')
    version('2.1.1', tag='v2.1.1', submodules='True')
    version('2.1.0', tag='v2.1.0', submodules='True')
    version('2.0.0', tag='v2.0.0', submodules='True')
    version('1.2.0', tag='v1.2.0', submodules='True')
    version('1.1.0', tag='v1.1.0', submodules='True')
    version('1.0', tag='v1.0', submodules='True')

    variant('shared', default=True, description='Build Shared Libs')

    depends_on('cmake@3.8:', type='build')
    depends_on('umpire')

    depends_on('cmake@3.9:', type='build', when="+cuda")
    depends_on('umpire+cuda', when="+cuda")

    phases = ['hostconfig', 'cmake', 'build','install']

    def _get_sys_type(self, spec):
        sys_type = str(spec.architecture)
        # if on llnl systems, we can use the SYS_TYPE
        if "SYS_TYPE" in env:
            sys_type = env["SYS_TYPE"]
        return sys_type

    def _get_host_config_path(self, spec):
        var=''
        if '+cuda' in spec:
            var= '-'.join([var,'cuda'])

        host_config_path = "hc-%s-%s-%s%s-%s.cmake" % (socket.gethostname().rstrip('1234567890'),
                                               self._get_sys_type(spec),
                                               spec.compiler,
                                               var,
                                               spec.dag_hash())
        dest_dir = self.stage.source_path
        host_config_path = os.path.abspath(pjoin(dest_dir, host_config_path))
        return host_config_path

    def hostconfig(self, spec, prefix, py_site_pkgs_dir=None):
        """
        This method creates a 'host-config' file that specifies
        all of the options used to configure and build CHAI.

        For more details about 'host-config' files see:
            http://software.llnl.gov/conduit/building.html

        Note:
          The `py_site_pkgs_dir` arg exists to allow a package that
          subclasses this package provide a specific site packages
          dir when calling this function. `py_site_pkgs_dir` should
          be an absolute path or `None`.

          This is necessary because the spack `site_packages_dir`
          var will not exist in the base class. For more details
          on this issue see: https://github.com/spack/spack/issues/6261
        """

        #######################
        # Compiler Info
        #######################
        c_compiler = env["SPACK_CC"]
        cpp_compiler = env["SPACK_CXX"]

        # Even though we don't have fortran code in our project we sometimes
        # use the Fortran compiler to determine which libstdc++ to use
        f_compiler = ""
        if "SPACK_FC" in env.keys():
            # even if this is set, it may not exist
            # do one more sanity check
            if os.path.isfile(env["SPACK_FC"]):
                f_compiler = env["SPACK_FC"]

        #######################################################################
        # By directly fetching the names of the actual compilers we appear
        # to doing something evil here, but this is necessary to create a
        # 'host config' file that works outside of the spack install env.
        #######################################################################

        sys_type = self._get_sys_type(spec)

        ##############################################
        # Find and record what CMake is used
        ##############################################

        cmake_exe = spec['cmake'].command.path
        cmake_exe = os.path.realpath(cmake_exe)

        host_config_path = self._get_host_config_path(spec)
        cfg = open(host_config_path, "w")
        cfg.write("###################\n".format("#" * 60))
        cfg.write("# Generated host-config - Edit at own risk!\n")
        cfg.write("###################\n".format("#" * 60))
        cfg.write("# Copyright (c) 2020, Lawrence Livermore National Security, LLC and\n")
        cfg.write("# other CHAI Project Developers. See the top-level LICENSE file for\n")
        cfg.write("# details.\n")
        cfg.write("#\n")
        cfg.write("# SPDX-License-Identifier: (BSD-3-Clause) \n")
        cfg.write("###################\n\n".format("#" * 60))

        cfg.write("#------------------\n".format("-" * 60))
        cfg.write("# SYS_TYPE: {0}\n".format(sys_type))
        cfg.write("# Compiler Spec: {0}\n".format(spec.compiler))
        cfg.write("# CMake executable path: %s\n" % cmake_exe)
        cfg.write("#------------------\n\n".format("-" * 60))

        #######################
        # Compiler Settings
        #######################

        cfg.write("#------------------\n".format("-" * 60))
        cfg.write("# Compilers\n")
        cfg.write("#------------------\n\n".format("-" * 60))
        cfg.write(cmake_cache_entry("CMAKE_C_COMPILER", c_compiler))
        cfg.write(cmake_cache_entry("CMAKE_CXX_COMPILER", cpp_compiler))

        # use global spack compiler flags
        cflags = ' '.join(spec.compiler_flags['cflags'])
        if cflags:
            cfg.write(cmake_cache_entry("CMAKE_C_FLAGS", cflags))

        cxxflags = ' '.join(spec.compiler_flags['cxxflags'])
        if cxxflags:
            cfg.write(cmake_cache_entry("CMAKE_CXX_FLAGS", cxxflags))

        if ("gfortran" in f_compiler) and ("clang" in cpp_compiler):
            libdir = pjoin(os.path.dirname(
                           os.path.dirname(f_compiler)), "lib")
            flags = ""
            for _libpath in [libdir, libdir + "64"]:
                if os.path.exists(_libpath):
                    flags += " -Wl,-rpath,{0}".format(_libpath)
            description = ("Adds a missing libstdc++ rpath")
            if flags:
                cfg.write(cmake_cache_entry("BLT_EXE_LINKER_FLAGS", flags,
                                            description))

        if "+cuda" in spec:
            cfg.write("#------------------{0}\n".format("-" * 60))
            cfg.write("# Cuda\n")
            cfg.write("#------------------{0}\n\n".format("-" * 60))

            cfg.write(cmake_cache_option("ENABLE_CUDA", True))

            cudatoolkitdir = spec['cuda'].prefix
            cfg.write(cmake_cache_entry("CUDA_TOOLKIT_ROOT_DIR",
                                        cudatoolkitdir))
            cudacompiler = "${CUDA_TOOLKIT_ROOT_DIR}/bin/nvcc"
            cfg.write(cmake_cache_entry("CMAKE_CUDA_COMPILER",
                                        cudacompiler))

            if not spec.satisfies('cuda_arch=none'):
                cuda_arch = spec.variants['cuda_arch'].value
                cuda_arch = "sm_{0}".format(cuda_arch[0])
                flag = '-arch {0}'.format(cuda_arch)
                cfg.write(cmake_cache_string("CUDA_ARCH",cuda_arch))
                cfg.write(cmake_cache_string("CMAKE_CUDA_FLAGS", flag))

        else:
            cfg.write(cmake_cache_option("ENABLE_CUDA", False))

        # shared vs static libs
        cfg.write(cmake_cache_option("BUILD_SHARED_LIBS","+shared" in spec))

        umpire_conf_path = spec['umpire'].prefix + "/share/umpire/cmake"
        cfg.write(cmake_cache_entry("umpire_DIR",umpire_conf_path))

        #######################
        # Close and save
        #######################
        cfg.write("\n")
        cfg.close()

        print("OUT: host-config file {0}".format(host_config_path))

    def cmake_args(self):
        spec = self.spec
        host_config_path = self._get_host_config_path(spec)

        options = []
        options.extend(['-C', host_config_path])

        return options
