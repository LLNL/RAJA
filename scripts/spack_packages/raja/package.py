# Copyright 2013-2020 Lawrence Livermore National Security, LLC and other
# Spack Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (Apache-2.0 OR MIT)


from spack import *

import socket
import os

from os import environ as env
from os.path import join as pjoin

import re

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


class Raja(CMakePackage, CudaPackage, ROCmPackage):
    """RAJA Performance Portability Abstractions for C++ HPC Applications."""

    homepage = "https://github.com/LLNL/RAJA"
    git      = "https://github.com/LLNL/RAJA.git"
    tags     = ['radiuss', 'e4s']

    maintainers = ['davidbeckingsale']

    version('develop', branch='develop', submodules='True')
    version('main',  branch='main',  submodules='True')
    version('0.14.1', tag='v0.14.1', submodules="True")
    version('0.14.0', tag='v0.14.0', submodules="True")
    version('0.13.0', tag='v0.13.0', submodules="True")
    version('0.12.1', tag='v0.12.1', submodules="True")
    version('0.12.0', tag='v0.12.0', submodules="True")
    version('0.11.0', tag='v0.11.0', submodules="True")
    version('0.10.1', tag='v0.10.1', submodules="True")
    version('0.10.0', tag='v0.10.0', submodules="True")
    version('0.9.0', tag='v0.9.0', submodules="True")
    version('0.8.0', tag='v0.8.0', submodules="True")
    version('0.7.0', tag='v0.7.0', submodules="True")
    version('0.6.0', tag='v0.6.0', submodules="True")
    version('0.5.3', tag='v0.5.3', submodules="True")
    version('0.5.2', tag='v0.5.2', submodules="True")
    version('0.5.1', tag='v0.5.1', submodules="True")
    version('0.5.0', tag='v0.5.0', submodules="True")
    version('0.4.1', tag='v0.4.1', submodules="True")
    version('0.4.0', tag='v0.4.0', submodules="True")

    variant('openmp', default=True, description='Build OpenMP backend')
    variant('shared', default=False, description='Build Shared Libs')
    variant('libcpp', default=False, description='Uses libc++ instead of libstdc++')
    variant('tests', default='basic', values=('none', 'basic', 'benchmarks'),
            multi=False, description='Tests to run')
    variant('desul', default=False, description='Build Desul Atomics backend')

    depends_on('cmake@3.9:', type='build')

    depends_on('blt@0.4.1', type='build', when='@main')
    depends_on('blt@0.4.1:', type='build')

    depends_on('camp')
    depends_on('camp@0.2.2')
    depends_on('camp+rocm', when='+rocm')
    depends_on('camp+openmp', when='+openmp')
    for val in ROCmPackage.amdgpu_targets:
        depends_on('camp amdgpu_target=%s' % val, when='amdgpu_target=%s' % val)

    depends_on('camp+cuda', when='+cuda')
    for sm_ in CudaPackage.cuda_arch_values:
        depends_on('camp cuda_arch={0}'.format(sm_),
                   when='cuda_arch={0}'.format(sm_))

    conflicts('+openmp', when='+rocm')

    phases = ['hostconfig', 'cmake', 'build', 'install']

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
        if '+libcpp' in spec:
            var='-'.join([var,'libcpp'])

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
        all of the options used to configure and build Umpire.

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
        cfg.write("# Copyright 2016-22, Lawrence Livermore National Security, LLC\n")
        cfg.write("# and RAJA project contributors. See the RAJA/LICENSE file\n")
        cfg.write("# for details.\n")
        cfg.write("#\n")
        cfg.write("# SPDX-License-Identifier: (BSD-3-Clause) \n")
        cfg.write("###################\n\n".format("#" * 60))

        cfg.write("#------------------\n".format("-" * 60))
        cfg.write("# SYS_TYPE: {0}\n".format(sys_type))
        cfg.write("# Compiler Spec: {0}\n".format(spec.compiler))
        cfg.write("# CMake executable path: %s\n" % cmake_exe)
        cfg.write("#------------------\n\n".format("-" * 60))

        cfg.write(cmake_cache_string("CMAKE_BUILD_TYPE", spec.variants['build_type'].value))

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
        if "+libcpp" in spec:
            cflags += ' '.join([cflags,"-DGTEST_HAS_CXXABI_H_=0"])
        if cflags:
            cfg.write(cmake_cache_entry("CMAKE_C_FLAGS", cflags))

        cxxflags = ' '.join(spec.compiler_flags['cxxflags'])
        if "+libcpp" in spec:
            cxxflags += ' '.join([cxxflags,"-stdlib=libc++ -DGTEST_HAS_CXXABI_H_=0"])
        if cxxflags:
            cfg.write(cmake_cache_entry("CMAKE_CXX_FLAGS", cxxflags))

        # TODO (bernede1@llnl.gov): Is this useful for RAJA?
        if ("gfortran" in f_compiler) and ("clang" in cpp_compiler):
            libdir = pjoin(os.path.dirname(
                           os.path.dirname(f_compiler)), "lib")
            flags = ""
            for _libpath in [libdir, libdir + "64"]:
                if os.path.exists(_libpath):
                    flags += " -Wl,-rpath,{0}".format(_libpath)
            description = ("Adds a missing libstdc++ rpath")
            #if flags:
            #    cfg.write(cmake_cache_string("BLT_EXE_LINKER_FLAGS", flags,
            #                                description))

        gcc_toolchain_regex = re.compile("--gcc-toolchain=(.*)")
        gcc_name_regex = re.compile(".*gcc-name.*")

        using_toolchain = list(filter(gcc_toolchain_regex.match, spec.compiler_flags['cxxflags']))
        if(using_toolchain):
          gcc_toolchain_path = gcc_toolchain_regex.match(using_toolchain[0])
        using_gcc_name = list(filter(gcc_name_regex.match, spec.compiler_flags['cxxflags']))
        compilers_using_toolchain = ["pgi", "xl", "icpc"]
        if any(compiler in cpp_compiler for compiler in compilers_using_toolchain):
            if using_toolchain or using_gcc_name:
                cfg.write(cmake_cache_entry("BLT_CMAKE_IMPLICIT_LINK_DIRECTORIES_EXCLUDE",
                "/usr/tce/packages/gcc/gcc-4.9.3/lib64;/usr/tce/packages/gcc/gcc-4.9.3/gnu/lib64/gcc/powerpc64le-unknown-linux-gnu/4.9.3;/usr/tce/packages/gcc/gcc-4.9.3/gnu/lib64;/usr/tce/packages/gcc/gcc-4.9.3/lib64/gcc/x86_64-unknown-linux-gnu/4.9.3"))

        compilers_using_cxx14 = ["intel-17", "intel-18", "xl"]
        if any(compiler in cpp_compiler for compiler in compilers_using_cxx14):
            cfg.write(cmake_cache_entry("BLT_CXX_STD", "c++14"))

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

            if ("xl" in cpp_compiler):
                cfg.write(cmake_cache_entry("CMAKE_CUDA_FLAGS", "-Xcompiler -O3 -Xcompiler -qxlcompatmacros -Xcompiler -qalias=noansi " + 
                                            "-Xcompiler -qsmp=omp -Xcompiler -qhot -Xcompiler -qnoeh -Xcompiler -qsuppress=1500-029 " +
                                            "-Xcompiler -qsuppress=1500-036 -Xcompiler -qsuppress=1500-030"))
                cuda_release_flags = "-O3"
                cuda_reldebinf_flags = "-O3 -g"
                cuda_debug_flags = "-O0 -g"

                cfg.write(cmake_cache_string("BLT_CXX_STD", "c++14"))
            elif ("gcc" in cpp_compiler):
                cuda_release_flags = "-O3 -Xcompiler -Ofast -Xcompiler -finline-functions -Xcompiler -finline-limit=20000"
                cuda_reldebinf_flags = "-O3 -g -Xcompiler -Ofast -Xcompiler -finline-functions -Xcompiler -finline-limit=20000"
                cuda_debug_flags = "-O0 -g -Xcompiler -O0 -Xcompiler -finline-functions -Xcompiler -finline-limit=20000"
            else:
                cuda_release_flags = "-O3 -Xcompiler -Ofast -Xcompiler -finline-functions"
                cuda_reldebinf_flags = "-O3 -g -Xcompiler -Ofast -Xcompiler -finline-functions"
                cuda_debug_flags = "-O0 -g -Xcompiler -O0 -Xcompiler -finline-functions"
   
            cfg.write(cmake_cache_string("CMAKE_CUDA_FLAGS_RELEASE", cuda_release_flags))
            cfg.write(cmake_cache_string("CMAKE_CUDA_FLAGS_RELWITHDEBINFO", cuda_reldebinf_flags))
            cfg.write(cmake_cache_string("CMAKE_CUDA_FLAGS_DEBUG", cuda_debug_flags))

            if not spec.satisfies('cuda_arch=none'):
                cuda_arch = spec.variants['cuda_arch'].value
                cfg.write(cmake_cache_string("CUDA_ARCH", 'sm_{0}'.format(cuda_arch[0])))

        else:
            cfg.write(cmake_cache_option("ENABLE_CUDA", False))

        if "+rocm" in spec:
            cfg.write("#------------------{0}\n".format("-" * 60))
            cfg.write("# HIP\n")
            cfg.write("#------------------{0}\n\n".format("-" * 60))

            cfg.write(cmake_cache_option("ENABLE_HIP", True))

            hip_root = spec['hip'].prefix
            rocm_root = hip_root + "/.."
            cfg.write(cmake_cache_entry("HIP_ROOT_DIR",
                                        hip_root))
            cfg.write(cmake_cache_entry("ROCM_ROOT_DIR",
                                        rocm_root))
            cfg.write(cmake_cache_entry("HIP_PATH",
                                        rocm_root + '/llvm/bin'))
            cfg.write(cmake_cache_entry("CMAKE_HIP_ARCHITECTURES", 'fx906'))

            hipcc_flags = ['--amdgpu-target=gfx906']
            if "+desul" in spec:
                hipcc_flags.append('-std=c++14')
            
            cfg.write(cmake_cache_entry("HIP_HIPCC_FLAGS", ';'.join(hipcc_flags)))

            #cfg.write(cmake_cache_entry("HIP_RUNTIME_INCLUDE_DIRS",
            #                            "{0}/include;{0}/../hsa/include".format(hip_root)))
            #hip_link_flags = "-Wl,--disable-new-dtags -L{0}/lib -L{0}/../lib64 -L{0}/../lib -Wl,-rpath,{0}/lib:{0}/../lib:{0}/../lib64 -lamdhip64 -lhsakmt -lhsa-runtime64".format(hip_root)
            if ('%gcc' in spec) or (using_toolchain):
                if ('%gcc' in spec):
                    gcc_bin = os.path.dirname(self.compiler.cxx)
                    gcc_prefix = join_path(gcc_bin, '..')
                else:
                    gcc_prefix = gcc_toolchain_path.group(1)
                cfg.write(cmake_cache_entry("HIP_CLANG_FLAGS",
                "--gcc-toolchain={0}".format(gcc_prefix))) 
                cfg.write(cmake_cache_entry("CMAKE_EXE_LINKER_FLAGS",
                " -Wl,-rpath {}/lib64".format(gcc_prefix)))
            #else:
            #    cfg.write(cmake_cache_entry("CMAKE_EXE_LINKER_FLAGS", hip_link_flags))

        else:
            cfg.write(cmake_cache_option("ENABLE_HIP", False))

        cfg.write("#------------------{0}\n".format("-" * 60))
        cfg.write("# Other\n")
        cfg.write("#------------------{0}\n\n".format("-" * 60))

        cfg.write(cmake_cache_string("RAJA_RANGE_ALIGN", "4"))
        cfg.write(cmake_cache_string("RAJA_RANGE_MIN_LENGTH", "32"))
        cfg.write(cmake_cache_string("RAJA_DATA_ALIGN", "64"))

        cfg.write(cmake_cache_option("RAJA_HOST_CONFIG_LOADED", True))

        # shared vs static libs
        cfg.write(cmake_cache_option("BUILD_SHARED_LIBS","+shared" in spec))
        cfg.write(cmake_cache_option("RAJA_ENABLE_OPENMP","+openmp" in spec))
        cfg.write(cmake_cache_option("RAJA_ENABLE_DESUL_ATOMICS","+desul" in spec))

        if "+desul" in spec:
            cfg.write(cmake_cache_string("BLT_CXX_STD","c++14"))
            if "+cuda" in spec:
                cfg.write(cmake_cache_string("CMAKE_CUDA_STANDARD", "14"))

        cfg.write(cmake_cache_option("ENABLE_BENCHMARKS", 'tests=benchmarks' in spec))
        cfg.write(cmake_cache_option("ENABLE_TESTS", not 'tests=none' in spec or self.run_tests))

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
