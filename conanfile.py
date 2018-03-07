from conans import ConanFile, CMake, tools


class RajaConan(ConanFile):
    name = "RAJA"
    version = "develop"
    license = "BSD 3-clause"
    url = "https://github.com/LLNL/RAJA"
    description = "Parallel programming performance portability layer written in C++"
    settings = "os", "compiler", "build_type", "arch"
    options = {"shared": [True, False], "openmp": [True, False]}
    default_options = "shared=False", "openmp=False"
    generators = "cmake"

    def source(self):
        self.run("git clone https://github.com/LLNL/RAJA.git")
        self.run("cd RAJA && git submodule init && git submodule update && git checkout develop")
        tools.replace_in_file("RAJA/CMakeLists.txt", "VERSION ${RAJA_LOADED})",
                '''VERSION ${RAJA_LOADED})
                include(${CMAKE_BINARY_DIR}/conanbuildinfo.cmake)
                conan_basic_setup()''')

    def build(self):
        cmake = CMake(self)
        cmake.verbose = True

        cmake.configure(args=['-DENABLE_OPENMP=Off'], source_folder="RAJA")

        # self.run('cmake %s/hello %s' % (self.source_folder, cmake.command_line))

        cmake.build()

    def package(self):
        self.copy("*.h", dst="include", src="hello")
        self.copy("*.dll", dst="bin", keep_path=False)
        self.copy("*.so", dst="lib", keep_path=False)
        self.copy("*.dylib", dst="lib", keep_path=False)
        self.copy("*.a", dst="lib", keep_path=False)

    def package_info(self):
        self.cpp_info.libs = ["RAJA"]
