find_program(SPHINX_EXECUTABLE
        NAMES sphinx-build sphinx-build2
        DOC "Path to sphinx-build executable")

# Handle REQUIRED and QUIET arguments
# this will also set SPHINX_FOUND to true if SPHINX_EXECUTABLE exists
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(Sphinx
        "Failed to locate sphinx-build executable"
        SPHINX_EXECUTABLE)