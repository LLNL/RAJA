#-----[Makefile]------
#$@ Contains the target file name
#$< Contains the first dependency name
#=====================

#----[Directories]----
oPath = obj
sPath = src
iPath = include
#====================

#---[Compiler and Settings]-----
compiler = g++
compilerFlags = -std=c++11 
paths = -I./$(iPath) -I./${pwd}
#======================

#----[Variable Names]-------
headers = $(wildcard $(iPath)/*.hpp)
sources = $(wildcard $(sPath)/*.cpp)
objects = $(subst $(sPath)/,$(oPath)/,$(sources:.cpp=.o))
#==========================

#----[Compilation]---------
reproducer:$(objects) $(headers) reproducer.cpp
	$(compiler) $(compilerFlags) -o reproducer $(objects) reproducer.cpp $(paths)

$(oPath)/%.o:$(sPath)/%.cpp $(subst $(sPath)/, $(iPath)/,$(<:.cpp=.hpp))
	@mkdir -p $(@D)
	$(compiler) $(compilerFlags) -o $@ -c $(paths) $<

#-----[Clean up]-------
clean:
	rm reproducer
	rm -rf $(oPath)/
