##
## File defining macro variables that specify RAJA behavior.
##
## This should be included in the makefile to build code that uses RAJA
## or replace with an equivalent substitute to customize behavior.
## 
## \author  Rich Hornung, Center for Applied Scientific Computing, LLNL
## \author  Jeff Keasler, Applications, Simulations And Quality, LLNL
##

##
## RAJA uses some STL features internally.  To turn them off,
## unset the variable below:
##
## -DRAJA_USE_STL
##
#RAJA_USING_STL   = -DRAJA_USE_STL
RAJA_USING_STL   = 

##
## Available options for RAJA scalar floating point types are:
##
## -DRAJA_USE_DOUBLE
## -DRAJA_USE_FLOAT
##
## Exactly one must be defined!!!
##
RAJA_FP_TYPE	= -DRAJA_USE_DOUBLE
#RAJA_FP_TYPE	= -DRAJA_USE_FLOAT

##
## Available options for RAJA floating point pointer types are:
##
## -DRAJA_USE_BARE_PTR
## -DRAJA_USE_RESTRICT_PTR
## -DRAJA_USE_RESTRICT_ALIGNED_PTR
## -DRAJA_USE_PTR_CLASS
##
## Exactly one must be defined!!!
##
#RAJA_FPPTR_TYPE	= -DRAJA_USE_BARE_PTR
RAJA_FPPTR_TYPE	= -DRAJA_USE_RESTRICT_PTR
#RAJA_FPPTR_TYPE	= -DRAJA_USE_RESTRICT_ALIGNED_PTR
#RAJA_FPPTR_TYPE	= -DRAJA_USE_PTR_CLASS

##
## To turn on RAJA fault tolerance functionality, add the following:
##
## -DRAJA_USE_FT
##
#RAJA_FT_OPT	= -DRAJA_USE_FT -DRAJA_REPORT_FT
#RAJA_FT_OPT	= -DRAJA_USE_FT
RAJA_FT_OPT	=


RAJA_RULES	= $(RAJA_USING_STL) $(RAJA_FP_TYPE) $(RAJA_FPPTR_TYPE) $(RAJA_FT_OPT)
