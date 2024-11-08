# This creates executables for tests of the contracted gaussian orbital integration class 

# GNU C++ Compiler
CPP         = @g++      

# compiler flags -O3 optimized version -c create a library object
CPPFLAGS    = -O0 -o

# The directory where our gaussian orbital main library include files are stored.
INC_DIR = ../include

# The directory where lib object files are stored
LIB_DIR     = ../lib

# Executable programs
BIN_DIR     = ../bin

# armadillo library & include
 ARMA_INC    = $(CONDA_PREFIX)/include
 EIGEN_INC 	 = $(CONDA_PREFIX)/include/eigen3
 ARMA_LIB    = $(CONDA_PREFIX)/lib -larmadillo

# The next line contains a list of executable object files 
# created by this Makefile.
SCF_EXECUTABLES = pt1

pt1: $(INC_DIR)/gaus_overlap.h
	$(CPP) $(CPPFLAGS) pt1 pt1.cpp \
	-I$(INC_DIR) -L$(LIB_DIR) -lscfclass \
	-I$(ARMA_INC) -L$(ARMA_LIB) \
	-I$(EIGEN_INC)
	mv pt1 $(BIN_DIR)

run_pt1:
	cd $(BIN_DIR); ./pt1

all:
	make pt1
	make run_pt1

	
cleanall:
	cd $(BIN_DIR); rm $(SCF_EXECUTABLES) 

