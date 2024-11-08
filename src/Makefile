# GNU C++ Compiler
CPP         = @g++      # C++ Compuler

CPPFLAGS    = -O3 -c

# The directory where the include files needed to create the library objects are
INC_DIR = ../include

# The directory where all object files are going to moved afer creation
LIB_DIR     = ../lib

# armadillo library
ARMA_INC	= $(CONDA_PREFIX)/include
EIGEN_INC 	= $(CONDA_PREFIX)/include/eigen3
ARMA_LIB	= $(CONDA_PREFIX)/lib -larmadillo

# linkable object files with class info
SCF_OBJS = scf_class.o

# Lennard Jones library containing the gaussian orbital class
SCF_LIB = libscfclass.a

scf_class.o:  $(INC_DIR)/gaus_overlap.h
	$(CPP) $(CPPFLAGS) scf_class.cpp \
	-I$(INC_DIR) \
	-I$(ARMA_INC) -L$(ARMA_LIB) \
	-I$(EIGEN_INC)

# The following target creates a static library 
libscfclass.a: $(SCF_OBJS)
	ar ru $(SCF_LIB) $(SCF_OBJS) 
	mv $(SCF_LIB) $(LIB_DIR)
	rm $(SCF_OBJS)

all: $(SCF_LIB)

cleanall:
	cd $(LIB_DIR); rm $(SCF_LIB)

