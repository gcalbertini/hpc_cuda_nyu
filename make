#Anaconda Python distribution is quite popular. Include path:
PYTHON_HOME = $(HOME)/anaconda3
PYTHON_INCLUDE = $(ANACONDA_HOME)/include 
# Uncomment to use Python 3 
# PYTHON_LIBRARIES := boost_python3 python3.5m
# PYTHON_INCLUDE := /usr/include/python3.5m \
#                 /usr/lib/python3.5/dist-packages/numpy/core/include
# PYTHON_LIB := /usr/lib
PYTHON_LIB = $(ANACONDA_HOME)/Lib
INC_DIRS = -I$(CUDA_PATH)/include -I. -I$(PYTHON_INCLUDE) -I.
LIB_DIRS = -L$(CUDA_PATH)/lib64 -lcudart -lcurand -L$(PYTHON_LIB) -lpython3.9 -L"\Users\gbert\anaconda3\include\Python.h"
# CUDA architecture setting: going with all of them for now but change later: https://arnon.dk/matching-sm-architectures-arch-and-gencode-for-various-nvidia-cards/
# For CUDA < 6.0, comment the *_50 through *_61 lines for compatibility.
# For CUDA < 8.0, comment the *_60 and *_61 lines for compatibility.
# For CUDA < 9.0, comment the *_20 to *_37 lines for compatibility.
# For CUDA < 11.0, comment the *_20 to *_53 lines for compatibility.
CUDA_ARCH = -arch=sm_61
CXX=g++
NVCC=nvcc
CXX_FLAGS = -O3 -Wextra -std=c++11
CUDA_FLAGS= -c -g $(CUDA_ARCH)
all: results
results: hogwild.o
	$(CXX) $(CXX_FLAGS) hogwild.o results.cpp -o results $(INC_DIRS) $(LIB_DIRS)
hogwild.o: hogwild.cu
	$(NVCC) $(CUDA_FLAGS) hogwild.cu -o hogwild.o -I -dlink
clean: 
	rm -rf *.o