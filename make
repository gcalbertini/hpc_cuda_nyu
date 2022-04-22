find_package(PythonLibs 3 REQUIRED)
cmake_minimum_required(VERSION 3.8 FATAL_ERROR)

# Anaconda Python distribution is quite popular. Include path:
PYTHON_HOME = $(HOME)/anaconda3
PYTHON_INCLUDE = $(ANACONDA_HOME)/include 

# Uncomment to use Python 3 
# PYTHON_LIBRARIES := boost_python3 python3.5m
# PYTHON_INCLUDE := /usr/include/python3.5m \
#                 /usr/lib/python3.5/dist-packages/numpy/core/include

# PYTHON_LIB := /usr/lib
PYTHON_LIB = $(ANACONDA_HOME)/Lib

INC_DIRS = -I$(CUDA_PATH)/include -I. -I$(PYTHON_INCLUDE) -I.
LIB_DIRS = -L$(CUDA_PATH)/lib64 -lcudart -lcurand -L$(PYTHON_LIB) -lpython3.9

# CUDA architecture setting: going with all of them for now but change later: https://arnon.dk/matching-sm-architectures-arch-and-gencode-for-various-nvidia-cards/
# For CUDA < 6.0, comment the *_50 through *_61 lines for compatibility.
# For CUDA < 8.0, comment the *_60 and *_61 lines for compatibility.
# For CUDA < 9.0, comment the *_20 to *_37 lines for compatibility.
# For CUDA < 11.0, comment the *_20 to *_53 lines for compatibility.
CUDA_ARCH = -arch=sm_52 \ 
-gencode=arch=compute_52,code=sm_52 \ 
-gencode=arch=compute_60,code=sm_60 \ 
-gencode=arch=compute_61,code=sm_61 \ 
-gencode=arch=compute_70,code=sm_70 \ 
-gencode=arch=compute_75,code=sm_75 \
-gencode=arch=compute_80,code=sm_80 \
-gencode=arch=compute_80,code=compute_80 

CC=g++
NVCC=nvcc
CXX_FLAGS = -O3 -Wextra -std=c++11
CUDA_FLAGS= -std=c++11 -c -g $(CUDA_ARCH)

all: results

results: hogwild.o
    $(CC) $(CXX_FLAGS) hogwild.o results.cpp -o results $(INC_DIRS) $(LIB_DIRS)

hogwild.o: hogwild.cu hogwild.h 
    $(NVCC) $(CUDA_FLAGS) hogwild.cu -o hogwild.o -I -dlink

clean: 
    rm -rf *.o