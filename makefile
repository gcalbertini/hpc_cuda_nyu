# https://stackoverflow.com/questions/9421108/how-can-i-compile-cuda-code-then-link-it-to-a-c-project
all: program
program: hogwild.o
	g++ -o program -I/Users/gbert/anaconda3/include/ -I. -L/usr/local/cuda/lib64 -lcuda -lcudart -L/Users/gbert/anaconda3/Lib results.cpp hogwild.o 
hogwild.o:
	nvcc -c -arch=sm_62 hogwild.cu 
clean: rm -rf *.o program

# If you can somehow link in Python.h you win; try python3-config --ldflags or --cflags, locate Python.h