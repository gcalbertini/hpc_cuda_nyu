#include <algorithm>
#include <iostream>
#include <vector>
#include <random>
#include <iterator>
#include <iostream>
#include <fstream>
#include <sstream>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "utils.h"
#include <cuda.h>
#include <assert.h>

// Convenience function for checking CUDA runtime API results
// can be wrapped around any runtime API call. No-op in release builds.
inline
cudaError_t checkCuda(cudaError_t result)
{
#if defined(DEBUG) || defined(_DEBUG)
  if (result != cudaSuccess) {
    fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result));
    assert(result == cudaSuccess);
  }
#endif
  return result;
}

template <typename T>
void swap(T *a, T *b)
{
    T temp = *a;
    *a = *b;
    *b = temp;
}

// Shuffle x and y in the same way, x is n * k, y is n
// SGD requires random selection for mini batches. Random shuffle the whole
// train data and looping from the start serves the same purpose of random selection
template <typename T>
void shuffleXY(T x, T y, size_t n, size_t k)
{
    if (n == 0)
        return;
    srand((unsigned)time(NULL));
    for (size_t i = n - 1; i > 0; --i)
    {
        size_t j = rand() % (i + 1);
        swap(y + i, y + j);
        for (size_t kk = 0; kk < k; kk++)
        {
            swap(x + i * k + kk, x + j * k + kk);
        }
    }
}

double *train_x_csv()
{
    std::ifstream f;
    std::string line; /* string for line & value */
    long nrows = 0;
    long ncols = 0;

    f.open("generated_data/df_X.csv"); /* open file with filename as argument */
    if (!f.is_open())
    { /* validate file open for reading */
        std::cerr << "error: file open failed!\n";
    }

    std::stringstream lineStream;
    std::string lastline;
    while (std::getline(f, line))
    {
        lineStream.clear();
        lineStream.str(line);
        // std::cout << "row=" << nrows++
        //           << " lineStream.str() = " << lineStream.str() << std::endl;
        nrows++;
    }

    // just reads last line to count columns just by counting commas+1
    while (std::getline(lineStream, lastline, ','))
    {
        // std::cout << "cell=" << lastline << std::endl;
        ++ncols;
    }
    f.close();

    f.open("generated_data/df_X.csv"); /* open file with filename as argument */
    if (!f.is_open())
    { /* validate file open for reading */
        std::cerr << "error: file open failed!\n";
    }

    // std::cout << ncols << std::endl;
    double *train_x = (double *)aligned_malloc(ncols * nrows * sizeof(double));
    long idx = 0;
    // read lines
    while (std::getline(f, line))
    {
        lineStream.clear();
        lineStream.str(line);
        // std::cout << "row=" << row++
        //   << " lineStream.str() = " << lineStream.str() << std::endl;
        while (std::getline(lineStream, line, ','))
        {
            // std::cout << "element=" << line << std::endl;
            train_x[idx] = atof(line.c_str());
            idx++;
        }
    }
    f.close();

    return train_x;
}

double *train_y_csv()
{
    std::ifstream f;
    std::string line; /* string for line & value */
    long nrows = 0;

    f.open("generated_data/df_y.csv"); /* open file with filename as argument */
    if (!f.is_open())
    { /* validate file open for reading */
        std::cerr << "error: file open failed!\n";
    }

    std::stringstream lineStream;
    std::string lastline;
    while (std::getline(f, line))
    {
        lineStream.clear();
        lineStream.str(line);
        // std::cout << "row=" << nrows++
        //           << " lineStream.str() = " << lineStream.str() << std::endl;
        nrows++;
    }
    f.close();

    f.open("generated_data/df_y.csv"); /* open file with filename as argument */
    if (!f.is_open())
    { /* validate file open for reading */
        std::cerr << "error: file open failed!\n";
    }

    double *train_y = (double *)aligned_malloc(nrows * sizeof(double));
    long idx = 0;
    // read lines
    while (std::getline(f, line))
    {
        lineStream.clear();
        lineStream.str(line);
        // std::cout << "row=" << row++
        //   << " lineStream.str() = " << lineStream.str() << std::endl;
        while (std::getline(lineStream, line, ','))
        {
            // std::cout << "element=" << line << std::endl;
            train_y[idx] = atof(line.c_str());
            idx++;
        }
    }
    f.close();

    return train_y;
}

__global__
void hogwild_kernel(...) {
  // to do
}

// maybe we need reduction kernel???
// Reduction kernel
__global__ void reduction_kernel(double *res_d, double *res_out_d, long Nt) {
	__shared__ double sdata[REDUCTION_BLOCK_SIZE];
	// each thread loads one element from global to shared mem
	int tid = threadIdx.x;
	int i = blockIdx.x*blockDim.x + threadIdx.x;

	sdata[tid] = res_d[i];
	__syncthreads();

	//reversed looping
	for (int s=blockDim.x/2; s>0; s>>=1) {
		if (tid < s) {
			sdata[tid] += sdata[tid + s];
		}
		__syncthreads();
	}

	if (tid == 0) {
		res_out_d[blockIdx.x] = sdata[0];
	}
}

int main(int argc, char * argv[])
{
    long numpredictors;
    int batch_size;
    long train_size;
    int num_epochs;
    double learning_rate = 0.05;

    if(argc != 5)
    {
        fprintf(stderr, "usage: hogwild train_size numpredictors batch_size num_epochs\n");
        fprintf(stderr, "train_size = number of data points\n");
        fprintf(stderr, "numpredictors = number of predictors for each data point\n");
        fprintf(stderr, "batch_size = number of data points in each batch\n");
        fprintf(stderr, "num_epochs = number of epochs for training\n");
        exit(1);
    }

    train_size = atol(argv[1]);
    numpredictors = atol(argv[2]);
    batch_size = atoi(argv[3]);
    num_epochs = atoi(argv[4]);

    //X, y comes from csv function now? Both now should be C array

    /// Assume the above is implemented

    // Are w_gradients and b_gradient here necessary if we only need them on GPU?
    double *w_gradients = (double *)malloc(sizeof(double) * numpredictors);
    double *b_gradient = (double *)malloc(sizeof(double));
    double *weights = (double *)malloc(sizeof(double) * (numpredictors + 1));

    double *train_batch_x = (double *)malloc(sizeof(double) * batch_size * numpredictors);
    double *train_batch_y = (double *)malloc(sizeof(double) * batch_size);
    double *pred = (double *)malloc(sizeof(double) * batch_size);

    /// todo: initialize random weights and prediction

    double *w_gradients_d, *b_gradient_d, *weights_d, *train_batch_x_d, *train_batch_y_d, *pred_d;

    checkCuda(cudaMalloc((void**)&w_gradients_d, numpredictors*sizeof(double)));
    checkCuda(cudaMalloc((void**)&b_gradient_d, 1*sizeof(double)));
    checkCuda(cudaMalloc((void**)&weights_d, (numpredictors+1)*sizeof(double)));

    checkCuda(cudaMalloc((void**)&train_batch_x_d, batch_size*numpredictors*sizeof(double)));
    checkCuda(cudaMalloc((void**)&train_batch_y_d, batch_size*sizeof(double)));
    checkCuda(cudaMalloc((void**)&pred_d, batch_size*sizeof(double)));

    // assume weights and prediction are initialized
    checkCuda(cudaMemcpyAsync(weights_d, weights, (numpredictors+1)*sizeof(double), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpyAsync(pred_d, pred, batch_size*sizeof(double), cudaMemcpyHostToDevice));
    checkCuda(cudaDeviceSynchronize());

    long start = 0;

    for (int epoch = 0; epoch < num_epochs; epoch++)
    {   
        // shuffle and run from the start of the training set
        shuffleXY(X, y, train_size, numpredictors);
        for (long i = 0; i < train_size / batch_size; i++)
        {
            train_batch_x = ...;
            train_batch_y = ...;
            checkCuda(cudaMemcpyAsync(train_batch_x_d, train_batch_x, batch_size*numpredictors*sizeof(double), cudaMemcpyHostToDevice));
            checkCuda(cudaMemcpyAsync(train_batch_y_d, train_batch_y, batch_size*sizeof(double), cudaMemcpyHostToDevice));
            // run some kernels
            

        }
    }


    free(w_gradients);
    free(b_gradient);
    free(weights);
    free(train_batch_x);
    free(train_batch_y);
    free(pred);

    cudaFree(w_gradients_d);
    cudaFree(b_gradient_d);
    cudaFree(weights_d);
    cudaFree(train_batch_x_d);
    cudaFree(train_batch_y_d);
    cudaFree(pred_d);

    return 0;

}

