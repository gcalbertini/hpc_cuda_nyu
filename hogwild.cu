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

#define threadsperblock 1000

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
void hogwild_kernel(int num_epochs, long train_size, long numpredictors, int batch_size, double learning_rate ,double *X, double *y, double *weights, double *w_gradients, double *pred) {
    double b_gradient = 0;

    long idx = blockIdx.x * blockDim.x + threadIdx.x;

    for (int epoch = 0; epoch < num_epochs; epoch++)
    {
        long start = batch_size*idx;
        long pred_start_index = idx*batch_size;
        for (long i = start; i < start + batch_size; i++)
        {

            weights[0] = weights[0] - (b_gradient / batch_size) * learning_rate;
            for (long k = 0; k < numpredictors; k++)
            {
                weights[k + 1] = weights[k + 1] - (w_gradients[idx*numpredictors+k] / batch_size) * learning_rate;
            }

            // update prediction using the new weights
            // y = a + b*(x_0) + c*(x_1)^2 + d*(x_2)^3 + ....
            // a, b, c, d ... are weights, x_0, x_1, x_2 are predictor_values
                
            double pred_reduction_sum = weights[0];
            for (long j = 0; j < numpredictors; j++)
            {
                pred_reduction_sum += weights[j + 1] * pow(X[i * numpredictors + j], j + 1);
            }
            pred[pred_start_index+i-start] = pred_reduction_sum;
            // loss += pow(pred_reduction_sum - train_y[i], 2);
                
        }

            
        for (long i = start; i < start + batch_size; i++)
        {
            for (long k = 1; k <= numpredictors; k++)
            {
                w_gradients[idx*numpredictors+k] += 2 * (pred[pred_start_index+i-start] - y[i]) * (k * weights[k] * pow(X[i * numpredictors + k], k - 1));
            
            }
            b_gradient += 2 * (pred[pred_start_index+i-start] - y[i]);
        }
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

    int numblocks;
    int total_threads = train_size / batch_size;

    if (train_size % batch_size != 0) {
        total_threads += 1;
    }
 
    if( (total_threads % threadsperblock) == 0 )
	    numblocks = total_threads/ threadsperblock;
    else 
      	numblocks = (total_threads/threadsperblock)>0? (total_threads/threadsperblock)+1:1 ;

    //X, y comes from csv function now? Both now should be C array
    double *X = train_x_csv();
    double *y = train_y_csv();
    // shuffleXY(X,y,train_size,numpredictors)

   double *weights = (double *)malloc(sizeof(double) * (numpredictors + 1));
    memset(weights, 0, numpredictors);

    double *weights_d, *w_gradients_d, *pred_d, *X_d, *y_d ;

    size_t wg_size = size_t(total_threads*numpredictors) * sizeof(double);
    size_t pred_size = size_t(total_threads*batch_size) * sizeof(double);

    checkCuda(cudaMalloc((void**)&w_gradients_d, wg_size));
    checkCuda(cudaMemset(w_gradients_d, 0, wg_size));
    checkCuda(cudaMalloc((void**)&pred_d, pred_size));
    // checkCuda(cudaMemset(pred_d, 0, pred_size));

    checkCuda(cudaMalloc((void**)&weights_d, (numpredictors+1)*sizeof(double)));
    checkCuda(cudaMalloc((void**)&X_d, train_size*(numpredictors)*sizeof(double)));
    checkCuda(cudaMalloc((void**)&y_d, train_size*sizeof(double)));

    // assume weights and prediction are initialized
    checkCuda(cudaMemcpyAsync(weights_d, weights, (numpredictors+1)*sizeof(double), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpyAsync(X_d, X, train_size*(numpredictors)*sizeof(double), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpyAsync(y_d, y, train_size*sizeof(double), cudaMemcpyHostToDevice));
    checkCuda(cudaDeviceSynchronize());

    printf("GPU: %d blocks of %d threads each\n", numblocks, threadsperblock); 

    hogwild_kernel<<<numblocks , threadsperblock>>>(num_epochs, train_size, numpredictors, batch_size, learning_rate, X_d, y_d, weights_d, w_gradients_d, pred_d);


    free(weights);
    //free(pred);

    cudaFree(weights_d);
    //cudaFree(pred_d);

    return 0;

}

