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
#include <cuda.h>
#include <math.h>
#include "utils.h"

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

// Assume equation of form y= a + bx + cx^2 + dx^3 + ....
// Squared L2 norm
// Loss for single example = (y* - y)^2
// Derivative of loss wrt coeff_i = 2(y* - y)*(i*coeffs[i]*predictor_values[i]^(i-1))

double difflinear(double *weights, double predictor_value, long desired_coeff, double pred, double train_y)
{
    if (desired_coeff == 0)
        return 2 * (pred - train_y);

    return 2 * (pred - train_y) * (desired_coeff * weights[desired_coeff] * pow(predictor_value, desired_coeff - 1));
}

// calculate gradient for each batch
void calc_gradient(double *train_x, double *train_y, int batch_size, long numpredictors, double *weights, double *pred, double *w_gradients, double *b_gradient)
{

    for (int b = 0; b < batch_size; b++)
    {
        for (long k = 1; k <= numpredictors; k++)
        {
            w_gradients[k] += difflinear(weights, train_x[b * numpredictors + k], k, pred[b], train_y[b]);
        }
        b_gradient[0] += difflinear(weights, train_x[0], 0, pred[b], train_y[b]);
    }
}

void calc_pred(double *train_x, double *weights, int batch_size, double *w_gradients, double *b_gradient, long numpredictors, double learning_rate, double *pred)
{

    // calculate the actual prediction using the gradients.
    // Calculate it in mini-batch gradients and then update

    // update weights based using the gradients
    weights[0] = weights[0] - (b_gradient[0] / batch_size) * learning_rate;
    for (long i = 0; i < numpredictors; i++)
    {
        weights[i + 1] = weights[i + 1] - (w_gradients[i] / batch_size) * learning_rate;
    }

    // update prediction using the new weights
    // y = a + b*(x_0) + c*(x_1)^2 + d*(x_2)^3 + ....
    // a, b, c, d ... are weights, x_0, x_1, x_2 are predictor_values
    for (long i = 0; i < batch_size; i++)
    {
        double pred_reduction_sum = weights[0];
        for (long j = 0; j < numpredictors; j++)
        {
            pred_reduction_sum += weights[j + 1] * pow(train_x[i * numpredictors + j], j + 1);
        }
        pred[i] = pred_reduction_sum;
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

// decide if all double or all type long throughout?
double *train_x_C(const unsigned long batch_size, const unsigned long numpredictors)
{
    /* Generate a new random seed from system time - do this once in your constructor */
    srand(time(0));

    double *train_x = (double *)aligned_malloc(batch_size * numpredictors * sizeof(double));
    for (long i = 0; i < batch_size * numpredictors; i++)
        train_x[i] = 123.4; // drand48();

    return train_x;
}

double *train_y_C(const unsigned long batch_size, const unsigned long numpredictors, double *train_x)
{

    double *train_y = (double *)(aligned_malloc(sizeof *train_y * batch_size));
    // Define random generator with Gaussian distribution
    double b = 123.;
    const double mean = 0.0;
    const double stddev = 0.2;
    std::default_random_engine generator;
    std::normal_distribution<double> dist(mean, stddev);

    // Add Gaussian noise too
    for (long i = 0; i < batch_size; i++)
    {
        for (long j = 0; j < numpredictors; j++)
        {
            train_y[j] = train_x[i * numpredictors + j] + b + dist(generator);
        }
    }

    return train_y;
}

int main(int argc, char *argv[])
{
    long numpredictors;
    int batch_size;
    long train_size;
    int num_epochs;
    double learning_rate = 0.05;

    train_size = atol(argv[1]);
    numpredictors = atol(argv[2]);
    batch_size = atoi(argv[3]);
    num_epochs = atoi(argv[4]);

    //X, y comes from csv function now? Both now should be C array

    /// Assume the above is implemented

    /// todo: initialize random weights and gradients
    double *w_gradients = (double *)malloc(sizeof(double) * numpredictors);
    double *b_gradient = (double *)malloc(sizeof(double));
    double *weights = (double *)malloc(sizeof(double) * (numpredictors + 1));
    ///

    double *train_batch_x = (double *)malloc(sizeof(double) * batch_size * numpredictors);
    double *train_batch_y = (double *)malloc(sizeof(double) * batch_size);
    double *pred = (double *)malloc(sizeof(double) * batch_size);
    long start = 0;

    for (int epoch = 0; epoch < num_epochs; epoch++)
    {
        for (long i = 0; i < train_size; i++)
        {
            for (long j = 0; j < numpredictors; j++)
            {
                train_batch_x[start * numpredictors + j] = X[i * numpredictors + j];
            }
            train_batch_y[start] = y[i];
            start++;
            if ((i + 1) % batch_size == 0)
            {
                calc_pred(train_batch_x, weights, batch_size, w_gradients, b_gradient, numpredictors, learning_rate, pred);
                calc_gradient(train_batch_x, train_batch_y, batch_size, numpredictors, weights, pred, w_gradients, b_gradient);
                start = 0;
            }
        }
    }

    return 0;
}
