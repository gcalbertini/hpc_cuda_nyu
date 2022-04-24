#include <vector>
#include <random>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <math.h>
#include "utils.h"

// Assume equation of form y= a + bx + cx^2 + dx^3 + ....
// Squared L2 norm
// Loss for single example = (y* - y)^2
// Derivative of loss wrt coeff_i = 2(y* - y)*(i*coeffs[i]*predictor_values[i]^(i-1))

float difflinear(long numpredictors, double *coeffs, double *predictor_values, long desired_coeff, double pred, double train_y)
{
    if (desired_coeff == 0)
        return 2 * (pred - train_y);

    return 2 * (pred - train_y) * (desired_coeff * coeffs[desired_coeff] * pow(predictor_values[desired_coeff], desired_coeff - 1));
}

void calc_gradient(double *train_x, double *train_y, double *weights, double *constant, long batch_size, long numpredictors, double *pred)
{
    // Note: have type long and type double mixing but I believe both 8 bytes here so happen to be ok? Best to chanage evertyhing to double or long double? Assuming we compile in C++ as his examples showed with CUDA
    double *temp_x = (double *)(malloc(sizeof *temp_x * numpredictors)); // frees us from having to worry about changing the RHS of the expression if ever change the type of temp_x: https://stackoverflow.com/questions/605845/do-i-cast-the-result-of-malloc
    double *w_gradients = (double *)(malloc(sizeof *w_gradients * numpredictors));
    double b_gradient = 0;

    for (long i = 0; i < batch_size; i++)
    {
        for (long j = 0; j < numpredictors; j++)
        {
            temp_x[j] = train_x[i * numpredictors + j];
        }

        for (long k = 1; k <= numpredictors; k++)
        {
            w_gradients[k] += difflinear(numpredictors, weights, temp_x, k, pred[i], train_y[i]);
        }
        b_gradient += difflinear(numpredictors, weights, temp_x, 0, pred[i], train_y[i]);
    }
}

// decide if all double or all type long throughout?
double *train_x_C(const unsigned long batch_size, const unsigned long numpredictors)
{
    /* Generate a new random seed from system time - do this once in your constructor */
    srand(time(0));

    double *train_x = (double *)aligned_malloc(batch_size * numpredictors * sizeof(double));
    for (long i = 0; i < batch_size * numpredictors; i++)
        train_x[i] = 123.4;//drand48();

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

double *train_x_csv()
{
    std::ifstream f;
    std::vector<std::vector<double>> array; /* vector of vector<double>  */
    std::string line;                  /* string for line & value */
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

    double *train_x = (double *)aligned_malloc(ncols * nrows * sizeof(double));
    //std::cout << ncols << std::endl;
    long col = 0;
    long row = 0;
    // read lines
    while (std::getline(f, line))
    {
        lineStream.clear();
        lineStream.str(line);
        //std::cout << "row=" << nrows++
        //           << " lineStream.str() = " << lineStream.str() << std::endl;
        while (std::getline(lineStream, line, ','))
        {
            std::cout << "element=" << line << std::endl;
            train_x[col * nrows + row] = atof(line.c_str());
            col++;
        }
        row++;
    }

    return train_x;
}

int train_y_csv()
{
    std::ifstream f;
    std::string line, val; /* string for line & value */
    std::vector<double> v;
    long nrows = 0;

    f.open("generated_data/df_Y.csv"); /* open file with filename as argument */
    if (!f.is_open())
    { /* validate file open for reading */
        std::cerr << "error: file open failed!\n";
    }

    while (std::getline(f, line))
    {                              /* read each line */
        std::stringstream s(line); /* stringstream line */
        nrows++;
    }

    return 0;
}

int main()
{
    auto X = train_x_csv();
    std::cout << X[5];

    return 0;
}