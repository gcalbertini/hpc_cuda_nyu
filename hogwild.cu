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

void calc_pred(double *train_x)
{

    // calculate the actual prediction using the gradients.
    // Calculate it in mini-batch gradients and then update
}

// May want to create namespaces such such that we can use things like GPU::calc_gradient and CPU::calc_pred calc_gradient