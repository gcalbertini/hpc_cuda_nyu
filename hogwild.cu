#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <math.h>
#include "utils.h"

#include <algorithm>
#include <iostream>
#include <random>
#include <iterator>

template <typename T> void swap(T* a, T* b) {
    T temp = *a;
    *a = *b;
    *b = temp;
}

// Shuffle x and y in the same way
// SGD requires random selection for mini batches. Random shuffle the whole
// train data and looping from the start serves the same purpose of random selection
template <typename T> void shuffleXY(T x, T y, size_t n, size_t k) {
    if (n == 0)
        return;
    srand((unsigned)time(NULL));
    for (size_t i = n - 1; i > 0; --i) {
        size_t j = rand() % (i + 1);
        swap(arr + i, arr + j);
        for (size_t kk = 0; kk < k; kk++) {
            swap(arr1 + i*k+kk, arr1 + j*k+kk);
        }
    }
}


// Assume equation of form y= a + bx + cx^2 + dx^3 + ....
// Squared L2 norm
// Loss for single example = (y* - y)^2
// Derivative of loss wrt coeff_i = 2(y* - y)*(i*coeffs[i]*predictor_values[i]^(i-1))

float difflinear(long numpredictors, double* coeffs, double* predictor_values, long desired_coeff, double pred, double train_y){
    if desired_coeff == 0
        return 2*(pred - train_y)

    return 2*(pred - train_y)*(desired_coeff*coeffs[desired_coeff]*pow(predictor_values[desired_coeff], desired_coeff-1))

}


void calc_gradient(double* train_x, double* train_y, double* weights, double* constant, long batch_size, long numpredictors, double* pred){

    double* temp_x = malloc((numpredictors)*size_of(double));
    double* w_gradients = malloc(numpredictors*size_of(double));
    double b_gradient = 0;

    for(long i = 0; i<batch_size; i++){
        for(long j = 0; j<numpredictors; j++){
            temp_x[j] = train_x[i*numpredictors+j];
        }

        for(long k = 1 ; k<=numpredictors; k++){
            w_gradients[k] += difflinear(numpredictors, weights, temp_x, k, pred[i], train_y[i]);
        }
        b_gradient += difflinear(numpredictors, weights, temp_x, 0, pred[i], train_y[i]);
    }


}

void calc_pred(double* train_x, double* weights, double* batch_size, double* w_gradients, double* b_gradient, long numpredictors, double learning_rate, double* pred){

    //calculate the actual prediction using the gradients.
    //Calculate it in mini-batch gradients and then update

    // update weights based using the gradients
    weights[0] = weights[0] - b_gradient * learning_rate;
    for (long i = 0; i<numpredictors; i++) {
        weights[i+1] = weights[i+1] - w_gradients[i] * learning_rate;
    }

    // update prediction using the new weights
    // y = a + b*(x_0) + c*(x_1)^2 + d*(x_2)^3 + ....
    // a, b, c, d ... are weights, x_0, x_1, x_2 are predictor_values
    for (long i = 0; i<batch_size; i++) {
        double pred_reduction_sum = weights[0];
        for (long j = 0; j<numpredictors; j++) {
            pred_reduction_sum += weights[j+1] * pow(train_x[i*numpredictors+j], j+1);
        }
        pred[i] = pred_reduction_sum;
    }
}
