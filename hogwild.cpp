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
#include <time.h>
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
        return -2 * (train_y - pred);

    return -2 * (train_y - pred) * pow(predictor_value, desired_coeff);
}

// calculate gradient for each batch
void calc_gradient(double *train_x, double *train_y, int batch_size, long numpredictors, double *weights, double *pred, double *w_gradients, double *b_gradient)
{

    for (int b = 0; b < batch_size; b++)
    {
        for (long k = 1; k < numpredictors; k++)
        {
            w_gradients[k] += difflinear(weights, train_x[b * numpredictors + k], k, pred[b], train_y[b]);
        }
        b_gradient[0] += difflinear(weights, train_x[0], 0, pred[b], train_y[b]);
    }
}

double calc_pred(int epoch, double *train_x, double *train_y, double *weights, int batch_size, double *w_gradients, double *b_gradient, long numpredictors, double learning_rate, double *pred)
{

    // calculate the actual prediction using the gradients.
    // Calculate it in mini-batch gradients and then update

    double loss = 0;
    // update weights based using the gradients
    // weights[0] = weights[0] - (b_gradient[0] / batch_size) * learning_rate;
    // for (long i = 0; i < numpredictors; i++)
    // {
    //     weights[i + 1] = weights[i + 1] - (w_gradients[i] / batch_size) * learning_rate;
    // }

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
        loss += pow(pred_reduction_sum - train_y[i], 2);
    }
    // printf("Epoch: %d loss: %f\n", epoch ,loss/batch_size);
    return loss;
}

void update_weights(double *weights, double *w_gradients, double *b_gradient, int batch_size, long numpredictors, double learning_rate)
{
    weights[0] = weights[0] - (b_gradient[0] / batch_size) * learning_rate;
    for (long i = 0; i < numpredictors; i++)
    {
        weights[i + 1] = weights[i + 1] - (w_gradients[i] / batch_size) * learning_rate;
    }
}


void train_x_csv(double * X, long nrows, long ncols)
{
    std::ifstream f;
    std::string line; /* string for line & value */
    std::stringstream lineStream;
  
    f.open("generated_data/df_X_train.csv"); /* open file with filename as argument */
    if (!f.is_open())
    { /* validate file open for reading */
        std::cerr << "error: file open failed!\n";
    }

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
            X[idx] = atof(line.c_str());
            idx++;
        }
    }
    f.close();
}

void train_y_csv(double * y, long nrows)
{
    std::ifstream f;
    std::string line; /* string for line & value */
    std::stringstream lineStream;
   
    f.open("generated_data/df_y_train.csv"); /* open file with filename as argument */
    if (!f.is_open())
    { /* validate file open for reading */
        std::cerr << "error: file open failed!\n";
    }

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
            y[idx] = atof(line.c_str());
            idx++;
        }
    }
    f.close();
}

int main(int argc, char *argv[])
{
    long numpredictors;
    int batch_size;
    long train_size;
    int num_epochs;
    double learning_rate = 0.05;

    if (argc != 5)
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

    
    double *X = (double *)malloc(sizeof(double)*numpredictors*train_size);
    double *y = (double *)malloc(sizeof(double)*train_size);
    train_x_csv(X, train_size, numpredictors);
    train_y_csv(y, train_size);

    /// Assume the above is implemented

    /// todo: initialize random weights and gradients
    srand(time(NULL));
    double *w_gradients = (double *)malloc(sizeof(double) * numpredictors);   // values after differentiation
    double *b_gradient = (double *)malloc(sizeof(double));                    // gradient for single bias term
    double *weights = (double *)malloc(sizeof(double) * (numpredictors + 1)); // weights minus learning rate

    // memset(weights, 0, numpredictors);
    // memset(w_gradients, 0, numpredictors);
    std::fill_n(weights, numpredictors + 1, 0.0);
    std::fill_n(w_gradients, numpredictors, 0.0);
    b_gradient[0] = 0;

    double *train_batch_x = (double *)malloc(sizeof(double) * batch_size * numpredictors);
    double *train_batch_y = (double *)malloc(sizeof(double) * batch_size);
    double *pred = (double *)malloc(sizeof(double) * batch_size);
    long start = 0;
    double loss_sum = 0;

    for (int epoch = 0; epoch < num_epochs; epoch++)
    {
        loss_sum = 0;
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
                //memset(w_gradients, 0, numpredictors);
                std::fill_n(w_gradients, numpredictors, 0.0);
                b_gradient[0] = 0;
                double loss = calc_pred(epoch + 1, train_batch_x, train_batch_y, weights, batch_size, w_gradients, b_gradient, numpredictors, learning_rate, pred);
                calc_gradient(train_batch_x, train_batch_y, batch_size, numpredictors, weights, pred, w_gradients, b_gradient);
                update_weights(weights, w_gradients, b_gradient, batch_size, numpredictors, learning_rate);
                start = 0;
                loss_sum += loss;
            }
        }
        printf("Epoch: %d Average loss: %f\n", epoch, loss_sum / ((train_size) / batch_size));
        learning_rate = learning_rate / 2;
    }

    free(train_batch_x);
    free(train_batch_y);
    free(pred);
    free(w_gradients);
    free(b_gradient);
    free(weights);
    free(X);
    free(y);

    return 0;
}
