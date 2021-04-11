//
// Created by ayoub on 20. 12. 25..
//

#ifndef TP_ML_ANN_H
#define TP_ML_ANN_H

#include <stdlib.h>
#include <stdio.h>
#include "../Maths/Math_Functions.h"
#include "../Maths/Distributions.h"
#include<time.h>

#define loop(i, a, b) for(int i=a;i<b;i++)
#define bloop(i, a, b) for(int i=a;i>=b;i--)

typedef long double type;

int dim, n;             // dimension of x +1, & how many samples; shape(x)==(n,dim-1);
type **x, *y;            // resp. sample of input (doesn't contain the bias 1) and labels;
// Every ANN is defined by:
int n_layers, *layers;  // n_layers: how many layers including input layer & output layer;
type (**activations)(type); // Activation function for each hidden layer;
type (**derivativeActivation)(type);    // Derivative of each activation function;
type ***W, **Z, **A;    // resp. Weights, linear combination of input and weights, activation. W[layer][unit][weight];
type **dF, ***dFW, ***global_dFW;      // resp. derivative of final activation with respect to each activation of a unit, with respect to each weight.

void allocate_params() {
    // dim, n_layers, layers must be specified before this method's invocation;
    W = (type ***) malloc(sizeof(type **) * (n_layers - 1));
    A = (type **) malloc(sizeof(type *) * n_layers);      // index 0 for input layer;
    Z = (type **) malloc(sizeof(type *) * (n_layers - 1));
    dF = (type **) malloc(sizeof(type *) * (n_layers - 1));     // index 0 for 1st hidden layer; same shape as Z;
    dFW = (type ***) malloc(sizeof(type **) * (n_layers - 1));  // same shape as W;
    global_dFW = (type ***) malloc(sizeof(type **) * (n_layers - 1));  // same shape as W;
    srand(time(0));

    loop(i, 0, n_layers - 1) {
        // First, hidden units for every hidden layer, knowing that layers[0] is dim-1;
        W[i] = (type **) malloc(sizeof(type *) * layers[i + 1]);
        A[i] = (type *) malloc(sizeof(type) * (1 + layers[i])); // include the bias in A as the last row;
        Z[i] = (type *) malloc(sizeof(type) * layers[i + 1]);
        dF[i] = (type *) malloc(sizeof(type) * layers[i + 1]);
        dFW[i] = (type **) malloc(sizeof(type *) * layers[i + 1]);
        global_dFW[i] = (type **) malloc(sizeof(type *) * layers[i + 1]);
        A[i][layers[i]] = 1;
        loop(j, 0, layers[i + 1]) {
            // Then, weights associated for every hidden unit in previous layer; including bias also in the last 'column';
            W[i][j] = (type *) malloc(sizeof(type) * (layers[i] + 1));
            dFW[i][j] = (type *) malloc(sizeof(type) * (layers[i] + 1));
            global_dFW[i][j] = (type *) malloc(sizeof(type) * (layers[i] + 1));
            // Initialize the vector of parameters;
            loop(k, 0, layers[i] + 1) W[i][j][k] = uniform_number();
        }
    }
    // Note: layers[n_layers - 1] == 1;
    A[n_layers - 1] = (type *) malloc(sizeof(type) * (1 + layers[n_layers - 1]));
    A[n_layers - 1][layers[n_layers - 1]] = 1;
}

type forwardPropagation(type *xx) {
    // xx is the input to work on;
    // Initialize the first layer of A
    loop(j, 0, layers[0]) A[0][j] = xx[j];
    // go through hidden layers;
    loop(k, 0, n_layers - 1) {
        loop(j, 0, layers[k + 1]) Z[k][j] = dot_product(layers[k] + 1, A[k], W[k][j]);
        loop(j, 0, layers[k + 1]) A[k + 1][j] = activations[k](Z[k][j]);
    }

    // Return estimation;
    return A[n_layers - 1][0];
}

void backPropagation() {
    // We suppose you run forwardPropagation on an input xx;
    // Calculate the gradient with respect to each activation in all layers except for the input layer;
    dF[n_layers - 2][0] = 1;
    bloop(k, n_layers - 3, 0) {
        loop(i, 0, layers[k + 1]) {
            dF[k][i] = 0;
            loop(t, 0, layers[k + 2]) dF[k][i] +=
                                              dF[k + 1][t] * derivativeActivation[k + 1](Z[k + 1][t]) * W[k + 1][t][i];
        }
    }

    // Calculate the gradient with respect to each weight w[k][i][j] in all layers except for the input layer;
    loop(k, 0, n_layers - 1) {
        loop(i, 0, layers[k + 1]) {
            type tmp = derivativeActivation[k](Z[k][i]);
            loop(j, 0, layers[k] + 1) { // to include the bias;
                dFW[k][i][j] = dF[k][i] * tmp * A[k][j];
            }
        }
    }
}

type optimizeWeights(type step) {
    // Optimize the weights by observing all the dataset x,y;
    // Initialize the global_dFW with 0;

    loop(k, 0, n_layers - 1)loop(i, 0, layers[k + 1])loop(j, 0, layers[k] + 1)global_dFW[k][i][j] = 0;
    // sum the sub-gradients with respect to each observation x[i],y[i];
    type cost = 0;
    loop(t, 0, n) {
        type label = forwardPropagation(x[t]);
        backPropagation();
        cost += pow(label - y[t], 2);
        double tmp = 2 * (label - y[t]);
        loop(k, 0, n_layers - 1) {
            loop(i, 0, layers[k + 1]) {
                loop(j, 0, layers[k]) {
                    global_dFW[k][i][j] += tmp * dFW[k][i][j];
                }
            }
        }
    }
    // Now optimize the weights using the gradient descent method + calculate the infinity norm of the gradient;
    type norm = 0;
    loop(k, 0, n_layers - 1) {
        loop(i, 0, layers[k + 1]) {
            loop(j, 0, layers[k]) {
                W[k][i][j] = W[k][i][j] - step * global_dFW[k][i][j] / n;
                norm = max(norm, abs(global_dFW[k][i][j] / n));
            }
        }
    }
    printf("Gradient Norm: %.20Lf, \t\t Cost Function:%.20Lf\n", norm, cost / n);
    return norm;
}

void fitData() {
    // Here, we suppose that all necessary informations have been provided;
    allocate_params();

    double norm = 100;
    while (norm > 65) {
        norm = optimizeWeights(5*1e-9);
    }
}

type generalizationError() {
    type cost = 0;
    loop(t, n, 1503) {
        type label = forwardPropagation(x[t]);
        cost += pow(label - y[t], 2);
    }
    return cost / (1503 - n);
}

void read_csv() {
    FILE *file = fopen("airfoil_self_noise.csv", "r");
    if (file == NULL) exit(-1);
    dim = 6;
    n = 1503;
    x = (type **) malloc(n * sizeof(type *));
    y = (type *) malloc(n * sizeof(type));
    char line[100];
    // Now Read each observation;
    double tmp;
    int itmp;
    loop(i, 0, n) {
        // Allocate memo for observation;
        x[i] = (type *) malloc((dim - 1) * sizeof(type));

        // Reading the rest of the variables
        loop(j, 0, dim - 1) {
            // Read the comma here.
            fscanf(file, "%lf", &tmp);
            x[i][j] = tmp;
            fscanf(file, "%c", line);
        }
        // Now reading the label
        fscanf(file, "%lf", &tmp);
        y[i] = tmp;
        fscanf(file, "%c", line);

    }
    fclose(file);
    n = 1052;
}

type sigmoid(type z) {
    return 1 / (1 + exp(-z));
}

type identity(type z) {
    return z;
}

type derIdentity(type z) {
    return 1;
}

type derSigmoid(type z) {
    type tmp = exp(-z);
    return tmp / pow(1 + tmp, 2);
}

#endif //TP_ML_ANN_H