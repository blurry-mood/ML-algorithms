//
// Created by ayoub on 20. 12. 8..
//

#ifndef TP_ML_LOGISTIC_H
#define TP_ML_LOGISTIC_H

#include <malloc.h>
#include <stdlib.h>
#include <bits/stdc++.h>
#include "../Maths/Math_Functions.h"
#include "../Maths/Distributions.h"
#include "../Maths/Optimizer.h"


#define loop(i, a, b) for(int i=a;i<b;i++)

int m;
long double **x; // TODO: each column is a feature; x should contain 1 at last column;
long double *y; // TODO: this y must be either 0 or 1;
int dim;    // Dimension of w.


double sigmoid(double z) {
    return 1 / (1 + exp(-z));
}

long double predict(long double *w, long double *xx) {
    return sigmoid(dot_product(dim, w, xx));
}

long double costFunction(long double *w) {
    long  double l = 0;
    long  double tmp;
    loop(i, 0, m) {
        tmp = predict(w, x[i]);
        l -= y[i] * log(tmp) + (1 - y[i]) * log(1 - tmp);
    }
    return l / m;
}

long double generalisationError(long double *w) {
    long double l = 0;
    long double tmp;
    loop(i, m, 118) {
        tmp = predict(w, x[i]);
        l -= y[i] * log(tmp) + (1 - y[i]) * log(1 - tmp);
    }
    // TODO: 118
    return l / (118 - m);
}

long double *gradientCost(long double *w) {
    // No need to allocate the same array for multiple uses unless it changes it's size.
    static long double *g = (long double *) malloc(dim * sizeof(long double));
    double tmp;
    loop(j, 0, dim) g[j] = 0;
    loop(i, 0, m) {
        tmp = predict(w, x[i]);
        loop(j, 0, dim) g[j] += (tmp - y[i]) * x[i][j];
    }
    loop(j, 0, dim) g[j] /= m;
    return g;
}

long double *fitData() {
    // Initialize W
    long double *w = (long double *) malloc(dim * sizeof(long double));
    generateUniformly(dim, w);

    // Execute Gradient Descent Algorithm using logistic loss;
    gradientDescent(dim, w, &gradientCost, &costFunction, 10.e-10);
    // Print final loss, and optimal parameters;
    printf("\n Set of optimal Parameters: \t");
    loop(i, 0, dim) printf("%Lf  ", w[i]);
    printf("\n Final Loss: \t %.12Lf", costFunction(w));

    return w;
}

void read_csv(char* filename, int dimm) {
    FILE *file = fopen(filename, "r");
    if (file == NULL) exit(-1);
    dim = dimm;
    m = 118;
    x = (long double **) malloc(m * sizeof(long double *));
    y = (long double *) malloc(m * sizeof(long double));
    char line[100];
    // Now Read each observation;
    double tmp;
    int itmp;
    loop(i, 0, m) {
        // Allocate memo for observation;
        x[i] = (long double *) malloc(dim * sizeof(long double));

        // Reading the rest of the variables
        loop(j, 0, dim ) {
            // Read the comma here.
            fscanf(file, "%lf", &tmp);
            x[i][j] = tmp;
            fscanf(file, "%c", line);
        }
        // Now reading the label
        fscanf(file, "%i", &itmp);
        y[i] = itmp;
        fscanf(file, "%c", line);

    }
    m = 90;
    fclose(file);
}


#endif //TP_ML_LOGISTIC_H
