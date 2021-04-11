#ifndef TP_ML_LOGISTICREGRESSION_H
#define TP_ML_LOGISTICREGRESSION_H

#include <stdio.h>
#include "../Maths/Optimizer.h"
#include "../Maths/Distributions.h"

int m;
long double **x; // TODO: each column is a feature; x should contain 1 at last column;
long double *y;     // this y must be either -1 or 1;
int dim;    // Dimension of w.


long double costFunction(long double *w) {
    long double l = 0;
    loop(i, 0, m) l += log(1 + exp(-y[i] * dot_product(dim, w, x[i])));
    return l / m;
}

long double *gradientCost(long double *w) {
    // No need to allocate the same array for multiple uses unless it changes it's size.
    static long double *g = (long double *) malloc(dim * sizeof(long double));
    long double tmp;
    loop(j, 0, dim) g[j] = 0;
    loop(i, 0, m) {
        tmp = -1 / (1 + exp(y[i] * dot_product(dim, w, x[i])));
        loop(j, 0, dim) g[j] += tmp * x[i][j] * y[i];
    }
    loop(j, 0, dim) g[j] /= m;
    return g;
}

long double predict(long double *w, long double *xx) {
    return 1 / (1 + exp(-dot_product(dim,w,xx)));
}

long double *fitData() {
    // Initialize W
    long double *w = (long double *) calloc(dim, sizeof(long double));
     generateUniformly(dim,w);
//     generateUsingStandardNormal(dim,w);
//    generateUsingBernouilli(dim, w,.4);

    // Execute Gradient Descent Algorithm using logistic loss;
    gradientDescent(dim, w, &gradientCost, &costFunction, 10.e-5);
    // Print final loss, and optimal parameters;
    printf("\n Set of optimal Parameters: \t");
    loop(i, 0, dim) printf("%Lf  ", w[i]);
    printf("\n Final Loss: \t %Lf", costFunction(w));

    return w;
}

void readcsv(int dimm) {
    FILE *file = fopen("binary.csv", "r");
    if (file == NULL) exit(-1);
    // The file contains 401 rows including the header; and 4 variables, the first one is the label;
    dim = dimm;
    m = 400;
    x = (long double **) malloc(m * sizeof(long double *));
    y = (long double *) malloc(m * sizeof(long double));
    char line[1024];
    // Read the header:
    fgets(line, sizeof(line), file);
    // Now Read each observation;
    float tmp;
    int itmp;
    loop(i, 0, m) {
        // Allocate memo for the whole observation;
        x[i] = (long double *) malloc(4 * sizeof(long double));
        x[i][0] = 1;      // Always
        // Now reading the label
        fscanf(file, "%i", &itmp);
        y[i] = 2 * itmp - 1;
        // Reading the rest of the variables
        loop(j, 1, 4) {
            // Read the comma here.
            fscanf(file, "%c", line);
            fscanf(file, "%f", &tmp);
            x[i][j] = tmp;
        }
        // read \n
        fscanf(file, "%c", line);
    }


    fclose(file);
}

#endif //TP_ML_LOGISTICREGRESSION_H
