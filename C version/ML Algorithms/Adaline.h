#ifndef TP_ML_ADALINE_H
#define TP_ML_ADALINE_H

#include <malloc.h>
#include <stdlib.h>
#include <math.h>
#include "../Maths/Math_Functions.h"
#include "../Maths/Distributions.h"
#include "../Maths/Optimizer.h"

#define loop(i, a, b) for(int i=a;i<b;i++)

int m;
long double **x; // Note: each column is a feature; x should contain 1 'bias' at last column;
long double *y;
int dim;    // Dimension of w.

long double predict(long double *w, long double *xx) {
    return dot_product(dim, w, xx);
}

long double costFunction(long double *w) {
    long double l = 0;
    loop(i, 0, m)l += pow(y[i] - predict(w, x[i]), 2);
    return l / m;
}


long double *gradientCost(long double *w) {
    // No need to allocate the same array for multiple uses unless it changes it's size.
    static long double *g = (long double *) malloc(dim * sizeof(long double));
    long double tmp;
    loop(j, 0, dim) g[j] = 0;
    loop(i, 0, m) {
        tmp = -2 * (y[i] - predict(w, x[i]));
        loop(j, 0, dim) g[j] += tmp * x[i][j];
    }
    loop(j, 0, dim) g[j] /= m;
    return g;
}

long double *fitData() {
    // Initialize W
    long double *w = (long double *) malloc(dim * sizeof(long double));
    generateUniformly(dim, w);

    // Execute Gradient Descent Algorithm using logistic loss;
    gradientDescent(dim, w, &gradientCost, &costFunction, 10e-10);
    // Print final loss, and optimal parameters;
    printf("\n Set of optimal Parameters: \t");
    loop(i, 0, dim) printf("%.20Lf  ", w[i]);
    printf("\n Final Loss: \t %.12Lf", costFunction(w));

    return w;
}

void read_Cars( int dimm, int mm) {
    FILE *file = fopen("cars.csv", "r");
    if (file == NULL) exit(-1);
    dim = dimm;
    m = mm;
    x = (long double **) malloc(m * sizeof(long double *));
    y = (long double *) malloc(m * sizeof(long double));
    char line[100];
    fgets(line, 100, file);
    // Now Read each observation;
    double tmp;
    loop(i, 0, m) {
        // Allocate memo for observation;
        x[i] = (long double *) malloc(dim * sizeof(long double));
        x[i][0] = 1;
        // ignore the id
        fscanf(file, "%lf", &tmp);
        fscanf(file, "%c", line);
        // Read the speed
        fscanf(file, "%lf", &tmp);
        x[i][1] = tmp;
        fscanf(file, "%c", line);
        // Now reading the distance
        fscanf(file, "%lf", &tmp);
        y[i] = tmp;
        fscanf(file, "%c", line);

    }
    fclose(file);
}
void read_Nasa() {
    FILE *file = fopen("airfoil_self_noise.csv", "r");
    if (file == NULL) exit(-1);
    dim = 6;
    m = 1503;
    x = (type **) malloc(m * sizeof(type *));
    y = (type *) malloc(m * sizeof(type));
    char line[100];
    // Now Read each observation;
    double tmp;
    int itmp;
    loop(i, 0, m) {
        // Allocate memo for observation;
        x[i] = (type *) malloc(dim * sizeof(type));
        x[i][dim - 1] = 1;
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
    m = 1052;
}


#endif //TP_ML_ADALINE_H
