//
// Created by ayoub on 20. 11. 20..
//

#ifndef TP_ML_DISTRIBUTIONS_H
#define TP_ML_DISTRIBUTIONS_H

#include <stdlib.h>
#include <math.h>
#include <time.h>

#define loop(i, a, b) for(int i=a;i<b;i++)

long double uniform_number(){
    return rand()/(long double) RAND_MAX;
}

void shuffle(int n, double **x, int *y) {
    loop(k, 0, 100)loop(i, 0, n) {
            int j = rand() % n;
            double *xx = x[i];
            x[i] = x[j];
            x[j] = xx;
            int yy = y[i];
            y[i] = y[j];
            y[j] = yy;
        }
}

void generateUniformly(int dim, long double *w) {
    loop(i, 0, dim) w[i] = rand() / (double) RAND_MAX;
}

double sampleNormal() {
    double u = ((double) rand() / (RAND_MAX)) * 2 - 1;
    double v = ((double) rand() / (RAND_MAX)) * 2 - 1;
    double r = u * u + v * v;
    if (r == 0 || r > 1) return sampleNormal();
    double c = sqrt(-2 * log(r) / r);
    return u * c;
}

void generateUsingStandardNormal(int dim, long double *w) {
    loop(i, 0, dim) w[i] = sampleNormal();
}

void generateUsingBernouilli(int dim, long double *w,double p) {
    loop(i, 0, dim) w[i] = (rand() / (double) RAND_MAX < p) ? 1 : 0;
}

#endif //TP_ML_DISTRIBUTIONS_H
