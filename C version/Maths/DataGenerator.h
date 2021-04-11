//
// Created by ayoub on 20. 12. 7..
//

#ifndef TP_ML_DATAGENERATOR_H
#define TP_ML_DATAGENERATOR_H

#define loop(i, a, b) for(int i=a;i<b;i++)

#include "math.h"

long minimumSampleComplexity(int vc, double eps, double delta) {
    if (eps * delta <= 0 || eps <= 0) return -1;
    return 8 / eps * (vc * log(16 / eps) + log(2 / delta));
}

double **generateInputs(int n, int d, double amplitude) {
    // Note: x[:][d]=1 always;
    // Generated data is centered around 0;
    double **x = (double **) malloc(n * sizeof(double *));
    loop(i, 0, n) {
        x[i] = (double *) (malloc((d + 1) * sizeof(double)));
        loop(j, 0, d) {
            x[i][j] = rand() / (double) RAND_MAX * amplitude;
            if (rand() % 2 == 0) x[i][j] *= -1;
        }
        x[i][d] = 1;
    }
    return x;
}

int *labelInputs(int n, int d, double **x, int(*label)(int d, double *x)) {
    // Note is the number of cols in x;
    int *y = (int *) malloc(sizeof(int) * n);
    loop(i, 0, n) y[i] = label(d, x[i]);
    return y;
}


#endif //TP_ML_DATAGENERATOR_H
