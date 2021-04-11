#ifndef TP_ML_PERCEPTRON_H
#define TP_ML_PERCEPTRON_H

#include <malloc.h>
#include <stdlib.h>
#include <bits/stdc++.h>

#define MAX_ITER 10e5

#define loop(i, a, b) for(int i=a;i<b;i++)

int per_n, per_m;
double *per_w;

void per_randomW();

double per_predict(const double *x) {
    double s = per_w[per_m];
    loop(i, 0, per_m) s += per_w[i] * x[i];
    return s;
}

double per_loss(double **x, const int *y) {
    int l = 0;
    loop(i, 0, per_n) if (per_predict(x[i]) * y[i] < 0) l++;
    return (double) l / per_n;
}

void per_updateW(const double *x, int y) {
    per_w[per_m] += y;
    loop(i, 0, per_m) per_w[i] += x[i] * y;
}

void per_fit(double **x, int *y, int n, int m) {
    per_n = n;
    per_m = m;
    per_w = (double *) (malloc(sizeof(double) * (per_m + 1)));
    per_randomW();
    double loss = per_loss(x, y);
    printf("loss after %i : %lf\n", 0, loss);
    for (int iter=0; iter < MAX_ITER && loss != 0; iter++) {
        loop(i, 0, per_n) if (per_predict(x[i]) * y[i] < 0) per_updateW(x[i], y[i]);
        loss = per_loss(x, y);
        printf("%lf,", loss);
    }
    printf("\n");
    loop(i, 0, m + 1)printf("%lf \t", per_w[i]);
}

void per_randomW() {
    loop(i, 0, per_m + 1) per_w[i] = rand() % 10;
}

#endif
