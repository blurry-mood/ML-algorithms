#ifndef TP_ML_POCKET_H
#define TP_ML_POCKET_H


#include <malloc.h>
#include <stdlib.h>
#include <bits/stdc++.h>

#define loop(i, a, b) for(int i=a;i<b;i++)

int poc_n, poc_m;
double *poc_w;


void poc_randomW();

double poc_predict(const double *ws, const double *x) {
    double s = ws[poc_m];
    loop(i, 0, poc_m) s += ws[i] * x[i];
    return s;
}

double poc_loss(double *ws, double **x, const int *y) {
    int l = 0;
    loop(i, 0, poc_n) if (poc_predict(ws, x[i]) * y[i] < 0) l++;
    return (double) l / poc_n;
}

void poc_updateWs(double *ws, const double *x, int y) {
    ws[poc_m] += y;
    loop(i, 0, poc_m) ws[i] += x[i] * y;
}

void poc_updateW(const double *ws) {
    loop(i, 0, poc_m + 1) poc_w[i] = ws[i];
}

static std::vector<double> poc_fit(double **x, int *y, int nn, int mm, int iter) {
    std::vector<double> losses;
    poc_n = nn;
    poc_m = mm;
    poc_w = (double *) (malloc(sizeof(double) * (poc_m + 1)));
    double *ws = (double *) (malloc(sizeof(double) * (poc_m + 1)));
    poc_randomW();
    loop(i, 0, poc_m + 1) ws[i] = poc_w[i];
    double l = poc_loss(poc_w, x, y);
    losses.push_back(l);
    loop(t, 0, iter) {
        loop(i, 0, poc_n)if (poc_predict(ws, x[i]) * y[i] < 0) poc_updateWs(ws, x[i], y[i]);
        if (l > poc_loss(ws, x, y)) poc_updateW(ws);
        l = poc_loss(poc_w, x, y);
        losses.push_back(l);
    }
    free(ws);
    return losses;
}

void poc_randomW() {
    loop(i, 0, poc_m + 1) poc_w[i] = rand() % 10;
}

#endif //TP_ML_POCKET_H
