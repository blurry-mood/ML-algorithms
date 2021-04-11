#include <math.h>
#include <stdio.h>
#include "Math_Functions.h"

#ifndef TP_ML_OPTIMIZER_H
#define TP_ML_OPTIMIZER_H


///////////////////////////////////////////////////////



//////////////////////////////////////// Calculation of phi function and its first and second derivatives;
long double phi(int dim, long double alpha, long double *gradW, long double *w, long double (*loss)(long double *)) {
    return loss(diff_vector(dim, w, scalar_vector(dim, alpha, gradW)));
}

long double
phiPr(int dim, long double alpha, long double *w, long double *gradW, long double *(*gradientCost)(long double *)) {
    static long double *grad = (long double *) malloc(dim * sizeof(long double));
    // Note: calculating the gradient from gradientCost function updates the previous value of the gradient; store it.
    loop(i, 0, dim) grad[i] = gradW[i];
    long double res = -dot_product(dim, gradW, gradientCost(diff_vector(dim, w, scalar_vector(dim, alpha, gradW))));
    loop(i, 0, dim) gradW[i] = grad[i];
    return res;
}

long double
phiSec(int dim, long double alpha, long double *w, long double *gradW, long double **(*hessienCost)(long double *)) {
    static long double **hess = (long double **) malloc(dim * sizeof(long double *));
    long double **hessien = hessienCost(diff_vector(dim, w, scalar_vector(dim, alpha, gradW)));
    // Note: calculating the hessien from hessienCost function updates the previous value of the hessien; store it.
    loop(i, 0, dim) loop(j, 0, dim) hess[i][j] = hessien[i][j];
    long double res = dot_product(dim, gradW, mat_vector(dim, hessien, gradW));
    loop(i, 0, dim) loop(j, 0, dim) hessien[i][j] = hess[i][j];
    return res;
}

//////////////////////////////////////// Calculation of optimal/approximated step;

long double armijo(int dim, long double *gradW, long double *w, long double (*loss)(long double *)) {
    // Armijo Hyperparameters.
    long double eps = .5 + rand() / (long double) RAND_MAX / 2;
    long double eta = 1.05 + rand() / (long double) RAND_MAX * 10;

    long double alpha = 10e-10;
    long double phiPrZero = -dot_product(dim, gradW, gradW) * eps;
    long double phiZero = loss(w);
    while (phi(dim, alpha, gradW, w, loss) > alpha * phiPrZero * eps + phiZero)
        alpha /= eta;
    while (phi(dim, alpha, gradW, w, loss) <= alpha * phiPrZero + phiZero)
        alpha *= eta;
    return alpha / eta;
}

long double regula_falsi(int dim, long double *gradW, long double *w, long double *(*gradientCost)(long double *)) {
    long double alpha2 = 10e-20;
    static long double *grad = (long double *) malloc(dim * sizeof(long double));
    // Calculating a Maths gradient updates the previous value of the gradient. so keep in memory the previous one.
    loop(i, 0, dim) grad[i] = gradW[i];
    double phiPr2 = phiPr(dim, alpha2, grad, w, gradientCost);
    long double tmp2;
    long double perturbation = 2;
    while (phiPr2 < -10e-10) {
        tmp2 = (phiPr(dim, alpha2 * (1 + perturbation), grad, w, gradientCost) -
                phiPr(dim, alpha2 * (1 - perturbation), grad, w, gradientCost)) /
               (alpha2 * (1 + perturbation) - alpha2 * (1 - perturbation));
        alpha2 = alpha2 - phiPr2 / (tmp2 == 0 ? -phiPr2 / (alpha2 * 0.01) : tmp2);
        phiPr2 = phiPr(dim, alpha2, grad, w, gradientCost);
    }
    // retrieve the old value of the gradient.
    loop(i, 0, dim) gradW[i] = grad[i];
    return alpha2;
}

long double
newton_raphson(int dim, double delta, long double *w, long double *gradW, long double *(*gradientCost)(long double *),
               long double **(*hessienCost)(long double *)) {
    long double alpha = 0;
    long double prime = phiPr(dim, alpha, w, gradW, gradientCost);
    while (prime < -delta) {
        alpha = alpha - prime / phiSec(dim, alpha, w, gradW, hessienCost);
        prime = phiPr(dim, alpha, w, gradW, gradientCost);
    }
    return alpha;
}

//////////////////////////////////////// Calculation of optimal set of parameters;
void
gradientDescent(int dim, long double *w, long double *(*gradient)(long double *), long double (*loss)(long double *),
                double delta) {
    long double *grad = gradient(w);
    long double step;
    printf("\n");
    while (norm(dim, grad) > delta) {
        // Update w
        step = armijo(dim, grad, w, loss);
        loop(i, 0, dim) w[i] = w[i] - step * grad[i];
        grad = gradient(w);
//        printf("%.20lf\n", loss(w));
        printf("Gradient Norm: %.20Lf\t \t Current Cost: %.20Lf\t \t Armijo Step: %.20Lf\n",
               norm(dim, grad), loss(w), step);
    }
    printf("\n");
}

void conjugateGradient(int dim, long double *w, long double **Q, long double *b, double delta) {
    // f(x)=1/2x^TQx +bx
    static long double *direction = (long double *) malloc(sizeof(long double) * dim);
    static long double *gradient = add_vector(dim, mat_vector(dim, Q, w), b);
    // Copy by  value not by reference;
    loop(i, 0, dim) direction[i] = -gradient[i];
    long double alpha, beta;
    loop(i, 0, dim) {
        // Calculate optimal step;
        alpha = -dot_product(dim, gradient, direction) / dot_product(dim, direction, mat_vector(dim, Q, direction));
        // Calculate next point;
        loop(j, 0, dim) w[j] += alpha * direction[j];
        // Calculate gradient of new point;
        gradient = add_vector(dim, mat_vector(dim, Q, w), b);
        beta = dot_product(dim, gradient, mat_vector(dim, Q, direction)) /
               dot_product(dim, direction, mat_vector(dim, Q, direction));
        // Calculate new direction;
        loop(j, 0, dim) direction[j] = -gradient[i] + beta * direction[j];
    }
}

#endif //TP_ML_OPTIMIZER_H
