#ifndef TP_ML_MATH_FUNCTIONS_H
#define TP_ML_MATH_FUNCTIONS_H

#include <math.h>

typedef long double type;

#define loop(i, a, b) for(int i=a;i<b;i++)

type max(type a, type b) {
    return a < b ? b : a;
}

type norm(int dim, type *vector) {
    type sum = 0;
    loop(i, 0, dim) sum += abs(vector[i]);
    return sum;
}

type *mat_vector(int dim, type **matrix, type *vect) {
    static type *res = (type *) malloc(dim * sizeof(type));
    type sum;
    loop(i, 0, dim) {
        sum = 0;
        loop(j, 0, dim) sum += matrix[i][j] * vect[j];
        res[i] = sum;
    }
    return res;
}

type dot_product(int dim, type *w, type *x) {
    type sum = 0;
    loop(i, 0, dim){
        sum += w[i] * x[i];
    }
    return sum;
}

type *scalar_vector(int dim, type scalar, type *vector) {
    static type *res = (type *) malloc(dim * sizeof(type));
    loop(i, 0, dim) res[i] = scalar * vector[i];
    return res;
}

type *diff_vector(int dim, type *vec1, type *vec2) {
    static type *res = (type *) malloc(dim * sizeof(type));
    loop(i, 0, dim) res[i] = vec1[i] - vec2[i];
    return res;
}

type *add_vector(int dim, type *vec1, type *vec2) {
    static type *res = (type *) malloc(dim * sizeof(type));
    loop(i, 0, dim) res[i] = vec1[i] + vec2[i];
    return res;
}


#endif //TP_ML_MATH_FUNCTIONS_H
