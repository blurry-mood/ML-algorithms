//
// Created by ayoub on 20. 12. 8..
//


#include "Adaline.h"
#include "ANN.h"


int main() {
    read_Nasa();
    fitData();
}


int main2() {
    read_csv();
    // Show the inputs
//    loop(i, 0, n) {
//        loop(j, 0, dim - 1)printf("%Lf, ", x[i][j]);
//        printf("%Lf\n", y[i]);
//    }
    n_layers = 3;
    layers = (int *) malloc(sizeof(int) * n_layers);
    layers[0] = 5;
    layers[1] = 3;
    layers[2] = 1;

    activations = (type (**)(type)) (malloc(sizeof(type(*)(type)) * (n_layers - 1)));
    derivativeActivation = (type (**)(type)) (malloc(sizeof(type(*)(type)) * (n_layers - 1)));
    activations[0] = &identity;
    derivativeActivation[0] = &derIdentity;
    activations[1] = &identity;
    derivativeActivation[1] = &derIdentity;

    fitData();

    printf("Testing error: %Lf ", generalizationError());

    return 0;
}
