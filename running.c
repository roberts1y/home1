#include <stdio.h>
#include <stdlib.h>

int main(int argc, char *argv[]) {
    int n = 10000;
    double e = 1.0;
    double factorial = 1.0;

    if (argc != 1){
	    printf("usage is ./run <num_loops>");
	}
    n = atoi(argv[1]);

    for (int i = 1; i < n; i++) {
        factorial *= i;
        e += 1.0 / factorial;
        printf("\rApproximating e (%d terms): %.15f", i, e);
        fflush(stdout);  // ensure it prints immediately
    }

    printf("\nFinal approximation of e: %.15f\n", e);
    return 0;
}

