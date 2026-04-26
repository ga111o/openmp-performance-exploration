#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

static int is_prime_trial_division(long long n) {
    if (n < 2) {
        return 0;
    }
    if (n == 2) {
        return 1;
    }
    if (n % 2 == 0) {
        return 0;
    }

    long long limit = (long long)sqrt((double)n);
    for (long long d = 3; d <= limit; d += 2) {
        if (n % d == 0) {
            return 0;
        }
    }

    return 1;
}

int main(int argc, char *argv[]) {
    long long n = atoll(argv[1]);
    long long count = 0;

    #pragma omp parallel for reduction(+:count) schedule(static)
    for (long long i = 1; i <= n; i++) {
        if (is_prime_trial_division(i)) {
            count++;
        }
    }

    printf("%lld\n", count);

    return 0;
}
