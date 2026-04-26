#include <omp.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

static inline uint64_t rdtsc(void) {
    uint32_t lo;
    uint32_t hi;
    __asm__ __volatile__("rdtsc" : "=a"(lo), "=d"(hi));
    return ((uint64_t)hi << 32) | lo;
}

static inline uint64_t fixed_asm_op(uint64_t x) {
    uint64_t out;
    __asm__ __volatile__(
        "mov %[in], %%rax\n\t"
        "imul $1664525, %%rax, %%rax\n\t"
        "add $1013904223, %%rax\n\t"
        "movabs $0x9e3779b97f4a7c15, %%rcx\n\t"
        "xor %%rcx, %%rax\n\t"
        : "=&a"(out)
        : [in] "r"(x)
        : "rcx", "cc"
    );
    return out;
}

int main(int argc, char *argv[]) {
    uint64_t iterations = strtoull(argv[1], NULL, 10);
    uint64_t acc = 0;

    int nthreads = omp_get_max_threads();
    uint64_t *tsc_create = (uint64_t *)calloc((size_t)nthreads, sizeof(uint64_t));
    uint64_t *tsc_destroy = (uint64_t *)calloc((size_t)nthreads, sizeof(uint64_t));

    uint64_t tsc_before = rdtsc();

    #pragma omp parallel num_threads(nthreads) reduction(^:acc)
    {
        int tid = omp_get_thread_num();
        tsc_create[tid] = rdtsc();

        #pragma omp for schedule(static)
        for (uint64_t i = 0; i < iterations; ++i) {
            acc ^= fixed_asm_op(i);
        }

        tsc_destroy[tid] = rdtsc();
    }

    uint64_t tsc_after = rdtsc();

    printf("acc=%llu\n", (unsigned long long)acc);
    printf("threads=%d\n", nthreads);
    for (int t = 0; t < nthreads; ++t) {
        uint64_t create_cycles = tsc_create[t] - tsc_before;
        uint64_t destroy_cycles = tsc_after - tsc_destroy[t];
        printf("tid=%d create=%llu destroy=%llu\n",
               t,
               (unsigned long long)create_cycles,
               (unsigned long long)destroy_cycles);
    }

    free(tsc_create);
    free(tsc_destroy);
    return 0;
}
