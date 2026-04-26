#include <omp.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#define WARM_TRIALS 64
#define WORK_TRIALS 5

static inline uint64_t rdtsc_s(void) {
    uint32_t lo;
    uint32_t hi;
    __asm__ __volatile__(
        "lfence\n\t"
        "rdtsc\n\t"
        "lfence"
        : "=a"(lo), "=d"(hi)
        :
        : "memory"
    );
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
    if (argc < 2) {
        fprintf(stderr, "usage: %s <iterations>\n", argv[0]);
        return 1;
    }

    uint64_t iterations = strtoull(argv[1], NULL, 10);
    int nthreads = omp_get_max_threads();
    uint64_t acc = 0;

    uint64_t t0 = rdtsc_s();
    #pragma omp parallel num_threads(nthreads)
    { (void)omp_get_thread_num(); }
    uint64_t t1 = rdtsc_s();
    uint64_t pool_create = t1 - t0;

    uint64_t fork_join_min = UINT64_MAX;
    for (int i = 0; i < WARM_TRIALS; ++i) {
        uint64_t a = rdtsc_s();
        #pragma omp parallel num_threads(nthreads)
        { (void)omp_get_thread_num(); }
        uint64_t b = rdtsc_s();
        uint64_t d = b - a;
        if (d < fork_join_min) fork_join_min = d;
    }

    uint64_t serial_min = UINT64_MAX;
    for (int i = 0; i < WORK_TRIALS; ++i) {
        uint64_t a = rdtsc_s();
        for (uint64_t k = 0; k < iterations; ++k) {
            acc ^= fixed_asm_op(k);
        }
        uint64_t b = rdtsc_s();
        uint64_t d = b - a;
        if (d < serial_min) serial_min = d;
    }

    uint64_t parallel_min = UINT64_MAX;
    for (int i = 0; i < WORK_TRIALS; ++i) {
        uint64_t a = rdtsc_s();
        #pragma omp parallel for num_threads(nthreads) reduction(^:acc) schedule(static)
        for (uint64_t k = 0; k < iterations; ++k) {
            acc ^= fixed_asm_op(k);
        }
        uint64_t b = rdtsc_s();
        uint64_t d = b - a;
        if (d < parallel_min) parallel_min = d;
    }

    long long pure_parallel = (long long)parallel_min - (long long)fork_join_min;
    double speedup_total = (double)serial_min / (double)parallel_min;
    double speedup_pure = pure_parallel > 0
        ? (double)serial_min / (double)pure_parallel
        : 0.0;

    printf("threads=%d iterations=%llu\n", nthreads, (unsigned long long)iterations);
    return 0;
}
