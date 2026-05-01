#include <errno.h>
#include <inttypes.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

static inline uint64_t rdtsc_serialized(void) {
    uint32_t lo, hi;
    __asm__ __volatile__(
        "lfence\n\t"
        "rdtsc\n\t"
        "lfence"
        : "=a"(lo), "=d"(hi)
        :
        : "memory");
    return ((uint64_t)hi << 32) | lo;
}

static int parse_u64(const char *s, uint64_t *out) {
    char *end = NULL;
    errno = 0;
    unsigned long long v = strtoull(s, &end, 10);
    if (errno != 0 || end == s || *end != '\0') return -1;
    *out = (uint64_t)v;
    return 0;
}

static double elapsed_seconds(const struct timespec *start,
                              const struct timespec *end) {
    const time_t sec = end->tv_sec - start->tv_sec;
    const long nsec = end->tv_nsec - start->tv_nsec;
    return (double)sec + (double)nsec / 1000000000.0;
}

static bool is_prime(uint64_t n) {
    if (n < 2) return false;
    if (n == 2) return true;
    if ((n & 1u) == 0) return false;

    for (uint64_t d = 3; d <= n / d; d += 2) {
        if (n % d == 0) return false;
    }
    return true;
}

static uint64_t count_primes(uint64_t limit) {
    uint64_t count = 0;
    for (uint64_t n = 2; n <= limit; ++n) {
        if (is_prime(n)) ++count;
    }
    return count;
}

int main(int argc, char *argv[]) {
    if (argc != 2) {
        fprintf(stderr, "usage: %s <limit>\n", argv[0]);
        return 1;
    }

    uint64_t limit = 0;
    if (parse_u64(argv[1], &limit) != 0) {
        fprintf(stderr, "error: invalid limit '%s'\n", argv[1]);
        return 1;
    }

    struct timespec start_time, end_time;
    if (clock_gettime(CLOCK_MONOTONIC, &start_time) != 0) {
        perror("clock_gettime");
        return 1;
    }
    const uint64_t start_cycles = rdtsc_serialized();

    const uint64_t primes = count_primes(limit);

    const uint64_t end_cycles = rdtsc_serialized();
    if (clock_gettime(CLOCK_MONOTONIC, &end_time) != 0) {
        perror("clock_gettime");
        return 1;
    }

    printf("RESULT"
           " limit=%" PRIu64
           " primes=%" PRIu64
           " cycles=%" PRIu64
           " seconds=%.9f"
           "\n",
           limit,
           primes,
           end_cycles - start_cycles,
           elapsed_seconds(&start_time, &end_time));

    return 0;
}
