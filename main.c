#include <errno.h>
#include <inttypes.h>
#include <limits.h>
#include <omp.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define BENCH_WARMUP_TRIALS  32
#define BENCH_MEASURE_TRIALS  3

#define LCG_MULTIPLIER_STR  "1664525"
#define LCG_INCREMENT_STR   "1013904223"
#define GOLDEN_RATIO_64_STR "0x9e3779b97f4a7c15"

typedef enum {
    WS_PARALLEL_FOR,
    WS_MANUAL,
    WS_TASKS,
} ws_kind_t;

typedef struct {
    const char *ws_str;
    const char *sk_str;
    ws_kind_t   ws;
    omp_sched_t sk;
    int         schedule_chunk;
    int         task_chunk;
    int         num_threads;
    uint64_t    iterations;
} config_t;

typedef struct {
    uint64_t pool_create;
    uint64_t fork_join_min;
    uint64_t serial_min;
    uint64_t parallel_min;
    int      threads_used;
    uint64_t checksum;
} measurement_t;

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

static inline uint64_t mix_kernel(uint64_t x) {
    uint64_t out;
    __asm__ __volatile__(
        "mov %[in], %%rax\n\t"
        "imul $" LCG_MULTIPLIER_STR ", %%rax, %%rax\n\t"
        "add $"  LCG_INCREMENT_STR  ", %%rax\n\t"
        "movabs $" GOLDEN_RATIO_64_STR ", %%rcx\n\t"
        "xor %%rcx, %%rax\n\t"
        : "=&a"(out)
        : [in] "r"(x)
        : "rcx", "cc");
    return out;
}

static int parse_work_sharing(const char *s, ws_kind_t *out) {
    if (strcmp(s, "parallel_for") == 0) { *out = WS_PARALLEL_FOR; return 0; }
    if (strcmp(s, "manual")       == 0) { *out = WS_MANUAL;       return 0; }
    if (strcmp(s, "tasks")        == 0) { *out = WS_TASKS;        return 0; }
    return -1;
}

static int parse_schedule_kind(const char *s, omp_sched_t *out) {
    if (strcmp(s, "static")  == 0) { *out = omp_sched_static;  return 0; }
    if (strcmp(s, "dynamic") == 0) { *out = omp_sched_dynamic; return 0; }
    if (strcmp(s, "guided")  == 0) { *out = omp_sched_guided;  return 0; }
    if (strcmp(s, "auto")    == 0) { *out = omp_sched_auto;    return 0; }
    return -1;
}

static int parse_long(const char *s, long lo, long hi, long *out) {
    char *end = NULL;
    errno = 0;
    long v = strtol(s, &end, 10);
    if (errno != 0 || end == s || *end != '\0' || v < lo || v > hi) return -1;
    *out = v;
    return 0;
}

static int parse_u64(const char *s, uint64_t *out) {
    char *end = NULL;
    errno = 0;
    unsigned long long v = strtoull(s, &end, 10);
    if (errno != 0 || end == s || *end != '\0') return -1;
    *out = (uint64_t)v;
    return 0;
}

static int parse_args(int argc, char **argv, config_t *cfg) {
    if (argc != 7) {
        fprintf(stderr,
                "usage: %s <work_sharing> <schedule_kind> <schedule_chunk> "
                "<task_chunk_size> <num_threads> <iterations>\n",
                argv[0]);
        return -1;
    }

    cfg->ws_str = argv[1];
    cfg->sk_str = argv[2];

    if (parse_work_sharing(argv[1], &cfg->ws) != 0) {
        fprintf(stderr, "error: unknown work_sharing '%s'\n", argv[1]);
        return -1;
    }
    if (parse_schedule_kind(argv[2], &cfg->sk) != 0) {
        fprintf(stderr, "error: unknown schedule_kind '%s'\n", argv[2]);
        return -1;
    }

    long schedule_chunk = 0, task_chunk = 0, nthreads = 1;
    if (parse_long(argv[3], 0, INT_MAX, &schedule_chunk) != 0 ||
        parse_long(argv[4], 0, INT_MAX, &task_chunk)     != 0 ||
        parse_long(argv[5], 1, 4096,    &nthreads)       != 0 ||
        parse_u64 (argv[6], &cfg->iterations)            != 0) {
        fprintf(stderr, "error: invalid numeric argument\n");
        return -1;
    }
    cfg->schedule_chunk = (int)schedule_chunk;
    cfg->task_chunk     = (int)task_chunk;
    cfg->num_threads    = (int)nthreads;
    return 0;
}

static uint64_t bench_pool_create(void) {
    uint64_t a = rdtsc_serialized();
    #pragma omp parallel
    { (void)omp_get_thread_num(); }
    uint64_t b = rdtsc_serialized();
    return b - a;
}

static uint64_t bench_fork_join_min(void) {
    uint64_t best = UINT64_MAX;
    for (int i = 0; i < BENCH_WARMUP_TRIALS; ++i) {
        uint64_t a = rdtsc_serialized();
        #pragma omp parallel
        { (void)omp_get_thread_num(); }
        uint64_t b = rdtsc_serialized();
        uint64_t d = b - a;
        if (d < best) best = d;
    }
    return best;
}

static uint64_t bench_serial(uint64_t iterations, uint64_t *checksum) {
    uint64_t best = UINT64_MAX;
    uint64_t acc  = 0;
    for (int i = 0; i < BENCH_MEASURE_TRIALS; ++i) {
        uint64_t local = 0;
        uint64_t a = rdtsc_serialized();
        for (uint64_t k = 0; k < iterations; ++k) {
            local ^= mix_kernel(k);
        }
        uint64_t b = rdtsc_serialized();
        uint64_t d = b - a;
        if (d < best) best = d;
        acc ^= local;
    }
    *checksum ^= acc;
    return best;
}

static uint64_t run_parallel_for(uint64_t iterations, int *threads_used, int probe) {
    uint64_t local = 0;
    #pragma omp parallel reduction(^:local)
    {
        if (probe && omp_get_thread_num() == 0) {
            *threads_used = omp_get_num_threads();
        }
        #pragma omp for schedule(runtime) nowait
        for (uint64_t k = 0; k < iterations; ++k) {
            local ^= mix_kernel(k);
        }
    }
    return local;
}

static uint64_t run_manual(uint64_t iterations, int *threads_used, int probe) {
    uint64_t local = 0;
    #pragma omp parallel reduction(^:local)
    {
        const int tid = omp_get_thread_num();
        const int nth = omp_get_num_threads();
        if (probe && tid == 0) *threads_used = nth;

        const uint64_t start = (iterations * (uint64_t)tid)       / (uint64_t)nth;
        const uint64_t end   = (iterations * (uint64_t)(tid + 1)) / (uint64_t)nth;

        uint64_t lo = 0;
        for (uint64_t k = start; k < end; ++k) {
            lo ^= mix_kernel(k);
        }
        local ^= lo;
    }
    return local;
}

static uint64_t run_tasks(uint64_t iterations, int task_chunk,
                          int *threads_used, int probe) {
    uint64_t result = 0;
    #pragma omp parallel
    {
        if (probe && omp_get_thread_num() == 0) {
            *threads_used = omp_get_num_threads();
        }
        #pragma omp single
        {
            if (task_chunk > 0) {
                #pragma omp taskloop grainsize(task_chunk) reduction(^:result)
                for (uint64_t k = 0; k < iterations; ++k) {
                    result ^= mix_kernel(k);
                }
            } else {
                #pragma omp taskloop reduction(^:result)
                for (uint64_t k = 0; k < iterations; ++k) {
                    result ^= mix_kernel(k);
                }
            }
        }
    }
    return result;
}

static uint64_t bench_parallel(const config_t *cfg, uint64_t *checksum,
                               int *threads_used) {
    uint64_t best        = UINT64_MAX;
    uint64_t acc         = 0;
    int      seen        = cfg->num_threads;

    for (int i = 0; i < BENCH_MEASURE_TRIALS; ++i) {
        const int probe = (i == 0);
        uint64_t local;
        uint64_t a = rdtsc_serialized();

        switch (cfg->ws) {
            case WS_PARALLEL_FOR:
                local = run_parallel_for(cfg->iterations, &seen, probe);
                break;
            case WS_MANUAL:
                local = run_manual(cfg->iterations, &seen, probe);
                break;
            case WS_TASKS:
                local = run_tasks(cfg->iterations, cfg->task_chunk, &seen, probe);
                break;
            default:
                local = 0;
                break;
        }

        uint64_t b = rdtsc_serialized();
        uint64_t d = b - a;
        if (d < best) best = d;
        acc ^= local;
    }

    *checksum     ^= acc;
    *threads_used  = seen;
    return best;
}

static void emit_result(const config_t *cfg, const measurement_t *m) {
    const long long pure_parallel =
        (long long)m->parallel_min - (long long)m->fork_join_min;
    const double speedup_total =
        (double)m->serial_min / (double)m->parallel_min;
    const double speedup_pure  = pure_parallel > 0
        ? (double)m->serial_min / (double)pure_parallel
        : 0.0;

    printf("RESULT"
           " work_sharing=%s"
           " schedule_kind=%s"
           " schedule_chunk=%d"
           " task_chunk_size=%d"
           " num_threads_req=%d"
           " num_threads_actual=%d"
           " iterations=%"      PRIu64
           " pool_create=%"     PRIu64
           " fork_join_min=%"   PRIu64
           " serial_min=%"      PRIu64
           " parallel_min=%"    PRIu64
           " pure_parallel=%lld"
           " speedup_total=%.6f"
           " speedup_pure=%.6f"
           " acc=%"             PRIu64
           "\n",
           cfg->ws_str, cfg->sk_str,
           cfg->schedule_chunk, cfg->task_chunk,
           cfg->num_threads, m->threads_used,
           cfg->iterations,
           m->pool_create, m->fork_join_min,
           m->serial_min,  m->parallel_min,
           pure_parallel,
           speedup_total, speedup_pure,
           m->checksum);
}

int main(int argc, char *argv[]) {
    config_t cfg;
    if (parse_args(argc, argv, &cfg) != 0) return 1;

    omp_set_num_threads(cfg.num_threads);
    omp_set_schedule(cfg.sk, cfg.schedule_chunk);

    measurement_t m = {0};
    m.pool_create   = bench_pool_create();
    m.fork_join_min = bench_fork_join_min();
    m.serial_min    = bench_serial(cfg.iterations, &m.checksum);
    m.parallel_min  = bench_parallel(&cfg, &m.checksum, &m.threads_used);

    emit_result(&cfg, &m);
    return 0;
}
