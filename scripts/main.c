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
#define CACHE_WALK_MIN_STEPS  (1ULL << 22)
#define CACHE_WALK_MAX_STEPS  (1ULL << 26)
#define CACHE_LINE_BYTES      64

typedef enum {
    WS_PARALLEL_FOR,
    WS_MANUAL,
    WS_TASKS,
    WS_CACHE_WALK,
    WS_CACHE_WALK_PRIVATE,
    WS_FALSE_SHARING,
    WS_FALSE_SHARING_PADDED,
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

static bool is_prime(uint64_t n) {
    if (n < 2) return false;
    if (n == 2) return true;
    if ((n & 1u) == 0) return false;

    for (uint64_t d = 3; d <= n / d; d += 2) {
        if (n % d == 0) return false;
    }
    return true;
}

static uint64_t prime_candidate_count(uint64_t limit) {
    return limit >= 2 ? limit - 1 : 0;
}

static inline uint64_t splitmix64(uint64_t *state) {
    uint64_t z = (*state += 0x9e3779b97f4a7c15ULL);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
}

static int parse_work_sharing(const char *s, ws_kind_t *out) {
    if (strcmp(s, "parallel_for") == 0) { *out = WS_PARALLEL_FOR; return 0; }
    if (strcmp(s, "manual")       == 0) { *out = WS_MANUAL;       return 0; }
    if (strcmp(s, "tasks")        == 0) { *out = WS_TASKS;        return 0; }
    if (strcmp(s, "cache_walk")   == 0) { *out = WS_CACHE_WALK;   return 0; }
    if (strcmp(s, "cache_walk_private") == 0) {
        *out = WS_CACHE_WALK_PRIVATE;
        return 0;
    }
    if (strcmp(s, "false_sharing") == 0) {
        *out = WS_FALSE_SHARING;
        return 0;
    }
    if (strcmp(s, "false_sharing_padded") == 0) {
        *out = WS_FALSE_SHARING_PADDED;
        return 0;
    }
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
    const uint64_t candidates = prime_candidate_count(iterations);
    for (int i = 0; i < BENCH_MEASURE_TRIALS; ++i) {
        uint64_t local = 0;
        uint64_t a = rdtsc_serialized();
        for (uint64_t offset = 0; offset < candidates; ++offset) {
            local += is_prime(2 + offset) ? 1ULL : 0ULL;
        }
        uint64_t b = rdtsc_serialized();
        uint64_t d = b - a;
        if (d < best) best = d;
        acc ^= local;
    }
    *checksum += acc;
    return best;
}

static uint64_t cache_walk_steps(uint64_t elements) {
    uint64_t steps = elements * 4ULL;
    if (steps < CACHE_WALK_MIN_STEPS) steps = CACHE_WALK_MIN_STEPS;
    if (steps > CACHE_WALK_MAX_STEPS) steps = CACHE_WALK_MAX_STEPS;
    return steps;
}

static uint64_t *make_cache_walk(uint64_t elements) {
    if (elements < 2 || elements > SIZE_MAX / sizeof(uint64_t)) {
        return NULL;
    }

    uint64_t *order = NULL;
    uint64_t *next  = NULL;
    if (posix_memalign((void **)&order, CACHE_LINE_BYTES,
                       (size_t)elements * sizeof(*order)) != 0 ||
        posix_memalign((void **)&next, CACHE_LINE_BYTES,
                       (size_t)elements * sizeof(*next)) != 0) {
        free(order);
        free(next);
        return NULL;
    }

    for (uint64_t i = 0; i < elements; ++i) {
        order[i] = i;
    }

    uint64_t rng = 0x123456789abcdef0ULL ^ elements;
    for (uint64_t i = elements - 1; i > 0; --i) {
        const uint64_t j = splitmix64(&rng) % (i + 1);
        const uint64_t tmp = order[i];
        order[i] = order[j];
        order[j] = tmp;
    }

    for (uint64_t i = 0; i < elements; ++i) {
        next[order[i]] = order[(i + 1) % elements];
    }

    free(order);
    return next;
}

static uint64_t *make_cache_walk_private(uint64_t elements_per_thread,
                                         int num_threads) {
    if (elements_per_thread < 2 || num_threads < 1) {
        return NULL;
    }

    const uint64_t threads = (uint64_t)num_threads;
    if (elements_per_thread > UINT64_MAX / threads ||
        elements_per_thread * threads > SIZE_MAX / sizeof(uint64_t)) {
        return NULL;
    }

    const uint64_t total = elements_per_thread * threads;
    uint64_t *order = NULL;
    uint64_t *next  = NULL;
    if (posix_memalign((void **)&order, CACHE_LINE_BYTES,
                       (size_t)elements_per_thread * sizeof(*order)) != 0 ||
        posix_memalign((void **)&next, CACHE_LINE_BYTES,
                       (size_t)total * sizeof(*next)) != 0) {
        free(order);
        free(next);
        return NULL;
    }

    for (uint64_t t = 0; t < threads; ++t) {
        const uint64_t base = t * elements_per_thread;
        for (uint64_t i = 0; i < elements_per_thread; ++i) {
            order[i] = i;
        }

        uint64_t rng = 0xfedcba9876543210ULL ^ elements_per_thread
            ^ (t * 0x9e3779b97f4a7c15ULL);
        for (uint64_t i = elements_per_thread - 1; i > 0; --i) {
            const uint64_t j = splitmix64(&rng) % (i + 1);
            const uint64_t tmp = order[i];
            order[i] = order[j];
            order[j] = tmp;
        }

        for (uint64_t i = 0; i < elements_per_thread; ++i) {
            next[base + order[i]] = base + order[(i + 1) % elements_per_thread];
        }
    }

    free(order);
    return next;
}

static uint64_t walk_cache(const uint64_t *next, uint64_t steps,
                           uint64_t start) {
    uint64_t idx = start;
    for (uint64_t i = 0; i < steps; ++i) {
        idx = next[idx];
    }
    return idx;
}

static uint64_t bench_cache_serial(const uint64_t *next, uint64_t elements,
                                   uint64_t *checksum) {
    uint64_t best = UINT64_MAX;
    uint64_t acc  = 0;
    const uint64_t steps = cache_walk_steps(elements);

    for (int i = 0; i < BENCH_MEASURE_TRIALS; ++i) {
        uint64_t a = rdtsc_serialized();
        uint64_t local = walk_cache(next, steps, 0);
        uint64_t b = rdtsc_serialized();
        uint64_t d = b - a;
        if (d < best) best = d;
        acc ^= local;
    }

    *checksum ^= acc;
    return best;
}

static uint64_t bench_cache_private_serial(const uint64_t *next,
                                           uint64_t elements_per_thread,
                                           int num_threads,
                                           uint64_t *checksum) {
    uint64_t best = UINT64_MAX;
    uint64_t acc  = 0;
    const uint64_t steps = cache_walk_steps(elements_per_thread);

    for (int i = 0; i < BENCH_MEASURE_TRIALS; ++i) {
        uint64_t local = 0;
        uint64_t a = rdtsc_serialized();
        for (int tid = 0; tid < num_threads; ++tid) {
            const uint64_t start = elements_per_thread * (uint64_t)tid;
            local ^= walk_cache(next, steps, start);
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
    const uint64_t candidates = prime_candidate_count(iterations);
    #pragma omp parallel reduction(+:local)
    {
        if (probe && omp_get_thread_num() == 0) {
            *threads_used = omp_get_num_threads();
        }
        #pragma omp for schedule(runtime) nowait
        for (uint64_t offset = 0; offset < candidates; ++offset) {
            local += is_prime(2 + offset) ? 1ULL : 0ULL;
        }
    }
    return local;
}

static uint64_t run_manual(uint64_t iterations, int *threads_used, int probe) {
    uint64_t local = 0;
    const uint64_t candidates = prime_candidate_count(iterations);
    #pragma omp parallel reduction(+:local)
    {
        const int tid = omp_get_thread_num();
        const int nth = omp_get_num_threads();
        if (probe && tid == 0) *threads_used = nth;

        const uint64_t start = (candidates * (uint64_t)tid)       / (uint64_t)nth;
        const uint64_t end   = (candidates * (uint64_t)(tid + 1)) / (uint64_t)nth;

        uint64_t lo = 0;
        for (uint64_t offset = start; offset < end; ++offset) {
            lo += is_prime(2 + offset) ? 1ULL : 0ULL;
        }
        local += lo;
    }
    return local;
}

static uint64_t run_tasks(uint64_t iterations, int task_chunk,
                          int *threads_used, int probe) {
    uint64_t result = 0;
    const uint64_t candidates = prime_candidate_count(iterations);
    #pragma omp parallel
    {
        if (probe && omp_get_thread_num() == 0) {
            *threads_used = omp_get_num_threads();
        }
        #pragma omp single
        {
            if (task_chunk > 0) {
                #pragma omp taskloop grainsize(task_chunk) reduction(+:result)
                for (uint64_t offset = 0; offset < candidates; ++offset) {
                    result += is_prime(2 + offset) ? 1ULL : 0ULL;
                }
            } else {
                #pragma omp taskloop reduction(+:result)
                for (uint64_t offset = 0; offset < candidates; ++offset) {
                    result += is_prime(2 + offset) ? 1ULL : 0ULL;
                }
            }
        }
    }
    return result;
}

static uint64_t run_cache_walk(const uint64_t *next, uint64_t elements,
                               int *threads_used, int probe) {
    uint64_t local = 0;
    const uint64_t steps = cache_walk_steps(elements);

    #pragma omp parallel reduction(^:local)
    {
        const int tid = omp_get_thread_num();
        const int nth = omp_get_num_threads();
        if (probe && tid == 0) *threads_used = nth;

        const uint64_t start = (elements * (uint64_t)tid) / (uint64_t)nth;
        const uint64_t my_steps = steps / (uint64_t)nth
            + ((uint64_t)tid < (steps % (uint64_t)nth));
        local ^= walk_cache(next, my_steps, start);
    }

    return local;
}

static uint64_t run_cache_walk_private(const uint64_t *next,
                                       uint64_t elements_per_thread,
                                       int *threads_used, int probe) {
    uint64_t local = 0;
    const uint64_t steps = cache_walk_steps(elements_per_thread);

    #pragma omp parallel reduction(^:local)
    {
        const int tid = omp_get_thread_num();
        if (probe && tid == 0) *threads_used = omp_get_num_threads();

        const uint64_t start = elements_per_thread * (uint64_t)tid;
        local ^= walk_cache(next, steps, start);
    }

    return local;
}

static uint64_t false_sharing_stride(ws_kind_t ws) {
    return ws == WS_FALSE_SHARING_PADDED
        ? (uint64_t)(CACHE_LINE_BYTES / sizeof(uint64_t)) * 2ULL
        : 1ULL;
}

static uint64_t false_sharing_slots(ws_kind_t ws, int num_threads) {
    const uint64_t stride = false_sharing_stride(ws);
    return ((uint64_t)num_threads - 1ULL) * stride + 1ULL;
}

static uint64_t bench_false_sharing_serial(volatile uint64_t *slots,
                                           uint64_t iterations,
                                           int num_threads,
                                           ws_kind_t ws,
                                           uint64_t *checksum) {
    uint64_t best = UINT64_MAX;
    uint64_t acc  = 0;
    const uint64_t stride = false_sharing_stride(ws);

    for (int trial = 0; trial < BENCH_MEASURE_TRIALS; ++trial) {
        for (uint64_t i = 0; i < false_sharing_slots(ws, num_threads); ++i) {
            slots[i] = 0;
        }

        uint64_t a = rdtsc_serialized();
        for (int tid = 0; tid < num_threads; ++tid) {
            volatile uint64_t *slot = &slots[(uint64_t)tid * stride];
            for (uint64_t k = 0; k < iterations; ++k) {
                *slot += 1ULL;
            }
            acc ^= *slot + (uint64_t)tid;
        }
        uint64_t b = rdtsc_serialized();
        uint64_t d = b - a;
        if (d < best) best = d;
    }

    *checksum ^= acc;
    return best;
}

static uint64_t bench_false_sharing_parallel(const config_t *cfg,
                                             volatile uint64_t *slots,
                                             uint64_t *checksum,
                                             int *threads_used) {
    uint64_t best = UINT64_MAX;
    uint64_t acc  = 0;
    int seen      = cfg->num_threads;
    const uint64_t stride = false_sharing_stride(cfg->ws);
    const uint64_t count  = false_sharing_slots(cfg->ws, cfg->num_threads);

    for (int trial = 0; trial < BENCH_MEASURE_TRIALS; ++trial) {
        for (uint64_t i = 0; i < count; ++i) {
            slots[i] = 0;
        }

        const int probe = (trial == 0);
        uint64_t a = rdtsc_serialized();
        #pragma omp parallel reduction(^:acc)
        {
            const int tid = omp_get_thread_num();
            if (probe && tid == 0) seen = omp_get_num_threads();

            volatile uint64_t *slot = &slots[(uint64_t)tid * stride];
            for (uint64_t k = 0; k < cfg->iterations; ++k) {
                *slot += 1ULL;
            }
            acc ^= *slot + (uint64_t)tid;
        }
        uint64_t b = rdtsc_serialized();
        uint64_t d = b - a;
        if (d < best) best = d;
    }

    *checksum    ^= acc;
    *threads_used = seen;
    return best;
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

    *checksum     += acc;
    *threads_used  = seen;
    return best;
}

static uint64_t bench_cache_parallel(const config_t *cfg, const uint64_t *next,
                                     uint64_t *checksum, int *threads_used) {
    uint64_t best = UINT64_MAX;
    uint64_t acc  = 0;
    int seen      = cfg->num_threads;

    for (int i = 0; i < BENCH_MEASURE_TRIALS; ++i) {
        const int probe = (i == 0);
        uint64_t a = rdtsc_serialized();
        uint64_t local = (cfg->ws == WS_CACHE_WALK_PRIVATE)
            ? run_cache_walk_private(next, cfg->iterations, &seen, probe)
            : run_cache_walk(next, cfg->iterations, &seen, probe);
        uint64_t b = rdtsc_serialized();
        uint64_t d = b - a;
        if (d < best) best = d;
        acc ^= local;
    }

    *checksum    ^= acc;
    *threads_used = seen;
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

    if (cfg.ws == WS_CACHE_WALK || cfg.ws == WS_CACHE_WALK_PRIVATE) {
        uint64_t *next = (cfg.ws == WS_CACHE_WALK_PRIVATE)
            ? make_cache_walk_private(cfg.iterations, cfg.num_threads)
            : make_cache_walk(cfg.iterations);
        if (next == NULL) {
            fprintf(stderr, "error: failed to allocate %s with %" PRIu64
                    " elements\n", cfg.ws_str, cfg.iterations);
            return 2;
        }
        m.serial_min   = (cfg.ws == WS_CACHE_WALK_PRIVATE)
            ? bench_cache_private_serial(next, cfg.iterations, cfg.num_threads,
                                         &m.checksum)
            : bench_cache_serial(next, cfg.iterations, &m.checksum);
        m.parallel_min = bench_cache_parallel(&cfg, next, &m.checksum,
                                              &m.threads_used);
        free(next);
    } else if (cfg.ws == WS_FALSE_SHARING ||
               cfg.ws == WS_FALSE_SHARING_PADDED) {
        const uint64_t slot_count = false_sharing_slots(cfg.ws, cfg.num_threads);
        if (slot_count > SIZE_MAX / sizeof(uint64_t)) {
            fprintf(stderr, "error: false-sharing slot count overflow\n");
            return 2;
        }

        uint64_t *slots = NULL;
        if (posix_memalign((void **)&slots, CACHE_LINE_BYTES,
                           (size_t)slot_count * sizeof(*slots)) != 0) {
            fprintf(stderr, "error: failed to allocate false-sharing slots\n");
            return 2;
        }
        m.serial_min = bench_false_sharing_serial(slots, cfg.iterations,
                                                  cfg.num_threads, cfg.ws,
                                                  &m.checksum);
        m.parallel_min = bench_false_sharing_parallel(&cfg, slots,
                                                      &m.checksum,
                                                      &m.threads_used);
        free(slots);
    } else {
        m.serial_min   = bench_serial(cfg.iterations, &m.checksum);
        m.parallel_min = bench_parallel(&cfg, &m.checksum, &m.threads_used);
    }

    emit_result(&cfg, &m);
    return 0;
}
