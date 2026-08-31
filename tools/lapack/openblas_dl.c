/* Same call, resolved at runtime.
 * This build prefixes its symbols (scipy_dgels_64_) to avoid clashing with a
 * system BLAS, which is why a plain -ldgels finds nothing.
 * from a plain -l; dlsym does, and it also proves we are calling the exact
 * shared object numpy loads rather than something else on the search path. */
#include <stdio.h>
#include <stdlib.h>
#include <dlfcn.h>
typedef long long i64;
typedef void (*dgels_t)(char *, i64 *, i64 *, i64 *, double *, i64 *,
                        double *, i64 *, double *, i64 *, i64 *);
int main(int argc, char **argv) {
    void *h = dlopen(argv[1], RTLD_NOW);
    if (!h) { fprintf(stderr, "dlopen: %s\n", dlerror()); return 3; }
    dgels_t dgels = (dgels_t) dlsym(h, "scipy_dgels_64_");
    if (!dgels) { fprintf(stderr, "dlsym: %s\n", dlerror()); return 4; }

    i64 m, n;
    if (scanf("%lld %lld", &m, &n) != 2) return 1;
    double *rm = malloc(sizeof(double) * m * n);
    for (i64 i = 0; i < m * n; i++) if (scanf("%lf", &rm[i]) != 1) return 1;
    double *b = malloc(sizeof(double) * m);
    for (i64 i = 0; i < m; i++) if (scanf("%lf", &b[i]) != 1) return 1;
    double *a = malloc(sizeof(double) * m * n);
    for (i64 i = 0; i < m; i++) for (i64 j = 0; j < n; j++) a[j * m + i] = rm[i * n + j];

    char trans = 'N'; i64 nrhs = 1, lda = m, ldb = m, lwork = -1, info = 0; double wq;
    dgels(&trans, &m, &n, &nrhs, a, &lda, b, &ldb, &wq, &lwork, &info);
    lwork = (i64) wq;
    double *work = malloc(sizeof(double) * lwork);
    dgels(&trans, &m, &n, &nrhs, a, &lda, b, &ldb, work, &lwork, &info);
    if (info != 0) { fprintf(stderr, "info=%lld\n", info); return 2; }
    printf("openblas\n");
    for (i64 j = 0; j < n; j++) printf("%.17g\n", b[j]);
    return 0;
}
