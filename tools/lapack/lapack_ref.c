/*
 * Least squares through LAPACK directly, with no Python in the way.
 *
 * numpy is a binding; this calls the same Fortran routines it binds to, so a
 * difference between the two would be numpy's doing rather than LAPACK's.
 * Prototypes are declared by hand because liblapack-dev is not installed --
 * the symbols are in the shared object regardless, and LAPACK's Fortran ABI
 * is stable.
 *
 * Two solvers, because they disagree on hard problems and it matters which
 * one is being called "the reference":
 *   dgels  - QR, what numpy.linalg.qr and lstsq's fast path use
 *   dgelsd - SVD with divide and conquer, the rank-revealing one
 *
 * Input on stdin:  m n  then m*n matrix values row-major, then m response
 * Output on stdout: solver name and n coefficients, one per line.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

extern void dgels_(char *trans, int *m, int *n, int *nrhs, double *a, int *lda,
                   double *b, int *ldb, double *work, int *lwork, int *info);
extern void dgelsd_(int *m, int *n, int *nrhs, double *a, int *lda, double *b,
                    int *ldb, double *s, double *rcond, int *rank, double *work,
                    int *lwork, int *iwork, int *info);

int main(int argc, char **argv) {
    int m, n;
    if (scanf("%d %d", &m, &n) != 2) return 1;

    double *rowmajor = malloc(sizeof(double) * m * n);
    for (int i = 0; i < m * n; i++)
        if (scanf("%lf", &rowmajor[i]) != 1) return 1;

    double *b = malloc(sizeof(double) * m);
    for (int i = 0; i < m; i++)
        if (scanf("%lf", &b[i]) != 1) return 1;

    /* Fortran is column-major. */
    double *a = malloc(sizeof(double) * m * n);
    for (int i = 0; i < m; i++)
        for (int j = 0; j < n; j++)
            a[j * m + i] = rowmajor[i * n + j];

    int nrhs = 1, lda = m, ldb = m, info = 0;
    const char *which = argc > 1 ? argv[1] : "dgels";

    if (strcmp(which, "dgelsd") == 0) {
        double *s = malloc(sizeof(double) * (m < n ? m : n));
        double rcond = -1.0, wq;
        int rank, lwork = -1, iwq;
        dgelsd_(&m, &n, &nrhs, a, &lda, b, &ldb, s, &rcond, &rank, &wq, &lwork, &iwq, &info);
        lwork = (int) wq;
        double *work = malloc(sizeof(double) * lwork);
        int *iwork = malloc(sizeof(int) * (iwq > 0 ? iwq : 1024 * 64));
        dgelsd_(&m, &n, &nrhs, a, &lda, b, &ldb, s, &rcond, &rank, work, &lwork, iwork, &info);
    } else {
        char trans = 'N';
        double wq;
        int lwork = -1;
        dgels_(&trans, &m, &n, &nrhs, a, &lda, b, &ldb, &wq, &lwork, &info);
        lwork = (int) wq;
        double *work = malloc(sizeof(double) * lwork);
        dgels_(&trans, &m, &n, &nrhs, a, &lda, b, &ldb, work, &lwork, &info);
    }

    if (info != 0) { fprintf(stderr, "LAPACK %s returned info=%d\n", which, info); return 2; }

    printf("%s\n", which);
    for (int j = 0; j < n; j++) printf("%.17g\n", b[j]);
    return 0;
}
