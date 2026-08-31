/*
 * Statistics through the GNU Scientific Library, called from C.
 *
 * A third independent opinion, and an independent *algorithm*: this library
 * solves least squares by Householder QR with iterative refinement, LAPACK by
 * blocked QR, and GSL by singular value decomposition. Three routes to the
 * same certified answer disagree in different ways, which is more informative
 * than any two of them agreeing.
 *
 * GSL also ships gsl_stats_lag1_autocorrelation, which neither numpy nor scipy
 * exposes as a primitive -- so it is the only reference available for one of
 * the NIST univariate quantities.
 *
 * Prototypes are declared by hand: libgsl-dev is not installed, the shared
 * object is, and the ABI is stable.
 *
 * Usage:
 *   gsl_ref univariate            < "n  v1 .. vn"
 *   gsl_ref regression            < "m n  A(row-major)  b"
 *   gsl_ref correlation           < "n  x1..xn  y1..yn"
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

typedef struct { size_t size1, size2, tda; double *data; void *block; int owner; } gsl_matrix;
typedef struct { size_t size, stride; double *data; void *block; int owner; } gsl_vector;

extern gsl_matrix *gsl_matrix_alloc(size_t n1, size_t n2);
extern gsl_vector *gsl_vector_alloc(size_t n);
extern void gsl_matrix_set(gsl_matrix *m, size_t i, size_t j, double x);
extern void gsl_vector_set(gsl_vector *v, size_t i, double x);
extern double gsl_vector_get(const gsl_vector *v, size_t i);
extern double gsl_matrix_get(const gsl_matrix *m, size_t i, size_t j);
extern void *gsl_multifit_linear_alloc(size_t n, size_t p);
extern int gsl_multifit_linear(const gsl_matrix *X, const gsl_vector *y,
                               gsl_vector *c, gsl_matrix *cov, double *chisq, void *work);

extern double gsl_stats_mean(const double *data, size_t stride, size_t n);
extern double gsl_stats_sd(const double *data, size_t stride, size_t n);
extern double gsl_stats_variance(const double *data, size_t stride, size_t n);
extern double gsl_stats_lag1_autocorrelation(const double *data, size_t stride, size_t n);
extern double gsl_stats_correlation(const double *d1, size_t s1, const double *d2, size_t s2, size_t n);
extern double gsl_stats_spearman(const double *d1, size_t s1, const double *d2, size_t s2, size_t n, double *work);

int main(int argc, char **argv) {
    const char *mode = argc > 1 ? argv[1] : "univariate";

    if (strcmp(mode, "univariate") == 0) {
        size_t n;
        if (scanf("%zu", &n) != 1) return 1;
        double *v = malloc(sizeof(double) * n);
        for (size_t i = 0; i < n; i++) if (scanf("%lf", &v[i]) != 1) return 1;
        printf("%.17g\n%.17g\n%.17g\n",
               gsl_stats_mean(v, 1, n), gsl_stats_sd(v, 1, n),
               gsl_stats_lag1_autocorrelation(v, 1, n));
        return 0;
    }

    if (strcmp(mode, "correlation") == 0) {
        size_t n;
        if (scanf("%zu", &n) != 1) return 1;
        double *x = malloc(sizeof(double) * n), *y = malloc(sizeof(double) * n);
        for (size_t i = 0; i < n; i++) if (scanf("%lf", &x[i]) != 1) return 1;
        for (size_t i = 0; i < n; i++) if (scanf("%lf", &y[i]) != 1) return 1;
        double *work = malloc(sizeof(double) * 2 * n);
        printf("%.17g\n%.17g\n",
               gsl_stats_correlation(x, 1, y, 1, n),
               gsl_stats_spearman(x, 1, y, 1, n, work));
        return 0;
    }

    /* regression: SVD, where we and LAPACK both use QR */
    size_t m, p;
    if (scanf("%zu %zu", &m, &p) != 2) return 1;
    gsl_matrix *X = gsl_matrix_alloc(m, p);
    for (size_t i = 0; i < m; i++)
        for (size_t j = 0; j < p; j++) {
            double v; if (scanf("%lf", &v) != 1) return 1;
            gsl_matrix_set(X, i, j, v);
        }
    gsl_vector *y = gsl_vector_alloc(m);
    for (size_t i = 0; i < m; i++) {
        double v; if (scanf("%lf", &v) != 1) return 1;
        gsl_vector_set(y, i, v);
    }
    gsl_vector *c = gsl_vector_alloc(p);
    gsl_matrix *cov = gsl_matrix_alloc(p, p);
    double chisq;
    void *work = gsl_multifit_linear_alloc(m, p);
    if (gsl_multifit_linear(X, y, c, cov, &chisq, work) != 0) return 2;

    printf("gsl\n");
    for (size_t j = 0; j < p; j++) printf("%.17g\n", gsl_vector_get(c, j));
    /* gsl_multifit_linear returns the covariance already scaled by the
       residual variance, so the standard errors are its diagonal directly. */
    double s2 = chisq / (double)(m - p);
    for (size_t j = 0; j < p; j++)
        printf("%.17g\n", sqrt(gsl_matrix_get(cov, j, j)));
    printf("%.17g\n", sqrt(s2));
    return 0;
}
