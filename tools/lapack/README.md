# Verifying against LAPACK itself

`tools/generate_lls_attainable.py` measures numpy, and numpy is a binding. These
two programs call the Fortran routines it binds to, directly, so the comparison
does not depend on what numpy does around them.

It turned out to matter. Three LAPACK builds give three different answers on the
same data:

| implementation                  | mean correct digits, NIST LLS |
|---------------------------------|-------------------------------|
| **Vegoia**                      | **11.40**                     |
| numpy (via OpenBLAS)            | 10.93                         |
| OpenBLAS called from C          | 10.84                         |
| ATLAS LAPACK called from C      | 10.79                         |
| LAPACK `dgelsd` (SVD)           | 9.64                          |

So "LAPACK" is not one accuracy. ATLAS scores 7.07 on Filip where OpenBLAS
scores 8.55 -- a difference larger than the one between Vegoia and either. The
SVD path, which is the textbook advice for ill-conditioned problems, is the
worst of all and collapses to 6.24 digits on Pontius.

## Building

No headers required; `liblapack-dev` is not installed on the development
machine either. The Fortran ABI is stable, so the prototypes are declared by
hand.

```sh
# ATLAS/reference LAPACK, 32-bit Fortran integers
gcc -O2 -o lapack_ref lapack_ref.c -l:liblapack.so.3 -l:libblas.so.3 -lm

# The exact OpenBLAS numpy loads: 64-bit integers, symbols prefixed scipy_,
# resolved with dlopen because the linker will not bind them from -l
gcc -O2 -o openblas_dl openblas_dl.c -ldl -lm
./openblas_dl "$(python3 -c 'import numpy,glob,pathlib;
print(glob.glob(str(pathlib.Path(numpy.__file__).parent.parent) + "/**/*openblas*.so*", recursive=True)[0])')"
```

Both read `m n`, then the design matrix row-major, then the response, and print
the coefficients.
