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

## What is not here: igraph's C library

`libigraph.so.3` is installed and exports everything needed, but calling it
from C with hand-declared prototypes did not work: igraph 0.10 widened
`igraph_integer_t` to 64 bits and reshaped several structs, so a guessed layout
segfaults rather than misbehaving visibly. Getting it right needs the headers,
which means installing `libigraph-dev`.

It would have added little. python-igraph is a thin binding, and the graph
operations here run for milliseconds to seconds -- long enough that the cost of
crossing into Python is a small fraction of the total, unlike the least squares
calls where the work is microseconds and the overhead would dominate. The
Python-side igraph timings are used for graphs; the C oracles are used for the
statistics, where the difference matters.
