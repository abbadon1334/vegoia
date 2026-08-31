# Vegoia

Graph algorithms and statistics for PHP — Leiden community detection, centrality,
shortest paths, and numerically sound statistics. **Zero runtime dependencies.**

Tested against the implementations everyone else checks against: `leidenalg` (by
the author of the Leiden algorithm), `networkx`, and the NIST Statistical
Reference Datasets.

---

## Why this exists

PHP has no Leiden. It has no Louvain worth using either, no Brandes betweenness,
and no statistics library that survives the NIST accuracy datasets. So a PHP
application that needs community detection — a GraphRAG pipeline, say, grouping
claims into topics before summarising them — shells out to Python:

```php
// The state of the art in PHP, until now
$process = new Process(['python3', 'sidecar/leiden.py']);
$process->setInput(json_encode(['nodes' => $nodes, 'edges' => $edges]));
$process->run();
```

That works, and it costs you a Python runtime in every container, a second
dependency tree, a serialisation boundary, and roughly half a second of process
startup on every single call. Vegoia removes it.

## Why "Vegoia"

Vegoia — Etruscan *Vecu* — is the prophetess whose doctrine survives in the
writings of the Roman land surveyors. What she is remembered for is the
*limitatio*: the division of land, the drawing and keeping of boundaries.

The Etruscans left us no mathematician; the language is only partly deciphered
and no such name comes down to us. But they did leave a discipline preoccupied
with partitioning — the bronze Liver of Piacenza divides the sky into sixteen
regions, and Vegoia's doctrine divides the earth. Community detection is the
same operation on a graph: take a whole, and find where it should be cut.

Pronounced *veh-GOY-ah*.

## Requirements

PHP 8.5 or later. No extensions required. No runtime dependencies.

```
composer require abbadon1334/vegoia
```

## Quick start

### Community detection

```php
use Vegoia\Graph\Graph;
use Vegoia\Graph\Community\Leiden;
use Vegoia\Graph\Community\Quality\Modularity;

$graph = Graph::undirected(nodes: 34, edges: [[0, 1, 1.0], [0, 2, 1.0], /* ... */]);

$partition = Leiden::modularity(resolution: 1.0, seed: 42)->partition($graph);

$partition->count();          // 4
$partition->communities();    // [[0, 1, 3, ...], [2, 8, ...], ...]
$partition->communityOf(9);   // 2

(new Modularity())->of($graph, $partition);   // 0.4449...
```

The same seed always gives the same partition. Different seeds give different,
equally valid ones — community detection optimises a rugged landscape, it does
not evaluate a function.

When your communities are small relative to the graph, modularity's resolution
limit will merge them however clearly they are separated. Use the Constant Potts
Model instead, where the resolution is a density threshold that means the same
thing at every scale:

```php
Leiden::constantPotts(resolution: 0.05, seed: 42)->partition($graph);
```

### Statistics

```php
use Vegoia\Stats\Descriptive;
use Vegoia\Stats\Regression\LeastSquares;

$stats = Descriptive::of($values);
$stats->mean();        // compensated summation
$stats->stdDev();      // corrected two-pass, not the textbook one-pass
$stats->quantile(0.95);
$stats->autocorrelation(lag: 1);

$fit = LeastSquares::fit($predictors, $response);
$fit->coefficients;
$fit->standardErrors;
$fit->rSquared;
$fit->predict([1.5, 2.0]);

LeastSquares::polynomial($x, $y, degree: 10);   // Householder QR; this actually works
```

### Centrality and paths

```php
use Vegoia\Graph\Centrality\{PageRank, Betweenness, Closeness};
use Vegoia\Graph\Path\{BreadthFirst, Dijkstra};
use Vegoia\Graph\Connectivity;

(new PageRank(damping: 0.85))->of($graph);   // sums to 1
Betweenness::of($graph);                     // Brandes, O(nm)
Closeness::of($graph);                       // Wasserman-Faust corrected

BreadthFirst::distancesFrom($graph, 0);      // hops, -1 where unreachable
Dijkstra::shortestPath($graph, from: 0, to: 42);

Connectivity::components($graph);
Connectivity::inducesConnectedSubgraph($graph, [3, 7, 9]);
```

### Retrieval

```php
use Vegoia\Rag\{Similarity, NearestNeighbours, MaximalMarginalRelevance};

Similarity::cosine($a, $b);
Similarity::jaccard(['a', 'b'], ['b', 'c']);       // 1/3

NearestNeighbours::cosine($query, $corpus, k: 10);

// Relevance alone returns near-duplicates; MMR trades a little for coverage.
MaximalMarginalRelevance::select($query, $corpus, k: 5, lambda: 0.7);
```

## What's inside

```
src/
├── Graph/
│   ├── Graph.php            immutable, compressed sparse row
│   ├── Partition.php        canonical community labelling
│   ├── Connectivity.php     components, induced-subgraph connectivity
│   ├── Community/
│   │   ├── Leiden.php       local moving → refinement → aggregation
│   │   └── Quality/         Modularity, ConstantPotts
│   ├── Centrality/          PageRank, Betweenness, Closeness
│   └── Path/                BreadthFirst, Dijkstra
├── Stats/
│   ├── Descriptive.php      Chan-Golub-LeVeque corrected two-pass
│   └── Regression/          Householder QR least squares
├── Rag/                     Similarity, NearestNeighbours, MMR
└── Support/
    └── CompensatedSum.php   Neumaier summation
```

## How it is tested

This is the part worth reading, because "it has tests" means nothing on its own.

**Statistics are checked against NIST certified values.** The Statistical
Reference Datasets exist to break statistical software. `NumAcc1` is three
8-digit integers whose standard deviation is exactly 1 — the textbook one-pass
variance returns 0. `Filip` is a degree-10 polynomial with a condition number
near 1e15, which the normal equations cannot touch at all. The certified values
are read out of the shipped `.dat` files, never transcribed, so a typo in a test
is not possible.

Accuracy is reported the way the literature reports it — Log Relative Error,
roughly "how many correct significant digits" — so one threshold is comparable
across datasets whose values span nine orders of magnitude. Run
`php tools/accuracy_report.php` to reproduce:

```
dataset        mean   stdDev     r(1)        dataset       p     coef   stdErr
PiDigits      exact    15.21    14.87        Norris        2    14.12    13.89
Lottery       15.18    15.71    14.99        Longley       7    11.68    14.08
NumAcc1       exact    exact    exact        Wampler2      6    13.14    14.54
NumAcc3       15.93     9.46    11.93        Wampler5      6     7.80    14.42
NumAcc4       15.73     8.25    10.73        Filip        11    13.75    13.89
```

The regression figures were checked against LAPACK called directly from C, not
just against numpy, because numpy is a binding and the difference turned out to
matter. Mean correct digits across the collection:

| implementation                | mean digits |
|-------------------------------|-------------|
| **Vegoia**                    | **11.94**   |
| numpy (via OpenBLAS)          | 10.93       |
| OpenBLAS called from C        | 10.84       |
| ATLAS LAPACK called from C    | 10.79       |
| LAPACK `dgelsd` (SVD)         | 9.64        |

There is no single "LAPACK accuracy" here to fall short of: ATLAS and OpenBLAS
differ from each other by more than either differs from this code, and the SVD
path -- the textbook advice for ill-conditioned problems -- is the worst of the
lot. Ahead of numpy on 18 of 22 certified quantities, behind on 2.

Filip is the clearest case. A degree-10 fit over x in [-8.8, -3.1] gives a
Vandermonde with condition number 1.8e15, and LAPACK gets 8.55 digits out of
it. Mapping the predictor onto [-1, 1] first brings the condition number to
2.9e3, and the same solver then returns 13.75 -- the change of variable is
exact, so the extra five digits are conditioning rather than cleverness.

The harness is in `tools/lapack/`, so the comparison can be re-run rather than
taken on trust.

### How these compare to what the reference libraries demand

Worth knowing before reading the table as a scorecard: the bar here is higher
than the one the reference libraries hold themselves to.

Of the NIST collections used, scipy tests only the ANOVA sets, at `rtol=1e-7`
— about seven correct digits. Neither scipy nor numpy tests the linear least
squares or univariate summary sets at all, and leidenalg ships no test suite in
its distribution. This library asserts against all four collections, and at
thresholds measured from what numpy attains rather than at a fixed epsilon.

On SmLs04, the one dataset where the two suites overlap, scipy asks for 7
digits and this reaches 10.43.

Where a figure is below 15, the limit is the arithmetic, not the algorithm:
`10000000.2` is not representable in binary64, so `NumAcc4` is capped near 8
digits for anyone. That is not asserted, it is *measured* —
`tools/generate_nist_attainable.py` records what numpy reaches on the same data,
and the suite requires Vegoia to match it. Lowering a threshold until the tests
pass is exactly the failure mode this design prevents.

**Graph results are checked against `leidenalg` and `networkx`**, which are
generated into `resources/fixtures/graph/` by `tools/generate_graph_fixtures.py`
and committed, so running the suite needs no Python. Centrality, distances and
component counts are asserted to 8–12 digits against networkx on eleven graphs,
including Zachary's karate club, Les Misérables, and a ring of cliques.

**Leiden is stochastic, so it is tested three ways rather than one:**

1. *What is deterministic is asserted exactly.* On three disjoint cliques there
   is one defensible answer and the algorithm must find it, on every seed.
2. *What is stochastic is asserted against a measured envelope.* The fixtures
   record the modularity leidenalg reaches over 50 seeds; Vegoia must land inside
   that band. A broken refinement drops well below it.
3. *What the paper guarantees is asserted as a property.* Leiden's headline
   result over Louvain is that no community is ever internally disconnected.
   That is checkable directly, with no golden values, on every graph and every
   seed — and it is the assertion most likely to catch a subtly wrong
   implementation.

```
$ composer test
OK (482 tests, 4209 assertions)

$ composer stan
[OK] No errors                      # PHPStan level max

$ php tools/jit_check.php
OK: identical results with and without the JIT.
```

That last guard is not paranoia. The tracing JIT is the recommended way to run
these kernels, and results must not depend on it: an earlier optimisation of
Leiden's inner loop produced a different -- and worse -- partition under the
JIT of PHP 8.5.10 than under the interpreter, on the same seed. PHPUnit cannot
catch that class of bug from inside one process, because the JIT is a
process-level setting, so the guard runs the same work in both modes and
compares checksums. It found one real bug before release; the offending
optimisation was removed.

## Performance

Measured on planted-partition graphs with an average degree around 11, median
of repeated runs; reproduce with `php tools/bench_compare.php`. The JIT column
is PHP's tracing JIT -- one ini flag, worth 2-5x on these kernels:

```
php -d opcache.enable_cli=1 -d opcache.jit=tracing -d opcache.jit_buffer_size=128M
```

Leiden community detection, on the SNAP graphs the literature benchmarks
(`python3 tools/fetch_benchmark_graphs.py`) -- real networks, heavy-tailed,
with the hubs a planted-partition generator never produces:

| graph             | nodes   | edges     | **Vegoia + JIT** | leidenalg (C) | ratio |
|-------------------|---------|-----------|------------------|---------------|-------|
| ca-GrQc           | 5 241   | 14 484    | **25 ms**        | 19 ms         | 1.32x |
| ca-HepTh          | 9 875   | 25 973    | **55 ms**        | 42 ms         | 1.31x |
| facebook_combined | 4 039   | 88 234    | **52 ms**        | 34 ms         | 1.53x |
| ca-CondMat        | 23 133  | 93 439    | **196 ms**       | 144 ms        | 1.36x |
| email-Enron       | 36 692  | 183 831   | **534 ms**       | 314 ms        | 1.70x |
| com-dblp          | 317 080 | 1 049 866 | **5 490 ms**     | 5 973 ms      | 0.92x |
| com-amazon        | 334 863 | 925 872   | **6 230 ms**     | 6 306 ms      | 0.99x |

Within 1.3-1.7x of the C reference on mid-sized graphs, and level with it past
300k nodes. Quality is the same or better -- modularity within 0.0002 on every
graph, and ahead on three of five:

| graph       | Vegoia    | leidenalg | difference |
|-------------|-----------|-----------|------------|
| ca-GrQc     | 0.866313  | 0.865842  | **+0.00047** |
| ca-CondMat  | 0.743108  | 0.742007  | **+0.00110** |
| email-Enron | 0.629078  | 0.627303  | **+0.00178** |
| ca-HepTh    | 0.777860  | 0.778026  | -0.00017   |
| facebook    | 0.835694  | 0.835815  | -0.00012   |

Zero internally disconnected communities across all of them.

Honesty requires the next number too: igraph also ships a natively tuned
`community_leiden` that does the 100k graph in about 520 ms -- roughly 2x ahead
of Vegoia with the JIT. That is what a C extension could still chase; nothing a
subprocess architecture can reach.

### Statistics, against Python and C

The graph timings above are the ones that decide whether this library is
usable. The statistical kernels are smaller and the picture is different:

| operation                    | Vegoia + JIT | numpy / scipy | ratio  |
|------------------------------|--------------|---------------|--------|
| ANOVA, 1809 observations     | **0.026 ms** | 0.469 ms      | 0.06x  |
| Pearson, 5000 pairs          | 0.699 ms     | 0.198 ms      | 3.5x   |
| least squares, Filip (82x11) | 0.422 ms     | 0.024 ms      | 18x    |
| lag-1 autocorrelation, 5000  | 1.019 ms     | 0.024 ms      | 43x    |

Where the work is a single pass over a few thousand values, numpy's vectorised
inner loops win by one to two orders of magnitude, and no amount of PHP is
going to change that. Where the work has structure numpy has no primitive for
-- one-way ANOVA is grouping plus two passes -- scipy pays for its generality
and this is sixteen times faster.

The autocorrelation is the clearest trade: it is the slowest ratio here
precisely because it computes exact products, which is what buys the five extra
digits over a plain implementation. Accuracy was chosen over speed knowingly,
in a routine that runs once per series rather than in a loop.

Reaching a C library from outside costs more than either: the GSL harness in
`tools/lapack/` takes 1.3-1.6 ms for the same calls, nearly all of it process
spawn. Below a millisecond of actual work, staying in-process wins whatever the
language.

Other operations at 100 000 nodes, with the JIT: PageRank 220 ms (igraph
385 ms), Dijkstra 51 ms (igraph 20 ms), graph construction 306 ms (igraph
49 ms). Betweenness is O(nm) -- 166 ms at 1 000 nodes and practical to a few
thousand, in any language. Standard deviation over 1 000 000 values: 100 ms.

Every number above holds with results **bit-identical between interpreter and
JIT**; `php tools/jit_check.php` guards exactly that, and the guard has already
caught one optimisation that broke it.

## Design notes

**Zero runtime dependencies, on purpose.** A graph of 100 000 edges is 200 000
integers and 200 000 floats in compressed sparse row form. The same graph as one
object per edge costs an allocation and a pointer chase per step, and Leiden
walks every edge many times per iteration — the abstraction would be paid for in
the inner loop. Typed-collection libraries are excellent for collections of
objects; this is a numeric kernel, and it uses packed arrays.

**Conventions are pinned and stated.** Every measure here has several defensible
definitions, and quietly picking a different one produces numbers that look
reasonable and are not comparable with anyone else's. Undirected self-loops count
twice toward degree (as igraph); betweenness is unnormalised and halved (as
networkx `normalized=False`); closeness uses the Wasserman-Faust correction;
quantiles use R's type 7. Each is documented where it is implemented and enforced
by a test.

**Degenerate input is refused, not guessed.** Cosine similarity against a zero
vector raises rather than returning 0.0, because 0.0 asserts orthogonality — a
claim the data does not support, which would silently rank an empty embedding
above a genuinely unrelated document.

## Regenerating fixtures

Only needed when adding graphs or datasets. Requires `python-igraph`,
`leidenalg`, `networkx`, `numpy`.

```
python3 tools/generate_graph_fixtures.py    # golden graph fixtures
python3 tools/generate_nist_attainable.py   # float64 accuracy ceilings
python3 tools/generate_lls_attainable.py    # ditto, for regression
```

NIST source data lives in `resources/fixtures/nist/` and comes from
<https://www.itl.nist.gov/div898/strd/>.

## References

- V.A. Traag, L. Waltman & N.J. van Eck (2019). *From Louvain to Leiden:
  guaranteeing well-connected communities.* Scientific Reports 9, 5233.
- V.A. Traag, P. Van Dooren & Y. Nesterov (2011). *Narrow scope for
  resolution-limit-free community detection.* Physical Review E 84, 016114.
- M.E.J. Newman & M. Girvan (2004). *Finding and evaluating community structure
  in networks.* Physical Review E 69, 026113.
- S. Fortunato & M. Barthélemy (2007). *Resolution limit in community
  detection.* PNAS 104(1), 36–41.
- U. Brandes (2001). *A faster algorithm for betweenness centrality.* Journal of
  Mathematical Sociology 25(2), 163–177.
- T.F. Chan, G.H. Golub & R.J. LeVeque (1983). *Algorithms for computing the
  sample variance.* The American Statistician 37(3), 242–247.
- J. Carbonell & J. Goldstein (1998). *The use of MMR, diversity-based reranking.*
  SIGIR '98, 335–336.

## Licence

MIT.
