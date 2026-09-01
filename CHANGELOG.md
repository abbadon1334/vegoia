# Changelog

Notable changes, newest first. Versions are calendar-based: `YY.M.patch`.

## 26.9.3 — 2026-09-01

### Fixed

- **`Correlation::kendall()` returned a number that was neither answer when
  two values collided as strings but not as doubles.** Tau-b counts tied pairs
  in its denominator and ordered pairs in its numerator, and the two halves
  disagreed about what a tie was: the denominator grouped values by
  `(string) $value`, and PHP's default `precision` of 14 makes distinct
  doubles collide.

  ```
  (string) 0.1 === (string) 0.10000000000000012   // true
  0.1 === 0.10000000000000012                     // false
  ```

  On `x = [0.1, 0.10000000000000012, 0.2, 0.3, 0.4, 0.5]` against
  `y = [1, 2, 3, 6, 4, 5]` it returned 0.75907, where the answer is 0.73333 and
  the answer had the pair genuinely been tied would have been 0.69007. Nothing
  about that input is contrived — it is 0.1 and a few units in the last place
  above it, which any arithmetic produces.

### Added

- `Vegoia\Stats\Ranks`, with `midranks()`, `tieSizes()` and `tiedPairs()`.
  Extracted so that everything needing to know what a tie is agrees, which is
  what fixes the above; `Correlation` delegates to it. It exists in its own
  right because Mann-Whitney and Kruskal-Wallis will need the same definition.

  Writing its tests found a second defect on the way: `midranks([])` called
  `range(0, -1)`, which counts downwards and returns `[0, -1]` with a warning.
  `Correlation` never reached it, refusing samples below two long before
  ranking them, but a public primitive has no such guard.

## 26.9.2 — 2026-09-01

### Changed

- **Iterative methods now raise instead of returning an unsettled answer.**
  PageRank, eigenvector centrality, HITS and Katz throw `DidNotConverge` if
  they reach their iteration ceiling without converging, where they used to
  return whatever they were holding. Katz already refused for its own
  divergence case, and the rest disagreed with it — the same class of failure
  answered two different ways. Katz's internal estimate of the largest
  eigenvalue refuses too: that number decides whether the caller's alpha is
  inside the radius of convergence, so an estimate that has not settled can
  wave through an alpha whose series diverges.

  Nothing in the test collection provokes it: on every graph, up to a thousand
  nodes and sixteen thousand edges, the defaults converge in a small fraction
  of the iterations allowed. Code that was getting a correct answer keeps
  getting it.

### Fixed

- Seven skipped tests were skipping on a false claim — that `predict()` cannot
  take a polynomial fit's predictors, which the test in the next file does.
  Nine more were skipped because "Dijkstra reduces to BFS on an unweighted
  graph", which is true of the mathematics and says nothing about whether this
  implementation does. Both now run. 23 skips down to 5, and 541 more
  assertions for no new test methods.

## 26.9.1 — 2026-09-01

First release.

### Graphs

- **Community detection.** Leiden (Traag, Waltman & van Eck 2019) against four
  objectives — modularity, Constant Potts, Erdős–Rényi Potts,
  Reichardt–Bornholdt — with the guarantee that distinguishes it from Louvain:
  every community it returns is internally connected. Surprise and
  Significance are available for scoring a partition. Label propagation, which
  optimises nothing and runs in linear time, for large graphs and for second
  opinions. NMI, adjusted Rand index and variation of information for
  comparing two partitions.
- **Centrality.** PageRank, personalised PageRank, Brandes betweenness (also
  weighted), closeness with the Wasserman–Faust correction, harmonic, Katz,
  eigenvector, HITS.
- **Structure.** Connected components, strongly connected components by
  Tarjan, bridges and articulation points by Hopcroft–Tarjan, k-cores,
  triangles, local clustering and transitivity, minimum and maximum spanning
  forests by Kruskal.
- **Paths.** Breadth-first distances, Dijkstra with paths.
- **Link prediction.** Common neighbours, Jaccard, Adamic–Adar, resource
  allocation and preferential attachment, pairwise or ranked over the
  candidates two hops away.
- **Interop.** `LabelledGraph` builds from edge lists, adjacency maps or
  adjacency matrices and gives results back under the names they came in with.
  `Graphp` reads a graph built with `graphp/graph`, which stays a development
  dependency.

### Statistics

- **Descriptive.** Compensated summation throughout; Chan–Golub–LeVeque
  corrected two-pass variance, skewness, kurtosis, quantiles (R type 7),
  autocorrelation. Two precision modes, since the accurate one costs about ten
  times the fast one and is not always what you need.
- **Correlation.** Pearson, Spearman, Kendall tau-b.
- **Regression.** Least squares by Householder QR with iterative refinement,
  polynomial fits that survive a condition number of 1.8e15, full coefficient
  covariance, t statistics and two-sided p-values per coefficient, confidence
  intervals, the overall F test, and intervals for both a fitted mean and a
  single new observation.
- **Analysis of variance.** One-way, with the p-value that makes an F readable.
- **Distributions.** Normal, Student's t, chi-squared and Fisher–Snedecor,
  with both tails first-class: `survival()` is computed directly rather than
  as `1 - cumulative()`, and quantiles are offered against either tail. At
  z = 10 the difference is between 7.6e-24 and exactly zero.
- **Special functions.** erf, erfc, log-gamma, and the regularised incomplete
  gamma and beta. PHP ships none of them, which is the concrete reason a PHP
  program could not turn an F statistic into a p-value without shelling out.

### Retrieval

- Cosine and Jaccard similarity, k nearest neighbours, maximal marginal
  relevance.

### On accuracy

Every numeric claim is checked against a certified value from an independent
source, with the reference implementation's own accuracy as the bar to clear:
NIST StRD for descriptive statistics and regression, mpmath at 50 digits for
the special functions and distributions with SciPy as the ceiling, statsmodels
for inference, and igraph, leidenalg and NetworkX for the graph algorithms.
Beating the reference passes rather than fails, which matters more often than
it sounds — at the first row of Wampler5 the true value is exactly 1,
statsmodels returns 1.0000011 and this library returns 1.0000000014.

On NIST's linear least squares collection this reaches 11.94 mean correct
digits against numpy's 10.93 and 10.84 for OpenBLAS called directly from C.

### Requirements

PHP 8.5 or later. No runtime dependencies.
