# Changelog

Notable changes, newest first. Versions are calendar-based: `YY.M.patch`.

## Unreleased

### Added

- **`Vegoia\Stats\MultipleTesting`** with `adjust()` and `rejected()`, and the
  `Adjustment` enum: Bonferroni, Holm, Benjamini-Hochberg. Twenty p-values at
  the five per cent level expect one false positive by chance alone, and this
  library hands out p-values freely, so it has to hand out the thing that makes
  a family of them readable.

  There is no default procedure. Bonferroni and Holm control the family-wise
  error rate, Benjamini-Hochberg the false discovery rate, and those are
  different experiments rather than settings of one.

  Checked against statsmodels on ten families chosen for the shapes that
  separate the three, plus the properties a caller relies on without stating
  them: the adjusted family is monotone in the raw p-value, Holm never rejects
  less than Bonferroni, and Benjamini-Hochberg never adjusts further than Holm.

- **Four hypothesis tests**, built on the distributions already here:
  `ChiSquaredTest::independence()`, `TTest::student()` and `::welch()` with
  confidence intervals, `MannWhitneyU::of()` and `KruskalWallis::of()`, plus
  the `Continuity` and `Alternative` enums so there are no boolean mode flags.

  Three conventions were verified against SciPy rather than taken from
  documentation, and two of them contradicted what a first reading suggested.
  Yates' correction is applied by default and only to tables with one degree
  of freedom, and its half-step is clamped to the difference being corrected --
  unclamped, on a table whose every difference is 0.244, the textbook formula
  reports 0.023 where the answer is exactly zero. The H that SciPy reports for
  Kruskal-Wallis is already divided by the tie-correction factor. And
  Mann-Whitney implements the asymptotic route only, with fixtures generated
  by passing `method='asymptotic'` explicitly, because `auto` switches route
  on sample size and on ties and the two routes disagree.

  Kruskal-Wallis uses the centred form of H rather than the textbook one,
  which differences 3(N+1) from a quantity of the same size while H stays
  O(1): measured against exact rationals it holds 11.95 digits at N = 60000
  where the centred form holds 13.54.

- **`Vegoia\Graph\Distance`** and **`Vegoia\Graph\Assortativity`**:
  eccentricity, diameter, radius, mean shortest path length, and Newman's
  degree assortativity.

  One object rather than four functions, because one sweep produces all four
  and it is the most expensive thing in the library -- 1.4 seconds on a
  thousand nodes -- so four static entry points would run it four times.

  NetworkX raises on a disconnected graph; this returns `INF`, which is the
  answer rather than a guess. The eccentricities are measured within each
  node's own component and so stay finite, which is what makes the result
  usable on the two fixtures that are disconnected, one of them the largest.

  Assortativity refuses a regular graph, where the correlation is 0/0 --
  `Correlation::pearson` already refuses a sample with no variation and this is
  literally that. A self-loop contributes no pair of endpoints; all three
  readings of that differ and NetworkX agrees with neither obvious one, so it
  is pinned by hand against a direct Pearson computation.

- **`Vegoia\Rag\ReciprocalRankFusion`**, for combining a vector ranking with
  a keyword one. A BM25 score and a cosine similarity are not comparable and do
  not become comparable by rescaling, so the scores are thrown away and only
  the positions kept.

  There is no canonical implementation to check against -- LangChain,
  Elasticsearch and Weaviate read the conventions differently -- so the fixture
  computes the scores as exact rationals instead.

- `InvalidArgument::notANumber()`. PHP 8.5 warns when NAN is coerced to a
  string, so a range message reporting a NAN emitted a warning while reporting
  an error. A NAN is a different failure from a value out of bounds anyway,
  and it reaches this library from its own output: `Fit::pValue()` returns one
  for a zero coefficient with a zero standard error.

### Fixed

- **`Fit::fStatistic()` and `::overallPValue()` refused a model fitted through
  the origin.** The reasoning was that the F compares against an
  intercept-only model, and without an intercept there is nothing to compare
  with. But the smaller model is the one with the *slopes* removed, which
  without an intercept is `y = 0` -- a real model, and the comparison every
  reference makes. NIST certifies 15750.25 for NoInt1 and 298.6666666666667
  for NoInt2, and statsmodels reports both; refusing made this library the odd
  one out, and contradicted the total sum of squares beside it, which is
  already measured about zero for such a model.

  Still refused where it is genuinely undefined: a model with no slopes at all
  is the smaller model, and comparing it with itself asks nothing.

- **`Similarity` and `LabelledGraph::fromMatrix` read arrays by key rather
  than by position**, and were silently wrong on anything that was not a list.

  `Similarity::dot`, `::euclidean` and `::cosine` walk two vectors together
  using the first one's keys. Given a vector left keyed 5, 9, 12 -- which is
  what `array_filter` returns after dropping a dimension -- every lookup
  missed, and PHP answers a missing key with null and a warning rather than an
  error, so the sum completed and returned a number. Worse, keys that all
  exist in the wrong order raised nothing at all: `[2 => 1.0, 1 => 2.0,
  0 => 3.0]` against `[4.0, 5.0, 6.0]` gave a dot product of 28 where the
  answer is 32. `MaximalMarginalRelevance` inherited it, since it scores
  candidates through the same door.

  `fromMatrix` used a row's key directly as a node index, so a matrix keyed
  2, 5, 9 came back as a graph with **no edges at all** -- every column index
  fell below its row index and was skipped as the lower triangle. Labels
  failed in the mirror image: three of them under keys 3, 7, 11 passed the
  count check, missed every lookup, and left the nodes named 0, 1, 2.

  Both now read by position, which is what the declared `list` types mean.
  This is the same defect `Correlation` had, where a Pearson coefficient came
  back 0.596 instead of 0.965; the signatures are widened to
  `array<array-key, ...>` to say that any keying is accepted and normalised.

- **`Katz::spectralRadius()` refused long paths.** It raised when its power
  iteration had not settled in a thousand steps, which a 201-node path does
  not: the two largest eigenvalues are 1.999758 and 1.999033, and separating
  them takes about 152,000 iterations. Raising the ceiling costs 34 seconds on
  a thousand-node path, so it now falls back on the maximum strength, which
  bounds the spectral radius without iterating. The bound can only reject an
  alpha that would have converged, never accept one that would not, and on the
  graphs where it triggers it gives up 0.012% of the usable range.

### Changed

- **The residual sum of squares is formed without collapsing the compensated
  fitted value first.** Subtracting a fitted value from an observed one is a
  cancellation, and cancellation is exactly when a discarded compensation stops
  being negligible; subtracting the head first also makes the subtraction exact
  by Sterbenz's lemma when the two are within a factor of two. Worth 0.17 of a
  digit on NIST Pontius, where the residual is 1e-7 of the response, and 0.06
  on Norris. `CompensatedSum::subtractedFrom()` is the new operation, alongside
  `exponentiated()` and `dividedBy()`, which exist for the same reason.


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
