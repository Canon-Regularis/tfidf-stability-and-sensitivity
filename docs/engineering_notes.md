# Engineering notes

Decisions the code makes that a comment of four lines cannot justify. Each
section is named by the slug that a `Rationale:` comment in the source cites:
the comment states what the code does, and this page states why.

- [Python package](#python-package): 78 notes
- [C++ core and binding](#c-core-and-binding): 59 notes
- [Test suite](#test-suite): 61 notes
- [Mutation tooling](#mutation-tooling): 6 notes
- [Scripts and gates](#scripts-and-gates): 25 notes
- [Build, CI and project configuration](#build-ci-and-project-configuration): 30 notes

## Python package

### A1 margin regimes

Source: `src/tfidf_stability/analysis/stability_profile.py`.

Section 4.4 proves a sufficient condition: if every score moves by less than m_k / 2, the top-k set is unchanged. The observed behaviour differs from the bound in two directions.

Worst case. The Stage 4 dyadic witness makes m_k / 2 the flip radius: a perturbation of size m_k / 2 + delta flips the boundary pair, so the bound is tight and needs no sampling.

Average case. Under independent uniform perturbation in [-eps, +eps], the flip rate stays at zero well past eps = m_k / 2. The adversarial configuration (the rank-k score pushed down and the rank-(k+1) score pushed up, both by the full eps) is a measure-zero corner of the perturbation cube. In the measurement recorded in [ranking_stability.md](ranking_stability.md) (k = 10, section 7.1 leave-one-out queries, 40 trials each), the flip rate is 0% for eps from 0.25 to 1.01 times the radius m_k / 2, 0.15% at 1.1 times, 22.35% at 2 times and 60.61% at 5 times.

Neither result implies the other. Reporting only the average case understates the risk; reporting only the worst case overstates how fragile rankings are. transition_curve measures the average case and certificate_audit measures the soundness and conservatism of the bound.

The certificate as a 2x2 table. certified_stable is a proof when true and only "not covered" when false, so plain accuracy would reward a certificate that always says no. The table separates the two error directions. The certified-but-changed cell must be zero: a non-zero entry falsifies section 4.4 and indicates a bug or a broken proof. The uncertified-but-unchanged cell is the conservatism and is expected to be large; reporting it prevents "not certified" being read as "will break". The audit classifies each trial by its realised movement, not the drawn eps, and draws eps uniformly from 0 to max_ratio times the certified radius so both cells of each row are populated; drawing only small perturbations would make the certificate trivially sound.

The A1/A2 boundary. Queries whose m_k is zero are excluded from the transition curve and the audit and are counted: in n_excluded by transition_curve and in n_exact_tie by certificate_audit. At m_k = 0 the boundary is an exact tie, no perturbation is needed to change the outcome, and the tie-break decides. That is A2's regime; including it would let a tie-break effect read as a numerical-stability effect. G3 makes the same separation for margin distributions: it keeps exact ties in them and reports P(m_k = 0) as a statistic of its own, and it sends the zero query vector, at which m_k = 0 for every k, to the tie-break ablations rather than to the margin distributions.

Per-query attribute tables. Under section 7.1's protocol each query excludes its own profile items and so ranks over a different candidate set (G19). A restricted score vector ranked against the full-corpus table would let the tie-break consider documents the query cannot retrieve, so both functions accept per-query tables. Queries with k at or beyond their candidate count are counted rather than clamped (in n_excluded by transition_curve, in n_undefined by certificate_audit), because a clamped k measures a different quantity.

### Abi version check

Source: `src/tfidf_stability/_native/__init__.py`.

The Python package and the compiled extension each hold a separate copy of the ABI string: REQUIRED_ABI in tfidf_stability._native and __abi__ in the extension. The loader compares the two at import, so a mismatch identifies a build that is stale against the current C++/Python contract. The ABI string is bumped whenever that contract changes shape: a binding is added, removed or changes signature. The check reads __abi__ and not the release __version__, because bindings can be added between releases: an extension that lacks them can still report the current release version, so a version check would accept a build that is missing bindings the Python side calls. A build that predates __abi__ reports None and is rejected in the same way as any other mismatch. require_native raises AbiVersionMismatchError for a stale build and NativeBackendUnavailableError for a missing one, since the fixes differ (rebuild against the current contract, or build at all); the former subclasses the latter, so callers that do not distinguish them are unaffected.

### Benchmark design

Source: `src/tfidf_stability/benchmarks/tfidf_perf.py`.

The C++20 core is an optimisation, so it must be fast and stay bit-identical. The differential suite enforces bit-identity; the benchmark module measures speed. Four rules shape it.

A benchmark is not a correctness escape hatch. Every comparison with a native side proves that the two backends produce bit-identical results before it times either side, and raises BitIdentityError otherwise. Timing output is not read for wrong answers, so the check must stop the run.

The minimum is reported, not the mean (see [Benchmark minimum not mean](#benchmark-minimum-not-mean)).

Reference-only is a first-class mode: on an install without the compiled extension, or with use_native=False, the benchmark reports reference-only timings rather than an error.

Comparisons are like-for-like or carry a note. The index-build row pairs operations of different scope: the native index constructor also builds the inverted index that TAAT scoring needs, which has no reference counterpart, and its note states that the ratio therefore understates the norm kernel alone. The ranking row pairs the native permutation kernel with the native score sort so that both sides do the same work as the reference rank. The reduction rows note that the native side receives a prepared float64 array and that marshalling is excluded from both sides.

The workload is the seeded synthetic corpus, which contains exact duplicates and twin pairs. The ranking rows need these ties: on distinctly scored documents the tie-break is never consulted, so random scores would time only the first component of the comparator.

### Benchmark minimum not mean

Source: `src/tfidf_stability/benchmarks/tfidf_perf.py`.

Timing noise is one-sided: a run interrupted by another process, a page fault or a cache eviction can only be slower. The mean therefore estimates the true cost plus a machine-dependent noise term, while the minimum is the sample closest to interference-free execution. The minimum is biased low against the cost seen on a busy machine. That is the correct trade when comparing two implementations, because a shared additive noise term only obscures the difference between them.

One untimed warm-up call precedes the calibration and the timed batches, so lazy imports, a cold branch predictor and freshly allocated arenas fall outside the measurement rather than on the first batch. Each batch runs with the garbage collector disabled, because a collection is charged to whichever call triggers it, unrelated to that call's cost.

### Bit exactness

Source: `src/tfidf_stability/utils/numerics.py`.

The Python module tfidf_stability.utils.numerics is the Python half of the bit-exactness contract. Its reduction policies have counterparts in `cpp/include/tfidf/core/reduction.hpp` that produce byte-identical results, and the differential suite asserts this. Its floating-point environment probes have a counterpart in `cpp/include/tfidf/core/fp_guard.hpp`. The logarithm has no C++ counterpart: the C++ core receives idf as data.

Summation order is part of the specification: the normative reading of README section 2's unbracketed sums is a left-to-right fold (Reduction.NAIVE), and the other policies are instruments for the noise floor. See [Reduction policies](#reduction-policies).

The logarithm is computed by correctly_rounded_log_ratio rather than by the platform libm. See [Exact log](#exact-log).

### Canonical hashing

Source: `src/tfidf_stability/utils/hashing.py`.

A published number is traced back to its data, config and build only if the same input always produces one digest. The module tfidf_stability.utils.hashing therefore fixes one hashing form for each kind of input. Some modules (tfidf_stability.preprocessing.lemmatise, preprocessing.pipeline, preprocessing.stopwords, ranking.attributes, ranking.sort_keys, vectorisation.tfidf and vectorisation.vocabulary) call hashlib.sha256 directly rather than through it.

Digests are taken over bytes, not over renderings. A float is hashed through its raw binary64 bit pattern, packed little-endian. The output of repr is lossless in CPython, but that is a language guarantee and not a file-format guarantee. The bit pattern also keeps a one-ulp difference visible in the digest, and one-ulp differences are what this project measures.

Digests are taken over a canonical form. JSON is rendered with sorted keys, no incidental whitespace and ensure_ascii=False, so two configs that differ only in key order share a digest, and a non-ASCII character is hashed as its UTF-8 bytes rather than as a \uXXXX escape, the same literal form that canonical_json in tfidf_stability.utils.io writes. Text has CRLF replaced by LF before encoding as UTF-8 (a lone CR is left unchanged), so a CRLF copy and an LF copy of the same file give the same digest. The .gitattributes rule `* text=auto eol=lf` makes git store and check out LF, but a file written on Windows and hashed before it is committed, or a file from outside the repository, can still carry CRLF.

### Certificate is a bound

Source: `src/tfidf_stability/perturbation/score_bounds.py`.

certified_radius reports the largest perturbation a ranking provably tolerates at rank k: the set radius m_k/2 and the order radius m_min^top/2. It claims nothing about typical perturbations. README section 3 describes observed margins as "empirical certificates of stability", and section 7.2 reports the corresponding flip radii; reading them as expected behaviour inverts their meaning. Both radii depend only on the sorted score multiset, so a certificate is a property of the scores and is the same under pi, pi_score and pi_alt.

Neither radius dominates the other: m_min^top minimises over the gaps strictly inside the top-k (ranks 1->2 through (k-1)->k) and m_k is the boundary gap (k->k+1), so the two gap sets are disjoint. A tight cluster at the top with a wide boundary gives order_radius < set_radius; a well-spread top with a near-tied boundary gives the reverse. A certificate quoted without naming the invariant it certifies is therefore ambiguous. StabilityCertificate carries both radii, and joint_radius covers callers who need both guarantees: it is their minimum, or the set radius alone when the order radius is undefined (k = 1).

### Cosine lipschitz constant

Source: `src/tfidf_stability/similarity/geometry.py`.

README section 4.3 states that |cos(u', v') - cos(u, v)| <= C (||u' - u|| + ||v' - v||) "for a constant C depending on lower and upper bounds on the norms" and leaves C unspecified, so it cannot be computed, tested or used. The module `similarity/geometry.py` fixes it (see docs/spec_addenda.md#g4).

Theorem. For non-zero u, v, u', v' and L = min(||u||, ||v||, ||u'||, ||v'||):

    |cos(u', v') - cos(u, v)| <= (1/L) (||u' - u|| + ||v' - v||),   i.e. C = 1/L.

Proof. Write u_hat = u / ||u||.

1. |<u_hat, v_hat> - <u_hat', v_hat'>| <= |<u_hat - u_hat', v_hat>| + |<u_hat', v_hat - v_hat'>| <= ||u_hat - u_hat'|| + ||v_hat - v_hat'||, by the triangle inequality and Cauchy-Schwarz with unit vectors.
2. In an inner-product space the Dunkl-Williams inequality holds with the sharp Hilbert-space constant 2: ||u/||u|| - u'/||u'|| || <= 2 ||u - u'|| / (||u|| + ||u'||).
3. ||u|| + ||u'|| >= 2L, so ||u_hat - u_hat'|| <= ||u - u'|| / L; likewise for v.

The module also provides a tighter non-uniform form, which applies Dunkl-Williams per vector without collapsing to a single L: 2 ||du|| / (||u|| + ||u'||) + 2 ||dv|| / (||v|| + ||v'||).

A bound that depends only on the corpus, with no reference to the perturbation, is C <= sqrt(max nnz). It follows because ||tf_i||_1 = 1 gives ||tf_i||_2 >= 1/sqrt(nnz_i) by Cauchy-Schwarz, and idf >= 1 termwise can only increase the norm, so ||w_i|| >= 1/sqrt(nnz_i). This makes the section 6 remark that "cosine similarity becomes unstable for low-norm vectors" a quantitative statement. The bound grows with the largest support: an L1-normalised tf spreads its mass over many distinct terms and has a small L2 norm, so the instability is worst for documents with many distinct terms, while a document with few distinct terms has a norm close to its idf weights.

Property tests check the perturbation bounds by searching for a violation, not by confirming a few examples: `test_lipschitz_bound_survives_an_adversarial_search` in `tests/test_perturbation_bounds.py` covers the uniform and per-vector forms, and `test_three_term_bound_is_never_violated` in `tests/test_cosine_edge_cases.py` covers the section 4.2 bound. The corpus-level bounds, norm_lower_bound and corpus_lipschitz_bound, are checked against the fitted vectors of the mini corpus fixture.

### Dataset provenance

Source: `src/tfidf_stability/datasets/loaders.py`.

`DATASET_NAMES` in `src/tfidf_stability/datasets/loaders.py` is the dataset registry, and `load_dataset` turns a registered name into records. `configs/datasets.yaml` documents the same datasets but no code loads it, so editing a value there changes no behaviour. The experiment scripts and the notebooks that use a registered dataset load it through `load_dataset`, and none of them depend on whether the corpus is generated, downloaded or read from disk. The CLI does not call `load_dataset`; it reads a JSONL corpus file directly. A corpus already on disk is accepted as `jsonl:<path>`; this is how a MovieLens-derived corpus is consumed once `scripts/build_corpus.py` has written it.

`LoadedDataset` carries a provenance block that goes verbatim into the run manifest. A synthetic result and a MovieLens result therefore stay distinguishable from the manifest alone, down to the archive digest or generator spec behind each. Without the block the two differ only by filename, and filenames are not preserved when results are copied.

The two datasets are handled differently because of their licences. Synthetic data is redistributable and is regenerated from its spec on demand, so a synthetic result is reproducible from the repository alone. MovieLens may not be redistributed, so it is loaded from a local archive that the repository does not contain, and its provenance records `redistributable: false` together with the archive SHA-256 and the interaction threshold. See `data/README.md`.

### Display tau

Source: `src/tfidf_stability/analysis/noise_floor.py`.

TauBand.display_tau returns a tau for prose and figure captions: the geometric midpoint sqrt(tau_floor * g_min) of the band. It is a presentation value, so that a caption can name a number without implying the number was fitted; it is not a derived constant. TauBand.is_invariant is the claim, and it states that the choice of value within the band is immaterial. An empty band returns NaN. Two degenerate bands are reachable. When no strictly positive gap exists, every score is identical and g_min is infinite; the midpoint would be inf (NaN if tau_floor is also zero), and at inf every tie ball contains the whole corpus. No gap exists to cross, so every admissible tau gives the same structure and the smallest, tau_floor, is returned; this case is checked first. When every compared policy (naive, Neumaier, pairwise) agrees with Reduction.EXACT on every score of the corpus, tau_floor is zero and the midpoint collapses to 0. That value is legal under G3 but is the exact-tie baseline, so g_min / 2 is returned instead.

### Exact log

Source: `src/tfidf_stability/utils/numerics.py`.

math.log delegates to the platform libm, which IEEE-754 does not require to be correctly rounded. Over N in {100, 610, 9742, 20000, 50000} and all valid df, the platform logarithm log((1 + N) / (1 + df)) differs from the correctly-rounded one in 44.5% of cases, and the idf value log((1 + N) / (1 + df)) + 1 in 15.16% of cases (docs/spec_addenda.md#g13). The +1 moves the value into a coarser binade, which rounds away most of the 1-ulp disagreements. The remaining difference makes idf, and every weight, norm and score computed from it, platform-dependent at the 1-ulp level, and UCRT, glibc and Apple libm each round differently.

correctly_rounded_log_ratio evaluates the exact rational ratio in decimal.Decimal at DECIMAL_LOG_PRECISION (60) digits and rounds once to binary64. Sixty digits is well above the roughly 17 needed to round binary64 correctly, and the result agrees with a 120-digit evaluation on every ratio the project produces. The idf vector holds O(|V|) values, computed once and outside any hot loop, so the cost of a few microseconds per vocabulary entry is immaterial.

Computing idf in Python also removes the only transcendental function from the native pipeline: the C++ core receives idf as data and computes with correctly rounded operations alone. LogImpl.PLATFORM keeps the libm path available so that the difference (the G13 gap) can be measured; only LogImpl.CORRECTLY_ROUNDED is valid for published results.

The division happens before the logarithm, matching section 2.1. log(a/b) and log(a) - log(b) differ in 94.53% of the ratios that arise at N = 9742.

Arguments: numerator is the exact integer 1 + N (section 2.1); denominator is the exact integer 1 + df(t). Both must be positive, otherwise ValueError is raised. The return value is the correctly-rounded natural logarithm of the exact rational ratio.

### Exact tf recovery

Source: `src/tfidf_stability/vectorisation/tfidf.py`.

The ``tf`` field reported by ``TfidfModel.intermediates`` (and displayed by ``tfidf-stability inspect``) is not computed as ``w / idf``. The stored weight is ``fl(tf * idf)``, and dividing it back by ``idf`` adds a second rounding that does not cancel the first. Over a sweep of 184,080 realistic (N, df, L, count) cases the round trip misses the true ``tf`` by one ulp in 9.01% of them; for example, N=100, df=1, L=6, count=5 gives 0x1.aaaaaaaaaaaaap-1 for a true 0x1.aaaaaaaaaaaabp-1. An error of that size is the quantity the project studies, so the inspected field must be exact. Because ``tf`` is ``count / L`` for an integer count, ``_exact_tf`` computes ``weight / idf * L``, whose error is far below 0.5, rounds it to the integer count and divides by ``L`` once. The result is exact on all 184,080 cases of the sweep. `tests/test_tfidf_sparse_ops.py` runs a smaller grid of the same kind (more than 10,000 cases), asserts that the reconstruction is bitwise exact on every case, and asserts that the naive form fails somewhere in that grid, which shows that the reconstruction is necessary.

### Field boundaries

Source: `src/tfidf_stability/preprocessing/pipeline.py`.

PreprocessingPipeline.preprocess_fields holds a hard boundary between the fields of a document assembled from separate parts, such as a title, a genre list and a set of free-text tags. No continuous text crosses those seams, so an n-gram spanning one is produced by the assembly order rather than found in the data. Each field is tokenised alone and the token streams are joined with the GAP sentinel, on which generate_ngrams segments. Joining the fields into one string first does not hold the seam: GAP is a control character, which normalise deletes under the default strip_control=True, and the seam closes. insert_gaps=False drops every sentinel, these included, and cross_gaps=True bridges them. Both are ablations of this boundary, so under either setting the seams are not held.

### Fks cases

Source: `src/tfidf_stability/ranking/distances.py`.

Every unordered pair drawn from the union of the two lists contributes according to how it is witnessed:

- Case 1: both elements appear in both lists. Contributes 1 if the lists order them oppositely, else 0.
- Case 2: both elements are in one list and exactly one is in the other. Contributes 1 if the list holding both ranks the element the other list is missing first, else 0.
- Case 3: one element appears only in the first list and the other only in the second. Contributes 1.
- Case 4: both elements are in one list and neither is in the other. Contributes p.

Cases 2 and 3 carry the generalisation. Both encode the reading that an element present in a list is ranked above one absent from it: case 3 then disagrees unavoidably, and case 2 disagrees when the list holding both puts the absent element first.

The implementation is a direct O(k^2) enumeration over the union. At k <= 50 the union holds at most 100 elements, so at most 4950 pairs, and legibility is preferred over asymptotics. `test_fks_reduces_to_inversion_count_when_the_sets_match` in `tests/test_ordering_distances.py` cross-checks it against inversion_count when both lists hold the same set, where only case 1 can apply.

Arguments: a and b are two top-k lists, best first, which may rank different sets; penalty is the case-4 penalty p in [0, 1] (G2); normalise divides by `fks_max(max(len(a), len(b)), penalty)` so the result lies in [0, 1], and gives 0.0 when that ceiling is not positive. The result is 0.0 when both lists are empty or identical.

A penalty outside [0, 1] raises ValueError. `kendall_fks` checks it before the enumeration, so the unnormalised path, which never calls `fks_max`, rejects it too. `fks_max` applies the same check because the final division cannot catch a bad penalty: unchecked, the ceiling would be NaN for a NaN penalty and negative for a sufficiently negative one, ceiling > 0.0 would be false in both cases, and the normalisation would report disjoint lists as being in agreement.

### Fks distance

Source: `src/tfidf_stability/ranking/distances.py`.

The paper asks for an ordering distance in two places, and they are different problems. Section 4.5 asks for a distance between orderings restricted to tie groups; section 7.3 asks for within-top-k reordering restricted to tie-affected subsets. Treating them as one problem produces a quantity that appears defined and is not.

Restricted to a tie group the problem is well-posed. A tie group is defined on the score vector, which pi, pi_score and pi_alt share, so the two orderings contain the same elements. `kendall_tau_distance` is plain normalised Kendall tau computed by a merge-sort inversion count in O(n log n).

Restricted to top-k the problem is ill-posed. When topk(pi) != topk(pi'), which is the case section 7.3 measures, the two lists rank different sets and the number of discordant pairs has no meaning. The module resolves this with the Fagin-Kumar-Sivakumar generalised Kendall distance (Comparing Top k Lists, SIAM J. Discrete Math. 17(1), 2003) at penalty p = 1/2.

K^(p) is a near-metric, not a metric, and p = 1/2 is chosen on bias grounds ([Fks penalty](#fks-penalty)). tests.test_ordering_distances.test_fks_is_a_near_metric_not_a_metric guards the triangle-inequality witness; see [Fks near metric and singleton maximum](#fks-near-metric-and-singleton-maximum).

No single measure is reported alone. `compare_top_k` returns the set-disagreement indicator section 7.3 asks for, normalised FKS, intersection Kendall with its support size, and Jaccard, because each is blind to something the others see. K_int cannot see membership change at all, and membership change is the effect under study.

### Fks penalty

Source: `src/tfidf_stability/ranking/distances.py`.

FKS_PENALTY is the Fagin-Kumar-Sivakumar penalty for a pair whose two elements both appear in one top-k list and neither appears in the other.

p = 1/2 is the neutral choice. With no information about the relative order of two elements absent from a list, they disagree with probability one half, so 1/2 is the unbiased estimate of the contribution. p = 0 assumes the unseen pair agrees and biases every measurement downwards, which is the wrong direction for a study of instability; p = 1 biases upwards.

No penalty in [0, 1] makes K^(p) a metric ([Fks near metric and singleton maximum](#fks-near-metric-and-singleton-maximum)). A near-metric has bounded distortion without the triangle inequality. This suffices for reporting disagreement rates, and for this reason the module never clusters on the quantity or treats it as a norm.

### Flip witness

Source: `src/tfidf_stability/perturbation/score_bounds.py`.

Section 4.4 of the README states eps < m_k/2 as a sufficient condition for top-k set invariance and does not address necessity. flip_witness(scores, order, k, delta=None) constructs a perturbation that breaks the condition at eps = m_k/2 + delta. It moves only the two documents straddling the boundary, past each other: s'[r_k] = s[r_k] - eps and s'[r_{k+1}] = s[r_{k+1}] + eps, so |ds_i| <= eps for every document while s'[r_{k+1}] > s'[r_k].

Arguments: scores is the unperturbed score vector, indexed by document; order is a ranking's document order, best first; k is the boundary rank, 1-indexed; delta is how far past the radius to go. The default delta is scaled to the margin (margin/2 x 1e-6) and floored at 4 ulp of max(|s[r_k]|, 1) so the excess survives rounding.

The function returns (perturbed_scores, eps), or None when no witness exists: k is out of range for the shorter of scores and order, or m_k = s[r_k] - s[r_{k+1}] <= 0 and there is no radius to exceed. It also returns None if the rounded result does not reverse the pair.

Where the margin is dyadic and normal the construction is exact: m/2 only shifts an exponent and the additions land on representable values. Otherwise the caller checks the realised perturbation against eps rather than assuming it, because fl(s + eps) can differ from s + eps by up to half an ulp; a test that assumes exactness is flaky.

### Full refit

Source: `src/tfidf_stability/perturbation/experiments.py`.

run_perturbation fits the baseline and the perturbed corpus from scratch with two independent TfidfVectoriser fits. The pipeline is not incremental. A partial update that shares state with the baseline is the most likely way to hide a real perturbation effect, because any stale document frequency, IDF value or norm carried over from the baseline hides the movement the experiment measures. The cost is acceptable: an experiment runs a handful of perturbations, not one per query.

### Idf correctly rounded log

Source: `src/tfidf_stability/vectorisation/idf.py`.

The idf vector evaluates log((1 + N) / (1 + df)) through the correctly rounded logarithm, not math.log, and keeps LogImpl.PLATFORM only to measure the G13 gap. See [Exact log](#exact-log).

### Idf union vocabulary

Source: `src/tfidf_stability/perturbation/idf_perturb.py`.

Section 4.1 defines delta_idf(t) = log((1 + N') / (1 + df'(t))) - log((1 + N) / (1 + df(t))) and states that this expression exhibits the competing effects of changes in corpus size and of changes in the document-frequency distribution. That expression, and the section 4.2 bound built on it, assume that idf and idf' are vectors over a common index set. Under a real corpus perturbation the vocabulary itself moves: tokens appear, and tokens fall below min_df and vanish. The paper does not define delta_idf in that case.

G5 (spec_addenda.md#g5) aligns both models on the union vocabulary V u V', with coordinates set to zero outside the respective vocabulary. For t in V' \ V this gives delta_idf(t) = idf'(t), which is large, so the section 4.2 bound remains valid but becomes loose. Alignment exposes the shared, gained and lost index sets. They partition the union, so the movement of a vector over the union splits exactly, by Pythagoras, into a part on shared tokens and a part on churned tokens. `perturbation/vector_perturb.py` applies this split to ||w' - w||, so a caller can separate genuine coordinate change from vocabulary churn. IdfPerturbation.looseness reports the ratio of the union L-infinity norm to the shared-token L-infinity norm.

The union is sorted in ascending UTF-8 byte order, the same ordering rule the vocabulary module uses, so alignment introduces no new convention and is a pure function of the two token sets.

### Leave one out protocol

Source: `src/tfidf_stability/profiles/query_modes.py`.

Section 7.1 names three query modes and evaluates two: user-profile queries, which aggregate a user's interacted items; leave-one-out, which holds one interacted item out as the target and builds the profile from the rest; and item-as-query, which the paper describes as "part of the implementation but not evaluated in the present experiments". The section describes leave-one-out in four sentences and leaves five decisions open, each of which changes the reported margin distribution. They are fixed as follows (spec_addenda.md#g10). query_modes applies decisions 1 to 4 and records them on each Query (target, excluded, n_candidates) and on its QuerySet (mode, aggregation, min_interactions, exclude_profile_items), so a result traces back to the protocol that produced it; group_interactions applies decision 5 before any query is built.

1. Which item is held out. Every interacted item in turn, all folds, no sampling. The query count is the sum of item counts over eligible users, a function of the data, not of a seed.
2. Whether the held-out item stays in the corpus. It stays. It is the retrieval target; removing it makes the fold unanswerable.
3. Whether the user's remaining profile items are excluded from the candidate set. They are excluded. This is the most consequential unstated choice in section 7.1. Those items contribute the query's text, so if they remain they score near the top, occupy the top-k and dominate the margin distribution with an artefact. Excluding them makes the measurement about retrieval rather than self-similarity. Setting exclude_profile_items to False is a declared ablation that changes what the margin distribution measures.
4. Eligibility. Users with at least min_interactions qualifying interactions, five by default (eligible_users, which rejects a threshold below two).
5. What counts as an interaction. A weight threshold from the dataset configuration, applied by group_interactions through min_weight; for MovieLens, interaction_min_weight is 4.0, inclusive, so rating >= 4.0.

Decision 3 has a consequence the paper does not state: the candidate set is smaller than the corpus and differs per fold, so N differs per query and any statistic aggregated across folds has a varying denominator. Query.n_candidates carries the per-fold count, QuerySet.candidate_spread gives its (min, median, max), and QuerySet.provenance writes those three values to the run manifest (G19), so the variation is recorded rather than assumed constant.

### Lemmatiser identity

Source: `src/tfidf_stability/preprocessing/lemmatise.py`.

Lemmatiser.name identifies the backend. It is the complete identity only for a backend whose output is fixed by its class: IdentityLemmatiser and Porter2Stemmer. LookupLemmatiser takes its answers from a table and a fallback backend handed in at construction, while its name is always "lookup". It therefore also carries a digest: a SHA-256 over compact, key-sorted UTF-8 JSON holding the table in sorted form and the fallback's own lemmatiser_identity. The fallback is included because it handles every token the table misses, which on a real corpus is most of them. lemmatiser_identity returns the bare name (such as "porter2") for a backend without a digest and `<name>:<digest>` (such as `lookup:<sha256>`) for one with, and PreprocessingPipeline.digest, like the fallback entry inside a LookupLemmatiser digest, uses it in place of name. PreprocessingPipeline.fingerprint lists the bare name beside that digest. With name alone, three pipelines that map "the cats running" to cat|running, feline|running and cat|run report the same digest, so a cache keyed on that digest serves one configuration's features for another. The digest attribute is kept out of the Lemmatiser Protocol so that adding one to a backend is not a breaking change for the backends that need none.

### Length filtered runs leave no gap

Source: `src/tfidf_stability/preprocessing/tokenise.py`.

The tokeniser applies the length bounds (min_token_length, max_token_length) itself rather than leaving them to a later stage. The bounds are inclusive, are therefore part of the tokenisation contract, and are hashed into the run manifest with the pattern (PreprocessingConfig.to_dict).

A run that the length filter drops leaves no gap sentinel, so an n-gram closes over it. Under the default bounds (1 to 64), the texts "king <66 z's> pop" and "king pop" produce identical features, including the bigram king|pop. Underscores, commas and any other run the pattern excludes behave the same way: the pattern and the bounds together define what a token is, and text that never became a token leaves no trace.

Among the stages that remove text, only stopword removal inserts a gap (docs/spec_addenda.md#g7), because only stopword removal deletes something that was already a token; the other gaps mark hard boundaries such as the join between fields. The effect is reachable through configuration: at min_token_length: 2, "vitamin c deficiency" and "vitamin deficiency" become the same document.

### Live rounding probe

Source: `src/tfidf_stability/utils/numerics.py`.

`_rounding_is_to_nearest` checks whether the process rounds to nearest, ties to even, at the moment of the call. It uses three probes; each detects one directed mode by its disagreement with round-to-nearest-even:

- 1 + ulp/2 is an exact tie between 1 and 1 + ulp. Nearest-even picks 1; upward picks 1 + ulp.
- The same probe on the negative side (-1 - ulp/2) detects downward rounding.
- 1 + 1.5 ulp is a tie between 1 + ulp and 1 + 2 ulp. Nearest-even picks 1 + 2 ulp (the even one); toward-zero and downward truncate to 1 + ulp.

All operands are variables, so CPython cannot constant-fold the expressions at compile time and answer under the compiler's rounding mode instead of the process's.

Python code cannot change the rounding mode, but a C extension can. A BLAS that changes it on load is the hazard this guard covers, so the probe queries the live process rather than the build. sys.float_info.rounds is FLT_ROUNDS captured when the interpreter was built and keeps the same value whatever the process does later, so it cannot answer this question. The C++ half of the guard reads std::fegetround() at run time.

### Logging provenance

Source: `src/tfidf_stability/utils/logging.py`.

A run manifest records the inputs a number depended on: data, config and build. The logging module in tfidf_stability.utils records the decisions taken while producing the number: the backend selected, the reduction policy in force, degenerate cases reached, diagnostics fired, and digests of the outputs. Together the two account for most numbers without a rerun. General-purpose tracing is out of scope.

Closed vocabulary. EventKind names every kind of event that can be recorded, and log_event is the only emitter. Each line renders as `kind key=value ...` with keys sorted. Two runs that take the same decisions therefore produce the same lines, whatever order a call site passes its fields in, while a change in the order of events changes the digest. This matches the key ordering canonical_json applies to configs.

Volatile state is kept off the reproducibility surface. A logging.LogRecord carries `created`, `msecs`, `relativeCreated`, `process` and `thread`, and all of these differ between two identical runs; none of them is read. DETERMINISTIC_FORMAT names only `levelname`, `name` and `message`. EventRecorder stores the structured payload the call site supplied, not the record it arrived on, so EventRecorder.digest covers the decisions alone. Wall-clock timestamps are opt-in (`configure(timestamps=True)`) and take the output out of the reproducible set.

Warnings are not routed through logging. The pytest `filterwarnings` setting in `pyproject.toml` starts with `"error"`, so `TauExceedsScoreRangeWarning` and `ChainInflationWarning` are test failures. logging.captureWarnings would turn each of them into a log line, which a test run does not fail on. An `EventKind.DIAGNOSTIC` event records that a diagnostic fired, alongside the warnings.warn call and never in place of it.

No handler is installed on import, not even a logging.NullHandler: a library that installs a handler takes the application's decision for it, and this package is imported by notebooks and by other test suites. With no handler on the chain, records at `WARNING` and above still reach stderr through logging.lastResort, so diagnostics, which log at `WARNING`, stay visible without any configuration; the other kinds log at `INFO` and do not. configure is the opt-in; the CLI calls it when `--log-level` is given.

### Lowercase not casefold

Source: `src/tfidf_stability/preprocessing/normalise.py`.

The lowercase option applies str.lower, not full Unicode case folding with str.casefold. Full case folding cannot be reproduced in C++ without ICU, and ICU tailorings vary by version. str.lower is stricter and portable, so a C++ implementation of the preprocessing map can reproduce the same token stream. spec_addenda G7 records this decision.

### Manifest digest

Source: `src/tfidf_stability/persistence/manifest.py`.

`RunManifest.digest` is a SHA-256 over the `canonical_json` rendering of the manifest with `VOLATILE_KEYS` and `_MACHINE_KEYS` stripped and `notes` excluded, so two identical runs on different machines at different times give the same digest. See [Manifest machine keys](#manifest-machine-keys) and [Two canonical renderings](#two-canonical-renderings).

### Margin stratification

Source: `src/tfidf_stability/analysis/stratify.py`.

README section 7.3 stratifies results by the boundary margin m_k relative to tau, because tie-break effects are expected to concentrate in the near-tie regime m_k <= tau. That expectation is research question A2, and the stratification tests it: disagreement spread uniformly across bands would falsify the claim that tie-breaking is a near-tie phenomenon.

Stratifying by m_k is valid because the margin is tie-break independent. It depends only on the sorted score multiset, so it is identical under all three operators. Stratifying an operator comparison by an operator-dependent quantity would be circular.

On short-text corpora the m_k == 0 band is not a small tail. Every document that shares no term with the query scores exactly zero, so these documents form one large exact-tie block, and the near-tie regime is in practice largely an exact-tie regime. For this reason EXACT_TIE_BAND is a separate band from the smallest positive-margin band: at m_k == 0 the tie-break decides membership outright.

### Margins

Source: `src/tfidf_stability/ranking/margins.py`.

Every function in tfidf_stability.ranking.margins takes a sorted score array rather than a Ranking. This makes a property structural: m_k depends only on the score multiset. The non-increasing rearrangement of a multiset is unique, and all three ranking operators (pi, pi_score and pi_alt) use score-descending as their primary key, so they produce identical score sequences and differ only in their document sequences. Research questions A1 (margins) and A2 (tie-breaking) are therefore independent, and the independence follows from the direction of the dependency between the modules, not from a runtime assertion. The paper does not state this; it is proposed as a note under section 2.3.2.

Undefined margins follow spec_addenda.md#g3: the value is NaN with an explicit validity flag. It is never coerced to 0 (which would read as an exact tie) or to infinity (which would read as perfect stability), and undefined margins are counted rather than dropped. Whether a degenerate query enters a reported distribution is a policy decision for analysis/stability_profile.py; the margins module reports the raw values so they stay auditable.

### Movielens ingestion

Source: `src/tfidf_stability/datasets/movielens.py`.

The MovieLens licence permits research use and prohibits redistribution. The data is therefore not committed, not used as a test fixture and not present in CI. The repository holds the parser in `src/tfidf_stability/datasets/movielens.py`, a pinned SHA-256 of the archive and the derived statistics. That is enough to reproduce the numbers from a copy of the archive obtained from GroupLens.

GroupLens updates `ml-latest-small` in place at the same URL, so without a pin the corpus behind a published result can change without notice. `load` verifies the digest and raises `DataIntegrityError` on a mismatch. It raises rather than warns, because a mismatch means the archive is not the one the published results were computed against.

The parser handles three determinism hazards itself rather than leaving them to downstream code:

1. Row order. CSV row order is a property of the GroupLens export. Films are sorted by `movieId` as an integer, so document indices depend on the data and not on the file or on string collation. Interactions are sorted as `(user, movie, weight)` tuples, so their order does not depend on `ratings.csv` row order. Tags for each film are sorted too, because their order in the concatenated text decides which tag-to-tag n-grams exist.
2. Half-integer ratings. Ratings are quantised to 0.5, so `2 * rating` is an exact integer. This makes the exact `(2*sum, count)` representation of `docs/spec_addenda.md#g8` available on real data: the mean is never formed as a float, so equal means always compare equal and distinct means never collide. Ratings are parsed by integer arithmetic on the decimal text. `float("3.5") * 2` is also exact for this input, but depending on it would tie correctness to the input alphabet; a rating off the half grid raises `DataIntegrityError` instead of being rounded.
3. Unrated films. `movies.csv` lists films that appear in no row of `ratings.csv`, so their mean rating is undefined. They carry `rating_count = 0`, become `has_value = False` and sort last under the default `MissingPolicy.LAST`, which exercises the G8 missing-value path on real data. `scripts/fetch_data.py` parses the archive and reports the count each time it runs past the digest check, since the repository holds no copy of the archive to check against.

The document text joins `title`, `genres` and the free-text `tags`. Genres are pipe-separated (`Action|Sci-Fi`) and the release year is parenthesised in the title (`Heat (1995)`). Both are split into tokens in the parser, because neither is natural language and the tokeniser carries no knowledge of dataset-specific conventions. The unjoined fields are also carried so that n-grams do not span the seam between two fields (`spec_addenda.md#g7`).

### Near tie floor

Source: `src/tfidf_stability/datasets/synthetic.py`.

Section 7.4 asks for two documents identified such that |s_A - s_B| <= tau. They are identified, not manufactured, because a fine near-tie cannot be constructed: section 2.2 normalises tf = count / L, so adding or removing one token moves every term frequency from c/L to c/(L+1), a relative perturbation of 1/(L+1). A separation of 1e-9 would need a document of roughly a billion tokens. The twin mechanism in `generate` therefore reaches the 1e-3 to 1e-1 range and no finer; this limit follows from the specification, not from the generator.

`docs/spec_addenda.md#g22` gives the adjacent-gap table for a 3000-document synthetic corpus (`SyntheticSpec(n_docs=3000, vocab_size=5000, n_exact_duplicates=30, n_twin_pairs=60, seed=20260811)`, queried with the first six features of document 0), and `tests/test_datasets.py` recomputes it: exactly 0, 18.4%; in (0, 1e-9), 0.0%; in [1e-9, 1e-6), 4.3%; in [1e-6, 1e-3), 76.4%; at least 1e-3, 0.8%. The shares depend on the configuration. On this synthetic corpus the near-tie regime is in practice the exact-tie regime: at tau = 1e-9 every within-tau pair has a gap of zero. MovieLens, by contrast, does place adjacent pairs in (0, 1e-9). With strictly_positive=False, `find_near_ties` keeps exact ties, which sort first, so it returns the exact-tie block, the dominant case.

In the returned list, rank is 1-indexed, so a pair is (r_rank, r_{rank+1}) and gap is m_rank; pairs are ordered tightest first, ties broken by rank.

### Nearest rank percentiles

Source: `src/tfidf_stability/analysis/summarise.py`.

The normative backend uses only the standard library, so percentiles are computed in tfidf_stability.analysis.summarise rather than taken from NumPy. The method is nearest-rank on the sorted sample: ``p50`` of an even-length sample is the lower of the two central values.

Interpolation is not used because it produces a value that no query produced. Every quantity summarised is a measured floating-point observation whose bit pattern matters elsewhere in the study (margins, flip radii, score gaps). An interpolated ``p50``, such as NumPy's default linear interpolation gives, cannot be looked up in the raw data, cannot be reproduced by inspection, and cannot be compared with ``same_bits`` against it. Each distribution record states the method as ``percentile_method``.

ExperimentResult is the envelope every runner script writes: the payload, the parameters in force, the provenance of the data it was computed from, and the environment. Its digest is taken over the payload and parameters after volatile fields are stripped, so two runs of the same experiment on the same data give the same digest despite different timestamps. A reader verifies a published number by comparing one hex string.

### Ngram joiner and gap boundaries

Source: `src/tfidf_stability/preprocessing/ngrams.py`.

Two choices in n-gram construction change the vocabulary and therefore every number downstream (see docs/spec_addenda.md#g7).

The joiner is ASCII Unit Separator (U+001F). After normalisation and tokenisation no token contains a control character, so U+001F cannot occur inside a token. The GAP sentinel (U+0000) is a control character but is never an n-gram constituent: generate_ngrams either splits on it or drops it. The mapping from token sequence to n-gram string is therefore injective: the bigram ("new", "york") and a hypothetical single token "new york" map to different vocabulary entries, and every n-gram splits back into its constituents. A space joiner would map both to the same entry.

N-grams do not span a gap sentinel. A gap marks a removed stopword or a hard boundary. Bridging a gap creates features that appear in no document ("king of pop" becomes "king pop"), which are an artefact of preprocessing order rather than of the text. The cross_gaps option of generate_ngrams enables bridging only so the effect can be measured as an ablation; it is off in the normative configuration.

### No backend registry

Source: `src/tfidf_stability/__init__.py`.

The package has two backends. The pure-Python reference is normative: it is a literal transcription of README sections 2-4. The compiled backend in tfidf_stability._native is an optional accelerator that the test suite holds to bit-for-bit agreement with the reference. A registry implies interchangeable implementations chosen at run time. Here one side defines the answer and the other is asserted to agree to the last bit, so selection is one availability check: native_available(), which returns False when the extension cannot be imported or when its __abi__ differs from REQUIRED_ABI. A numpy evaluator would have to reproduce the reference's summation order. numpy reductions are pairwise, with a block size and an unrolled base case that numpy does not document as part of its interface. At the same block size of 128 the two summation orders still disagree at 208 of 262 lengths tested (see [Pairwise reduction order](#pairwise-reduction-order)). Adding such an evaluator is therefore a research decision rather than an implementation task.

### Non finite json

Source: `src/tfidf_stability/utils/io.py`.

RFC 8259 has no token for NaN or Infinity. Python's json encoder emits them as a non-standard extension. Python's own json.loads reads that extension back, but strict parsers (JSON.parse, jq, Go, Rust) reject the whole document. Experiment results can contain NaN, for example an undefined margin, which G16 requires to be reported as undefined rather than coerced. They can also contain Infinity, for example g_min on a corpus with no strictly positive score gap. `_json_safe` maps every non-finite float to null, so every result file parses outside Python. No information is lost. A non-finite float in this position always means undefined, and the defined flags, n_nan counts and `*_hex` fields still separate the undefined, not-applicable and infinite cases. canonical_json leaves allow_nan at its default, so a value that escapes the sanitiser is written rather than raising mid-experiment. tests/test_canonical_bytes.py asserts that canonical_json output carries no NaN or Infinity token, at the top level and at every nesting depth.

### Norms

Source: `src/tfidf_stability/analysis/noise_floor.py`.

The weights w = tf * idf are policy-independent: each is one division, one correctly rounded logarithm (G13) and one multiplication, with no summation. One fit therefore serves every reduction policy, and all disagreement between policies arises in the two reductions downstream of the weights: the row norms and the dot product. Both must vary. TfidfModel.norms is precomputed under the model's own reduction, so passing a different policy to cosine_against_corpus alone varies only the dot product and holds the norms fixed. That understates eta: a query dot product runs over the intersection of query and document terms (a handful), whereas a norm sums the whole document vector (tens of terms), and the longer summation is where rounding error accumulates. measure_noise_floor therefore recomputes the row norms under each policy and compares against norms and scores computed under Reduction.EXACT.

### Pipeline digest identity

Source: `src/tfidf_stability/preprocessing/pipeline.py`.

PreprocessingPipeline.digest binds the lemmatiser and the stopword list into PreprocessingConfig.digest. PreprocessingPipeline accepts an injected lemmatiser that bypasses the config's lemmatiser field; without binding it, PreprocessingPipeline(cfg) and PreprocessingPipeline(cfg, lemmatiser=IdentityLemmatiser()), which turn "running cats" into run|cat and running|cats, would hash to the same string. The lemmatiser is bound by lemmatiser_identity rather than by Lemmatiser.name, because the name of LookupLemmatiser is the constant "lookup" while its output depends on the table passed at construction ([Lemmatiser identity](#lemmatiser-identity)). The binding is omitted only when the identity equals both the backend name and the config's lemmatiser kind. That holds for IdentityLemmatiser and Porter2Stemmer when the config names their kind, since their output is fixed by their class. A backend carrying content, such as LookupLemmatiser, is bound unconditionally, so a config naming LOOKUP plus an injected table is bound by its table rather than matched by name. No config-only route to a lookup pipeline exists, because make_lemmatiser refuses that kind without a table, so a config-only pipeline carries no lemmatiser_override key. The stopword list is bound by its two digests, `digest` and `content_digest`, joined with a colon in the same way lemmatiser_identity joins a backend name to its content hash; neither subsumes the other ([Stopword digests](#stopword-digests)). In PreprocessingConfig.digest, the stopword_digest and lemmatiser_override keys are omitted when None, so a config hashed without them has the same digest as a config whose payload never had those keys.

### Porter2 default backend

Source: `src/tfidf_stability/preprocessing/lemmatise.py`.

README section 2 requires lemmatisation inside a fixed, deterministic preprocessing map. NLTK's WordNet lemmatiser needs a downloaded, versioned corpus and a POS tagger. spaCy needs a statistical model whose output changes between releases. Neither is reproducible and neither ports to C++. Porter2 (the Snowball English algorithm) is a complete published algorithmic specification with no data files. One upstream source generates both the Python and the C implementation, and snowballstem.org publishes a canonical 42 649-word test-vector pair (voc.txt / output.txt), which tests/test_preprocessing_determinism.py replays, so this step is machine-verified. The implementation is vendored in tfidf_stability.preprocessing._snowball, which records its provenance. Porter2 is a stemmer, not a lemmatiser: it maps "happy" to "happi", which is not a word. That limitation and the proposed wording change to the paper are recorded in docs/spec_addenda.md#g7. LookupLemmatiser serves callers who need true lemmas. It takes its table as a mapping at construction and carries a SHA-256 digest of that table and its fallback, so the mapping is part of the run's recorded provenance and not a property of the machine.

### Preprocessing order

Source: `src/tfidf_stability/preprocessing/pipeline.py`.

The preprocessing map of README section 2 runs normalise, tokenise, remove stopwords, lemmatise, n-grams. Section 2 requires the map to be deterministic and fixed across all perturbation experiments: stable across processes, Python versions and platforms, not only within one run. The C++ core has no preprocessing stage; choices such as str.lower over str.casefold and byte-order sorting keep the map reproducible by a C++ implementation. Every input that could vary is pinned in PreprocessingConfig and folded into PreprocessingConfig.digest, which every run manifest records. The stage order is fixed. Stopwords are removed before n-gram construction and leave a gap sentinel, so no n-gram spans a removed token (docs/spec_addenda.md#g7). Lemmatisation runs after stopword removal, so the stopword list is matched against surface forms; in the other order "willing" stems to "will" and is deleted as a stopword. N-grams are built last, from lemmatised tokens, so "running shoes" and "run shoe" give the same bigram.

### Profile aggregation

Source: `src/tfidf_stability/profiles/user_profile.py`.

README section 3 constructs user-specific documents from interactions such as liked, viewed or favourited items, and embeds them with the same vocabulary and IDF mapping as the corpus. Section 7.1 builds a profile by aggregating the text of a user's interacted items, which makes ProfileAggregation.TEXT_CONCAT the normative mode and the default. Concatenating token streams makes tf a length-weighted average, so one verbose item can dominate a profile regardless of how many other items the user interacted with. A profile whose direction is set by one long document is fragile in the sense of section 1.1. ProfileAggregation.VECTOR_MEAN and ProfileAggregation.VECTOR_SUM therefore exist as ablations that weight each item equally in the embedding space rather than by token count. VECTOR_SUM has the same direction as VECTOR_MEAN, so cosine similarities are identical; it is kept because the norm differs and the section 4.2 bounds are stated in terms of norms.

### Profile eligibility

Source: `src/tfidf_stability/profiles/user_profile.py`.

group_interactions keeps only interactions whose weight is at or above min_weight (G10(5)). The test is spelled `weight < min_weight` rather than the negated `not (tau >= 0.0)` form used for the tau thresholds, because weights are not confined to a non-negative domain: a negative threshold is legal, and with non-negative weights it keeps everything. A NaN threshold also keeps everything, because the comparison is false for every weight; this is the same no-op as None rather than an error, and no caller constructs one. For MovieLens, min_weight is the `rating >= 4.0` threshold that decides what 'interacted' means, a choice section 7.1 leaves open and one that materially changes the profile. Each user's items are deduplicated and returned in ascending UTF-8 byte order of the identifier rather than arrival order (G20): interaction files are not order-stable and a concatenated profile's feature tuple follows item order, so the order is canonicalised at grouping rather than inherited.

### Profile streams

Source: `src/tfidf_stability/profiles/user_profile.py`.

build_profile concatenates items in the order given, which group_interactions canonicalises into ascending identifier byte order (spec_addenda.md#g20). features_by_doc holds streams that have already passed the preprocessing map (PreprocessingPipeline.preprocess, or preprocess_fields for a record with fields), so the n-grams exist before aggregation and no n-gram pass runs over the joined result (spec_addenda.md#g28). Three consequences follow. First, there are no seam n-grams: text aggregation of 'quick brown fox' and 'lazy sleeping dog' would yield the bigram fox|lazi, whereas feature aggregation yields the union of the two streams and nothing spanning the join. Second, concatenation is order-insensitive: under TEXT_CONCAT, reordering items permutes the feature tuple but leaves the multiset, and hence the embedding, bit-identical; the G20 canonical ordering makes the tuple itself reproducible, which the digest depends on. The vector aggregations sum each term's values in item order, so their bits do depend on that order. Third, separate_items is inert: it inserts a gap sentinel between consecutive items to stop seam n-grams, but there are none, and the sentinel is out of vocabulary, so it is dropped before it reaches a count or the in-vocabulary length L. It is kept because it becomes meaningful under text aggregation, and test_separate_items_is_inert_while_aggregation_joins_features asserts the inertness. Moving to text aggregation is a research decision: it adds one bigram per item boundary, changes every profile's L, and moves every number in section 7.1. An item with no feature stream raises KeyError rather than being skipped, because skipping would shrink the profile without record and profile size is part of what the leave-one-out protocol reports.

### Python cpp type correspondence

Source: `src/tfidf_stability/types.py`.

The aliases in src/tfidf_stability/types.py record the width the native side uses wherever a value crosses the Python/C++ boundary. Python does not check them at run time. They do not mirror cpp/include/tfidf/core/types.hpp one-to-one. The correspondence is:

| Python | C++ | Note |
|---|---|---|
| `TermId` | `TermId` | same concept, `int32` in C++ |
| `DocIndex` | `DocId` | the names differ: a positional row index |
| `DocId` | no counterpart | the stable external identifier, a `str` |
| `Rank` | no counterpart | 1-indexed, Python-side reporting only |
| `Offset` | `Offset` | `int64` in C++ |
| `Real` | `Real` | `double` |
| `Score` | `Score` | `double` |

The same name, `DocId`, denotes a different concept on each side. C++ `DocId` is a positional row index, which the Python package calls `DocIndex`. Python `DocId` is the stable external string that survives a corpus reordering; its UTF-8 byte-order rank is the last component of every sort key. A signature read across the boundary therefore needs this table to be interpreted correctly. tests/test_public_api_surface.py parses the C++ header and asserts this correspondence.

### Query grid

Source: `src/tfidf_stability/analysis/query_grid.py`.

The experiment runners `scripts/run_stability_profile.py` and `scripts/run_tie_break_ablations.py` build their queries through the section 7.1 grid: user-profile queries and leave-one-out folds. Item-as-query is implemented but not evaluated. `scripts/run_similarity.py`, the unperturbed baseline, scores truncated document prefixes instead. Truncated document prefixes are a different and easier protocol: a prefix retrieves its own source document at high similarity, whereas a fold has to find a held-out item from the rest of a user's history. The choice of protocol changes the score distribution, the margin distribution and every A1 number.

Each query carries its own exclusions, so the candidate set differs per query (G19). With `exclude_profile_items` set (G10 decision 3, the default), a fold excludes the remaining profile items, which contributed the query text and would otherwise retrieve themselves, and keeps the target scoreable. Consequences: N differs between queries, so k can exceed the candidate count, and those queries are reported rather than clamped. Margins are computed over the candidate scores only; over the full corpus they would include documents the query cannot retrieve and would inflate the apparent separation. The attribute table is restricted to the same subset, so the tie-break does not rank non-candidates.

A profile can be empty after preprocessing, and a non-empty query can embed to the zero vector when every feature is out of vocabulary. Both are legitimate section 7.1 outcomes and are counted separately: G3 excludes them from margin distributions and keeps them in ablations.

### Rank encoding

Source: `src/tfidf_stability/ranking/attributes.py`.

Score ties are resolved lexicographically on a fixed attribute tuple: popularity, rating, engagement, identifier. Each attribute is converted to a dense int32 rank at construction time, in which smaller means earlier, with direction (desc/asc) and missing-value placement folded into the rank.

Consequences:

1. The comparator is a plain ascending lexicographic comparison of (float, int, int, int, int): no direction branch, no missing-value branch, and no rational arithmetic inside the sort.
2. No floating point remains in the tie-break, as G8 requires. The only double in the key is the similarity score.
3. A NaN cannot enter the sort key through an attribute, because every attribute component of the key is an integer rank. A FLOAT64 attribute column is also checked for finiteness when its ranks are built. A NaN in a sort key breaks the strict weak ordering, which is undefined behaviour in std::sort and can produce an out-of-bounds write.
4. The C++ mirror receives the rank matrix as data and never re-derives it, so cross-language agreement reduces to the integer equality py_ranks == cpp_ranks rather than to whether two rational-comparison implementations agree. spec_addenda.md#g13 applies the same approach to idf. The native hot path compares ranks and never multiplies attribute values, so no overflow can occur there.

fractions.Fraction is not used in the sort key. It has no C++ counterpart, so the two languages would run structurally different algorithms over a result the test suite asserts is identical, and two legitimate formulations can agree on most inputs yet diverge on some. Fraction also runs math.gcd and allocates an object per element, and Fraction(0, 0) raises, so an absent rating would still need a separate has_value bit. Cross-multiplication is the comparison rule; it is applied in the rank construction, once over the distinct values, rather than inside a sort that runs per query.

### Ranking independent of k and tau

Source: `src/tfidf_stability/ranking/ranker.py`.

The sort depends on neither k nor tau: a ranking is a function of (scores, table, priority) alone. The section 7 grid is therefore queries x 3 operators rather than queries x k-values x tau-values, and every k and tau is read off the result afterwards. The saving is a factor of the number of k values times the number of tau values: 32 for the four k values of section 7.2 and the eight tau_sweep values in `configs/default.yaml`. For this reason rank() takes no k, and the margin and tie-group modules take a sorted score array rather than a Ranking.

### Ratio dense ranks

Source: `src/tfidf_stability/ranking/attributes.py`.

`_order_distinct` deduplicates with dict.fromkeys, which uses tuple equality, so (14, 2) and (21, 3) both survive even though `_ratio_cmp` reports them equal (14/2 == 21/3). Numbering the survivors enumerate-style would give two documents with the same mean rating two different ranks.

Two guarantees depend on equal ratios sharing a rank. First, the rating component must tie where G8 says it ties, so that the tie-break falls through to engagement. Second, dict.fromkeys preserves insertion order, so with distinct ranks the choice of which equal representation sorts first would depend on the order the records arrived in; the ranking would then depend on corpus order, which the total-order argument of README section 2.3.1 forbids.

Ranks are therefore dense: consecutive entries that compare equal share a rank, and the next class takes the next integer. Only RATIO_I64 can hold two distinct representations of one value; every other dtype is deduplicated by == under an ordering consistent with its sort key.

### Recompose after control strip

Source: `src/tfidf_stability/preprocessing/normalise.py`.

When strip_control is set, normalise deletes every non-whitespace character in the Cc, Cf, Co and Cs categories after the first normalisation and case pass. Whitespace control characters such as tab and newline are kept for the whitespace collapse. Deleting a format character can bring a base character and a combining mark into contact, which produces a sequence the first pass did not see. For example, "e", U+200B, U+0301 becomes "e" followed by U+0301. That sequence is not NFKC, and under the default tokeniser pattern it tokenises to "e", while the composed "e-acute" tokenises to itself. For this reason, whenever stripping changes the text, normalise runs `_canonical_case` on the stripped text again. One recomposition is enough for two reasons. First, composition only merges a starter with a following mark, so it cannot create a character that the stripping stage deletes. Second, the whitespace collapse that follows always leaves a space between the characters it separates, so it cannot bring a base character and a mark into contact.

### Resizing edits

Source: `src/tfidf_stability/perturbation/experiments.py`.

PerturbationReport.certified_stable returns None at every k, before it looks up a certificate, for any edit that changes the corpus size, because section 4.4 bounds only the movement of documents present on both sides of the edit. See [Certified stable size changing edits](#certified-stable-size-changing-edits).

### Rho order

Source: `src/tfidf_stability/ranking/tie_groups.py`.

rho(tau) = |largest chain| / |largest clique|. On non-increasing input a clique is an interval whose adjacent gaps are all <= tau, so it lies inside some chain and rho >= 1. Large values mean the reported tie groups are held together by chaining rather than by indistinguishability.

The bound depends on that order and the function does not check it. On unsorted scores the chains and cliques are computed over intervals that do not correspond, and rho can fall below 1: on [0.07, 0.01, 0.84, 0.26, 0.23, 1.0] at tau = 0.2 the chains are [0, 3) and [3, 6), each of width 3, while the clique sweep returns the single interval [0, 6), so rho = 3/6 = 0.5. The C++ `chain_inflation_ratio` behaves the same way, and both return NaN on empty input. The function cannot distinguish that from a legitimate result, so establishing the order is the caller's responsibility. The package itself does not call chain_inflation_ratio; the sorted score arrays it builds, such as Ranking.sorted_scores, come from tfidf_stability.ranking.ranker.sorted_scores_desc. `scripts/run_tie_break_ablations.py` calls it on scores sorted with sorted(scores, reverse=True).

### Run manifest contents

Source: `src/tfidf_stability/persistence/manifest.py`.

README section 3 states that all stages of the pipeline are deterministic given a fixed corpus, configuration and software environment, including library versions. The run manifest makes this checkable: it records everything the numbers depend on, so a rerun can be verified against it.

A value goes into the manifest when changing it could move a published number. This includes values that do not look like data: the compiler's floating-point flags, the reduction policy, the stopword list's digest and the `tau` in force. Editing the stopword list moves the vocabulary and every number computed from it, so its digest is part of the preprocessing fingerprint.

Values that vary between identical runs (timestamps, paths, hostnames) are written for a human reader. Before the digest is taken, `strip_volatile` removes them, together with the machine-identity keys in `RunManifest._MACHINE_KEYS`, so two identical runs on different machines produce the same `RunManifest.digest`.

### Runtime float environment probe

Source: `src/tfidf_stability/cli/main.py`.

Of the three layers of the floating-point guard ([Fp guard layers](#fp-guard-layers)), only the runtime probe sees the process, and it is the one that detects the flush-to-zero (FTZ) and denormals-are-zero (DAZ) modes a BLAS loaded with numpy can set. The CLI runs the probe after the dispatch guard, so `--version` and `--help` do not pay for it. It runs before the command, so the repair precedes any number that the command produces. Each backend has its own check, and the CLI runs the one for the backend in use. When the compiled backend is available, `check_float_environment()` clears the FTZ and DAZ bits of MXCSR on targets that have it, re-runs the self-test, and issues a `RuntimeWarning` if the environment is still untrustworthy. Otherwise `assert_sane_float_environment()` runs; it cannot clear MXCSR from Python, so it raises `NumericEnvironmentError` in a flushed environment instead of running.

### Scoring term order

Source: `src/tfidf_stability/similarity/scoring.py`.

The module src/tfidf_stability/similarity/scoring.py is the normative Python reference for the two scoring kernels in cpp/include/tfidf/similarity/scoring.hpp. Three implementations compute s_i = cos(q, w_i) (README section 2.3): TAAT (term at a time) walks the postings list of each query term into a dense accumulator; DAAT (document at a time) merges each document's row against the query independently and builds no inverted index; and cosine_against_corpus in tfidf_stability.similarity.cosine performs the same per-document merge as DAAT over a materialised sequence of rows. Neither DAAT nor cosine_against_corpus shares loop nesting or a data structure with TAAT. All three produce identical bit patterns, not values that agree to within rounding, and tests/test_scoring_kernels.py asserts this with tfidf_stability.utils.numerics.same_bits.

Why they agree. The TAAT outer loop runs over query terms in ascending term identifier, and a canonical query stores each term once, so a given term contributes at most one addition to any one accumulator. Document d therefore sees the addend sequence: ascending term id over supp(q) INTERSECT supp(w_d), starting from 0.0. This is the sequence the merge in tfidf_stability.vectorisation.sparse.dot produces. Every reduction policy in tfidf_stability.utils.numerics.Reduction is a pure function of that ordered sequence, so equal sequences give equal bits.

The property depends on the ascending order. Blocking the term loop, or visiting postings in any order other than the one InvertedIndex.from_csr lays down, reassociates the sum and changes the digits, which is the subject of this project. No such reordering is applied in the reference, and none may be applied on the normative path.

Difference in form from the C++. For every policy other than NAIVE, the native TAAT kernel keeps one incremental accumulator object per touched document, with one add call per product. The Python reducers are batch functions over a sequence, so the non-naive TAAT branch buffers each document's products and reduces them once at the end. Each policy consumes its addends strictly in order and depends on nothing else, so buffering does not change any bit. For NAIVE both implementations add directly into the dense accumulator array.

Cost, where N is the number of documents. TAAT costs O(sum of df(t) over query terms t) multiply-adds plus O(|touched|) divisions rather than O(N), since an untouched document scores exactly 0. DAAT costs O(sum over every document d of (nnz(d) + nnz(q))), because each row is merged with the query. Both kernels also write all N output slots, and under a compensated policy the native TAAT kernel allocates one policy object and one flag per document on every query. TAAT is faster when the query's terms are individually rare, which is the usual case for the TF-IDF profile queries of section 7.1, so it is the default of `score`.

### Sort key ordering

Source: `src/tfidf_stability/ranking/sort_keys.py`.

The three ranking operators of README sections 2.3.1 and 4.5 are: pi = Sort(s_i, a_i), the full attribute tuple compared lexicographically; pi_score = Sort(s_i, id_i), the attribute-independent baseline; and pi_alt = Sort(s_i, a_i reordered), an alternate priority. Each sort key is (-score, rank_1, ..., rank_m, id_rank), compared ascending and lexicographically. Negating a binary64 value flips the sign bit and never rounds, so -s is exact and order-reversing. Python's tuple < and C++'s operator< are then the same relation, so only one comparator needs to be reasoned about. Identifiers are unique and terminate every key, so the key is injective and the comparator is a strict total order: every correct sorting algorithm produces the identical permutation, sort stability is irrelevant, and the output does not depend on the input order. If the identifier is dropped, keys that tie on score and every attribute compare equal, the order is not total and the sorted permutation is not unique; assert_strict_total_order checks this. If the identifier is moved ahead of an attribute, the key stays injective but that attribute never decides a tie, so build_keys rejects a priority that names the identifier. pi_score is the empty-priority case of pi, not a separate code path (spec_addenda.md G15); [Tie break ablation isolation](#tie-break-ablation-isolation) states what that means for the ablation.

### Sparse index order

Source: `src/tfidf_stability/vectorisation/sparse.py`.

Index order and reduction policy together make the reference and native backends bit-identical. A SparseVector stores its indices strictly ascending, which fixes the order in which terms accumulate in a dot product. Floating-point addition is not associative, so a different order can give a different number. The native backend's term-at-a-time scoring loop also visits terms in ascending identifier order, so it matches the reference dot product bit for bit. Every sum takes an explicit Reduction policy. The default, NAIVE, is the plain left-to-right fold, the normative reading of the unbracketed sums in README section 2.

### Stopword asset layout

Source: `src/tfidf_stability/preprocessing/stopwords.py`.

The frozen stopword list and its manifest are looked up first at `tfidf_stability/data/assets` inside the package, where `wheel.force-include` copies them ([Wheel force include assets](#wheel-force-include-assets)), and then at the repository root, by a resolver that takes the module's own path as an argument. See [Packaged asset resolution](#packaged-asset-resolution).

### Stopword digests

Source: `src/tfidf_stability/preprocessing/stopwords.py`.

A `StopwordSet` carries two digests because they answer different questions and neither subsumes the other. `digest` is provenance: `load_stopwords` sets it from the asset's raw file bytes, so an in-place edit of the file is detectable even when the parsed set is unchanged, and it goes into the run manifest so a result traces back to the word list that produced it. `content_digest` is identity: the constructor derives it as the SHA-256 of the canonical sorted words, one per line with a trailing newline, the same form `from_iterable` digests. A caller cannot supply it, so it cannot be wrong. `PreprocessingPipeline` includes the identity in its digest, joined to `digest` with a colon; if the identity were caller-supplied, two sets holding different words could carry the same `digest` and hash alike. `LookupLemmatiser` derives its identity for the same reason.

### Stopword load verification

Source: `src/tfidf_stability/preprocessing/stopwords.py`.

The stopword set determines the vocabulary and therefore every document frequency, idf value and score the package produces. It is treated as data with an identity: a versioned file whose SHA-256 is verified at load against the `MANIFEST.sha256` in the asset directory (`data/assets/MANIFEST.sha256` in a source checkout) and recorded in every run manifest, never a library default that can shift between releases. The manifest uses the format `scripts/check_vendored.py` parses, so the asset is covered both at load and by the repository-wide gate. The digest covers the raw file bytes rather than the parsed set, so a changed comment or a reordering also shows. `load_stopwords` raises `DataIntegrityError` when the asset has no recorded digest or its bytes do not match it. This check is required because the reproducibility snapshot compares runs against each other rather than against a pinned value, so a one-word edit to the list would change every df, idf and score without failing the snapshot. The result is cached because it is immutable, and the file cannot change within a run without invalidating that run's provenance.

### Stratify bucket k

Source: `src/tfidf_stability/analysis/stratify.py`.

An OperatorPair carries two k values. ablate_query builds the comparison at k_eff = min(k, N) and the margin at the requested k, and pair.k forwards k_eff. For k above the corpus size N the two values differ. ablate_query treats a k larger than the corpus as a legitimate grid point in a sweep, not a configuration error, so this is a supported input.

The band is derived from the margin, so the bucket key is the margin's k as well. Keying by pair.k fails in two ways. Where the clamped value is itself in ks, pairs from several grid points merge: with 10 documents and the default grid (5, 10, 20, 50), k = 10, 20 and 50 all land in bucket 10, which then reports n = 3 while buckets 20 and 50 report n = 0, giving a disagreement rate whose denominator is three times too large and drawn from other grid points. Where the clamped value is not in ks, the membership test acts as a filter and drops pairs: with 8 documents and ks = (5, 10, 20), k = 10 and 20 clamp to 8, two of the three pairs are discarded, and the per-query totals reconcile to 1 instead of 3.

### Streaming pairwise

Source: `src/tfidf_stability/utils/numerics.py`.

pairwise_sum is pairwise summation in the streaming form, with blocks of `_PAIRWISE_BLOCK` (128) elements merged through a binary-counter stack, and is the pinned specification that the C++ `Pairwise` matches bit for bit. See [Pairwise reduction order](#pairwise-reduction-order).

### Strict and lenient modes

Source: `src/tfidf_stability/utils/validation.py`.

Each condition in docs/spec_addenda.md#g3 that raises or warns has a named exception or warning class in tfidf_stability.utils.validation. Two modes apply. In strict mode the condition raises. Strict is the default of `rank`, `boundary_margin` and `min_adjacent_margin_top`, for interactive and CLI use, because an unexpected corpus shape there is more likely a mistake than an intention; `margin_profile`, which evaluates every k of a sweep, defaults to lenient. In lenient mode the condition is flagged and the affected value is returned as NaN; resolve_k instead clamps k to n, and the caller records k_effective. Lenient is the default inside experiment sweeps (the tie-break ablations and score bounds, and ranking.strict_mode in configs/default.yaml). Degenerate points, such as tau larger than the score range, are legitimate members of the grid there, and they must not abort the run. The configured mode, ranking.strict_mode, is recorded in the run manifest's config block, because the mode changes which queries contribute to a reported distribution.

### Synthetic prng

Source: `src/tfidf_stability/datasets/synthetic.py`.

The synthetic generator provides three things the MovieLens corpus does not. First, near-ties at chosen magnitudes: section 7.4 is a constructed near-tie case study, and on real data near-ties in the relevant range are rare enough that finding one is a search problem. The generator builds exact ties (duplicated documents, gap 0) and twin pairs (a copy plus one extra token) whose gaps span about 1e-3 to 1e-1; [Near tie floor](#near-tie-floor) explains why no finer gap can be constructed. Second, scaling: sweeping N, |V| and document length shows how margins scale. Third, a redistributable corpus: MovieLens may not be redistributed, so CI cannot use it, whereas the synthetic corpus can be committed and generated offline.

Determinism. `random.choice`, `random.sample` and `random.shuffle` are not guaranteed stable across CPython versions, so a corpus generated with them does not necessarily regenerate identically on another interpreter. Only the Mersenne Twister core is stable, so draws are restricted to `random.Random.random` and `random.Random.getrandbits` (the module calls only `getrandbits`), and the selection logic (`_pick` for weighted sampling from an integer cumulative distribution, `_uniform_int` for inclusive uniform integers) is written out with integer comparisons and rejection sampling instead of `random.choices` or `randrange`.

Transcendental functions are avoided for a related reason: `pow` with a non-integer exponent calls the platform libm, and `docs/spec_addenda.md#g13` records that platform libms round transcendental functions (there, `log`) differently. The default Zipf exponent of 1 is computed in exact integer arithmetic, as `_ZIPF_SCALE // (rank + 1)` for 0-based rank. A non-integer exponent is permitted but makes the generated files, not the spec, the reproducible artefact.

`write_corpus` writes a generated corpus, its interactions and its spec, with a digest manifest, so a corpus can be consumed from files rather than regenerated. The registered synthetic datasets are not consumed that way: `load_dataset` regenerates them from their spec at experiment time (see [Dataset provenance](#dataset-provenance)). Their reproducibility therefore rests on the restrictions above: draws from the Mersenne Twister core only, integer selection logic, and the integer default Zipf exponent.

### Tau band

Source: `src/tfidf_stability/analysis/noise_floor.py`.

Section 7.1 constrains tau qualitatively from both sides: it should exceed floating-point noise while remaining small relative to typical score separations. That is an interval with no procedure for choosing a point inside it, so configs/default.yaml has no tau key (only a tau_provisional placeholder and a tau_sweep grid), and every tau-dependent function in the repository, such as tie_chains, tie_cliques and tie_ball_interval, takes tau as an explicit argument. tfidf_stability.analysis.noise_floor measures both endpoints and shows that the choice between them does not affect the result.

Lower endpoint (arithmetic). tau decides whether |s_i - s_j| <= tau indicates a real difference or an artefact of arithmetic, so the quantity to bound is the error in a margin. A score error of at most eta gives a margin error of at most `2 * eta`, and both signs are attainable, so `tau_floor = 2 * eta`. This is the same factor 2 as in section 4.4's eps_k^flip = m_k / 2. eta is measured rather than bounded a priori: the reduction policies run over one corpus with Reduction.EXACT (Shewchuk summation via math.fsum) as correctly rounded ground truth, and eta is the worst per-score absolute disagreement (max_abs). The ulp distance max_ulps is reported alongside but does not enter eta or tau_floor.

Upper endpoint (structure). g_min is the smallest strictly positive adjacent gap observed. Below it, no pair of distinctly scored documents lies within tau.

Why the band needs no sweep. Every tau-dependent object in the repository is piecewise constant in tau, with breakpoints only at observed gap values: tie_chains cuts wherever an adjacent gap exceeds tau, so it changes only when tau crosses a gap; tie_cliques admits an interval when its diameter is <= tau, and every diameter is a sum of gaps; tie_ball_interval(S, j, tau) is delimited by |s_i - s_j| <= tau, again a sum of gaps. If the half-open band [tau_floor, g_min) contains no observed gap, every tau in it yields bit-identical tie structure, by argument rather than by sweep. tau_band returns the band, and verify_band_invariance recomputes tie_chains at log-spaced probes across it.

The derivation is not circular: tau_floor comes from arithmetic and g_min from the score lattice, and neither is a function of tau.

### Tf exact integer denominator

Source: `src/tfidf_stability/vectorisation/tf.py`.

Term frequency is tf_i(t) = count_i(t) / L_i with L_i = sum_{s in V} count_i(s) (README section 2.2). The denominator counts in-vocabulary tokens only. Out-of-vocabulary tokens are discarded after vocabulary construction, so a mostly-filtered document still has term frequencies summing to 1 over the tokens that remain, and a document with no in-vocabulary tokens maps to the zero vector, as section 2.2 states.

L_i is accumulated as an exact Python integer, so each tf value is a single correctly-rounded binary64 division. The metamorphic property "concatenating a document with itself leaves tf unchanged" therefore holds exactly: (2c) / (2L) and c / L are equal rationals, and correct rounding maps equal rationals to the same binary64 value.

The paper does not state it, but tf rescales each document by the positive scalar 1 / L_i. Cosine similarity is invariant under positive per-vector scaling, so this normalisation changes no similarity score and no ranking. It changes only the vector norms, and hence only the perturbation bounds of sections 4.2 and 4.3. term_frequencies returns L_i alongside the vector because the section 4.2 bounds need ||tf|| and because L_i is the scalar relating these vectors to scikit-learn's.

### Tfsx container format

Source: `src/tfidf_stability/persistence/save_load.py`.

Model serialisation uses a custom container because the standard choices break the reproducibility snapshot:

* `numpy.savez` writes a zip, and a zip embeds a per-member modification timestamp, so two saves of the same model differ in bytes.
* `pickle` is version-sensitive, unsafe to load, and its output tracks Python's object layout rather than the data.
* JSON cannot carry a `double` without losing bits or bloating the file with 17-digit decimals whose parse-back is a separate risk.

`.tfsx` is a plain uncompressed container: a fixed-width header, raw little-endian arrays and UTF-8 string blocks. Saving the same model twice gives identical bytes, and a one-ulp change anywhere changes the file, which is what `test_reproducibility_snapshot.py` requires.

Layout. All integers are little-endian. The header is fixed width so it can be read without parsing:

```
magic          8   b"TFIDFSTB"
format_version u32
n_docs         u32
n_terms        u32
nnz            u64
flags          u32   bit 0: idf computed with a correctly-rounded logarithm
reduction      u32   the Reduction policy every sum was taken under
token_bytes    u64   length of the token block
doc_id_bytes   u64   length of the document-id block
reserved       u32 x 2   must be zero

indptr    i64 x (n_docs + 1)
indices   i32 x nnz
values    f64 x nnz
idf       f64 x n_terms
norms     f64 x n_docs
lengths   i64 x n_docs
df        i64 x n_terms
cf        i64 x n_terms
tokens    utf-8, LF-separated, n_terms entries
separator 1 byte, LF
doc_ids   utf-8, LF-separated, n_docs entries
```

The file length must equal the size the header describes exactly; a truncated file or trailing bytes are rejected.

A JSON sidecar carries the same metadata in readable form. It is written alongside the container and never inside it, so the container stays byte-stable whatever fields the sidecar gains.

### Tfsx idf sign

Source: `src/tfidf_stability/persistence/save_load.py`.

README section 2.1 fixes idf(t) = log((1 + N) / (1 + df(t))) + 1, and df <= N always, so the ratio is at least 1, the logarithm at least 0 and idf at least 1. A negative entry cannot come from any fit, only from a corrupt file.

Nothing downstream detects one. `similarity/cosine.py` relies on non-negative coordinates so that cos lies in [0, 1], and `check_non_negative` exists to enforce that; `ranker.py` checks scores for finiteness but not for sign, so negative scores sort like any other. For example, setting idf[0] to -5.0 in a container over ('aa', 'ab', 'cc') makes the query ['aa'] score [-0.894, 0.0, -0.707] instead of [+0.894, 0.0, +0.707]: the ranking inverts, no error is raised on load, scoring or ranking, and the two non-zero scores lie outside the range [0, 1] that README section 2.3 documents.

The check is non-negativity rather than the tighter `>= 1`: it matches what the sibling checks assert, it catches the sign flip that does the damage, and it bakes no smoothing convention into the parser.

### Tfsx integer arrays

Source: `src/tfidf_stability/persistence/save_load.py`.

The three integer arrays `lengths`, `df` and `cf` are validated alongside the float arrays and the string blocks, so all ten fields `model_bytes` writes are checked on load. Each carries a constraint a real fit cannot violate:

* `lengths[i]` counts in-vocabulary terms, so it is non-negative; it is zero for a document whose tokens are all stopwords, which the mini corpus fixture `tests/fixtures/mini_corpus.jsonl` contains (document d5). It is also at least the row's non-zero count, since each distinct term occurs at least once. This is a lower bound only: a term occurring once and one occurring three times are one non-zero each, so an inflated length cannot be detected from the container. In the three-token test container, raising `lengths[0]` from 3 to 7 makes `intermediates(0)` report tf = 0.714 for `aa` (a count of 5 out of 7) against the genuine tf = 0.667 (2 out of 3), and the format cannot refute it.
* `df[t]` counts documents containing `t`, and a term enters the vocabulary only by appearing, so `1 <= df[t] <= N`.
* `cf[t]` counts total occurrences, and a term occurs at least once in each document holding it, so `cf[t] >= df[t]`.

A wrong length changes the `tf` that `TfidfModel.intermediates()` returns and the `count` that `scripts/export_intermediates.py` derives from it. These are the two fields that need `_exact_tf` to be recovered, and that export publishes each value's `float.hex` bit pattern beside the decimal.

### Tfsx semantic validation

Source: `src/tfidf_stability/persistence/save_load.py`.

The structural checks in `model_from_bytes` confirm the container is shaped correctly; the semantic checks confirm its contents could have come from a real fit. A file can satisfy every count and still carry NaN weights (undefined behaviour for the sort) or repeated ids (which destroy the ranking operator's strict total order). Both produce a plausible-looking ranking rather than an error.

The three CSR arrays are read independently, with only their lengths fixed by the header, so a corrupt container can carry an `indptr` that is non-monotonic or does not start at zero. `CsrMatrix` is frozen with no `__post_init__` and nothing downstream re-checks it, so without `_check_csr` such a model is returned intact and misbehaves far from its cause.

### Tfsx token order

Source: `src/tfidf_stability/persistence/save_load.py`.

The token block gets the same treatment as the document-id block. `Vocabulary` is frozen with no `__post_init__`, so constructing it in the parser bypasses the `is_sorted` guard that `build_vocabulary` runs, which is the check standing between a mis-sorted vocabulary and silently wrong weights. The parser therefore runs it explicitly.

Without the check, a single changed byte can turn the vocabulary ('aa', 'ab', 'cc') into ('aa', 'aa', 'cc'). `_index`, built by enumeration, then maps 'aa' to column 1, so the query ['aa'] scores the document containing "aa" twice at 0.0 and a document containing no "aa" at 0.707. That corrupt container re-serialises to itself, so the round-trip property that the fuzz suite checks still holds and cannot detect it. The equivalent change in the document-id block is caught by `check_unique_ids`; `load_model` reads a container from disk through `model_from_bytes`, which treats its input as untrusted, so both blocks need the guard.

Strictly ascending UTF-8 byte order is required rather than uniqueness alone: it implies uniqueness and also rejects a permuted block, which passes a uniqueness test while breaking the byte order that `Vocabulary.digest` and the determinism guarantee rest on.

### Three term vector bound

Source: `src/tfidf_stability/perturbation/vector_perturb.py`.

Section 4.2 writes ``w = tf (*) idf`` and expands

    w' - w = (dtf) (*) idf + tf (*) (didf) + (dtf) (*) (didf)

The inequality ``||a (*) b||_2 <= ||a||_2 ||b||_inf`` applied termwise gives

    ||w' - w||_2 <= ||dtf||_2 ||idf||_inf
                  + ||tf||_2  ||didf||_inf
                  + ||dtf||_2 ||didf||_inf

The three terms separate a local document edit, a global corpus change, and their interaction.

`tfidf_stability.perturbation.vector_perturb` evaluates all three terms alongside the quantity they bound, so a caller can assert the inequality and report its tightness (``observed / total``). It also reports which term dominates. Section 4.2 states that the interaction term provides "a natural mechanism for perturbation amplification" but gives no evidence; the per-term values make that claim measurable.

All quantities are computed on the union vocabulary (``spec_addenda.md#g5``). A corpus perturbation changes the vocabulary, while the bound as stated assumes a fixed index set. The module also reports the exact Pythagorean split ``||w' - w||^2 = shared^2 + gained^2 + lost^2``, which separates coordinate change on shared tokens from vocabulary churn (mass on tokens present on one side only). The three parts lie on disjoint coordinate sets, so the identity is exact in the reals and holds to rounding in binary64; it also detects alignment errors, since a misaligned index breaks the identity before it breaks any inequality.

### Threshold rounding

Source: `src/tfidf_stability/vectorisation/vocabulary.py`.

A proportional min_df or max_df decides vocabulary membership, and membership decides every number downstream, so the proportion is resolved exactly. The float is first snapped with Fraction(value).limit_denominator(1_000_000). Fraction(0.1) is the exact binary64 value 3602879701896397/36028797018963968, slightly above one tenth, so `Fraction(0.1) * 30` exceeds 3 and its ceiling is 4: a stricter filter than the 0.1 the caller wrote. Snapping to 1/10 first gives 3. Without limit_denominator, min_df=0.1 changes the vocabulary. Plain float multiplication is not the hazard here: `0.3 * 10` and `0.1 * 30` are both exactly 3.0 in binary64. The rounding direction is a parameter because the two ends of the filter differ. min_df keeps `df >= p * n` and rounds up; max_df keeps `df <= p * n` and rounds down. Rounding an upper bound up admits tokens the caller asked to exclude: at p=0.5, n=3 it keeps a token present in 2 of 3 documents (66.7%), and at p=0.95, n=7 it resolves to 7 and filters nothing.

### Tie ball search

Source: `src/tfidf_stability/ranking/tie_groups.py`.

tie_ball_interval takes a non-increasing score array S, a 0-indexed position j and a tolerance tau (tau = 0 is legal and gives exact ties), and returns the half-open rank range [lo, hi) of {i : |S[i] - S[j]| <= tau}. The range is contiguous because the array is sorted.

A binary search for the values S[j] + tau and S[j] - tau evaluates the wrong predicate: those bounds are themselves rounded, so the test becomes S[i] <= fl(S[j] + tau), which differs from the |s_i - s_{r_j}| <= tau pinned by spec_addenda.md#g9. The two tests disagree only at the boundary, which is the only place tie groups are of interest.

The search therefore runs on the difference itself. The lower bound is the first i in [0, j] with S[i] - S[j] <= tau; the upper bound is the first i in [j, n) with S[j] - S[i] > tau. On a non-increasing array, S[i] - S[j] is non-increasing in i and S[j] - S[i] is non-decreasing, so both remain binary-searchable and both evaluate the subtraction G9 specifies. The monotonicity holds in binary64 as well as in the reals, because IEEE subtraction is monotone: a >= b implies fl(a - c) >= fl(b - c) and fl(c - a) <= fl(c - b).

### Tie break ablation isolation

Source: `src/tfidf_stability/analysis/tie_break_ablations.py`.

Section 7.3 recomputes each ranking under three sorting operators: pi = Sort(s_i, a_i), the full attribute tuple; pi_score = Sort(s_i, id_i), the attribute-independent baseline; and pi_alt = Sort(s_i, a_i reordered), an alternate priority. The ablation measures how far apart the results are, which separates decision-level instability (change that comes from the choice of secondary ordering rule) from numerical instability in the scores.

The separation is structural in this implementation, although the paper does not state it. All three operators share the same sorted score array, so every margin is identical between them (see tfidf_stability.ranking.margins). Any disagreement is therefore due to the tie-break alone, with delta s = 0. This keeps research questions A1 and A2 independent rather than confounded.

Two limits apply to the interpretation. First, the comparison has an effect only where ties exist: with all scores distinct the three operators coincide, and AblationResult.scores_all_distinct records that case, so a zero disagreement rate stays distinguishable from a query with nothing to break. Second, pi_score is the empty-priority special case of pi, so it agrees with pi whenever the attributes fail to discriminate. This weakens the ablation slightly and is the correct reading of README section 4.5.

### Tie clique sweep

Source: `src/tfidf_stability/ranking/tie_groups.py`.

Cliques are the sets in which every pair is within tau, so they are the sets for which "mutually indistinguishable" holds; in a chain only neighbours need be close.

The graph |s_i - s_j| <= tau is an indifference graph, so every maximal clique is a contiguous interval of the sorted order: if i < m < j and i, j are both in a clique, then S[i] >= S[m] >= S[j] gives |S[i] - S[m]| <= |S[i] - S[j]| <= tau, and likewise for m, j, so m belongs to the clique. An interval graph on N vertices has at most N maximal cliques, so taking one candidate per left endpoint misses none.

Let R(a) be the largest b with S[a] - S[b] <= tau. R is non-decreasing, so a two-pointer pass computes all of it in O(N), and [a, R(a)] is maximal exactly when a == 0 or R(a) > R(a - 1); otherwise the previous interval strictly contains it.

### Tie groups

Source: `src/tfidf_stability/ranking/tie_groups.py`.

README section 2.3.3 defines, for a tolerance tau > 0 and a rank j, the tie group G_tau(j) = { i : |s_i - score(r_j)| <= tau }, and states that documents inside one are "indistinguishable at the level of similarity scores". This definition is a ball, not an equivalence class. The relation |s_i - s_j| <= tau is reflexive and symmetric but not transitive: for scores {0, tau, 2*tau} the outer pair is unrelated. Balls therefore overlap and do not partition the corpus, "the tie group of document i" is not well defined, and two members of one ball can differ by up to 2*tau, so the quoted sentence does not hold strictly.

G1 resolves this with three separately named objects that are never conflated:

| object | definition | cost | partition |
|---|---|---|---|
| tie_ball_interval | section 2.3.3 verbatim | O(log N) | no |
| tie_chains | single linkage: adjacent gaps <= tau | O(N) | yes |
| tie_cliques | complete linkage: diameter <= tau | O(N) | no |

The ball is the primary reported object, because it is what the paper defines. The chain is the only one of the three for which "the group containing document i" is well defined, because the chains partition the ranks, so chains are reported alongside the ball wherever a partition is required. The clique is the set in which "mutually indistinguishable" holds: every pair in it is within tau. The chain-inflation ratio rho(tau) = |largest chain| / |largest clique| shows how far the chain and the clique diverge.

### Unavailable reason derivation

Source: `src/tfidf_stability/_native/__init__.py`.

During import, the loader sets `_REASON` on the two failure branches (import error, ABI mismatch) and `_MODULE` only on the success branch, so on every state reachable by import alone the two globals are mutually exclusive. unavailable_reason nevertheless returns None if native_available() else `_REASON`, instead of returning `_REASON` directly. The reason is that tests substitute a stub backend by setting `_MODULE` without clearing `_REASON`. On a machine without a built extension, returning `_REASON` directly would report a stale explanation beside a backend that reports itself available, and any manifest that records the reason would carry that contradiction; on a machine with a built extension `_REASON` is already None and the defect is hidden. The outcome would then depend on the runner's build state rather than on the code, so reference-backend CI legs and native-backend CI legs would disagree. tests/test_native_selection.py asserts the derived behaviour. Deriving the answer from native_available makes the documented contract hold for every state, and changes nothing for states reachable by import.

### Vocabulary byte order

Source: `src/tfidf_stability/vectorisation/vocabulary.py`.

The vocabulary, the identifier assignment, and therefore every document frequency, IDF value, weight, norm, score and ranking are pure functions of the multiset of preprocessed documents. They do not depend on document presentation order, hash seeds, dictionary iteration order, PYTHONHASHSEED or thread count. Two phases give this property. During accumulation any hash map is acceptable, because its iteration order does not leave the function. At freeze time the surviving tokens are sorted into UTF-8 byte order and identifiers are assigned by position, so the mapping from token to identifier depends only on the token set. Byte order is chosen over locale collation and Unicode collation, whose tailorings vary by ICU version and platform. Tokens are NFKC-normalised upstream, so byte order is well defined on a canonical form, and a memcmp comparison in C++ reproduces it. The paper does not specify the max_features truncation rule; docs/spec_addenda.md#g6 defines the total order used (TFIDF-SPEC-01).

### Zero vector convention

Source: `src/tfidf_stability/similarity/cosine.py`.

Cosine similarity is defined as 0 when either vector is zero. This convention shapes the study. A document with no in-vocabulary tokens maps to the zero vector and scores exactly 0 against every query. A document that shares no term with the query also scores exactly 0. On short-text corpora these documents form a large block of exact ties at the bottom of a ranking. That block is the regime the tie-break analysis of README section 4.5 addresses. TF-IDF coordinates are non-negative, so cosine similarity lies in [0, 1]; a negative coordinate upstream breaks this bound silently, and tfidf_stability.utils.validation.check_non_negative raises NegativeValueError on such input.

## C++ core and binding

### Compare top k negative k

Source: `cpp/tests/test_distances.cpp`.

In C++, `prefix` takes `std::max(k, 0)`, so a negative k compares two empty prefixes. Every field then reports no evidence (sets_differ false, intersection_size 0, swapped 0, kendall_intersection NaN, fks and jaccard equal to their values on two empty lists), while the `k` field carries the negative value through unclamped so that a caller can see what it asked for. Python slicing reads `a[:-1]` as all but the last element, which for the three-element test lists is the k = 2 prefix, and at k = 2 the same lists give sets_differ true and intersection_size 1, so the two implementations would give different answers. The divergence cannot be reached from Python, because both Python entry points refuse a negative k: `compare_top_k` raises `ValueError`, and the binding raises before the header is reached (`cpp/bindings/module.cpp`, "k must be non-negative"). Relaxing either check, or adding a second C++ caller, makes the divergence reachable.

### Core types integer widths

Source: `cpp/include/tfidf/core/types.hpp`.

Identifier and offset widths are fixed rather than left to the platform. Document and term identifiers (`TermId`, `DocId`) are `int32_t`. This addresses up to 2^31 - 1 (about 2.1 billion) documents and terms. Compared with 64-bit identifiers, it halves the memory traffic of the identifiers the postings loop reads, and that traffic dominates term-at-a-time scoring. Offsets into the flat index and value arrays of a sparse structure (`Offset`) are `int64_t`, because a count such as the total number of non-zeros can exceed 2^31.

### Csr indptr monotonicity

Source: `cpp/include/tfidf/vectorisation/sparse.hpp`.

`CsrView::is_canonical` checks that `indptr` is non-decreasing, and checks it before the per-row loop. `row(i)` computes `hi - lo` on `std::size_t`. A decreasing segment makes that subtraction wrap. At `hi == lo - 1` it wraps to exactly `std::dynamic_extent`, which `subspan` reads as "to the end", so the row spans the rest of the arrays and overlaps the rows after it. A larger decrease gives a count past the end of the arrays, which is undefined behaviour. The per-row loop calls `row(i)`, so the monotonicity check must run first.

The Python loader's check, `_check_csr` in `persistence/save_load.py`, raises "indptr decreases at row {row}: {lo} then {hi}". The native check matches it, so both reject such input. For example, `indptr = [0, 3, 2, 4]` over four non-zeros satisfies `front == 0` and `back == nnz`, but is not canonical. The check belongs in `is_canonical` rather than in `row()`, because `row()` may assume a canonical matrix.

### Distances positions last wins

Source: `cpp/include/tfidf/ranking/distances.hpp`.

`detail::positions` maps each item to its index, and when an item repeats, the last occurrence wins. This matches the reference's dict comprehension `{item: i for i, item in enumerate(items)}`. A ranking never repeats a document, so the pipeline cannot reach this case. The two implementations still resolve it the same way, because an unreachable case that they resolve differently is a divergence that the differential suite must rule out.

### Exact reduction non finite

Source: `cpp/include/tfidf/core/reduction.hpp`.

`Exact` keeps a list of non-overlapping partial sums whose exact total equals the exact sum of the inputs, then rounds once (Shewchuk's expansion algorithm). It is the same algorithm as Python's `math.fsum`, so the two agree bit-for-bit on finite inputs, which gives the noise-floor study one ground truth in both languages.

It supports finite inputs only. CPython's `math_fsum` tracks infinities and intermediate overflow separately and raises an exception: `math.fsum([inf, -inf])` raises ValueError and `math.fsum([1e308, 1e308])` raises OverflowError. `Exact` has no such handling and returns NaN for both of these inputs, and also for `[inf, 1.0]`, where `math.fsum` returns inf. The other three policies (`Naive`, `Neumaier`, `Pairwise`) agree with their Python counterparts on these inputs, so the difference is specific to `Exact`.

The published pipeline cannot reach this case. `l2_norm` squares each value before it sums, so finite inputs can produce an infinite addend: over `[1e200, 1e200]`, `l2_norm` under `Exact` gives NaN where the reference gives inf, and over `[1e154, 1e154]` the two finite squares overflow when summed, where the reference raises `OverflowError`. A TF-IDF weight is `tf * idf` with `tf <= 1` and `idf = ln((1 + N) / (1 + df)) + 1 <= ln((1 + N) / 2) + 1` for `df >= 1` (README sections 2.1 and 2.2), so it grows only as ln(N), and squaring one overflows binary64 only above `sqrt(DBL_MAX)`, about 1.34e154, that is for a corpus of roughly e^(1e154) documents. The case is reachable through the Python bindings that accept arbitrary values, such as `reduce_sum`, `dot` and `l2_norm`.

The difference is pinned by a test in `tests/test_native_differential.py` rather than closed. Closing it would place exception machinery inside a `noexcept` accumulator on the hot path to serve an input the pipeline cannot produce.

### Exact reduction rounding

Source: `cpp/include/tfidf/core/reduction.hpp`.

The expansion held by `Exact` is exact, but a plain fold over its partials can still round the total one ulp away from the correctly rounded result, which makes the result depend on the order in which the inputs arrived. `Exact` is the ground truth against which the other policies' error is measured and must match `math.fsum` bit-for-bit, so `value()` reproduces CPython's algorithm, including its half-even correction.

The algorithm sums from the largest partial downward until the running total becomes inexact, that is until the two-sum remainder `lo` is non-zero. If `lo` and the next remaining partial have the same sign, the exact sum lies further from `hi` than `hi + lo` does. When `lo` is exactly half the gap between `hi` and its neighbour in the direction of `lo`, round-half-even may have resolved that tie towards `hi`, which is then the wrong neighbour. The algorithm computes `y = 2 * lo` and `x = hi + y`, and replaces `hi` with `x` when `y == x - hi`, which holds exactly in that half-gap case. CPython's own regression case needs the correction: {1e-16, 1.0, 1e16} must sum to 1.0000000000000002e16, not 1e16, because the 1e-16 places the exact sum just past the halfway point between the two. With the correction, the result is the correctly rounded sum and so does not depend on input order.

### Exact reduction signed zero

Source: `cpp/include/tfidf/core/reduction.hpp`.

`Exact::add` appends the running total `x` to the expansion only when `x != 0.0`, as CPython's `math_fsum` does. Appending unconditionally preserves the value of the expansion but can change the sign of a zero result: over an input of only negative zeros, `x` is -0.0, so `value()` would return -0.0 where `math.fsum` returns +0.0. With the conditional append, `reduce_sum([-0.0], exact)` returns the bit pattern 0000000000000000, matching the reference; an unconditional append gives 8000000000000000. Because -0.0 == 0.0 under IEEE 754 comparison, no tolerance or equality check detects this difference; only a bitwise comparison does.

### Excess precision detection

Source: `cpp/include/tfidf/core/fp_guard.hpp`.

`selftest()` sets `kConstantFolding` from the macro `FLT_EVAL_METHOD`, not from an arithmetic probe. `FLT_EVAL_METHOD` is 0 when each operation is performed in its own type, and 2 when every operation is evaluated as long double. It reads 0 under `-mfpmath=sse -fexcess-precision=standard` and 2 under `-mfpmath=387 -fexcess-precision=fast`. The flag is set when the macro is undefined or non-zero.

An arithmetic probe such as `volatile double a = 0.1, b = 0.2; if (a + b == 0.3)` cannot detect the hazard. The exact sum of the doubles 0.1 and 0.2 differs from the double 0.3 at every precision, so the comparison is false whether or not intermediates are widened. Such a probe fires under no tested configuration, including `-ffast-math` and `-mfpmath=387 -fexcess-precision=fast`, so `kConstantFolding` would never be set and the matching branch of `describe()` would be unreachable.

### Fks generalised kendall cases

Source: `cpp/include/tfidf/ranking/distances.hpp`.

`kendall_fks` computes the generalised Kendall distance of G2b by the same four-case O(k^2) pair enumeration and the same normalisation as the normative Python. See [Fks cases](#fks-cases).

### Fks max guard equivalence

Source: `cpp/tests/test_distances.cpp`.

`fks_max` returns 0 wherever there is no pair to disagree about, that is for every k below one. Its guard is `k < 1`. Relaxing it to `k < 0` changes behaviour only at k == 0, where the unguarded formula `k*k + p*k*(k - 1)` computes `0*0 + p*0*(0 - 1) = 0.0 + (-0.0) = +0.0`, which is the same double the guard returns. `configs/equivalent_mutants_cpp.txt` records this mutant as equivalent. For every admissible penalty p in [0, 1] the term `p * 0.0 * -1.0` is +0.0 or -0.0 (+0.0 only for p = -0.0, which the domain check admits), and the sum is bit-identically +0.0. A non-finite penalty would make that term NaN, but `checked_penalty` refuses one before k is tested. Without any guard, at p = 0.5, k = -1 gives `1 + 0.5*(-1)*(-2) = 2.0` and k = -2 gives `4 + 0.5*(-2)*(-3) = 7.0`, so the guard is observable for negative k. The Python `fks_max` has the same guard, and `configs/equivalent_mutants.txt` carries the matching entry for it; its test pins the positive-zero result with `same_bits`. k = 1 is not degenerate ([Fks maximum](#fks-maximum)).

### Fks maximum

Source: `cpp/include/tfidf/ranking/distances.hpp`.

The maximum of `K^(p)` over two top-k lists is attained when the lists are disjoint. Their union then has 2k elements, and every pair falls into one of two cases:

- case 3, one element in each list: k^2 pairs, each contributing 1;
- case 4, both elements in the same list: `2 * C(k, 2) = k * (k - 1)` pairs, each contributing `p`.

The maximum is therefore `k^2 + p * k * (k - 1)`. The C++ computes `k^2` in int64 and adds `p * k * (k - 1)` in floating point.

`k = 1` is not degenerate: two disjoint singletons contribute one case-3 pair, so the maximum is 1. A guard of the form `if k < 2: return 0` would normalise two entirely disjoint lists to distance zero. Only `k = 0` has no pairs; it, and every negative k, returns 0.0.

`fks_max` is not `noexcept`: `checked_penalty` throws `std::invalid_argument` for a penalty outside `[0, 1]`, where the reference raises `ValueError` ([Fks penalty domain check](#fks-penalty-domain-check)).

### Fks near metric and singleton maximum

Source: `cpp/tests/test_distances.cpp`.

Two properties of the FKS generalised Kendall distance K^(p) are asserted as specified behaviour.

1. K^(p) is a near-metric, not a metric. G2 gives a witness where the triangle inequality fails: A = [3, 1, 0], B = [5, 3, 4], C = [5, 4, 2]. At p = 1/2 the unnormalised distances are d(A, B) = 6, d(B, C) = 2 and d(A, C) = 12 = fks_max(3), so d(A, C) > d(A, B) + d(B, C). A and C are disjoint, so their distance is the maximum, while B shares one element with each. In general d(A, B) = 5 + 2p, d(B, C) = 2 and d(A, C) = 9 + 6p, so the inequality fails by 2 + 4p at every penalty in [0, 1]; the test checks p in {0, 0.25, 0.5, 1}. Changing the distance to satisfy the triangle inequality would change every published K^(p) value.

2. fks_max(1) == 1.0, because two disjoint singletons form exactly one case-3 pair ([Fks maximum](#fks-maximum)). In general `fks_max(k) = k^2 + p * k * (k - 1)`, which at p = 1/2 is `k(3k - 1)/2`.

### Fks penalty domain check

Source: `cpp/include/tfidf/ranking/distances.hpp`.

G2 fixes the domain of the case-4 penalty as `[0, 1]`. The check is written as `!(penalty >= 0.0 && penalty <= 1.0)`. The equivalent-looking `penalty < 0.0 || penalty > 1.0` would accept NaN, because every comparison with NaN is false. Without the check, a NaN or a sufficiently negative penalty makes `fks_max` NaN or non-positive. The guard `ceiling > 0.0` in `kendall_fks` then fails, and two disjoint lists yield a distance of 0.0, the value that otherwise means agreement. Both `fks_max` and `kendall_fks` therefore refuse a penalty outside [0, 1], including NaN and both infinities, with `std::invalid_argument`. The endpoints 0 and 1 stay admissible, because G2 discusses them as the two biased readings.

### Fks penalty domain guard

Source: `cpp/tests/test_distances.cpp`.

The tests require `fks_max` and `kendall_fks` to refuse NaN, both infinities and every penalty outside [0, 1] with `std::invalid_argument`, and to accept the endpoints 0 and 1. See [Fks penalty domain check](#fks-penalty-domain-check).

### Fks penalty one half

Source: `cpp/include/tfidf/ranking/distances.hpp`.

`kFksPenalty` is 1/2, the unbiased contribution of a case-4 pair, on which the other list gives no evidence. See [Fks penalty](#fks-penalty).

### Fks union first appearance order

Source: `cpp/tests/test_distances.cpp`.

`kendall_fks` enumerates the union of the two lists in first-appearance order (the `uni` vector), not in sorted order and not by walking the `seen` set. At a dyadic penalty this order cannot be observed: `kFksPenalty` is 0.5 and the penalty sweeps use 0, 0.25, 0.5 and 1, so every partial sum is exact and any enumeration order gives the same double. A penalty of 1/3 is not dyadic, so the order of the additions decides the last bit. The test lists a = [5, 3, 1] and b = [4, 2, 0] are disjoint, so every pair falls in case 3 or case 4 and the addend sequence depends only on the union order. At p = 1/3 the expected unnormalised result is 11.000000000000002, which is the value the normative Python returns when it sums with `+` in the same order. Enumerating the sorted union instead gives 10.999999999999998. At p = 0.5 the same input gives exactly 12.0, so no ordering can be distinguished there.

### Fp guard layers

Source: `cpp/include/tfidf/core/fp_guard.hpp`.

A single stray compiler flag invalidates every bit-reproducibility claim in the native core. The code still runs and still looks correct, but it returns different digits. `fp_guard.hpp` therefore checks the floating-point environment at three points.

- Layer 1, compile time: `#error` directives reject flags known to break IEEE-754 semantics (`-ffast-math`, `-ffinite-math-only`, `-fno-signed-zeros`, MSVC `/fp:fast`). Defining `TFIDF_ALLOW_FAST_MATH` suppresses them. No build in the repository defines it: the CI job `guards-reject-unsound-builds` in `.github/workflows/determinism.yml` builds with `-ffast-math` and requires the build to fail with that flag named in the log.
- Layer 2, build time: the resolved compiler flag string is compiled into the binary (`build_config.hpp`) and recorded in every manifest.
- Layer 3, run time: `selftest()` checks the live environment. It reads `FLT_EVAL_METHOD` and `__ASSOCIATIVE_MATH__`, probes for FMA contraction with an arithmetic test, reads the rounding mode, and reads the FTZ and DAZ bits of MXCSR, or runs `subnormals_survive()` on targets without MXCSR.

The run-time FTZ and DAZ checks cover what no compile-time check can. A BLAS loaded with numpy (MKL, and some OpenBLAS builds) can set flush-to-zero (FTZ) and denormals-are-zero (DAZ) for the whole process. Under those modes a subnormal difference between two scores is flushed to zero, and near-tie margins are the subject of the study, so a flushed margin changes results.

### Fp guard portable flush probe

Source: `cpp/tests/test_fp_guard.cpp`.

The test sets FTZ (MXCSR bit 0x8000) and DAZ (MXCSR bit 0x0040) in every combination on x86 and checks both halves of `subnormals_survive()`, restoring MXCSR after each case and asserting that the flush bits (mask 0x8040) match their original state, so later tests in the binary do not run under a flushing mode. See [Subnormal survival probe](#subnormal-survival-probe).

### Kendall tau multiset check

Source: `cpp/include/tfidf/ranking/distances.hpp`.

`kendall_tau_distance` compares sorted copies of its inputs, not sets, as the Python compares `Counter`s, and on a mismatch throws `std::invalid_argument` naming `kendall_fks` as the function for lists that differ in membership. See [Kendall multiplicity guard](#kendall-multiplicity-guard).

### Margins sorted input

Source: `cpp/include/tfidf/ranking/margins.hpp`.

The margin functions in `tfidf::ranking` (README sections 2.3.2 and 4.4) compute three quantities from a score array sorted in descending order, where r_j is the document at rank j:

- the boundary margin m_k = score(r_k) - score(r_{k+1}), which governs top-k membership;
- the minimum adjacent margin inside the top k, m_min^top = min over 1 <= j < k of (score(r_j) - score(r_{j+1})), which governs the ordering within the top k;
- the flip radius eps_k^flip = m_k / 2.

Every function takes the sorted score array and never a ranking, so the three quantities are identical under every tie-break operator by the direction of the dependency between the modules; see [Margins](#margins).

### Native abi version

Source: `cpp/bindings/module.cpp`.

The native module exports `__abi__` (the constant `kAbi`), a version string for the shape of the binding surface that is independent of the project version and that the loader compares with `REQUIRED_ABI`. See [Abi version check](#abi-version-check).

### Native bench scope

Source: `cpp/bench/bench_main.cpp`.

The native microbenchmarks in `cpp/bench/bench_main.cpp` time the kernels directly, without FFI overhead. They measure what each reduction policy (Naive, Neumaier, Pairwise, Exact) costs. The Python benchmarks time the whole call, so argument marshalling dominates the measurement and the four policies show similar timings. In the native benchmarks the policies share only the input data.

Timings compare only within one run of one binary. The bench preset enables architecture-specific tuning, so a timing from one machine does not predict the timing on another.

### Native binding scope

Source: `cpp/bindings/module.cpp`.

`cpp/bindings/module.cpp` is the nanobind surface of the native backend. It converts arguments between Python and C++, and it validates these inputs before it calls the core: reduction policy, selection strategy, scoring algorithm, k and m bounds, CSR shape and canonical form, query canonical form, term identifier and rank index ranges, rank-matrix shape, identifier-rank bijection, attribute priority range and count, tie tolerance and score finiteness. It contains no arithmetic. All mathematics lives in cpp/include/tfidf/, so it can be tested from C++ alone and the binding technology can be replaced without touching it. Python owns every input buffer. At construction, `NativeIndex` and `NativeRanker` copy what they need into storage they own, so no C++ object keeps a borrowed pointer into a numpy array that Python might garbage-collect.

### Native k bounds

Source: `cpp/bindings/module.cpp`.

`checked_k` rejects k <= 0 at the margin entry points (`boundary_margin`, `min_adjacent_margin_top`). The normative Python reference function `resolve_k` raises `KOutOfRangeError` for k <= 0 in both strict and lenient mode, so a reference Margin never carries a negative `k_effective`. The margin entry points return `k_effective` to Python, so the native side refuses the same inputs. The other entry points that take a k-like argument have their own rules. `compare_top_k` must accept k == 0, so it does not use `checked_k`. Its own guard refuses only a negative k: in Python a negative k is a slice counted from the end, which drops elements from the end rather than taking a prefix, while the C++ header clamps it to an empty prefix, so the two would give different answers (see [Compare top k negative k](#compare-top-k-negative-k)). `fks_max` returns 0.0 for k < 1, as the reference does. `NativeRanker::top_k` checks its `m` argument in place: it refuses m < 0 and m greater than the number of ranked documents, matching `Ranking.top_k`.

### Ordering distances scope and bit exactness

Source: `cpp/include/tfidf/ranking/distances.hpp`.

The header splits the two ordering-distance problems of README sections 4.5 and 7.3 as the reference does, plain Kendall tau within a tie group and the near-metric `K^(p)` at `p = 1/2` across top-k lists; see [Fks distance](#fks-distance).

The C++ header mirrors the normative `ranking/distances.py` operation by operation, not only result by result. Two choices give this:

1. The union of the two lists is walked in first-appearance order (a, then b). Pairs are therefore enumerated in the sequence `itertools.combinations` produces, and the penalties accumulate through the same additions in the same order.
2. Every division sits where the reference puts one, so the roundings coincide exactly, not merely to within one ulp.

At `p = 1/2` every addend is dyadic and the sums are exact anyway. The implementation does not depend on that, so a non-default penalty is also bit-exact against the reference.

### Pairwise reduction order

Source: `cpp/include/tfidf/core/reduction.hpp`.

`Pairwise` is pairwise summation in streaming form. Values accumulate into blocks of `kPairwiseBlock`, and completed blocks merge through a binary-counter stack that combines partial sums of equal weight, one per level. This gives a balanced tree without knowing the length in advance, and it allocates nothing. Its error grows as O(log n) rather than O(n). The streaming form is required because the dot-product kernel consumes products one at a time and cannot see the intersection size in advance; the recursive form (split at n / 2 and recurse) needs the length up front. Both are valid pairwise schemes, and the streaming one is the pinned specification, but they do not build the same tree. The two coincide when n is at most the block size, where both are a plain fold, and when n is a power of two times the block size. For other lengths the trees differ, first at n = 129: the streaming form adds the 129th value to the sum of the first 128, while the recursion adds the sum of the first 64 values to the sum of the next 65. The Python reference (`pairwise_sum` in `utils/numerics.py`) uses the same streaming form.

The base case is `kPairwiseBlock`, the same block size numpy uses, but this does not make the result agree with `np.sum`. numpy unrolls its base case into eight independent accumulators, so its order of additions diverges from a straight fold well before the block boundary. On random inputs the two results first differ at n = 8, and over 262 tested lengths they differ at 208.

The contract is with this repository's Python reference implementation, and the differential suite in `tests/test_native_differential.py` checks it bit for bit: under every policy over 60 random lengths from 0 to 600, and for `Pairwise` at the lengths 0, 1, 127, 128, 129, 255, 256, 257, 384, 385 and 1000, where a mismatch in block size or merge rule would first appear.

### Pairwise weight equivalence

Source: `cpp/include/tfidf/core/reduction.hpp`.

`Pairwise::push` merges the new partial sum with the top of the stack while the two have equal weight, and the merged weight is the old weight multiplied by 2. The weight is only ever compared for equality, so its job is to give subtrees of equal size equal weights, not to record the size. A growth factor f assigns the weight f^k, computed in `std::size_t` and so modulo 2^64, to a subtree of 2^k blocks. The merges happen at the same points, and the summation tree is the same, whenever these weights are distinct across the levels that occur. For f = 3 they are: 3 is odd, and its powers modulo 2^64 do not repeat before 3^(2^62), far beyond the 64 levels the stack holds. Multiplying by 3 instead of 2 therefore produces the same results bit for bit, and `configs/equivalent_mutants_cpp.txt` records that mutation as equivalent. Factors 0 and 1 are not equivalent: each gives subtrees of different sizes the same weight. Nor is a factor whose powers repeat at a reachable size. With f = 2^32, the weight of a 4-block subtree wraps to 0, the same as that of an 8-block subtree, so at 12 blocks the two merge where the doubling rule keeps them apart.

### Ranker injective key stability

Source: `cpp/include/tfidf/ranking/ranker.hpp`.

The ranking operator turns scores plus tie-break ranks into a permutation of document identifiers. All five selection strategies (`FullSort`, `StableSort`, `PartialSort`, `NthElement`, `BoundedHeap`) must produce the identical permutation. This is the ranking analogue of the `TAAT == DAAT` check in the scoring layer, and it is the operational content of the claim that sort stability cannot matter for ranking.

The claim holds because the sort key is injective: identifier ranks form a bijection, so no two documents compare equal under `key_less`. The stability guarantee "elements that compare equal keep their input order" therefore quantifies over the empty set, and every correct sorting or selection algorithm satisfies it vacuously. Any two correct algorithms then return the same permutation.

### Ranker nth element hazards

Source: `cpp/include/tfidf/ranking/ranker.hpp`.

`select_top` handles two properties of the standard algorithms explicitly.

1. `std::nth_element` specifies nothing about the order of the remainder `[m, n)`, and standard libraries partition it differently. `select_top` therefore never reads `[m, n)`.
2. `std::nth_element` leaves the selected prefix `[0, m)` unordered, so `select_top` sorts the prefix afterwards.

The only postcondition of a partial selection that holds across all standard-library implementations is: every key before position `m` compares less than every key at or after it. `partition_is_valid` checks exactly this postcondition in the test build. Strategies other than `PartialSort` and `NthElement` rank all keys with a full `std::sort` and take the first `m`; the strategy-agreement test relies on this.

### Ranker selection strategies

Source: `cpp/include/tfidf/ranking/ranker.hpp`.

Python never hard-codes a `Selection` integer. As with `Reduction`, it looks each value up by name in a dictionary the module exports, here `SELECTION`, so the names are the binding contract. The enumerations differ in how far they mirror Python. `Reduction` has the same four members as the Python `Reduction`. The Python `Selection` is a string enumeration whose members past the first two (`full_sort`, `stable_sort`) name other algorithms (`heap_all`, `heap_top_k`, `insertion`) than the native enumeration does (`partial_sort`, `nth_element`, `bounded_heap`), so the two sides share only those first two names. Because the sort key is injective, every strategy yields the same permutation, and the choice of strategy affects only running time. `benchmarks/tfidf_perf.py` looks up the strategy for each native timing by its `SELECTION` name.

### Ranker stable score sort

Source: `cpp/include/tfidf/ranking/ranker.hpp`.

`sorted_scores_desc` sorts the raw score doubles in non-increasing order, independently of any ranking. One sorted array serves every operator, every `k` and every `tau`, which makes boundary margins provably independent of the tie-break.

The sort is `std::stable_sort` because the normative Python implementation is `sorted(..., reverse=True)`, and Python's `sorted` is stable. For doubles, stability matters in exactly one case: `-0.0 == 0.0` compares equal while the two values differ in bits, so an unstable sort may return them in either order.

On an input of 17 alternating signed zeros, an unstable `std::sort` returns `0.0, 0.0, -0.0, 0.0, ...` where the reference returns `0.0, -0.0, 0.0, ...`. A boundary margin is a difference of adjacent sorted scores, and 0.0 - (-0.0) is +0.0 while -0.0 - 0.0 is -0.0, so the arrangement of the zeros decides the sign of a zero margin. `boundary_margin` over the two arrangements differs in bits at k = 2, 3 and 4. The effect appears from 17 elements upwards because libstdc++'s introsort leaves ranges of 16 elements or fewer to insertion sort, and insertion sort is stable; differential fixtures with fewer than 17 elements cannot expose the difference. `tests/test_native_ranking_differential.py` therefore compares the two sorts bit for bit at 17, 33, 64 and 129 alternating signed zeros, and the boundary margins at k = 1 to 4 over 17.

Scoring itself never produces `-0.0`: TF-IDF coordinates are non-negative, so a dot product is `+0.0` or positive, and the zero-vector convention returns a literal `0.0`. However, `sorted_scores_desc` and `boundary_margin` are both exposed through the bindings and accept caller-supplied arrays, and both the bindings' finiteness check and the Python `check_finite` admit `-0.0`, so the stable sort is required for agreement with the reference.

### Ranking tests two score minimum

Source: `cpp/tests/test_ranking.cpp`.

The guards in `min_adjacent_margin_top` have three clauses: `k <= 0`, `k_effective < 2` and `n < 2`. The `n < 2` clause is unreachable. After refusing k <= 0, the function clamps k to k_effective = min(k, n) and refuses k_effective < 2. Reaching the `n < 2` clause therefore needs min(k, n) >= 2, and that forces n >= 2, so `n < 2` is never true at that point. The Python mirror marks its copy of the clause `pragma: no cover - defensive` for the same reason. If the clause were relaxed to `n <= 2`, it would become a reachable guard that refuses a two-element span, the smallest corpus that has a gap. Every other call to this function in the test file passes a span of three to five elements, so this test passes a two-element span: `min_adjacent_margin_top({1.0, 0.75}, 2)` must be defined with value 0.25. The normative Python returns the same value, and `boundary_margin({1.0, 0.75}, 1)` is also 0.25, so a guard that refuses this input diverges from the reference on a two-document corpus.

### Ratio less native cross check

Source: `cpp/include/tfidf/ranking/attributes.hpp`.

`ratio_less` compares two ratios exactly by evaluating `a_num * b_den < b_num * a_den`. The cross-multiplication equals the ratio comparison only when both denominators are strictly positive; the Python reference (`tfidf_stability.ranking.attributes.ratio_less`) states the same precondition, which the column invariant guarantees because a zero denominator means no value and never reaches the comparison. The published path never calls it, because Python has already reduced the ratios to ranks. It exists so that the C++-only tests can check that the two languages agree on the comparison rule itself, not only on the pre-computed ranks. Where the compiler provides a 128-bit integer type (`__SIZEOF_INT128__` is defined), both products are formed in `__int128`, so the product of two `int64_t` values cannot overflow. Elsewhere the products are formed in 64 bits. The Python side separately guarantees that the products fit in 64 bits (`check_ratio_fits_int64` rejects a column whose largest cross-product in absolute value over distinct documents exceeds `2^63 - 1`), so the 128-bit intermediate is a second safeguard on a path that is not performance-critical.

### Reduction constexpr magnitude

Source: `cpp/include/tfidf/core/reduction.hpp`.

`std::abs` is `constexpr` in libstdc++ (C++23, P0533) but not in MSVC's STL, which rejects it in a constant expression with error C3615, "cannot result in a constant expression". On Windows, a direct CMake invocation can select MinGW GCC to build the C++ tests, while scikit-build-core can select MSVC to build the Python extension. A header that calls `std::abs` in a constant expression can therefore compile in one build and fail in the other, so the reduction header uses its own `magnitude(x)`, defined as `x < 0 ? -x : x`.

The only difference from `std::abs` that the call site could see is at -0.0: `magnitude(-0.0)` returns -0.0, where `std::abs(-0.0)` returns +0.0. The one call site, in `Neumaier::add`, compares two magnitudes with `>=`, and -0.0 and +0.0 compare equal, so the result is unchanged. `magnitude(-inf)` is +inf, as with `std::abs`. A NaN keeps its sign bit, where `std::abs` clears it, but every comparison with a NaN is false either way.

### Reduction policies

Source: `cpp/include/tfidf/core/reduction.hpp`.

Summation order is part of the specification. In binary64, `(a + b) + c` and `a + (b + c)` are different computations. README section 2 writes its sums without bracketing, so the normative reading is a plain left-to-right fold. The `Naive` policy is that fold, and every published result uses it.

The other three policies (`Neumaier`, `Pairwise`, `Exact`) are measurement instruments. The spread of their results measures the floating-point noise floor, and the near-tie tolerance tau of README section 7.1 is derived from that noise floor rather than asserted. `Exact` is correctly rounded, so the difference between any policy and `Exact` is an absolute error.

The policies are classes, not `std::function` objects or virtual calls. Each kernel is a template on the accumulator type and is instantiated once per policy, so the accumulator stays in registers and the inner loop contains no dispatch. The policy is selected once per call, at the boundary, by a switch over the `Reduction` enumeration.

### Reduction test exact half even sign

Source: `cpp/tests/test_reduction.cpp`.

`Exact` collapses its expansion of partials from the top and then applies a round-half-even correction. The correction runs only when the remainder `lo` and the next partial have the same sign (`lo < 0 && partials < 0`, or `lo > 0 && partials > 0`), and an inner test `y == x - hi` then decides whether it changes the result. Because the inner test refuses most spurious corrections, the documented CPython case {1e-16, 1.0, 1e16} cannot separate the sign condition from the correction itself: a condition that always fires still gives the right answer on an input that needs the correction. The test inputs separate them. In each, the value is exactly halfway between two doubles once the leading partial is collapsed, because 1.0 is half an ulp at 1e16. The sign of what remains therefore decides the direction:

    {1e16, 1.0, 1e-16}     -> 1.0000000000000002e16
    {1e16, 1.0, -1e-16}    -> 1e16
    {-1e16, -1.0, -1e-16}  -> -1.0000000000000002e16
    {-1e16, -1.0, 1e-16}   -> -1e16
    {1e16, 1.0}            -> 1e16 (exact tie, round-half-even goes down)

The negative pair exercises the other arm of the condition. In the exact tie the descent consumes every partial, so the guard on the remaining count is what keeps the correction from reading index -1. All expected values are those returned by Python's `math.fsum`, which `Exact` is specified to match. Naive arithmetic returns a value of magnitude 1e16 for all five, so every difference between them comes from the correction.

### Reduction test exact last bit cases

Source: `cpp/tests/test_reduction.cpp`.

Six inputs whose correctly rounded sum is decided in the last bit. Each agrees with Python's `math.fsum`, which `Exact` is specified to match, and each gives a different result from a version of `Exact` with one specific error. The naive fold gives the same double as `Exact` on several of them, so they test `Exact` against a slightly wrong `Exact`, not against Naive.

- Expansion {-0.5, 3.0, 0.1, 1e16, 1e8, 0.5} -> 1.0000000100000004e16. Writing the surviving remainders from slot 1 in place of slot 0 leaves a stale partial; forming the two-sum remainder in `add` as `y + (hi - x)` in place of `y - (hi - x)` inverts it.
- Descent {-0.3, 3.0, 0.1, -1e8, 1e8} -> 2.8. The collapse uses a separate two-sum, `lo = y - (hi - x)`. Formed as a sum, `y + (hi - x)`, it is not the discarded part, and both the stopping point of the descent and the correction read it.
- Correction condition: `lo` and the next partial must both be negative or both positive; one alone is not enough, and the partial read must be the next one down, not the one just consumed or the one past it.
  - {1e-16, -2.0, -0.5, 1.0} -> -1.5 (both, not either)
  - {-1e-16, 3.0, -1e-16, 1e16} -> 1.0000000000000002e16 (reads the next partial)
  - {0.5, 1e16, -1.0, -1.0, 1e16} -> 2e16 (not the one just consumed)
  - {1.0000000000000002e16, -1e8, 1e16, -0.5, 1e16} -> 2.99999999e16 (not the one past it)

Each result is also required to be unchanged when the input is reversed. Order independence is the property the correction provides, and `Exact` is the ground truth against which the other policies are measured in the noise-floor study that derives the tau of README section 7.1, so a one-ulp error in it affects every such measurement.

### Reduction test ill conditioned sum

Source: `cpp/tests/test_reduction.cpp`.

The naive left-to-right fold of {1e100, 1.0, -1e100, 1.0} runs as follows:

    0 + 1e100      -> 1e100
    1e100 + 1.0    -> 1e100   (1.0 is below half an ulp of 1e100, so it is lost)
    1e100 - 1e100  -> 0
    0 + 1.0        -> 1.0     (the second 1.0 arrives after the cancellation and survives)

Naive therefore returns 1.0 and loses one of the two units. The exact sum is 2.0, and the Exact policy returns 2.0.

### Reduction test neumaier branch

Source: `cpp/tests/test_reduction.cpp`.

Neumaier's `add` step forms the compensation term by subtracting the operand of smaller magnitude from the one of larger magnitude. In that order the subtraction is exact. In the other order it loses the low-order bits that the compensation term exists to recover. The branch depends only on a comparison of two `magnitude` results, and no other case in this file separates a correct `magnitude` from one that returns its argument unchanged or one that compares against 1.0 in place of 0.0. Mixed magnitudes with cancellation separate the first. The input {0.5, -1e16, -0.5, -2.3, 1e16} sums to -2.3 under Neumaier and under Exact. Naive and Pairwise return -2.0 on it, because they lose the 0.3, and so does Neumaier when `magnitude` returns its argument unchanged. A threshold of 1.0 negates every value below 1.0, so it gives a positive operand below 1.0 a negative magnitude; on this input it still selects the right branches and Neumaier still returns -2.3. The second input, {-0.1, 0.5, 0.25, -0.5}, keeps every operand below 1.0 in magnitude and is one ulp wide: Neumaier and Exact return 0.15, while Naive, and Neumaier with a threshold of 1.0, return 0.15000000000000002. It fixes the threshold at 0.0. How `magnitude` treats -0.0 is not observable in the sum (see [Reduction constexpr magnitude](#reduction-constexpr-magnitude)).

### Reduction test pairwise merge rule

Source: `cpp/tests/test_reduction.cpp`.

Eight block sums of similar magnitude add to the same double in most association orders, so a random-data test does not fix the merge rule. The test uses eight block values chosen so that three associations give three different doubles, each one ulp from the next:

    balanced tree:        -151.17863766795136
    right-to-left fold:   -151.17863766795134
    left-to-right fold:   -151.1786376679513

The balanced-tree value is written as a literal so that the test does not agree with the implementation only by recomputing it the same way. Each block is kPairwiseBlock - 1 zeros (127) followed by one value. The block sum is then exactly that value, and every value sits at the end of its block. A block boundary off by one moves every value into the neighbouring block, which a layout with the value first in each block would not reveal.

### Reduction test pairwise tree shape

Source: `cpp/tests/test_reduction.cpp`.

`Pairwise` is a streaming accumulator: each completed block of `kPairwiseBlock` values is pushed onto a stack of partials with weight 1, and two partials merge when their weights are equal. When the number of blocks is a power of two, this builds the same tree as summing each block left to right and then combining the block sums two at a time, level by level, until one value remains. The test helper `pairwise_reference` computes that level-by-level tree independently. The accumulator matches it only if three things hold: the block size is `kPairwiseBlock`, each completed block is pushed with weight 1, and two partials merge exactly when their weights are equal. The factor by which a merged weight grows is not observable (see [Pairwise weight equivalence](#pairwise-weight-equivalence)). A test that only checks that pairwise error does not exceed naive error on a long uniform sum is satisfied by any tree shape, so it does not constrain these three. The test sums 8 * kPairwiseBlock random values (each a uniform draw from [-1, 1) times 10^k for an integer k from -12 to 3, so both signs and magnitudes below 1e3) and requires the result to be bit-identical to `pairwise_reference`. Eight blocks make the tree three rounds deep, so an error confined to the last round is still visible. The test also requires that folding the same eight block sums left to right gives a different double, which shows that the tree shape is observable on this input.

### Scoring taat determinism

Source: `cpp/include/tfidf/similarity/scoring.hpp`.

The TAAT kernel is bit-identical to the DAAT kernel because its outer loop ascends over query terms and each document keeps its own policy object, so every document receives the addend sequence a row-wise dot product produces; blocking or reordering the term loop would break this. See [Scoring term order](#scoring-term-order).

### Sort key explicit less than

Source: `cpp/include/tfidf/ranking/sort_keys.hpp`.

`key_less` compares the fields with an explicit `<` instead of a defaulted `operator<=>`. This is a correctness requirement, not a style choice. A defaulted `<=>` on a struct that contains a `double` yields `std::partial_ordering`, which `std::sort` cannot consume. Forcing `std::strong_ordering` instead would reopen the question of NaN ordering, which the finiteness guard at the boundary (`all_finite`) settles. Because every score is guaranteed finite on entry, plain `<` on the fields, applied lexicographically, is a strict total order.

### Sort key finiteness guard

Source: `cpp/include/tfidf/ranking/sort_keys.hpp`.

`all_finite` costs O(N) once per query, against the O(N log N) sort it protects. It is the only check between an arbitrary caller and undefined behaviour. A NaN score breaks the strict weak ordering that `std::sort` requires. With libstdc++, the final insertion-sort pass then walks off the front of the array. The result is an out-of-bounds write, not merely a wrong ranking.

### Sort key strict total order

Source: `cpp/include/tfidf/ranking/sort_keys.hpp`.

Each document receives the key (-score, rank_1, ..., rank_m, id_rank) that the Python reference builds, and because identifier ranks are a bijection onto 0..N-1 the key is injective, which the tests check through a unique sorted sequence and an output independent of input order. See [Sort key ordering](#sort-key-ordering).

### Sparse dot merge advance

Source: `cpp/include/tfidf/vectorisation/sparse.hpp`.

`dot_with` merges the two ascending index lists. When the indices match, it adds the product and advances both positions, although the sum does not require both increments. Indices are strictly ascending, so dropping either increment gives the same result: the next iteration compares the index that did not move against the successor of the other, takes the `a < b` or `a > b` branch, and advances it there. Removing one increment cannot be observed from outside the function, so `configs/equivalent_mutants_cpp.txt` records those mutations as equivalent and no test asserts them. In the same way, the `a < b` comparison is never reached with `a == b`, so `<=` in its place gives the same behaviour.

### Sparse l2 norm no rescaling

Source: `cpp/include/tfidf/vectorisation/sparse.hpp`.

The Euclidean norm is computed as sqrt of the sum of squares, in that order: ||v||_2 = sqrt(sum_i v_i^2). It does not use hypot-style rescaling. Rescaling is more robust to overflow but produces different digits, and README section 6 states that no stabilising transformation or numerical optimisation technique is applied. IEEE-754 requires `sqrt` to be correctly rounded, so for the same sum of squares the native and Python results agree to the last bit. `log` has no such requirement, so idf is computed on the Python side and passed to the native code as data (spec_addenda G13).

### Sparse layout invariants

Source: `cpp/include/tfidf/vectorisation/sparse.hpp`.

The sparse structures rest on two invariants.

1. Indices are strictly ascending within every vector and every CSR row. This fixes the order in which products are accumulated. binary64 addition is not associative, so a different accumulation order can give a different result. A fixed ascending order lets the native backend produce results bit-identical to the Python reference. `SparseView::is_canonical` and `CsrView::is_canonical` check this invariant.

2. Storage is structure-of-arrays: one array of indices and a separate array of values. A packed {int32, double} record is 16 bytes with padding, of which 4 bytes are padding (25% waste), and a traversal over packed records engages one prefetch stream instead of two.

### Subnormal survival probe

Source: `cpp/include/tfidf/core/fp_guard.hpp`.

`subnormals_survive()` is the portable fallback inside `selftest()`. It is the only FTZ/DAZ detection on targets without MXCSR, such as aarch64, which ships as a macOS wheel. It is a separate function so that it can be tested on an x86 machine, where MXCSR can set each mode directly.

The probe answers two separate questions:

- inputs: does the smallest subnormal, `tiny = denorm_min()`, read as non-zero? A DAZ-like mode reads subnormal operands as zero.
- results: does `tiny * 2.0`, which is itself a subnormal, stay non-zero? An FTZ-like mode flushes subnormal results to zero.

A single-condition probe of the form `tiny > 0.0 && tiny * 2.0 == 0.0` asks only whether a subnormal result collapses. Under a mode that also zeroes subnormal inputs, `tiny` itself reads as zero, so the `&&` short-circuits and the probe reports a clean environment. With MXCSR set directly on x86, the two probes report:

    mode          subnormals flushed   single-condition probe   two-half probe
    clean         no                   no                       no
    FTZ only      yes                  yes                      yes
    DAZ only      yes                  no                       yes
    FTZ + DAZ     yes                  no                       yes

FTZ alone leaves `inputs` true and makes `results` false; DAZ, alone or with FTZ, makes `inputs` false. The last row is the mode a BLAS sets, and the behaviour of AArch64's single FPCR.FZ bit. The two-half probe therefore computes both halves unconditionally and reports each one. In the fallback, `selftest()` maps zeroed inputs to `kDenormalsAreZero`, and flushed results with surviving inputs to `kFlushToZero`; on AArch64 the one bit does both, so it reports as DAZ, which is accurate and sufficient to warn.

`denorm_min()` returns the smallest positive subnormal, or `min()` on a type without subnormals. Both are non-zero, so a platform without subnormals is not reported as flushing.

### Test scoring csr envelope guards

Source: `cpp/tests/test_scoring.cpp`.

The CSR envelope check is four separate conditions: indptr has n_rows + 1 entries, indptr starts at 0, indptr ends at nnz, and indices and values have the same length. A malformed matrix usually breaks only one of them. A single test input that breaks all four at once would still pass with three of the four conditions deleted. Also, the front and back conditions are joined by one `||`, and an input that violates both cannot tell `||` from `&&`. So each subcase breaks exactly one condition. In the size, front and back subcases the other conditions stay true, so the broken condition is the only one that can reject the input. The length subcase shortens `values`, and `nnz()` is measured on `values`, so `indptr.back() == nnz` fails as well. The next test case ("the CSR guards past the envelope reject on their own too") supplies a length mismatch with the envelope intact. The first subcase confirms that the unmodified matrix is canonical, so every other subcase differs from it in exactly one respect.

### Test scoring csr indptr monotonicity

Source: `cpp/tests/test_scoring.cpp`.

The test input indptr = {0, 3, 2, 4} over four non-zeros passes the envelope checks and has rows that are each canonical on their own, so only the monotonicity check in `CsrView::is_canonical()` can reject it. See [Csr indptr monotonicity](#csr-indptr-monotonicity).

### Test scoring sparse guard boundaries

Source: `cpp/tests/test_scoring.cpp`.

The general canonical-form test uses index 99 against dim 8. That is far enough out of range that a guard relaxed from `>= dim` to `> dim` still rejects it. Its first out-of-order pair sits at position 2, so an ascent check restricted to `i > 1` (one that skips the first adjacent pair) still rejects it too. Both relaxations are real defects: the first admits an index one past the end, and the second never reads the first pair. So each bound is tested at the value that separates the correct guard from the relaxed one. An index equal to dim must be rejected while dim - 1 is accepted. A descending or equal pair at positions 0 and 1 must be rejected.

### Test scoring taat daat cross check

Source: `cpp/tests/test_scoring.cpp`.

The C++ scoring tests centre on one assertion, that TAAT and DAAT return identical binary64 scores for every document, because two traversals that share no data structure and no loop nesting are unlikely to make the same mistake. See [Native differential bit exact](#native-differential-bit-exact).

### Test scoring zero query poisoned output

Source: `cpp/tests/test_scoring.cpp`.

A zero query scores 0 against every document and never produces NaN (the zero-vector convention of section 2.3). The test fills the output with quiet NaN before scoring and checks each result on its bits. `std::vector<Real> out(n)` value-initialises to 0.0, so an unfilled buffer would also pass a plain `== 0.0` check: it cannot tell "wrote zeros" from "never ran". With the NaN fill, a kernel that writes nothing fails. The test also asserts `std::signbit(s) == false`, because `s == 0.0` is true for -0.0 as well. This repository treats -0.0 as a distinct value: score comparisons elsewhere are bitwise because float `==` conflates -0.0 with +0.0 (`bits_of` in `utils/numerics.py`), and `cpp/include/tfidf/core/fp_guard.hpp` refuses to compile under `-fno-signed-zeros`. So the expected result is exactly +0.0.

### Tie break ranks cross the boundary

Source: `cpp/include/tfidf/ranking/attributes.hpp`.

Tie-break attributes cross the language boundary as dense `int32` ranks that Python builds once per corpus, so the native side never builds a rank encoding or compares a rational. See [Rank encoding](#rank-encoding).

### Tie groups ball difference search

Source: `cpp/include/tfidf/ranking/tie_groups.hpp`.

`tie_ball_interval` returns { i : |S[i] - S[j]| <= tau } as a rank interval by binary searches on the differences S[i] - S[j] and S[j] - S[i], not on the rounded bounds S[j] +/- tau. See [Tie ball search](#tie-ball-search).

### Tie groups chain inflation bound

Source: `cpp/include/tfidf/ranking/tie_groups.hpp`.

`chain_inflation_ratio` returns rho(tau) = |largest chain| / |largest clique|, or NaN when the input is empty, and the bound rho >= 1 holds only on the non-increasing input it does not check. See [Rho order](#rho-order).

### Tie groups clique sweep

Source: `cpp/include/tfidf/ranking/tie_groups.hpp`.

`tie_cliques` returns the maximal intervals of diameter <= tau by the same complete O(N) two-pointer sweep as the reference. See [Tie clique sweep](#tie-clique-sweep).

### Tie groups objects

Source: `cpp/include/tfidf/ranking/tie_groups.hpp`.

The tie-group layer keeps the three objects of G1, the ball, the chain and the clique, apart, as the reference does. See [Tie groups](#tie-groups).

## Test suite

### Allowlist drift check

Source: `tests/test_mutation_gate.py`.

Each allowlist entry names `(path, line, kind, before, after)`. An edit above an entry moves the mutation it describes. The entry then points at a different expression on that line, or at nothing. A campaign reports the result as two findings, a stale claim at the old line and an undocumented survivor at the new one, when neither is a real finding. A campaign is the only other thing that would notice, and the nightly matrix in `nightly.yml` covers fewer modules than the allowlist argues about (12 against 34). A small edit, such as adding a few comment lines to one module, can shift several of its entries, and no campaign reports the shift when that module is outside the matrix. Whether a line still carries the mutation an entry describes can be answered from the AST alone, in about a second for the whole allowlist, so the check runs in the PR tier rather than waiting for a campaign.

### Allowlist site enumeration

Source: `tests/test_mutation_gate.py`.

`_Mutator` applies a single mutation chosen by index. Enumerating sites through it therefore re-parses the module once per site, which costs about 23 seconds over the documented modules through `_apply` (which also unparses) and about 11 seconds through the mutator alone. `_hit` is called at every site whatever the target, so a subclass that overrides it records every site in a single visit. With `target=-1` no index matches, so the base call always declines and nothing is mutated. The recorded count is cross-checked against `_count_sites`, which walks the same visitor independently. If the two disagree, the enumeration has missed a site and the dangling-entry check would stop covering it.

### Asymmetric stray seed

Source: `tests/test_analysis_harness.py`.

The noise-floor tests use two corpus builders. `_diverging_corpus` (seed 11, 50 documents of 80 tokens over a 150-term vocabulary, 5 queries of 40 tokens) makes the naive reduction policy stray from exact arithmetic symmetrically, +4 ulps and -4 ulps. On that corpus the largest signed offset and the largest magnitude are both 4, so a measurement that takes `max` over signed `ulps_between` results reports the same figure as one that takes the magnitude, and a test on it cannot tell them apart.

`_asymmetrically_straying_corpus` (seed 2, 20 documents of 60 tokens over a 120-term vocabulary, 3 queries of 30 tokens) separates the two readings. Its most negative offset is -6 ulps and its most positive is +2, so a `max` over signed values reports 2 where the true distance is 6. `ulps_between(a, b)` returns a negative offset when the policy value `a` sits above the exact value `b`, so a signed `max` starting at zero discards every stray in that direction and reports the worst stray in the other direction only. On this corpus that figure is a third of the real distance. The affected field is `PolicyError.max_ulps`. `eta`, and the tau band of section 7.1 derived from it, come from `max_abs`, which takes `abs(a - b)` and does not depend on the sign.

Seed 2 is load-bearing: at this corpus shape, seeds 0 and 1 give equal signed and unsigned quantities through the real code path, and 2 is the first seed for which they differ. The test that compares the two readings asserts against a recomputation rather than the literal 6.0, so it stays meaningful if the corpus construction shifts. It also asserts the asymmetry itself, so a corpus that stops straying asymmetrically fails the test rather than passing vacuously.

### Attribute digest is positional

Source: `tests/test_ranking_tie_breaks.py`.

The AttributeTable digest is taken over the column specs and over the per-column rank tuples, presence-bit tuples and identifier-rank tuple, all of which are indexed by document position. Supplying the same corpus in a different record order therefore permutes the tuples and changes the digest, even though the per-document mapping from identifier to rank, and so every ranking built from the table, is identical. The digest identifies this table rather than the corpus in the abstract. That makes it usable as a manifest key for a run that also records the corpus order. The test states both directions, the digest changing and the per-document ranks not changing, because a neighbouring test shows that reordering does not change the ranking and it is easy to assume the digest is invariant too.

### Benchmark tests assert structure

Source: `tests/test_benchmark_smoke.py`.

A benchmark prints numbers whatever happens, so a harness that has stopped comparing the two backends still looks healthy. The property under test in tests/test_benchmark_smoke.py is the bit-identity guard that stands between a reported speedup and a wrong answer.

No test in that file asserts a wall-clock duration as a performance threshold. Wall-clock time is a property of the machine: a threshold flakes on a loaded runner and says nothing about the code. The tests assert structure instead: every measured path appears in the report, reference-only mode runs to completion, a speedup is reported only beside a completed identity check, and the check fires on a one-ulp divergence, a signed-zero divergence, a truncated result, a transposed permutation and a ranking of a different length.

The bounds on measured values are sanity checks, not performance thresholds. A no-op must measure under one second per call, which shows the timer reports a duration rather than a clock reading; a batched no-op must measure under a millisecond per call, which shows the reported cost is per call rather than per batch. A no-op that takes that long is broken by any reading.

Rendering tests construct reports from plain dataclasses rather than measuring them, because a nanosecond operation, a multi-second one and a run that found a compiler are properties of the machine when measured and properties of the renderer when constructed. Batch-calibration boundary tests replace `_time_batch`, because a reading equal to the threshold to the last bit cannot be arranged on real hardware. The batch-cap tests lower `_MAX_INNER` and raise `_MIN_BATCH_SECONDS` to a value no batch reaches, so the cap is the only exit; at the real cap of 2^22 a test would make about four million calls.

### Canonical bytes and digests

Source: `tests/test_canonical_bytes.py`.

`utils/hashing.py` and `utils/io.py` fix the two things every reproducibility claim rests on: the exact bytes written to disk and the digest that identifies them.

A digest over floats sees the bit pattern. `hash_floats` packs `<d` rather than formatting. A digest over `repr` would depend on the interpreter's float formatting; the bit pattern does not, and it keeps a one-ulp difference visible, which is the difference the project measures. Positive and negative zero, and two distinct NaNs, hash by their bits. `hash_ints` packs a fixed width (`<i` or `<q`), and the width is part of the identity: an unsupported width raises `KeyError` and an out-of-range value raises `struct.error` rather than truncating.

Text hashing does not vary with the checkout platform. `hash_text` normalises CRLF to LF, so a digest computed on Windows equals one computed on Linux for the same tracked file. A lone CR is data and is left alone. A binary file is never normalised, because normalisation would corrupt it; `hash_file` keeps the text and binary paths separate, and the binary path streams in `_CHUNK` reads.

A write is atomic or it does not happen. `atomic_write_bytes` writes to a temporary file in the destination directory, fsyncs, then renames. If anything fails in between, the temporary file is removed (with `missing_ok=True`, so a vanished temporary does not mask the real error) and the original is left intact. A half-written report that still parses never reaches the destination path.

Fault injection in the tests patches the OS boundary (`os.replace`) and never a function inside the package, so a test cannot pass by agreeing with a mock of the thing under test. One test reaches the cleanup arm through a real failure (writing onto an existing directory) with nothing patched at all.

### Certified stable size changing edits

Source: `tests/test_perturbation_experiments.py`.

Section 4.4 bounds how far the scores of existing documents move. `max_score_shift` measures that quantity: the maximum score change over the documents present on both sides of the edit. A document that is absent from the 'before' ranking lies outside that set, so an added document can take rank 1 however little the surviving scores move. In the five-document fixture of `test_an_edit_that_changes_the_corpus_size_certifies_nothing`, only d0 shares terms with the query, and its score moves by about 0.06 (0.059) through the idf shift the addition causes: N rises from 5 to 6, and the df of alpha and beta rises from 1 to 2. That shift is inside the certified radius at k=1 (about 0.41, from a margin of 0.816), yet the added document scores 1.0 to within one ulp and takes rank 1 outright. A removal has the same gap from the other side: a document can leave the top-k without any score moving. `True` from `certified_stable` is documented as a proof, in its docstring and in `docs/perturbation_notes.md`. A `True` that does not hold is worse than no verdict, because `False` (the certificate does not cover the edit, and the ranking may still be unchanged) and `None` both invite a further check and `True` ends it. For this reason `certified_stable` returns `None` at every k for any edit whose `changes_corpus_size` is `True`, that is, whose N differs before and after; add, remove and duplicate edits all do. An edit that replaces a document's features keeps the candidate set fixed, which is the hypothesis of the section 4.4 theorem, so a verdict is still available for it.

### Chains and cliques empty together

Source: `tests/test_tie_groups_tau.py`.

`chain_inflation_ratio` and `TieGroupIndex.rho` both return NaN under a guard of the form `not chains or not cliques` (on `self.chains` and `self.cliques` in the property). The two operands always agree. Every score belongs to at least one chain and at least one clique, so both collections are empty exactly when the score vector is empty. Because of this invariant, replacing the `or` with an `and` in either guard has no observable effect, and both mutants are recorded as equivalent in `configs/equivalent_mutants.txt`, one entry per guard. The equivalence depends on the invariant, not on how the guards are worded, so `tests/test_tie_groups_tau.py` asserts the invariant directly in `test_chains_and_cliques_are_empty_together` and `test_every_score_lands_in_a_chain_and_in_a_clique`. If the invariant stopped holding, the recorded equivalence argument would be wrong even though the guards would still look reasonable.

### Conftest src path guard

Source: `tests/conftest.py`.

`tests/conftest.py` inserts `src/` at the front of `sys.path` only when `importlib.util.find_spec("tfidf_stability")` returns `None` and `src/` is not already on the path. The insertion lets the suite run with no install, including in a checkout where the native backend has not been built. The guard matters more than the insertion. An unconditional insert makes the suite test the source tree even when it is pointed at an installed distribution. The `CIBW_TEST_COMMAND` in `.github/workflows/build-wheels.yml` (the reusable workflow that `release.yml` and `wheels.yml` both call) runs `python -m pytest {project}/tests` against a freshly built wheel. With an unconditional insert the source tree shadows the wheel, nothing in the wheel is under test, and a wheel missing its data files passes. The check is by import rather than by a marker file because the question being asked is whether this interpreter can import the package as it stands. An editable install resolves back to `src/`, so a developer checkout is unaffected by the guard.

### Cpp mutation matrix is explicit

Source: `tests/test_cpp_mutation_gate.py`.

The nightly `cpp-mutation` job in `.github/workflows/nightly.yml` is a matrix of explicit header paths rather than a glob over `cpp/include`. A glob would also pick up whatever lands in `cpp/include` next, and there is no way to spell "this header is excluded" inside one. The cost of an explicit list is that it can fall behind the headers that exist. `tests/test_cpp_mutation_gate.py` refuses that in both directions: every header under `cpp/include` outside `third_party` must be either scheduled in the matrix or deferred in `_NOT_YET_RECONCILED` with a reason, and the matrix and the deferrals may name only headers that exist.

The matrix holds only headers whose survivors are all killed or argued in `configs/equivalent_mutants_cpp.txt`. A header whose campaign has not been triaged fails the nightly whatever the state of the tests, and a gate that always fails stops being read, which is the same argument `configs/equivalent_mutants.txt` makes about itself. A deferral with no reason is indistinguishable from an oversight, so each entry in `_NOT_YET_RECONCILED` must carry one. The mapping is empty: every header is scheduled.

### Csr canonical check

Source: `tests/test_tfidf_sparse_ops.py`.

`CsrMatrix` is a frozen dataclass with no validating `__post_init__`, so construction accepts any arrays. `is_canonical` is the opt-in integrity check for a malformed matrix: the loader rejects a corrupt container's CSR arrays with `_check_csr` ([Tfsx semantic validation](#tfsx-semantic-validation)), and `is_canonical` detects the same defects in a `CsrMatrix` constructed directly. A check whose reject arms never fire gives no assurance, so the tests exercise each arm by constructing a malformed matrix directly, without mocking.

### Csr invariants asserted in the fuzz helper

Source: `tests/test_fuzz_parsers.py`.

``_assert_typed`` in ``tests/test_fuzz_parsers.py`` treats an accepted mutation as legitimate only when the restored matrix satisfies the CSR invariants: ``indptr`` has ``n_rows + 1`` entries, starts at zero and ends at ``nnz``, is non-decreasing, and each row's column indices are in range and strictly increasing. A flipped mantissa bit gives a different but valid model, so acceptance alone is not a defect; acceptance of a container whose ``indptr`` is non-monotonic or does not start at zero is, and nothing downstream re-checks it. The invariants are spelled out in the helper rather than delegated to the parser's own ``_check_csr``, because a delegated check disappears with the function it delegates to and the test then passes without asserting anything. Idempotence is not a substitute for the invariant check: a model whose ``indptr`` starts at 255 re-serialises and reparses to byte-identical output, so a fixed-point assertion passes over it.

### Degenerate range guards

Source: `tests/test_datasets.py`.

`_uniform_int` rejection-samples until a draw lands below `span = high - low + 1`, and `_pick` rejection-samples until a draw lands below `cumulative[-1]`. `generate` validates the one spec field that feeds `_uniform_int`, and every other call site passes a range derived from an already-checked count; that is an invariant a new caller can break, so the samplers carry their own guards. Two degenerate shapes fail to terminate for different reasons, and a guard written for one can miss the other. At `span == 0` (the ranges [5, 4] and [0, -1]), `(0).bit_length()` is 0, so `getrandbits(0)` returns 0 and `0 < 0` is false. At `span < 0` (the ranges [5, 3] and [-3, -9]), every draw is non-negative and so never below the span; at `span == -1`, `(-1).bit_length()` is 1 and the draw is 0 or 1. The tests cover both shapes. `_pick` has the same hang at a total of zero, where `bit_length()` of 0 applies in the same way. Every weight `generate` passes to `_pick` is floored at 1, so no current call site reaches a zero total, and no spec field sets the total, so `_pick` guards it itself rather than relying on a spec check. A hang cannot be caught by a test that has to finish and presents in CI as a timeout with no stack, so both tests are driven by `_BoundedRandom`, which raises after a fixed number of draws; a weakened guard then fails the test rather than hanging the suite. Both guards refuse before any draw is asked for, and the tests assert zero draws.

### Df counts oracle

Source: `tests/test_df_counts_oracle.py`.

`tfidf_stability.vectorisation.df_counts` is a second implementation of document frequency and collection frequency. Nothing in the package imports it: `vocabulary.py` computes `df` and `cf` itself during vocabulary construction. A disagreement between the two would make every idf, and therefore every score, depend on which implementation a caller reached for. `tests/test_df_counts_oracle.py` therefore turns the duplicate into a differential oracle in the same sense as the reference-versus-native suite: the counts the vocabulary builds are compared against `document_frequencies` and `collection_frequencies` on the `mini_features` fixture and under Hypothesis over generated corpora. The two differ by design where the vocabulary is empty: a corpus with no features has empty counts, but `build_vocabulary` raises `EmptyVocabularyError`, because counting is total and vocabulary construction treats an empty result as a configuration error, which usually means `min_df` is too high. `df_after_edit` is the incremental update for the perturbation analysis of section 4.1. Its result is checked against a from-scratch recomputation on the edited corpus, and its bound, that a single-document edit moves no term's document frequency by more than one, is the premise of section 4.1, which bounds the induced idf shift; section 4.2's perturbation bounds rest on it.

### Dyadic flip witness

Source: `tests/test_margins_and_flip_radii.py`.

Section 4.4 gives eps < m_k / 2 as sufficient for top-k invariance and says nothing about tightness. The test suite shows the bound is tight with a witness built entirely from dyadic values, so every step is exact in binary64 and the test needs no floating-point tolerance.

The witness uses score(r_k) = 0.5 and score(r_{k+1}) = 0.25, so m = 0.25 and m / 2 = 0.125. With delta = 2^-30 the perturbation is eps = 0.125 + 2^-30. The rank-k document a moves down to s'_a = 0.5 - eps = 0.375 - 2^-30, and the rank-(k+1) document b moves up to s'_b = 0.25 + eps = 0.375 + 2^-30.

|ds_i| <= eps holds for every document, yet s'_b > s'_a and b displaces a from the top-k. At eps = m_k / 2 exactly the two perturbed scores meet at 0.375, bit for bit. The bound m_k / 2 is therefore tight: it is the flip radius, and no larger radius can be certified. The delta of 2^-30 is small enough to sit just past the radius and large enough that the excess is representable at the scale of the scores, so neither perturbed value rounds.

### Exact overflow divergence

Source: `tests/test_native_differential.py`.

The pinning test passes the finite inputs ``(1e200, 1e200)`` and ``(1e154, 1e154)`` to ``l2_norm``, so the overflow arises inside the squaring and summing rather than being supplied by the caller, and asserts that ``Exact`` alone departs from the reference there. See [Exact reduction non finite](#exact-reduction-non-finite).

### Exact tf reconstruction

Source: `tests/test_tfidf_sparse_ops.py`.

The test asserts that `_exact_tf` recovers tf bitwise on every case of a grid of more than 10,000 `(N, df, L, count)` combinations and that the naive `weight / idf` fails somewhere in it. See [Exact tf recovery](#exact-tf-recovery).

### Export intermediates tf

Source: `scripts/export_intermediates.py`, `tests/test_cli.py`.

`scripts/export_intermediates.py` publishes every intermediate quantity for one document, with raw bit patterns beside the decimals. Its docstring promises six quantities: term counts, term frequencies, document frequencies, IDF, the tf-idf weights and the norm. The record is built through `TfidfModel.intermediates` rather than by zipping the CSR row, because the row holds only the weight and carries neither counts nor term frequencies.

`tf` cannot be recovered from the weight by division ([Exact tf recovery](#exact-tf-recovery)). The export therefore carries `tf` itself, as `_exact_tf` recovers it, with `count` and `in_vocabulary_length` (L) beside it, so that `tf = count / L` and `w = fl(tf * idf)` are checkable bit for bit from the published record. Every float (`tf`, `idf`, `weight` and `norm`) is paired with its hex bit pattern, because a decimal rendering of a binary64 is a lossy summary at whatever precision the formatter chose and two values one ulp apart can print identically.

The export keeps its own key names (`term`, `column`) instead of the method's (`token`, `term_id`), because the shipped artefact `reports/intermediates_d000000.json` and the `tfidf-stability inspect` payload are read by different consumers. Renaming the export's fields to match the latter would break the former.

`ci.yml` runs the script and asserts only that it exits 0, so the content of the record is checked by `tests/test_cli.py` (the `_export_record` section) rather than by the workflow.

### Fks max zero guard

Source: `tests/test_ordering_distances.py`.

The mutation that narrows the `fks_max` guard from `k < 1` to `k < 0` survives in both languages, because the closed form gives bit-identically `+0.0` at `k = 0`, and the test pins that result with `same_bits`. See [Fks max guard equivalence](#fks-max-guard-equivalence).

### Float environment self test

Source: `tests/test_native_selection.py`.

Some BLAS builds set flush-to-zero and denormals-are-zero process-wide when they load. Under those flags a subnormal margin, the near-tie case that README section 7.4 measures, becomes zero without any error. The test suite imports scikit-learn, so a BLAS is loaded in the same process as the tests, and the hazard applies to the suite itself. ``check_float_environment`` runs the native ``fp_selftest`` and returns a bitmask; zero means the environment is sound. With ``restore=True`` (the default), a non-zero mask triggers a call to ``fp_restore_subnormals`` and a second ``fp_selftest``, so a BLAS that flipped the flags at load does not condemn an unattended run. A mask that is non-zero after that is returned and reported as a ``RuntimeWarning`` with ``stacklevel=2``, not raised: the caller decides whether to proceed, the manifest records the float environment, and the warning names the code that called the check rather than ``_native/__init__.py``. ``restore=False`` observes without repairing, for diagnostics that need the unrepaired state. ``test_a_subnormal_survives_arithmetic_in_this_process`` checks the same property with plain Python arithmetic, so the self-test is corroborated independently of the extension.

### G22 gap table

Source: `tests/test_datasets.py`.

G22 publishes a five-row table of adjacent-gap counts for one seeded synthetic corpus: n_docs=3000, vocab_size=5000, n_exact_duplicates=30, n_twin_pairs=60, seed=20260811, with the default preprocessing pipeline. The query is the first six features of document 0 after preprocessing, not its first six raw words: document 0 is five words, which become nine features once bigrams are added, so "six tokens" has no reading in raw words, and the two possible readings give different tables. G22 names the spec and writes the query as the expression the test uses. The table is 553 gaps of exactly 0, 0 in (0, 1e-9), 130 in [1e-9, 1e-6), 2292 in [1e-6, 1e-3) and 24 at or above 1e-3, with a smallest strictly-positive gap of 1.575373397705304e-08 and 2524 of 3000 documents scoring above zero. The 553 and 2524 figures identify the corpus; no other seed tried reproduces either. The smallest gap is asserted exactly rather than within a tolerance, because the corpus is seeded and the arithmetic is fixed, so it is a reproducibility assertion like every other float in the repository. The test is marked slow because the 3000-document fit takes about six seconds, almost all of it preprocessing. There is no cheaper form, since the table is a property of that corpus at that size; the sibling test covers the load-bearing claim, that (0, 1e-9) is empty, on a 600-document corpus, so no line depends on the gap-table test alone. A change that moves the table is applied to `docs/spec_addenda.md` and the test in the same commit, with the addendum stating what moved.

### Hypothesis health checks

Source: `tests/test_cosine_edge_cases.py`.

A per-test `@settings` replaces the profile's `suppress_health_check` list rather than extending it, so a test that names only `filter_too_much` drops the `too_slow` suppression the ci and nightly profiles set. Both checks are therefore named on `test_cosine_is_invariant_under_positive_scaling`. The three `assume()` calls in that test exclude zero norms, an overflowed scaled norm and coordinates below `NORM_UNDERFLOW_THRESHOLD` (the low-norm regime, G18), and together reject about 41% of generated examples (697 invalid of 1697 at `max_examples=1000`). That rate sits close to the `filter_too_much` threshold, so whether the check fires depends on machine, seed and Hypothesis version. Suppressing the check cannot hide a counterexample: rejections are counted before the test body runs, and every example that reaches the body is asserted in full.

### Idf linf pinned not repaired

Source: `tests/test_df_idf_correctness.py`.

`idf_linf` returns `max(x_i)` where the infinity norm `||x||_inf` is `max |x_i|`. On genuine idf values the two agree: smoothing floors every entry at 1.0, and the parameter is named `idf`, so the gap is unreachable through the documented use.

The function is exported and its docstring accepts a raw sequence. A caller that passes a delta vector, which section 4.1 makes freely negative, receives the largest positive entry instead of the largest magnitude. `vector_perturb` computes `max(abs(v) for v in ...)` inline rather than calling `idf_linf`, and nothing in the package calls `idf_linf` at all.

The behaviour is pinned by a test rather than changed: a change to `idf_linf` fails that test and so cannot pass unnoticed. A companion test sweeps a genuine idf vector and asserts the two spellings agree on it.

### Jaccard form

Source: `tests/test_ordering_distances.py`.

The reference computes the Jaccard distance as `1 - |A n B| / |A u B|`. The form `(|A u B| - |A n B|) / |A u B|` is algebraically equal but rounds differently in binary64: for `[1, 2]` against `[2, 3]` the first gives 0.6666666666666667 and the second gives 2/3 = 0.6666666666666666, one ulp apart. The native core must reproduce the reference bit for bit, so the choice of expression is part of the cross-language contract. The test compares with `same_bits` against `1.0 - 1 / 3` and asserts inequality with `2 / 3`, because `pytest.approx` would accept either form.

### Kendall multiplicity guard

Source: `tests/test_ordering_distances.py`.

`kendall_tau_distance` requires both arguments to be orderings of the same multiset. A guard that checks only equal length and equal `set()` accepts `[1, 1, 2]` and `[1, 2, 2]`. The `position` map then keeps only the last index of each repeated element, so the inversion count runs against a mapping that has lost a document and returns `0.0`: a confident result for inputs where one list is not an ordering of the other's multiset. `[1, 2, 1]` passes the same two checks against `[1, 2, 2]` and returns 1/3, an equally meaningless value. The guard therefore compares `Counter`s in Python, and the C++ mirror compares sorted copies of both lists, so both languages refuse the same inputs. No caller in the package can reach this case, since every argument comes from a `Ranking` whose order is a permutation; the guard exists because the failure mode is a wrong number rather than an error. Lists that carry the same repeats on both sides, such as `[1, 1, 2]` and `[2, 1, 1]`, are still accepted and return a value from the same last-wins map, which has no canonical meaning for a multiset. Rejecting any repeated element would close that case but is a larger change to the function's contract.

### Line keyed allowlist

Source: `tests/test_mutation_gate.py`.

An allowlist entry identifies a mutation by `(path, line, kind, before, after)`. A mutation's index in document order would separate two identical mutations on the same line, but that index shifts whenever anything earlier in the file is edited, so every claim in a module would go stale after an unrelated change. The line-based key accepts the opposite cost: two mutable sites on one line that yield the same mutation share a key and cannot be told apart. `@dataclass(frozen=True, slots=True)`, for example, yields two identical `constant True -> False` keys. The header of `configs/equivalent_mutants.txt` therefore requires a claim on such a line to justify every mutation that matches it. Moving the expression to another line invalidates the claim. Each entry carries a `src=` stamp, a fingerprint of the source line it describes, and `tooling/renumber_allowlists.py` moves an entry onto the line its stamp names.

### Logging as a provenance channel

Source: `tests/test_logging.py`.

The package logger is a provenance channel with three properties, and tests/test_logging.py checks each of them.

The tests check that importing configures no logging, which only a fresh interpreter can show because by collection time the package is imported and pytest's logging plugin has a handler on the root logger, that captured output is reproducible (field order does not change the digest, event order does), and that warnings are not routed through logging. See [Logging provenance](#logging-provenance).

### Lookup lemmatiser digest literals

Source: `tests/test_lemmatiser_backends.py`.

The identity tests in `tests/test_lemmatiser_backends.py` compare one identity against another, so they hold under any canonical form the digest is taken over: a change to `sort_keys` or `ensure_ascii` in `LookupLemmatiser` shifts every identity together and leaves every relative assertion true. Those two flags are therefore invisible to relative tests, and only a pinned absolute value can detect a change to them.

The canonical form is not free to change. A digest that shifts silently turns every recorded identity into a miss: a cache keyed on it re-runs work it already holds, and a manifest compared against an older one reports a difference where the features are identical. This is the same reason `configs/pipeline_digest.txt` holds literals rather than recomputing them.

So `_PINNED_IDENTITIES` holds four literal SHA-256 digests: an empty table, an ASCII table, a non-ASCII table and a table with a stemming fallback. The non-ASCII case is the only one from which `ensure_ascii` is observable: under `ensure_ascii=True` the table serialises as escape sequences rather than the characters themselves, giving a different digest for the same mapping while every test that does not compare against this literal still passes.

The literals are not values to refresh when they fail. A failure means the canonical form moved, and the question to answer first is whether every lookup identity the project has recorded (in caches and run manifests) is meant to move with it.

### Manifest machine keys

Source: `tests/test_reproducibility_snapshot.py`.

The manifest digest excludes three environment keys, `RunManifest._MACHINE_KEYS`: `platform`, `machine` and `python`. `environment["platform"]` is `platform.platform()`, which includes the operating system build number, so a digest that covered it would differ between machines on different operating system builds, and on one machine across an operating system update. The keys remain in the written manifest, the same treatment `hostname` and `cwd` receive as volatile keys, because a reader tracing a discrepancy needs to know which machine produced the file; deciding whether two runs match is a separate job. The machine keys are stripped in `digest` rather than added to `VOLATILE_KEYS`, because that set is shared with every other report the package hashes, and a machine identity is irrelevant only to this digest.

A test that builds two manifests in one process cannot check this, because `environment_block()` returns identical dictionaries within a process. The test therefore replaces the three keys with values of the shape a Linux x86_64 leg of `determinism.yml` reports, and asserts that `float` is unchanged. A separate test substitutes each key on its own, so an unstripped key cannot hide behind a value that happens to match. That workflow requires one identical pipeline digest across 18 legs (three operating systems by three interpreter versions, and three operating systems by three native build types), which is the evidence that these three keys move no number.

The other environment keys stay in the digest because they can move numbers. A fast-math or arch-tuned build changes `native`, and a flush-to-zero mode changes `float`. `implementation` stays covered because no claim is made that CPython and PyPy agree bit for bit; `determinism.yml` varies the interpreter version, never its implementation.

### Min df default keeps hapaxes

Source: `tests/test_df_idf_correctness.py`.

`VocabularyConfig.min_df` defaults to 1 (`min_df: int | float = 1`). One is the identity threshold: every term appears in at least one document, so the default filters nothing.

A hapax is not noise in this pipeline. A term in exactly one document has the highest idf in the corpus, and idf is what separates near-ties between documents. A higher default would discard the terms that carry the most weight, and the discard would be invisible because the vocabulary would still build.

The default is a value, not a separate code path: `build_vocabulary(docs)` and `build_vocabulary(docs, VocabularyConfig(min_df=1))` produce the same vocabulary digest. A test at `min_df=2` on the same corpus discards both hapaxes, which confirms the corpus has something to filter and the default test is not vacuous.

### Mutation baseline without werror

Source: `tests/test_cpp_mutation_gate.py`.

Warnings-as-errors and mutation testing have incompatible purposes. A mutation writes code that trips a warning by construction: `< -> <=` makes a comparison always true, `+ -> -` converts a sign, `|| -> &&` makes a branch unreachable. Under `-Werror` those mutants fail to compile and score `stillborn`, counted neither killed nor survived, so the tests never run against them and the campaign reports nothing about whether the suite would catch them.

On `sort_keys.hpp` the difference is large: with `TFIDF_WERROR` off the campaign scores 26 killed and 23 stillborn; with it on, 16 killed and 35 stillborn, and two argued survivors score stillborn instead of surviving. The mutation preset therefore sets `TFIDF_WERROR` to `OFF` and inherits `TFIDF_BUILD_TESTS` `ON` from `base`, since the campaign's verdict is ctest.

The workflow names the preset and the preset holds the cache variables, so the two are one fact in two files. The gate reads the preset the `cpp-mutation` job names in `.github/workflows/nightly.yml` and resolves its `inherits` chain in `CMakePresets.json`, rather than restating the preset name, so a renamed preset or a value inherited from `base` is still seen.

### Nan weight threshold

Source: `tests/test_profiles_and_loo.py`.

The interaction filter in `group_interactions` drops an interaction when `weight < min_weight`. That comparison is false for every weight when the threshold is NaN, so a NaN threshold keeps every interaction, exactly as no threshold does. Four other guards in the package, on the tau thresholds in `ranking/tie_groups.py` and `analysis/stratify.py`, are written as `not (tau >= 0.0)` so that a NaN is refused rather than ignored. The weight threshold does not follow that shape, for three reasons: it has no non-negative domain, a negative threshold is a legal way to keep every interaction, and no caller constructs a NaN. The difference from those four guards is a design choice, not an oversight. The test pins it alongside a negative threshold that also keeps everything and a real threshold that does filter.

### Native differential bit exact

Source: `tests/test_native_differential.py`.

The C++20 evaluator is required to perform the same floating-point operations in the same order as the pure-Python reference. This keeps the statement in README section 6 that no numerical optimisation techniques are employed literally true while the experiment grid still runs in minutes.

Because the claim is operation-for-operation equivalence, the differential tests compare IEEE 754 bit patterns through ``same_bits`` and never through ``pytest.approx``. A tolerance would accept the kind of divergence the study exists to detect, so a tolerance-based differential test would pass while hiding the signal.

Three implementations are compared:

* the Python reference, which is normative;
* the native TAAT kernel, which walks postings lists out of an inverted index into a dense accumulator;
* the native DAAT kernel, which merges each document's row against the query and builds no inverted index.

TAAT and DAAT share no data structure and no loop nesting. Bit-for-bit agreement between them is therefore independent evidence of correctness in a way that agreement with a recorded expectation is not: an indexing or accumulation defect in one traversal has no counterpart in the other. The same argument governs `tests/test_scoring_kernels.py`, which compares the Python TAAT, DAAT and cosine_against_corpus (a per-document merge over a materialised sequence of rows), and the C++ scoring tests, which compare the native kernels. [Scoring term order](#scoring-term-order) explains why the traversals agree to the bit.

### Native distances differential

Source: `tests/test_native_distances_differential.py`.

The differential for the ordering distances makes the same claim as the scoring differential (same operations, same order), reached by two different routes, which separates a logic defect from an accumulation-order defect when a failure is diagnosed.

``inversion_count`` and the set measures are integer quantities with a float only at the final division, so agreement between the reference and the native core is combinatorial and any divergence is a logic defect.

``kendall_fks`` accumulates floats. Its total sums ones and penalties over the pairs of the union, so it depends on the order in which the union is walked. At ``p = 1/2`` every addend is dyadic and the sum is exact in any order, so the default penalty cannot expose a mis-ordered enumeration. The suite therefore drives non-dyadic penalties as well (``1/3``, ``0.1``, ``0.7``): the additions round, and only the reference's first-appearance order reproduces its bits. Without these penalties the suite passes on an implementation that walks the union in sorted order.

NaN is compared as NaN rather than as a bit pattern. ``kendall_intersection`` is undefined below two shared elements, and the contract stops at "undefined": no particular quiet-NaN payload is promised, so the differential compares definedness and compares bits only where a value is claimed.

### Native package coverage

Source: `tests/test_native_selection.py`.

``src/tfidf_stability/_native/__init__.py`` is project Python that the coverage gate measures ([Coverage omits only compiled extension](#coverage-omits-only-compiled-extension)). Its two module-level failure arms (the ``ImportError`` arm and the ABI-mismatch arm) run once at import, cannot be replayed in a process that has already imported the module, and carry ``pragma: no cover``. The functions that read the resolved state are exercised instead by substituting that state rather than by re-importing: ``tests/test_native_selection.py`` substitutes ``_MODULE``, ``_REASON`` and ``require_native``, and ``tests/test_validation_contracts.py`` also substitutes ``_ABI_MISMATCH``.

### Native ranking permutation identity

Source: `tests/test_native_ranking_differential.py`.

The native ranker and the reference ranker produce element-by-element identical permutations. A ranking is a sequence of int32, so the claim is exact equality of the two orders; margins and sorted scores are floats and are held to the bitwise standard.

Permutation identity holds under four separately testable conditions:

1. The sort key is injective. Identifier ranks are a bijection, which the native constructor validates and refuses to proceed without.
2. The key inputs are identical. Scores are bit-exact by the scoring differential tests, and the integer attribute and identifier ranks cross the language boundary as data with no re-derivation on the native side.
3. The comparison relation is the same in both languages: IEEE `<` on finite doubles, with negation (a sign-bit flip) the only arithmetic applied.
4. The build is not fast-math, per `test_build_is_reproducible` in `tests/test_native_differential.py`.

A uniform-random double vector contains a tie with probability near zero, so a differential test over random scores exercises none of the tie-break and passes with a broken attribute table. The fixture scores therefore come from a small discrete alphabet (0.0, 0.25, 0.5, 0.75), and `n // 5` scores of each corpus are fixed at zero before the alphabet draws, which add further zeros. The zero block stands in for the zero-norm documents that short-text corpora produce in bulk.

### Near tie interval synthetic only

Source: `tests/test_datasets.py`.

On a seeded synthetic corpus, adjacent cosine score gaps land either at exactly zero or above 1e-9, with the interval between them empty. The CI experiments run on this generator, so the test pins the lattice there. The property does not generalise to real data. On MovieLens under the normative naive reduction, 197 of 114,504 adjacent pairs fall in (0, 4.44e-16) and the smallest strictly-positive gap is 8.67e-19, below the arithmetic noise floor. Recomputed exactly, those gaps disappear, so they are artefacts of naive summation. Section 7.4's regime therefore differs on real data, and G22 records both regimes.

### Negative index guards

Source: `tests/test_query_grid_protocol.py`.

Five sites in the package take a count or index and refuse a negative value: `short` in `utils/hashing.py`, `Ranking.top_k`, `compare_top_k`, `CsrMatrix.row`, and the `limit` argument of `build_query_grid`. In Python a negative value is a legal index or slice bound, so an unguarded site silently counts from the end instead of raising.

The `limit` site is the one a driver can reach. Both experiment drivers (`scripts/run_stability_profile.py` and `scripts/run_tie_break_ablations.py`) pass `--queries` straight through as `limit`, and argparse accepts `type=int` with no lower bound. Without the guard, `folds[:-1]` trims one query rather than raising or meaning unlimited, and the reduced count is reported as the protocol's own. The guard is written `not limit >= 0` rather than `limit < 0` so that a NaN, which compares false against every bound, is refused with the same `ValueError`. Under `limit < 0` a NaN passes the guard, the cap test `len(queries) > limit` is also false, and the full grid is returned uncapped with no error.

### Packaged asset resolution

Source: `tests/test_packaged_assets.py`.

Two files the package cannot work without sit outside the package directory in a checkout: the frozen stopword list under ``data/assets`` and the normative ``configs/default.yaml``. In a wheel both are copied inside ``tfidf_stability/``. A path computed as ``parents[3]`` from the module file is the repository root from ``src/tfidf_stability/<sub>/`` but the directory above ``site-packages`` from an installed distribution, so a fixed-depth walk cannot serve both layouts. The resolvers in ``preprocessing.stopwords`` and ``cli.commands`` therefore take the module's path as an argument, check the packaged location first and fall back to the repository layout. The packaged copy is preferred so that an installed distribution is self-contained; preferring the repository would make a wheel read whatever tree sits above ``site-packages``. For the configuration the check is on the file, not the directory, so a packaged ``configs/`` without ``default.yaml`` does not shadow the checkout's copy. A checkout presents only the repository layout, so the wheel branch is unreachable from the suite unless a layout is built to order under ``tmp_path``; ``tests/test_packaged_assets.py`` builds both. The wheel's own test command then covers the packaged layout because ``tests/conftest.py`` does not shadow an installed package with ``src/`` ([Conftest src path guard](#conftest-src-path-guard)).

### Pairwise tree oracle

Source: `tests/test_reduction_policies.py`.

`pairwise_sum` is a streaming binary counter. Blocks of 128 elements accumulate, each completed block merges with any stored partial of equal weight, and the weight doubles on each merge. The shape of that tree is part of the specification: the C++ core is required to build the identical tree, bit for bit, and `tests/test_native_differential.py` checks it at the lengths where the tree changes shape. A comparison against `exact_sum` cannot pin the shape, because many different trees land within a few ulps of the correctly rounded answer. A reversed carry comparison, or a merge that does not double the weight, still passes such a comparison. The test oracle is therefore a second implementation of the same scheme, written from the description rather than from the code, and the comparison is on bits. The oracle keeps `(weight, value)` pairs in one stack and derives each new weight from the stored one, while the module keeps two parallel stacks and doubles a running weight. The tree is the same and the bookkeeping differs, so a bookkeeping defect in either implementation appears as a disagreement and is not reproduced in both. On smooth input such as `1/(i+1)` many regroupings land on identical bits, so the oracle is also run on a fixed-seed mix of magnitudes spanning 32 decades, where any length that carries makes the association observable.

### Parser fuzz contract

Source: `tests/test_fuzz_parsers.py`.

The ``.tfsx`` container parser and the tokeniser consume bytes the package does not write itself: a ``.tfsx`` file can arrive from anywhere, and document text is arbitrary. The fuzz tests in ``tests/test_fuzz_parsers.py`` hold the container parser to one property: it either round-trips its input correctly or raises a typed exception from ``tfidf_stability.utils.validation``. The tokeniser is held to accepting any text without raising. Three outcomes count as defects. A silently wrong model is the worst of them, because a plausible ranking computed from corrupt weights is indistinguishable from a real one. An untyped exception (``struct.error``, ``IndexError``, ``UnicodeDecodeError``, ``MemoryError``) is a defect because a caller cannot handle it. Unbounded time or memory on a small input is a defect because a hostile length field such as ``nnz`` sizes an array read, so a trusted value lets a twelve-byte edit request terabytes. Truncation is checked at every offset rather than sampled: the document-id block is the last block in the container, so a cut at the final byte is the one case that satisfies every count check while still corrupting the last identifier.

### Pinned tiny corpus digests

Source: `tests/test_datasets.py`.

The digests of `synthetic_tiny` are recorded as constants in the test rather than recomputed, because "the same spec gives the same corpus" is satisfied by any deterministic generator, including a corrupted one: a change to the twin source index, the attribute ranges or the interaction sampler leaves every determinism test green. Pinning is safe because the generator draws only from `random.Random.getrandbits`, whose Mersenne Twister stream CPython keeps stable across versions, and at the default Zipf exponent of 1, which `synthetic_tiny` uses, every weight and comparison on the way is integer. A non-integer exponent goes through `pow` and the platform libm. That is the reason the module is written as it is, and the digest assertion holds it to that. The corpus digest is taken over the records alone, so two loads producing identical documents agree whatever their provenance; the interactions therefore carry a separate digest. If a change to the generator moves the digests, they are updated in the same commit and anything that quotes a number from the corpus is re-run. The pins make such a change deliberate; they do not forbid it.

### Public api surface tests

Source: `tests/test_public_api_surface.py`.

`tests/test_public_api_surface.py` pins three properties that no other test file asserts.

First, the aliases in `types.py` correspond to `cpp/include/tfidf/core/types.hpp` as a checked table, not as a prose claim. `DocId` denotes a row index (`std::int32_t`) in C++ and an external string identifier in Python, and the Python row index is `DocIndex`. The same name with different meanings across a language boundary gives a plausible wrong answer rather than a crash, so the correspondence is asserted in both directions.

Second, every enum in the package is a `class X(str, Enum)` with an explicit `__str__`. `pyproject.toml` justifies not using `StrEnum` because the explicit method makes the serialised form part of the visible contract rather than an inherited behaviour. These values are written to run manifests and read from YAML configs, so a member whose `str()` differs from its value changes a published file.

Third, every name in every module's `__all__` resolves. Several modules are imported by nothing else, so a stale entry raises only when a user tries the documented import.

The file does not import a module only to execute its lines: `types.py` is seven `TypeAlias` assignments that run at import, so a bare import scores it fully covered while asserting nothing. The tests that check `__all__` and walk the package for enums and dataclasses import every module outside `_snowball` and `_native`, which marks import-time statements across the package as covered without exercising behaviour. A module's coverage is therefore judged by its own owning test file, not by this one.

### Query grid protocol

Source: `tests/test_query_grid_protocol.py`.

`analysis/query_grid.py` decides which queries the reported experiments run and, for each query, which documents it may retrieve. Every A1 and A2 number is computed over the grid this module builds, so a change to it moves every published figure. `tests/test_query_grid_protocol.py` covers four properties of the module.

A per-query candidate set, not a corpus-wide one. `_score_one` restricts the attribute table to the same subset as the scores, so the tie-break cannot reach a document the query is not allowed to retrieve. If scores and table described different sets, a candidate would be ranked against an outsider's attributes.

Degenerate and zero-vector queries are different findings. A query with no features at all is counted in `n_degenerate` and skipped. A query whose features are all out of vocabulary is scored, comes back all-zero, and is counted in `n_zero_vector`. Scored, both would give an all-zero vector and a ranking decided by attributes alone, so conflating them would report one cause for two.

The emptiness guard on `is_zero_vector`. `all()` over no scores is `True`, so without the guard a query with no candidates answers yes and enters the published `n_zero_vector` count as a pure-attribute ranking with nothing to rank.

A deterministic prefix under `limit`. Runs at different caps must remain comparable, so the cap takes a prefix of the grid in a fixed order rather than a sample.

### Ratio less unchecked denominators

Source: `tests/test_ranking_tie_breaks.py`.

`ratio_less` compares two ratios by cross-multiplication, `a_num * b_den < b_num * a_den`, and requires both denominators to be strictly positive. It does not check this. A zero denominator means "no rating"; that state is carried by the column's `has_value` bit, so such rows never reach the comparison. `ratio_less` is the innermost comparison of every tie-break, and a check there would run on every pair of every sort, so the precondition is pinned by tests rather than guarded at run time. The tests record the consequences of a violation. A zero denominator raises no `ZeroDivisionError` and gives a defined but meaningless answer: `ratio_less(1, 0, 1, 1)` and `ratio_less(0, 0, 0, 0)` both return `False`. A negative denominator flips the inequality and gives a confidently wrong answer: 1/-2 is less than 1/1, and `ratio_less(1, -2, 1, 1)` returns `False`.

### Ratio overflow scan two largest

Source: `tests/test_ranking_tie_breaks.py`.

`check_ratio_fits_int64` refuses a ratio column unless every cross-product `num_i * den_j` with `i != j` fits in int64, so the C++ mirror can skip a checked multiply. Only products between different documents occur, since nothing is compared with itself. The scan therefore takes the two largest numerators and the two largest denominators by absolute value (`sorted(..., reverse=True)[:2]`) and maximises over the admissible combinations among them.

Keeping only the single largest of each fails when both belong to the same document. With pairs (2^62, 2^62) and (2^40, 2^40), the maximising product is the largest numerator against the second largest denominator, 2^102, which overflows; a top-one guard finds no admissible combination and passes the column.

The maximum is taken over all combinations rather than the first admissible one. With pairs (4e9, 4e9) and (3e9, 2), the first admissible pair the loop reaches is `4e9 * 2`, which fits, and the overflowing pair is `3e9 * 4e9` = 1.2e19, above the int64 maximum of about 9.22e18. A rule that ranked candidates by anything other than their product, such as the quotient, would settle on the first pair and pass a column the mirror cannot represent.

The direction of the sort matters only with more than two documents: with two pairs both orderings reach the same four combinations. The test therefore uses three pairs, (2^62, 2^62), (2^40, 2^40) and (1, 1), the fewest for which taking the two smallest pairs excludes the overflow.

### Reproducibility snapshot

Source: `tests/test_reproducibility_snapshot.py`.

README section 3 states that all stages of the pipeline are deterministic given a fixed corpus, configuration and software environment. tests/test_reproducibility_snapshot.py tests that statement end to end. The snapshot is one SHA-256 over every intermediate the pipeline produces: vocabulary, document frequencies, IDF, weights, norms, scores, rankings and margins. Every floating-point quantity in it is hashed from its raw binary64 bit pattern, not from any decimal rendering, so a one-ulp change anywhere moves it. A changed snapshot therefore means a published number has moved, and the commit that moves it states why.

The digest is insensitive to everything that is not a number: working directory, time, document arrival order and interpreter hash seed. Document order has its own test. The hash seed and the working directory vary together across three subprocesses. Time is covered by the manifest tests, since the manifest digest strips timestamps and durations.

The pipeline digest value is not hard-coded in this file. The tests recompute the digest under varied conditions and require the results to agree, which checks determinism without fixing the expected value. The expected values are recorded separately, one stage per line, in `configs/pipeline_digest.txt`, which `scripts/snapshot.py` checks with `--check`.

The module imports `pytest` only inside the tests that need it, never at module scope. `test_the_digest_is_stable_across_processes_and_hash_seeds` imports the module in a plain interpreter subprocess whose environment holds only `PYTHONHASHSEED` and an empty `PATH`, so that no inherited variable can reach a digest. With a top-level `import pytest`, the module imports there only when `pytest` is installed in that interpreter's own site-packages; a `pytest` reached through `PYTHONPATH` is absent, and the failure surfaces as a CalledProcessError from the subprocess rather than as an ImportError naming the cause. For the same reason the per-key manifest tests use a loop in place of `pytest.mark.parametrize`, since a decorator is evaluated at import time. Each loop runs over a literal tuple, and the count asserted after it shows that every key was exercised.

### Sandbox import precedence

Source: `tests/test_mutation_gate.py`.

Two tests check that a test run inside the mutation sandbox imports the sandbox's copy of the package rather than the working tree that an editable install puts on the path. See [Sandbox sitecustomize](#sandbox-sitecustomize).

### Score dispatch fallthrough

Source: `tests/test_scoring_kernels.py`.

`score()` in `similarity/scoring.py` checks `algorithm is ScoringAlgorithm.DAAT` and runs TAAT for every other value. That includes the bare string "daat": it equals the str enum member's value but is not the member. Any non-member value runs TAAT rather than raising. The test pins this behaviour and does not treat it as a defect. The two traversals must agree to the last bit, so taking the wrong branch costs time but never changes an answer. The C++ `score` in `cpp/include/tfidf/similarity/scoring.hpp` falls through in the same way, and raising on the Python side alone would make the two implementations disagree about an input neither is asked for. The binding is the one place a value is refused: `checked_algorithm` in `cpp/bindings/module.cpp` rejects an integer outside the enum, so a manifest never names an algorithm that did not run.

### Scoring kernels three way bit identity

Source: `tests/test_scoring_kernels.py`.

The scoring tests compare the Python TAAT, DAAT and cosine_against_corpus on raw bit patterns, with no tolerance, for the reason the native differential compares three implementations. See [Native differential bit exact](#native-differential-bit-exact).

### Signed zero sort stability

Source: `tests/test_native_ranking_differential.py`.

The signed-zero fixtures are sized 17, 33, 64 and 129, because libstdc++'s `std::sort` leaves 16 elements or fewer to a stable insertion sort, and the margin test compares `m_k` bitwise for k = 1 to 4 on the 17-element fixture. See [Ranker stable score sort](#ranker-stable-score-sort).

### Sklearn oracle scaling

Source: `tests/test_tfidf_sparse_ops.py`.

scikit-learn is the external oracle for TF-IDF construction. It is an independent implementation of the same weighting scheme, so it catches errors that a comparison between the project's two backends cannot: both backends can share the same misreading of the specification. scikit-learn differs from section 2.2 of the paper in two ways. It uses raw counts where section 2.2 uses `count / L`, and it L2-normalises rows where section 2.2 applies no normalisation. Both differences are positive per-vector scalings, and cosine similarity is invariant under positive scaling. Three properties follow. First, the `idf` formula is the same, because `smooth_idf=True` is the formula of section 2.1: under `LogImpl.PLATFORM` the values agree bitwise, and under the default correctly-rounded logarithm (spec_addenda G13) they differ by at most one ulp in a minority of entries. Second, the vectors differ by the scalar `1 / L_i`: each weight equals scikit-learn's unnormalised weight divided by `L_i`, to within 2 ulps. Third, the similarities agree, to within `1e-14`. A separate test guards the difference in normalisation: norms all equal to 1, as scikit-learn's are, would indicate that scikit-learn's convention had been implemented in place of the paper's.

### Str enum field coercion

Source: `tests/test_public_api_surface.py`.

All enums in the package are str enums, so a string such as "vector_mean" compares equal to its member without being that member. Dispatch sites such as `similarity/scoring.py` select with `is`. A dataclass field left uncoerced therefore takes a different branch from the one its value names, with no error. For an uncoerced `Reduction` such as "naive", `similarity/scoring.py` fails `policy is Reduction.NAIVE` and falls through to the per-document arm written for compensated and tree-shaped reductions, while `reduce_sum` in `utils/numerics.py` looks the same value up in a dict and selects the naive reducer, so one string gives two behaviours. The record also keeps the requested value, so nothing downstream can detect the mismatch. Each dataclass therefore normalises its str-enum fields to their member in `__post_init__`. `test_every_str_enum_field_of_a_dataclass_is_coerced` walks the package source with `ast` and fails on any such field that no `__post_init__` coerces, unless the class is listed with a reason in `_UNCOERCED_BY_DESIGN`, which is empty.

### Strict total order oracle specimens

Source: `tests/test_ranking_tie_breaks.py`.

`assert_strict_total_order` in `ranking/sort_keys.py` is an oracle. Its job is to catch a comparator that is not a strict total order. Its failure arms must therefore be shown to fire, or the oracle is not shown to detect anything. Coverage pragmas on those arms would exclude them permanently and leave the detector unverified. The oracle checks injectivity, irreflexivity, asymmetry, trichotomy and transitivity in that order, the last only for at most 40 keys because it is O(n^3). Each check gets an input that fails it and passes every check before it. A duplicated tuple key fails injectivity. A single `_ReflexivelyLess` key is less than itself. Two `_MutuallyLess` keys are each less than the other. Two separately constructed NaNs in real tuple keys hash differently, so injectivity passes, and neither is less than the other, so trichotomy fails. Three `_CyclicKey` keys form a rock-paper-scissors cycle that passes the pairwise checks and fails only transitivity. The specimen classes are the inputs under test, not stand-ins for collaborators.

### Subnormal flip radius

Source: `tests/test_margins_and_flip_radii.py`.

`Margin.flip_radius` halves the margin. For a margin of at least 2^-1021 the half is still a normal number, so halving only decrements the exponent and `2 * flip_radius` recovers the margin bit for bit. Below 2^-1021 the half is subnormal, there is no exponent left to decrement, and halving rounds to nearest even in the significand.

Every such margin is an integer multiple m of 5e-324 (2^-1074). For odd m the half lies exactly between two representable values and rounds to the even neighbour; 5e-324 itself halves to zero. Even multiples halve exactly and round-trip. The boundary of the exactness claim is therefore the parity of the multiple, not only the smallest subnormal.

Halving rounds down when m is congruent to 1 mod 4 and up when m is congruent to 3 mod 4, so a subnormal flip radius can exceed the true radius by 2^-1075, half the smallest subnormal. Reaching this case requires two adjacent scores less than 2^-1021 apart, which is far below the floor of the tau band used elsewhere in the analysis.

### Subnormal square band

Source: `tests/test_cosine_edge_cases.py`.

Section 4.2 is an algebraic statement about Euclidean norms. Below `NORM_UNDERFLOW_THRESHOLD` (`sqrt(DBL_MIN)`, about 1.49e-154) `l2_norm` is not a Euclidean norm: each coordinate's square is subnormal and only a handful of bits survive the squaring (G18). At `tf = {1.5e-161}` with `delta_tf = 0` the inequality is an equality over the reals while the computed sides differ by 0.5%, six significant bits surviving the squaring (`docs/spec_addenda.md#g27`). That is a property of the norm and is pinned by `test_cosine_degrades_for_vectors_whose_squares_underflow`, so the `normal_sparse_map` strategy draws coordinates that are either exactly zero or at least the threshold. Subtraction re-enters the band, because two coordinates above the threshold can differ by far less than it, and the realised difference `w' - w` is smaller than `tf` by roughly the idf perturbation; `assume()` calls in `test_three_term_bound_is_never_violated` therefore also require each of the four norms (`tf`, `tf'`, `delta_tf` and `w' - w`) to be zero or at least the threshold.

### Synthetic movielens archive

Source: `tests/test_datasets.py`.

The MovieLens tests build a synthetic archive in memory, because the GroupLens licence forbids committing the real one, and a test that required the real archive would skip everywhere, including CI. The archive reproduces the quirks of ``ml-latest-small`` that the parser handles: a UTF-8 BOM on every CSV, which corrupts the first column's name while leaving its value intact; pipe-separated genres and parenthesised release years; films with no ratings at all, which upstream contains; half-star ratings, which make G8's exact ``(2*sum, count)`` integer pair available; and ``movieId`` values whose string and integer collation disagree (the file lists 3 and 9 after 10, and string order puts "10" before "9").

### Tfsx container

Source: `tests/test_persistence_roundtrip.py`.

The tests require a model loaded from a .tfsx file to be bit-identical to the model that was saved, and two saves of one model to give identical bytes, because everything downstream compares floats bitwise. See [Tfsx container format](#tfsx-container-format).

### Tfsx field validation

Source: `tests/test_persistence_roundtrip.py`.

`load_model` reads untrusted input, so the tests check, on a 301-byte three-token container, that it validates each of the ten fields `model_bytes` writes and treats sibling fields alike. See [Tfsx token order](#tfsx-token-order), [Tfsx idf sign](#tfsx-idf-sign) and [Tfsx integer arrays](#tfsx-integer-arrays).

### Two canonical renderings

Source: `tests/test_canonical_bytes.py`.

There are two paths from a payload to bytes and they are not the same path. `canonical_json` sanitises non-finite floats to `null` before rendering, so what lands on disk is strict JSON; RFC 8259 has no token for NaN or Infinity. `hash_json` calls `json.dumps` directly, and Python's encoder emits the non-standard `NaN` and `Infinity` tokens by default.

For every finite payload the two agree, so `hash_text(canonical_json(payload, indent=None)) == hash_json(payload)`. For a payload holding an undefined margin, which G16 requires be reported as undefined rather than coerced, they disagree: the digest of the file differs from the digest of the payload written into it, and neither function raises.

`RunManifest.digest` therefore renders with `canonical_json`, the same path `write` uses, so a manifest holding an undefined value verifies against its own file. An outside verifier parses the file, removes `manifest_digest`, reproduces the stripping rule (`strip_volatile` with `_MACHINE_KEYS`, then drop `notes`) and recomputes the digest over the compact `canonical_json` rendering (`indent=None`) of what remains.

`json.loads` accepts the `NaN` extension by default, so the divergence is visible only to a strict parser. The tests use a `parse_constant` hook that refuses the tokens. A round trip through JSONL is lossy at exactly this point: an undefined margin goes out as `null` and comes back as `None`, so a reader must check `is None` rather than compare against NaN.

### Validation contracts

Source: `tests/test_validation_contracts.py`.

The typed exceptions and validators in `utils/validation.py` are imported by many test files, and `tests/test_validation_contracts.py` is the file that owns them. It tests three properties.

The hierarchy is the contract. Callers catch `TfidfStabilityError` to mean a condition the package detected and reported, and catch `NativeBackendUnavailableError` to mean that the reference backend should be used instead. An error class defined outside that tree escapes both handlers. The tree is therefore checked by walking `__all__`, not by listing known classes, so a new class is covered as soon as it is exported. `AbiVersionMismatchError` is a subclass of `NativeBackendUnavailableError`, so fallback handlers also catch a stale extension.

The validators divide the work between them. `-0.0` and NaN pass `check_non_negative`, and NaN fails `check_finite`. `check_non_negative` owns the sign; `check_finite` owns representability. Both exist because each covers a case the other does not, and a caller that needs both properties calls both. The tests state this contrastively so that the guard that rejects NaN is clear.

Guard order is observable. `resolve_k` has three checks, and a caller sees only the first one that fires. Which error a bad `k` produces is part of the interface, so each check is reached at an input that the others do not claim. The positivity check runs before the range check, so `k = 0` on an empty corpus reports "k must be positive".

The callers are not retested in this file. `boundary_margin` and `rank_top_k` handle their own `k`; `resolve_k` is the subject here.

## Mutation tooling

### Cpp mutation score scope

Source: `tooling/mutation_cpp.py`.

The verdict for each C++ mutant is ctest. `cpp/tests/CMakeLists.txt` registers the whole `tfidf_tests` doctest binary as a single CTest test, `cpp.unit`, so the verdict is the native suite alone. The bit-for-bit agreement between the C++ and the normative Python is asserted in pytest, in `tests/test_native_differential.py`, `tests/test_native_distances_differential.py` and `tests/test_native_ranking_differential.py`. Those tests need the extension rebuilt for each mutant, which takes minutes rather than seconds, so they are not part of the score. A survivor therefore means that the native suite cannot observe the mutation, not that nothing can. A mutation in a function the bindings export may still be caught by the differential tests. A mutation in a function the bindings do not export has no other coverage: `is_exact_tie`, `adjacent_gaps` and `partition_is_valid` appear nowhere in `cpp/bindings/module.cpp`, so the native suite is their only coverage in either language.

The tool has no C++ parser, so mutation is lexical. `scan` tracks whether it is inside a line comment, a block comment, a string literal or a character literal, and skips preprocessor lines together with their backslash continuations, so no candidate falls in text the compiler does not read as code. Every candidate is a single-token replacement. `<=` and `<`, `>=` and `>`, `==` and `!=`, `&&` and `||`, `+` and `-`, and `*` and `/` replace each other; `true` and `false` replace each other; and the integer literals `0`, `1` and `2` become `1`, `0` and `3`. Operators are matched longest first, so `<=` is never read as `<`. The tokens `->`, `<<` and `>>` and compound assignments such as `+=` are skipped, because splitting them is a syntax error rather than a mutation. A literal is replaced only when it stands alone, so the digits in `1.0` or `x1` are left unchanged.

A mutant that fails to compile is stillborn and counted neither way, because a compile failure is a limitation of lexical mutation rather than evidence about the tests. The score is killed / (killed + survived) as a percentage, and `score` is `None` when nothing was scored. A mutant that compiles and passes ctest is a survivor. Every survivor must be argued in `configs/equivalent_mutants_cpp.txt` or the run exits non-zero, and over a complete run a claim in that file that matches no survivor also exits non-zero. The build directory must already be configured and its unmutated baseline must pass: `killed` means only that ctest returned non-zero, so a tree whose tests already fail would score every mutant as killed. `verify_baseline` refuses to start a campaign on such a tree.

### Ctest timeout bounds

Source: `tooling/mutation_cpp.py`.

A mutant can make the native suite loop forever, for example by removing the increment that ends a merge or by relaxing a bound a loop counts against. A suite that does not terminate has not passed, so a timeout is a kill. `cpp/tests/CMakeLists.txt` sets a `TIMEOUT` property of 300 s on `cpp.unit`. CMake documents `--timeout` as applying only to tests that carry no `TIMEOUT` property, so the inner bound is 300 s whatever `--test-timeout` is passed. The flag still sets the outer backstop, and it bounds the tests of a tree that does not set the property.

The Python timeout on the ctest subprocess is `max(test_timeout, 300) * 2` seconds. That is 600 s both for the default `--test-timeout` of 180 and for the 60 that the nightly C++ mutation job passes. The backstop is kept above 300 s so that ctest, not Python, kills the test binary. When the Python timeout fires it kills the ctest process only, which can leave the test binary running as an orphan that holds a lock on the executable the next mutant's build needs to relink. When ctest does not return within the backstop the mutant is still counted as killed. The build has its own limit of 1800 s. A build that fails or times out is stillborn rather than killed, because nothing is learned about the tests, and counting it as a kill would inflate the score with infrastructure failures.

### Mutation operator set

Source: `tooling/mutation_python.py`.

The operator set is narrow rather than generic. It targets the places where the package's results turn: comparison boundaries (`tie_cliques` extends a clique while a score difference is `<= tau`, and `tie_chains` breaks a chain where a gap is `> tau`), constants that are exactly 0, 1 or 2, and the direction of a subtraction asserted bit-for-bit. A generic tool such as mutmut spends most of its budget on string and container mutations that nothing in this package depends on.

The operators are comparison flips (`<` and `<=`, `>` and `>=`, `==` and `!=`), arithmetic swaps (`+` and `-`, `*` and `/`), `and` and `or` exchanged, a boolean constant negated, and a fixed set of numeric constants: the integers 0, 1 and 2 become 1, 0 and 1, and the floats 0.0, 1.0 and 0.5 become 1.0, 0.0 and 1.0. Other constants, such as a digest length or a display width, are not load-bearing, and mutating them produces survivors that mean nothing.

The narrow set also bounds run time. The cost of a campaign is the suite's run time multiplied by the number of mutable sites, so each nightly job mutates one module against its own test files and `tests/test_public_api_surface.py`, which keeps the job within its 90-minute limit ([Nightly python mutation scope](#nightly-python-mutation-scope)).

Constants are keyed by `(type, value)` because `0 == 0.0` and the two hash alike, so a dict keyed by value alone collapses the int and float entries into one. Boolean constants are negated before that lookup is reached, since `True == 1` and `False == 0` as well.

### Mutation sandbox copy

Source: `tooling/mutation_python.py`.

The Python mutation campaign mutates a temporary copy of the checkout, never the working tree. `_make_sandbox` copies the paths in `_SANDBOX_CONTENTS` into a directory created by `tempfile.mkdtemp`, and pytest runs with that directory as its working directory. `tests/conftest.py` computes `src/` from its own location, as `parents[1] / "src"`, so a tree containing `src/` and `tests/` is self-contained: the copy's conftest names the copy's `src/`. Conftest inserts that path only when `find_spec("tfidf_stability")` is None, and the sandbox's `sitecustomize.py` makes the copy win when an editable install is present (see [Sandbox sitecustomize](#sandbox-sitecustomize)).

Editing the module in place and restoring it from a backup in a `finally` block is unsafe. `finally` does not run when the process is killed, and on Windows `Popen.terminate()` calls `TerminateProcess`, so neither a SIGTERM handler nor an `atexit` hook runs either. A killed in-place campaign would leave a mutated module in the working tree, where it could be committed. With the copy, a killed run leaves a temporary directory behind and nothing else; a run that exits normally removes the directory in a `finally`.

### Sandbox contents

Source: `tooling/mutation_python.py`.

The sandbox must be a working checkout, not only the package and the suite. Every top-level path that a test, or a gate script a test runs, reads from the repository must be in `_SANDBOX_CONTENTS`, or the baseline run in the sandbox fails before any mutant is applied.

`data/` is required because `preprocessing/stopwords.py` resolves the stopword list as `parents[1] / "data" / "assets"` in a wheel and otherwise as `parents[3] / "data" / "assets"`, the repository root. The sandbox has the in-tree layout, so without `data/` the fixture that builds the normative pipeline fails. `cpp/` is required because `tests/test_public_api_surface.py` checks the aliases in `types.py` against `cpp/include/tfidf/core/types.hpp`. That test file also asserts package-wide contracts, frozen dataclasses and enum serialisation among them, that no single module's test file covers, which is why every nightly Python mutation job runs it beside the module's own tests. `scripts/` is required because `tests/test_benchmark_smoke.py` loads `scripts/benchmark.py` as a module and runs it, and because other tests load the gate scripts. `docs/`, `examples/` and `CITATION.cff` are required by two test files that read repository metadata rather than the package: `tests/test_doc_references.py` runs the link check in `scripts/check_docs.py` over `docs/`, and `docs/index.md` links into `examples/`; `tests/test_repository_gates.py` runs `scripts/check_versions.py`, which compares the version in `CITATION.cff`. `README.md`, the specification that the docs and docstrings cite, is copied with them. `tooling/` is required because the suite imports `tooling.allowlist` and the gate scripts import `tooling.gate`. The remaining entries carry their reasons beside them in `_SANDBOX_CONTENTS`. Among them, `pyproject.toml` holds the pytest configuration, which declares the markers under `--strict-markers` and turns warnings into errors.

The compiled extension lives in `src/tfidf_stability/_native/` and is copied with the rest of `src/`: the copy ignores only `__pycache__`, `*.pyc` and `.hypothesis`. Without the extension, `native_available()` is false in the sandbox and every test marked `native` or `differential` skips under its `skipif`, so those tests do not run against any mutant.

`tests/test_mutation_gate.py` keeps the list complete. It scans every `.py` file in `tests/`, `scripts/` and `tooling/` for a top-level path written as `REPO / "<name>"` and asserts that `_SANDBOX_CONTENTS` holds each one. The check over-approximates, so a path that is only written to, such as `.mutation-cpp-interrupted.json`, is listed too; `_make_sandbox` skips a listed path that does not exist.

### Sandbox sitecustomize

Source: `tooling/mutation_python.py`.

The project is installed editable for development, and scikit-build-core's `.pth` file has two lines. The first, `import _editable_skbc_tfidf_stability`, inserts a `ScikitBuildRedirectingFinder` at the front of `sys.meta_path`. The second adds the real `src/` directory as a plain `sys.path` entry. Both point at the working tree, and either alone makes every mutant survive. `tests/conftest.py` adds the sandbox's `src/` only `if find_spec("tfidf_stability") is None`; with either effect in place that spec is found, conftest inserts nothing, every test imports the working tree and the mutated copy is never executed. In that state no mutant can be killed. Every mutant is reported as a survivor, which is an ordinary outcome, so the output does not show that the tests never ran the mutated code, and a module scores 0 killed even for mutants that its own test files fail on when run directly. Removing only the finder still leaves the working tree on the path through the second line.

The nightly mutation job installs only `requirements-dev.txt`, which does not install the package, so there `find_spec` returns None and the sandbox's own conftest inserts `parents[1] / "src"`, the sandbox's `src/`. The competing path exists on a checkout with an editable install, such as a developer's, and in the CI jobs that run `pip install -e .` before the suite, such as the `native` job in `ci.yml`, which then runs `pytest tests/` over the whole suite.

`sitecustomize` is imported during interpreter startup, after the `.pth` files are processed and before pytest and conftest, so it is the one place early enough to undo both lines. `_run_tests` sets `PYTHONPATH` to the sandbox so that the sandbox's `sitecustomize.py` is found at startup. That module drops from `sys.meta_path` every finder whose class name contains `Redirecting` or whose module name contains `editable`, removes the real `src/` from `sys.path`, and prepends the sandbox's own `src/`, so the campaign does not depend on what conftest's guard decides. Paths are compared through `os.path.normcase(os.path.abspath(p))`, because the `.pth` entry and the generated string come from different tools and can differ in drive case or separator on Windows while naming one directory. The real tree is removed rather than placed behind the sandbox, so a later change to the order of `sys.path` cannot put it ahead. Where nothing installed the package, the two removals are no-ops and only the prepend has an effect. `tests/test_mutation_gate.py` checks the result with two tests. The slow-marked test adds a module that exists only in the sandbox and asserts that a test run there can import it, so a run that imports the working tree fails on `ImportError`. Every module a campaign mutates exists in both trees, so reachability alone does not show precedence, and the PR-tier test is the stronger of the two: it replaces a real module in a sandbox with one that raises on import and asserts that a test run there fails, which holds only if the sandbox copy is imported ahead of the working tree.

## Scripts and gates

### Audit trials not forwarded

Source: `scripts/run_stability_profile.py`.

The certificate audit runs at `_AUDIT_TRIALS` (20) rather than at `--trials`, and the parameters block records the audit count under its own key, `audit_trials`. The transition curve takes `--trials` (default 40). The two sample independently, so a single `trials` key cannot describe both counts. The audit and the transition curve both exclude the exact-tie queries, which leaves 33 of the 40 queries in the published run. Multiplying `trials` by those 33 queries gives the 1320 trials behind each transition point; multiplying `audit_trials` by them gives the 660 cases in the 2x2 certificate table. At 40 audit trials the audit gives 172 certified perturbations and 51.39% conservatism, against the 81 and 51.12% published in `reports/summary.md` and `docs/ranking_stability.md`. Forwarding `--trials` to the audit is a defensible change, but it republishes those figures, so the script records the count instead of forwarding it.

### Binding abi checked separately

Source: `scripts/check_versions.py`.

The binding ABI is a version claim independent of the release version ([Abi version check](#abi-version-check) gives the reason), so the two are compared on separate axes. `cpp/bindings/module.cpp` declares `kAbi` and `src/tfidf_stability/_native/__init__.py` declares `REQUIRED_ABI`. `check_abi` in `scripts/check_versions.py` compares the two literals textually rather than importing the built extension, so the check holds on a checkout with nothing built, which is the state of the reference CI legs. When the two disagree, the loader rejects the extension, every process runs on the reference backend, and the differential suite skips rather than fails, so a run can stay green while exercising none of the native code. The check also requires the line `m.attr("__abi__") = kAbi;` in `module.cpp`, because without it the two literals can agree while the extension exports no ABI.

### Case study pair search

Source: `scripts/run_tie_break_ablations.py`.

Section 7.4 of the README describes two documents "identified such that |s_A - s_B| <= tau", and [G22](spec_addenda.md#g22) requires "identified" to be read literally, because a fine near-tie cannot be constructed by editing a document ([Near tie floor](#near-tie-floor)). `_case_study` therefore searches the ranked scores for the closest pair with `find_near_ties`, and reports whatever separation the corpus offers. The report carries the closest strictly positive gap, the tightest gap overall and whether it is an exact tie, and both documents' tie-break attributes. At `s_A == s_B` the attributes are the whole explanation of the outcome.

### Categorical palette validation

Source: `scripts/make_figures.py`.

`_SERIES` is the categorical ramp in the palette's fixed order and the one place a series colour is defined. `_PRIMARY` and `_LANDMARK` alias into it, so no hex is written twice.

The ramp is validated as a set, measuring each adjacent pair as Euclidean distance in OKLab scaled by 100. Under protanopia and deuteranopia, simulated with the Machado (2009) matrices at full severity, the worst adjacent pair is 9.1 against a target of at least 8. With normal vision it is 19.6 against a floor of 15. The same measure gives 5.8 under the Machado tritanopia simulation, so the 9.1 covers the red-green deficiencies only.

Against the white figure surface, three slots sit below 3:1 contrast (`#1baf7a` at 2.82:1, `#e87ba4` at 2.69:1 and `#eda100` at 2.17:1). The set is not separable in greyscale either: by luminance contrast the closest pair is 1.05:1 and the widest 2.04:1. Colour is therefore never the only encoding. Every categorical series carries a direct label or a legend entry, and that is what a low-contrast, colour-blind or greyscale reader depends on.

### Clang format subset check

Source: `scripts/check_cpp_format.py`.

`scripts/check_cpp_format.py` enforces the subset of `.clang-format` that can be checked without clang-format installed: ``ColumnLimit`` (no line wider than the limit), ``UseTab: Never`` (no tab character in any line) and the trailing-whitespace removal that clang-format always applies. It scans the `.hpp`, `.cpp`, `.h` and `.cc` files under `cpp/`. The script does not invoke clang-format. clang-format is not installed on every machine that builds the project, and a gate that skips silently when its tool is missing provides no guarantee about the tree. The checked values are read from `.clang-format` at run time rather than duplicated as constants, so an edit to the style file changes the check with it. A style file with no ``ColumnLimit`` makes the gate exit with an error rather than pass. Vendored code under `cpp/third_party` is excluded because its formatting belongs to upstream; `scripts/check_vendored.py` verifies each vendored file against the SHA-256 digest recorded in its directory's `MANIFEST.sha256`. The success message prints the number of files scanned, so a run that matched no sources is visible rather than passing silently.

### Cpp layout check is one directional

Source: `scripts/check_layout.py`.

`scripts/check_layout.py` asserts that every directory under `cpp/include/tfidf/` has a same-named subpackage under `src/tfidf_stability/`. The converse is not asserted. Python subpackages without a C++ mirror are expected, because only the hot paths are ported to C++ and the remaining subpackages are orchestration. Python directories whose names start with `.` or `_`, such as `_native`, are not counted. The `cpp/include/tfidf/core/` directory is exempt from the check. It holds the reduction policy, the floating-point guard, the build configuration and the scalar types, all of which the Python reference backend inherits from the interpreter, so no Python counterpart exists. The set of Python-only subpackages is computed and printed at run time rather than listed in the docstring, so the description cannot drift from the tree.

### Digest query coverage

Source: `scripts/snapshot.py`.

The digest queries are raw text and are run through `pipeline.preprocess` before `TfidfVectoriser.transform_query`. `transform_query` takes features: the stemmed, n-grammed tokens the vocabulary is built from. A raw token such as `cosine` does not match the vocabulary entry `cosin`. Handed over unprocessed, `q2` and `q3` embed to the zero vector; only `q1`, whose words stemming leaves unchanged, would still score. For a zero-vector query the `scores` and `margins` stages hash zeros, and the `rankings` stage hashes an order set by the document attributes alone. Such a query still yields a platform-stable digest, so stage coverage cannot be read from digest stability; it has to be set by the queries themselves. With preprocessing applied, `q1` (quick brown fox), `q2` (numerical stability sparse) and `q3` (cosine similarity vectors) all hit the vocabulary and produce non-zero score vectors, so those three stages exercise the scoring, ranking and margin code on real inputs. `q4` (the of and) is all stopwords and is the single intended zero-vector case: it exercises the ranking operators ordering an all-zero score vector by document attributes alone.

### Docs check imports without install

Source: `scripts/check_docs.py`.

`scripts/check_docs.py` puts `src/` on `sys.path` itself, as `tests/conftest.py` does for the test suite when the package is not installed, so the check runs with no install and no compiler. The docs CI job installs nothing, not even `requirements-dev.txt`. Depending on `pip install -e .` would build the package through scikit-build-core and nanobind, and so bring a CMake toolchain into a job whose inputs are Markdown and Python source. Resolving a docstring cross-reference needs the package importable (see [Docstring targets resolve by import](#docstring-targets-resolve-by-import)), so the script adds the path rather than asking the job to install the package. Modules under `_snowball` are excluded from the import pass.

### Docs extra outside dependency check

Source: `scripts/check_dependencies.py`.

`_DEV_GROUPS` in `scripts/check_dependencies.py` lists the extras a development install carries: `dev`, `viz` and `nb`. These extras, together with the runtime dependencies and the build-system requires, are compared against `requirements-dev.txt`. The `docs` extra is excluded because no CI job builds the documentation. Requiring `mkdocs-material` and `mkdocstrings` in a development install would add an install cost for every contributor with no job that consumes it. A change that adds a documentation job adds `docs` to `_DEV_GROUPS` in the same commit, so the two declarations stay in step from the moment the job exists.

### Docstring targets resolve by import

Source: `scripts/check_docs.py`.

`scripts/check_docs.py` resolves every `:func:`, `:class:`, `:meth:`, `:attr:`, `:data:`, `:mod:` and `:exc:` target found in a source docstring by importing the package and walking attributes, not by parsing the source tree. A cross-reference names a runtime object: `Reduction.EXACT` is an enum member and `TauBand.g_min` is a dataclass field, and neither is visible to a purely syntactic check. A target of the form `tests.<module>.<name>` is the one exception: it is looked up by parsing `tests/<module>.py`, since importing a test module would load fixtures and plugins. For every other target the lookup tries, in order, the class the docstring is attached to, the enclosing module, the builtins, an importable dotted path (which covers the standard library and fully qualified package names), and finally any public name exported by any module in the package, because docstrings name a sibling module's class without importing it. Sphinx would report a dangling target, but no job in this repository runs Sphinx, so this script carries that check.

### Figure falsification map

Source: `scripts/make_figures.py`.

Every figure is drawn from recorded experiment JSON, never from a live computation, and its caption carries the result digest, so a plot traces back to the run that produced it. A figure that recomputed its own data would drift from the numbers in the text whenever a default changed.

`fig_transition`: flip rate against `eps / (m_k / 2)`. A false A1 shows as a curve rising before 1.0. Each point is 1320 trials, so one flip is 0.076%. On a linear axis from 0 to 100% in the 800 px tall figure, one flip moves the marker by less than one pixel. The y axis is therefore symlog, linear up to one flip, and the caption carries the recorded violation count. That count, rather than the curve's shape, is what falsifies section 4.4.

`fig_margins`: the distribution of `m_k` at each `k`, log-scaled, with the exact-tie share annotated. The exact-tie mass is the finding.

`fig_tau_band`: the admissible band for `tau`, from twice the arithmetic noise floor to the smallest non-zero gap, on a log axis. An empty band means no `tau` separates numerical error from tie structure, which is a finding about the corpus.

`fig_rho_discontinuity`: `rho(tau)` as a step function, so interpolation does not assert values the function never takes. `rho = 1` means the largest chain and the largest clique are the same size; any step above it is single-linkage chaining. Every riser sits at the exact `tau` where `rho` jumps, because the sweep evaluates the adjacent gaps and the clique thresholds together rather than a grid.

`fig_ablation`: disagreement rate by operator pair and `k`. The three operators consume one shared score array, so a non-zero bar can only come from the tie-break.

`fig_rank_cascade`: the rank of each document against perturbation size, with the documents that cross the top-k boundary picked out. Section 4.4 certifies set membership below `eps = m_k/2`, so no line may cross the k boundary left of 1.0. Crossings wholly inside the top-k are permitted there, because the certificate does not cover the order within the set.

`fig_stratified`: the same disagreement conditioned on the margin. A non-zero rate right of the exact-tie band contradicts A2. The empty bands either side of `tau` show that the choice of `tau` within them cannot move a result.

### Intermediates export source

Source: `scripts/export_intermediates.py`.

`scripts/export_intermediates.py` builds each term record from `TfidfModel.intermediates` rather than by zipping the CSR row, because the row carries only the weight, and it keeps its own key names. See [Export intermediates tf](#export-intermediates-tf).

### Linters pinned exactly

Source: `scripts/check_dependencies.py`, `tests/test_repository_gates.py`.

`_EXACT` in `scripts/check_dependencies.py` names ruff and mypy. Other packages carry a floor (`>=`), which is a constraint on the resolver. A floor on a linter is instead a promise that every future release of that linter agrees with the current one. Linter releases add and change rules, so that promise does not hold. A floor such as `ruff>=0.6` accepts a release that stabilises a rule the tree has not been checked against: ruff 0.16 stabilises PLR0917, which the `PL` selection in `pyproject.toml` enables, so under a floor the lint job fails on a commit that changes no Python. Both `pyproject.toml` and `requirements-dev.txt` therefore pin these packages with `==`. The CI jobs install `requirements-dev.txt`, and the `dev` extra in `pyproject.toml` states the same exact pins, so a contributor installing `.[dev]` gets the linters CI runs, and a clean tree reports the same findings locally as in CI.

Two checks hold the pins. `scripts/check_dependencies.py` requires each package in `_EXACT` to be named in both files with an `==` specifier, and its general comparison requires the two specifiers for any package to be equal. `test_a_dependency_named_in_both_requirement_files_agrees` in `tests/test_repository_gates.py` requires ruff and mypy to be named in both files and compares the specifiers of every package the two share. Both compare specifiers rather than resolved versions, so they need no network and fail on the disagreement itself.

### Materialise corpus to file

Source: `scripts/build_corpus.py`.

`scripts/build_corpus.py` materialises a registered dataset as JSONL, so that an experiment can read the file through `load_dataset("jsonl:<path>")` rather than regenerate or re-parse its source on every run. The written file is the reproducibility boundary. A regenerated corpus is only as reproducible as its generator and the interpreter it runs on; `datasets/synthetic.py` holds its own output stable by drawing only from the Mersenne Twister core with integer selection logic, and `load_dataset` regenerates the registered synthetic datasets from their spec on that basis ([Synthetic prng](#synthetic-prng)). Writing the corpus once moves the boundary from "same generator and interpreter" to "same bytes", which also covers a generator configured outside those restrictions, such as a non-integer Zipf exponent. Materialising also gives the `build-corpus` command a JSONL corpus to read, and avoids re-parsing the MovieLens archive on every experiment run. The script keeps the data boundary explicit: upstream handles zip archives, CSV parsing and seeded generators, and downstream sees only `{doc_id, text, ...}` records. The interactions go to a separate `.interactions.jsonl` file. Provenance is written to a separate `.provenance.json` sidecar so the corpus digest covers the documents alone, and two corpora with identical content agree even when produced differently.

### Movielens digest pin

Source: `scripts/fetch_data.py`.

`scripts/fetch_data.py` is the only code in the repository that opens a network connection. It lives outside the `tfidf_stability` package, so the question of whether the library makes network requests is answered by reading one short file, and the library remains usable on an air-gapped machine once the archive is present.

GroupLens replaces `ml-latest-small.zip` in place at the same URL, so the script verifies the archive against a digest pinned in the repository, `movielens.MOVIELENS_SHA256` in `src/tfidf_stability/datasets/movielens.py`. When the pin is `None`, the script prints the archive's SHA-256 and the assignment to paste into that file. Once a pin is set, every run computes the digest of the archive and exits with status 1 on a mismatch. The pin guarantees that the corpus underneath a published number is the same bytes the experiments ran against. A mismatch means upstream has changed. The two valid responses are to obtain the pinned archive, or to re-run the experiments and update the pin in the same commit. The download is streamed to a `.partial` file and verified before it is renamed into place, so a mismatched download never replaces a pinned archive already at the destination. The rejected bytes are kept beside it with a `.rejected` suffix for inspection. A verified archive is then parsed, so a corrupt archive fails at fetch time rather than in the middle of an experiment.

The script writes the archive to `data/raw/`, which is gitignored, and writes elsewhere only when `--output` is given. The MovieLens licence prohibits redistribution of the data, so the archive must never enter the git history. The gitignore and the fixed default destination keep a broad `git add -A` from staging it.

### Operator score array identity

Source: `scripts/run_tie_break_ablations.py`.

`rank_all_operators` computes `sorted_scores_desc` once and hands the same tuple to every operator, so the three `sorted_scores` attributes are one object. The harness in `scripts/run_tie_break_ablations.py` checks this with `is` rather than with an elementwise `same_bits` comparison against `rankings["pi"].sorted_scores`. While the sharing holds, an elementwise comparison compares the object with itself, and `same_bits(x, x)` is true for every double including NaN, so that predicate is a tautology and the abort it guards is unreachable. The elementwise comparison also passes when the sharing is removed: each operator then recomputes `sorted_scores_desc` from the same score vector and produces three distinct, bit-identical tuples. Identity is the property that makes a disagreement attributable to the tie-break, because two equal arrays computed separately leave a second explanation for any difference. `test_every_operator_sees_one_and_the_same_score_array` in `tests/test_ranking_tie_breaks.py` asserts the same property with `id()`, and the harness applies the standard the unit test sets.

### Pair geometry projection

Source: `scripts/run_tie_break_ablations.py`.

A query ranks A above B if and only if `q . d > 0` for `d = w_A/||w_A|| - w_B/||w_B||`. `d` lies in the plane spanned by the two document directions, so the out-of-plane component of q is orthogonal to `d` and contributes zero to the decision statistic. Projecting q onto that plane therefore preserves the ranking decision exactly, unlike a t-SNE or PCA scatter, where apparent distance is an artefact of the embedding. The projection loses part of the magnitude of q, so angles measured from the drawn q to each document are not the true angles. `_pair_geometry` therefore takes the scores from the recorded pair and reports the share of `||q||` that lies in the plane. The output is presentation only, so `math.fsum` is used throughout, and the quoted scores come from the run's own reduction policy. The projection error is reported relative to `||q||` rather than to the decision statistic, because the statistic is 0.0 for an exact tie and a relative error against it is undefined; `q . d` divided by `||q||` is the score gap `s_A - s_B`. `angle_is_renderable` records whether the angle between the two document directions exceeds 0.05 degrees (about 8.7e-4 rad), below which a drawing of the pair shows no separation. For the recorded pair, an exact tie, the two directions are 1.5e-8 rad apart and the flag is false.

### Pipeline digest check

Source: `scripts/snapshot.py`.

`determinism.yml` computes the pipeline digest on every leg of its two matrices (operating systems by interpreters, and operating systems by native build types) and fails when any two legs disagree. That proves the digest is the same on every platform; it does not prove the digest is the published value. A change that moves every platform identically, such as an edited stopword list that alters every df, idf and score the same way on all of them, passes the cross-leg comparison. The test suite compares runs against each other rather than against a recorded value, so it passes there too. `scripts/snapshot.py --check` closes that gap. It compares each stage's digest against `configs/pipeline_digest.txt`, the record of which code produced the published numbers, and a change to that file belongs in the same commit as the code change that moves the digest. The comparison is reported per stage because the stage that moves says what changed. `idf` alone points at the logarithm: per [G13](spec_addenda.md#g13), the platform logarithm differs from the correctly rounded value in about 15% of IDF entries, and the cross-platform digest comparison is the test of the correctly rounded replacement. `preprocessing` points at the tokeniser, the n-gram map or the stopword list, a change first seen at `weights` points at the vectoriser, and one first seen at `norms` or later points at the reduction policy. A stage present on one side only is also reported: a computed stage with no recorded digest is unchecked, and a recorded stage that is not computed checks nothing. An empty record file is reported as a problem rather than passing vacuously.

### Python floor call arg check

Source: `scripts/check_python_floor.py`.

`pyproject.toml` sets ``requires-python = ">=3.11"``, and the reference CI legs run 3.11, 3.12 and 3.13. A contributor on a newer interpreter can write a call that their Python accepts and 3.11 rejects. Such a call passes every local check and fails on each supported interpreter that rejects it. For example, ``Path.write_text`` accepts a ``newline`` argument from 3.10, while ``Path.read_text`` accepts it only from 3.13, so ``read_text(encoding=..., newline="")`` runs on 3.14 and raises ``TypeError`` on 3.11 and 3.12.

mypy models each stdlib signature per version through typeshed, and the project's main mypy run executes at the floor over `src/tfidf_stability` and `tooling` only. Strict mypy over `tests/`, `scripts/` and `examples/` reports close to three hundred findings, mostly unused `type: ignore` comments, untyped definitions and calls into untyped code, which concern typing style rather than whether a call is possible. `scripts/check_python_floor.py` therefore runs mypy over those roots at the floor with `--ignore-missing-imports` and reports only the `call-arg` code: a call passing an argument that the signature at the floor does not accept. The floor is read from `pyproject.toml` rather than repeated in the script, so raising `requires-python` moves the check with it. The script exits non-zero when any such call exists, so it gates CI beside the other checks.

### Release tag joins version set

Source: `scripts/check_versions.py`.

On a tag build the tag is part of the version set. `release.yml` triggers on `v*` tags and passes ``--tag "${{ github.ref_name }}"`` to `scripts/check_versions.py`, which strips the `v` prefix and compares the tag against the four in-tree statements in `_SOURCES` (`pyproject.toml`, `src/tfidf_stability/__init__.py`, `CITATION.cff`, `CMakeLists.txt`). Without this comparison a tag such as `v0.3.0` on a tree declaring 0.2.0 builds, tests, gates and publishes 0.2.0 under a 0.3.0 release. `CITATION.cff` is in the set because a citation carrying a wrong version is a claim about which code produced a published number, and the project's results are defined as a function of the repository's contents. The `CMakeLists.txt` `VERSION` becomes `PROJECT_VERSION`, then `kVersion` in `build_config.hpp`, then the native module's `__version__` and `build_info()["version"]`. The latter is hashed into the digest of every RunManifest recorded with the native backend loaded. The script exits non-zero on any disagreement, so it can gate CI beside `scripts/check_docs.py`.

### Rho sweep breakpoints

Source: `scripts/make_figures.py`, `scripts/run_tie_break_ablations.py`.

`rho(tau) = |largest chain| / |largest clique|`, and the numerator and the denominator step at different values of `tau`. A chain is single-linkage, so its size changes only when `tau` crosses an adjacent gap between sorted scores. A clique is complete-linkage. With the scores sorted in decreasing order as `S`, the largest clique reaches size `m + 1` exactly when some window of `m + 1` consecutive sorted scores has diameter at most `tau`. That happens at `c_m = min over a of (S[a] - S[a + m])`, a span between scores `m` apart rather than between neighbours. A sweep over adjacent gaps alone locates the chain steps and misses most clique steps. On the recorded corpus only 18 of the 129 jumps in `rho` fall at an adjacent gap, so such a sweep locates about one jump in seven. A log grid laid over the gaps lands between breakpoints and records each jump at whichever sample comes next.

`_rho_sweep` evaluates `rho` at the union of the adjacent gaps and the `c_m` spans, which is every `tau` at which `rho` can change and nothing else, so the recorded curve is exact rather than sampled. On the recorded corpus that union is 215 values. In general it holds at most `2N - 2` values, against the `N(N-1)/2` distinct pairwise differences that enumerating every span evaluates for the same answer. Building the `c_m` costs `O(N^2)` comparisons and `O(N)` space. The chain and clique breakpoints are recorded separately in the output, so a step that sits on a clique span and not on a chain gap can be identified.

Every riser in the step plot therefore sits at the exact `tau` where `rho` jumps. The figure title counts risers off the curve rather than from the adjacent-gap list, because the adjacent-gap count measures how often the chain count changes, not how often `rho` does.

`rho = 1` means the largest chain and the largest clique are the same size, which does not by itself make the tie groups agree: 7 of the 24 sampled values at `rho = 1` have differing chain and clique counts. Any step above 1 is single-linkage chaining, and its height is how much that `tau` inflates the largest tie group.

The record's operative `tau` lies seven decades below the sweep's lower end, so the peak belongs to a regime none of the results occupy. Extending the axis to reach it would compress the informative decades into nothing, so the relation is stated in the stamp instead, using the `rho_below_range` value the sweep records.

### Tau band centre marker

Source: `scripts/make_figures.py`.

`display_tau` is the geometric centre of the admissible band `[2 eta, g_min)`, not the value any experiment ran with. `tie_break_ablations.json` uses its own `parameters.tau` of 4.8e-13, and the two differ by a factor of 13.

Nothing needs reconciling. The upper endpoint `g_min` is the smallest observed positive gap, so every observed positive gap is at least `g_min`. For any `tau` in the band, the relation `abs(s_i - s_j) <= tau` therefore holds only where the scores are equal, and the tie groups are the exact-equality classes whichever admissible `tau` is picked. The marker is labelled 'band centre', and the figure title states that consequence rather than printing `n_gaps_in_band`, which is 0 for any valid band and would read as a measurement.

The same fact empties the inner bands of `fig_stratified`, which span `(0, 100 tau]`. E0 measures `g_min` at 3.7e-08, against `100 tau` of 4.8e-11, a factor of about 770. In the recorded run every boundary margin is either exactly zero or above `100 tau`, which is why the position of `tau` inside the band cannot move a published number. An empty band is drawn as an explicit gap, never as a rate of zero, because 'no query was tested here' and 'every query tested here agreed' are opposite claims.

### Test vacuity scope

Source: `scripts/check_test_vacuity.py`.

A green suite is evidence only if the tests could have failed. Two run-level guards answer whether anything ran. `ctest --no-tests=error` fails an empty C++ suite, and `determinism.yml` counts executed tests, because every differential test is skipif-guarded and pytest exits 0 when all of them skip. `scripts/check_test_vacuity.py` answers the question one level down: whether what ran asserts anything. It reports an empty test, a test whose body cannot fail, an assertion on a literal or a non-empty tuple, a `parametrize` with no cases, a `@given` test without the `property` marker, a `pytest.raises` on a broad type or without `match=`, and a test whose every assertion sits inside a loop that a filter or an `if` can skip. Hand-written guards in individual tests, such as a count asserted after a loop, protect the one test they sit in; the script applies the same checks to every test. Assertion strength is out of scope: `assert x == approx(y, rel=1)` is nearly content-free, and no parser can tell. Mutation testing is the tool for that question. The script exits non-zero when any finding is reported, so it gates CI beside `scripts/check_layout.py` in the lint job.

### Undefined margins not degenerate queries

Source: `scripts/run_stability_profile.py`, `scripts/run_tie_break_ablations.py`.

The filter in `_margin_distributions` excludes undefined margins, not degenerate queries, and it counts the exclusions for both margin lists under `E1_excluded_undefined`. The boundary margin m_k is defined whenever k < n, which the `k >= n_documents` guard ensures, so its exclusion count is structurally zero. The interior margin m_min^top is the minimum over the adjacent gaps inside the top-k, an empty set at k = 1, so [G16](spec_addenda.md#g16) leaves it undefined there, and every query drops out of the interior distribution at k = 1. A feature-less profile never reaches the function: `evaluate` skips it and reports it as `n_degenerate_profiles`, in both experiments. A query whose features are all out of vocabulary does reach it, with an all-zero score vector and a boundary margin of exactly 0.0, which is defined and counted in the distribution. [G3](spec_addenda.md#g3) excludes a zero query vector from margin distributions; this filter does not apply that exclusion, and both published runs record `n_zero_vector_queries: 0`, so no published distribution contains one.

The tie-break ablations (E3/E4) keep zero-vector queries. G3 includes them in tie-break ablations: with every score zero, the ranking is the attribute order alone, so a zero-vector query is the most informative case for A2. The ablation script applies no zero-vector filter, and the `E3_degenerate_queries_included` field in its result payload records this.

### Unmatched raises exemptions

Source: `scripts/check_test_vacuity.py`.

`pytest.raises(X)` without `match=` asserts only the exception type, so two guards of the same class are interchangeable: swapping them, or reaching the second guard when the test targets the first, still passes. Most raise statements in the package use a handful of classes, `ValueError` and the project exceptions chief among them, so the message is the part that identifies the guard. The check exempts types that usually carry no message to match: `SystemExit`, `KeyboardInterrupt`, `StopIteration` and `GeneratorExit`. A `SystemExit` from argument parsing carries an exit code rather than prose, and the tests that expect it assert `.code` directly, which is stricter than any pattern; a `SystemExit` raised with a message can still take `match=`. A tuple of acceptable types is also exempt, because that form means any of several failure modes, which a single `match=` pattern cannot express.

## Build, CI and project configuration

### Ci bash default shell

Source: `.github/workflows/ci.yml`.

ci.yml sets `defaults.run.shell: bash` on every runner, including Windows, where the default is pwsh. GitHub runs bash with `set -eo pipefail`, so any failed command aborts the step. The pwsh wrapper sets `$ErrorActionPreference`, which applies to cmdlets only, and checks `$LASTEXITCODE` only at the end of the block. Under pwsh, a native command that fails partway through a multi-line block therefore does not fail the step. In the `native` job this would combine with the skipif-guarded native tests: a failed `pip install -e .` leaves the extension unimportable, every differential test skips, pytest exits 0, and the job reports success without having tested the native backend.

### Ci build isolation

Source: `.github/workflows/ci.yml`.

The `native` job installs the extension with `python -m pip install -e .`, which keeps pip's default build isolation. With isolation, pip builds in a clean environment that holds only the packages in `[build-system] requires`, which is the environment a user's install gets. `--no-build-isolation` resolves the build backend from the ambient environment instead. On windows-latest that fails with `BackendUnavailable: Cannot import 'scikit_build_core.build'`. On a Windows machine with scikit-build-core installed, the isolated install succeeds and the non-isolated one fails. A non-isolated build also tests a configuration that no user has. The step runs under bash, so a failure here stops the job before pytest runs. determinism.yml and nightly.yml install the extension the same way.

### Ci coverage floor

Source: `.github/workflows/ci.yml`.

The coverage job fails below 100% line and branch coverage (`--cov-fail-under=100`). The floor is never lowered to make a build pass. At 100%, every new line is either covered or carries a pragma that states why it cannot be reached, and the pragmas in src/ can be counted. The job runs with `-m "not slow"`, which deselects the stress tier. If a slow test were the only cover for a line, this job would fail, and a fast test is then needed. Stress tests widen the input domain and do not add coverage. The slow tests still run in the `reference` and `native` jobs, which run `pytest tests/` with no marker filter. `--cov-branch` is passed even though pyproject.toml sets `branch = true`, so branch coverage stays on if that setting is edited.

### Ci coverage native backend

Source: `.github/workflows/ci.yml`.

The coverage job installs the compiled extension before it measures coverage. Some Python code runs only when the native backend is present: the native comparison in `benchmarks/tfidf_perf.py`, the build-info block in `cli/commands.py`, and the branch of `persistence/manifest.py` that records a build. This code ships and runs for users, but it cannot execute on a machine without a compiler. A 100% figure measured without the extension would cover only the code reachable without it. The pure-Python branches remain covered because the tests reach them by monkeypatching `native_available`, not because the machine lacks a compiler. The step after the install runs `info --require-native`, so a failed build is reported as a build failure and not as a coverage shortfall in the Python code.

### Ci pytest report settings

Source: `.github/workflows/ci.yml`.

ci.yml sets PYTEST_ADDOPTS once for every pytest run in the workflow. The verbosity setting is an `-o` ini override, not a -v flag, because pytest computes verbosity as count(-v) minus count(-q). Every run in the workflow passes -q, so an added -v would cancel against it and make the output quieter. `verbosity_test_cases=1` names each test as it starts, so the last line of a truncated log names the test that hung. Without it, the log holds only progress dots, and the hung matrix cell can only be found by comparing job durations. `faulthandler_timeout=300` with `faulthandler_exit_on_timeout=1` dumps a stack trace for every thread after 300 seconds in one test and then ends the run. pytest includes this feature, so pytest-timeout is not needed and no dependency is added. `--durations=25` prints the 25 slowest tests, so an outlier is visible in the log and does not have to be worked out from the job list.

The cap applies per test and is well under every job's timeout-minutes (the shortest is 10 minutes), so it fires first and leaves a diagnosis instead of a cancelled job. Every job also carries timeout-minutes, because an untimed hang becomes a run cancelled hours later with no diagnosis. determinism.yml, release.yml, nightly.yml and build-wheels.yml set the same PYTEST_ADDOPTS block. nightly.yml raises the cap to 3600 seconds, because one property test at 100,000 examples runs for many minutes.

### Ci third party action pins

Source: `.github/workflows/ci.yml`.

Third-party actions such as seanmiddleditch/gha-setup-ninja are pinned to a full commit hash, with the release tag in a trailing comment, as every external `uses:` line is. See [Dependabot scope](#dependabot-scope).

### Codeql manual build

Source: `.github/workflows/codeql.yml`.

The C++ core runs under ASan, UBSan, TSan and cppcheck, and the Python code runs under ruff and strict mypy. The sanitizers report what one run did, and cppcheck and ruff work on one file at a time. CodeQL follows data across call boundaries. A use-after-free in the bindings, or an unvalidated path that reaches a file read, shows up only at that level.

The `c-cpp` analysis needs a real build, so it uses manual build mode with the `ci` preset. That preset is the configuration ci.yml's `native` job builds and tests: a Release build with the extension, the C++ tests and warnings-as-errors enabled (the C++ tests are off by default in CMakeLists.txt). Autobuild chooses its own CMake invocation, so the analysed build could differ from that configuration. The floating-point flags themselves come from cpp/cmake/NumericsFlags.cmake, which every configuration of the project's CMake build includes.

### Coverage omits only compiled extension

Source: `pyproject.toml`.

The pattern `*/_native/_tfidf_native*` matches the compiled extension module (the `.pyd` or `.so`) and nothing else in the `_native` package. A broader pattern such as `*/_native/*` also matches `_native/__init__.py`. That file holds about 40 measured statements of first-party Python that choose the backend, guard the ABI and self-test the floating-point environment. One of them is `check_float_environment`, the guard against a BLAS library that sets flush-to-zero and so flushes the subnormal margins this project studies. If that file is omitted, the 100% coverage gate can pass while the file is only partly tested. The omit pattern therefore names the extension alone, and the coverage gate measures the package Python beside it.

### Cross platform digest

Source: `.github/workflows/determinism.yml`.

The repository states that its numbers are bit-reproducible across operating systems, compilers and optimisation levels. An in-process test cannot check this, because one interpreter cannot show how a different libm rounds. determinism.yml computes the pipeline digest (`scripts/snapshot.py`) in every leg of two matrices: the reference backend on three operating systems and three Python versions, and the native backend on three operating systems and three build types. The compare job requires all 18 digests to be identical. In the native legs the digest comes from the reference path, and the differential suite checks that the two backends agree bit for bit.

For about 15% of IDF entries, the platform `log` gives a different value from the correctly rounded one. UCRT, glibc and Apple libm also round differently from each other, so a native `log` gives different weights, norms and scores on Linux and Windows. For this reason, IDF is computed once in Python with `decimal.Decimal.ln()`, and the native core does not call `log`. The native core then uses only operations that IEEE-754 requires to be correctly rounded (spec_addenda.md#g13). A failed determinism run means that a published number has changed.

### Dependabot scope

Source: `.github/dependabot.yml`.

A tag such as `@v4` can be moved to a different commit, and a branch reference such as `@master` cannot be reproduced at all and can be ahead of the latest release, so a workflow that names an action by tag or branch makes a CI run depend on code outside the repository, against the project's requirement that every result is a function of the repository's contents. Every external `uses:` line is therefore pinned to a full commit SHA, with the tag as a trailing comment. Dependabot reads the `<sha>  # v4` form and updates both halves, which keeps the pins maintainable. The configuration watches the `github-actions` ecosystem only, and groups every action bump into one weekly pull request.

Dependabot does not watch Python dependencies by version. requirements-dev.txt states floors for libraries and exact pins for ruff and mypy. A linter floor assumes that every later release accepts the current tree, and a new release can add rules that it does not accept. A weekly proposal to bump ruff would be merged unread or ignored. Security advisories reach the repository as Dependabot alerts, which need no configuration in this file.

### Differential requires native

Source: `.github/workflows/determinism.yml`.

Each differential test is skipped unless native_available() is true, and pytest exits 0 when every selected test is skipped. An extension that builds but does not load would then skip the whole comparison, and the job would report the native backend as bit-identical to the reference without comparing anything. A missing runtime DLL is a common cause, and the job builds on three platforms. For this reason, `tfidf_stability.cli.main info --require-native` runs before the suite. After the suite, the JUnit XML is parsed and at least 20 executed tests are required, because a marker rename or a collection error can select no tests even when the backend is present.

### Digest compare sort not cat

Source: `.github/workflows/determinism.yml`.

The compare job counts unique digests with `sort -u collected/*.txt`, not `cat collected/*.txt | sort -u`. Each digest file holds one line. Under `cat`, a file without a trailing newline runs into the next file. For example, files holding AAA, BBB and AAA can join into the single line AAABBBAAA, which `sort -u` counts as one unique value. If no file ends in a newline, every digest joins into one line, the count is always 1, and the gate cannot fail. `sort` reads each path as a separate input and ends the last line of each file itself, so the count stays correct even if a writer leaves out the newline. The job also requires exactly 18 collected digests, one per matrix leg (3 operating systems by 3 Python versions for the reference, 3 operating systems by 3 build types for native), because a single collected digest passes the uniqueness check by itself.

### Figures committed

Source: `.gitignore`.

matplotlib stamps its version into every PNG, so a figure's bytes depend on the plotting library as well as on the result. Two people who regenerate from identical JSON with different matplotlib versions can get different files. This is a reason to keep figures out of the tree, but they are committed. `reports/text/main.tex` embeds seven figures, and a tree without them builds the report with a "Figure not generated" box in place of each, so readers who only read the repository cannot see the results. A figure that every reader can see is worth more than byte-reproducibility of the image. The committed JSON is the reproducible artefact, and the result digest that `scripts/make_figures.py` stamps at the bottom of each figure ties the image to that JSON, so the provenance claim still holds. A regenerated PNG shows as modified whenever matplotlib changes, even when the result is identical. To compare figures, check the stamped digest, not the bytes.

### Mingw static link

Source: `cpp/CMakeLists.txt`.

CPython on Windows is built with MSVC. GCC runtime DLLs are not on the interpreter's DLL search path, so a MinGW-built .pyd that depends on one fails to import, and the only error is "DLL load failed while importing ...". The posix thread model also pulls in libwinpthread-1.dll, so -static-libgcc and -static-libstdc++ alone do not remove every GCC runtime dependency. Plain -static covers libgcc, libstdc++ and libwinpthread. With it, the module imports only python3xx.dll, KERNEL32 and the UCRT, which is all a self-contained extension needs. On Windows, the `native` job in ci.yml reads the extension's import table with objdump and fails if it lists libwinpthread, libstdc++ or libgcc.

### Minimal corpus demo

Source: `examples/minimal_corpus_demo.py`.

`examples/minimal_corpus_demo.py` runs the pipeline of README sections 2 and 3 on seven inline documents. It prints the ranking together with its margins, because the margins state how far the ranking is from becoming a different list, and sections 4.4 and 4.5 reason about them. A top-k printed without its margins is the misleading artefact this repository argues against.

Seven documents contain both regimes the paper separates, in one ranking:

- a boundary with a small, strictly positive margin, which certifies a tolerance: any perturbation with every `|ds_i|` below `m_k / 2` leaves the top-k set unchanged (section 4.4, research question A1);
- a boundary with a margin of exactly zero, where the scores certify nothing and the deterministic tie-break alone settles membership, with no numerical error involved (section 4.5, research question A2).

Both regimes occur together at this scale. This is why they are studied separately rather than as one notion of stability, and why no dataset is needed to show them. In the printed ranking, the boundary margin at k = 3 is about 1.0e-2, and the three documents that share no feature with the query score exactly 0, so m_k = 0 at k = 5 and 6 and the three operators order them differently. The example also prints every intermediate quantity that section 1.2 requires (tf, df, idf, weight, norm) for one document, and loads nothing from disk.

### Msvc fp strict

Source: `cpp/cmake/NumericsFlags.cmake`.

The MSVC path uses /fp:strict, not /fp:precise with a contraction switch. Under /fp:precise, some MSVC versions contract a*b+c to an FMA, which breaks bit-exactness against the reference. /fp:contract has no negative form. MSVC does not accept `/fp:contract-`: it prints `cl : command line warning D9002: ignoring unknown option '/fp:contract-'` and keeps contracting. D9002 is only a warning, so the build still succeeds, and only the Windows build log shows that the option was ignored. /fp:strict is the documented way to forbid contraction. It also pins floating-point exception and rounding semantics, which the project needs anyway. It costs some speed on the MSVC path in exchange for reproducibility.

### Nightly cpp mutation scope

Source: `.github/workflows/nightly.yml`.

The C++ mutation matrix has one job per header, as the Python matrix has one per module. Cost is the number of sites multiplied by the suite time, and a survivor names its own header. The C++ suite is a single ctest target, so the whole native suite judges every mutant.

Each header in the matrix has a triaged campaign: configs/equivalent_mutants_cpp.txt argues every mutant that scores as `survived`. An entry in that file that matches no survivor also fails the job, so the file cannot grow into a blanket suppression. All 11 headers under cpp/include/ outside `third_party` are scheduled; [Cpp mutation matrix is explicit](#cpp-mutation-matrix-is-explicit) gives the reason the matrix lists them explicitly and admits only triaged headers.

### Nightly python mutation scope

Source: `.github/workflows/nightly.yml`.

Mutation cost is the number of mutable sites multiplied by the suite time. Each nightly mutation job therefore runs one module against its own test files and tests/test_public_api_surface.py, not the whole suite, and a survivor names the module it comes from.

A scope that includes every test able to reach a module transitively reports no false survivors, but it does not fit the 90-minute job cap. utils/numerics.py is reachable from 27 test files, with 62 sites at 210 s a run, which gives 62 × 210 s = 217 minutes. vectorisation/sparse.py is reachable from 20 files, which gives 182 minutes. Interpreter start-up and collection take 7 to 9 seconds of every run, so a narrow scope saves less time than the test count suggests.

Widening a module's scope kills mutants that a narrower scope records as survivors. configs/equivalent_mutants.txt then fails on an entry that matches no survivor. A module is added to the matrix only after its campaign has run and its allowlist entries are reconciled.

tests/test_mutation_gate.py lists the 22 modules that no nightly job re-runs in its `_NOT_YET_SCHEDULED` map, each with a reason. 61 of the 105 entries in configs/equivalent_mutants.txt belong to those modules, and no nightly run rechecks them.

test_public_api_surface.py is in every scope because it asserts invariants that every module shares: frozen dataclasses, slots and `__all__`. Without it, a mutated `@dataclass(frozen=True)` survives in every module, and sort_keys.py has two surviving decorator mutants.

Recorded campaign results: analysis/noise_floor.py has 63 sites, run in 346 s, with 54 killed and 9 survivors, all in the allowlist. analysis/stability_profile.py has 72 sites, with 65 killed and 7 survivors, all in the allowlist. perturbation/vector_perturb.py has 51 sites, with 46 killed and 5 survivors, all in the allowlist, and no unmatched entry.

### Raw bit patterns

Source: `examples/inspect_intermediates.py`.

Section 1.2 requires intermediate quantities to stay inspectable. `examples/inspect_intermediates.py` shows why they are inspected as bit patterns: a decimal rendering is a lossy summary and discards the differences the study measures. The example demonstrates three properties on its twelve-document corpus.

First, one library call makes the pipeline platform-dependent. `log` is the only transcendental in the pipeline, and IEEE-754 leaves its rounding to the libm. At N = 12 the platform value and the correctly rounded value can already differ on a real vocabulary entry, before any question of scale (`docs/spec_addenda.md#g13`). With the Windows UCRT libm the example finds one: `alien`, at df = 6, where the platform idf is one ulp above the correctly rounded one. That one ulp carries through the weight and puts the score two ulps away, while the norm absorbs it.

Second, the difference is invisible in short decimal renderings and breaks a digest. The example formats the two scores five ways (`.6f`, `.12f`, `.15g`, `.16g`, `.17g`). In the measured case the first four print them identically; `.17g` separates them, as it separates any two binary64 values, and so does `float.hex`. An equality check against such a formatted string therefore reports agreement where `same_bits` reports a difference. For this reason `TfidfModel.digest` hashes `struct.pack("<d", ...)` and never a rendering.

Third, the place where the bits are measured decides what is measured. Varying only the reduction policy passed to the scorer, with the model's norms fixed, finds no difference on this corpus; refitting, so that the norms move as well, finds the noise floor at once. This is the trap in `docs/spec_addenda.md#g23`, and it separates a measured tau_floor from a vacuous one.

The closing sections place the effect in the two research questions. Bit-level disturbance is about thirteen decades below the smallest real score gap, so A1's ranking instability comes from corpus perturbation, not arithmetic. Exact ties are a bit-level property, and the block of bit-identical scores at the bottom of the ranking is where A2's deterministic tie-break chooses the order.

The vocabulary entry that the libm rounds differently depends on the libm, so the example hardcodes nothing: it searches the corpus and reports what it finds. With a libm that rounds every idf of this corpus correctly, it reports that none differ and skips the trace.

### Release licence split

Source: `.github/workflows/release.yml`.

The code is licensed under Apache-2.0. The prose and generated reports are licensed under CC BY 4.0 (see LICENSE.md). Creative Commons advises against its licences for software, because they give no patent grant and their attribution terms do not map onto compiled binaries. The Apache-2.0 licence on the code means wheels can be published without that conflict.

Before release, each wheel is tested as the installed package, not the source tree: build-wheels.yml, which release.yml calls, runs the test suite against each installed wheel through cibuildwheel's test command. A wheel that builds but leaves out a data file, such as the stopword list or the vendored stemmer, would pass every source-tree test and fail at install time.

### Release tag version check

Source: `.github/workflows/release.yml`.

release.yml triggers on `v*` tags, and `scripts/check_versions.py --tag` adds the tag to the set of versions the script compares ([Release tag joins version set](#release-tag-joins-version-set)). ci.yml calls the script without `--tag` and does not trigger on tags, so release.yml must call it with `--tag`. The check runs in the `sdist` job, and every later job depends on that job, directly or through `wheels`, so a wrong tag stops the run before any artefact is built. The step runs only on tag refs. A `workflow_dispatch` run from a branch has `github.ref_name` set to a branch name such as `main`, so there is no tag to compare.

### Reuse annotations not headers

Source: `REUSE.toml`.

LICENSE.md states the licence split in prose and gives the SPDX identifiers in a table. REUSE.toml states the same split in a form a tool can resolve, and `reuse lint`, which ci.yml runs, checks that every file in the tree is covered, so no reader has to inspect the tree by hand. The file uses `[[annotations]]` blocks instead of per-file SPDX headers for two reasons. A `.json` or `.jsonl` fixture cannot hold a header. A header added to a hashed asset changes the digest that the vendored-code gate (`scripts/check_vendored.py`) checks. The annotations cover every file without changing any byte that a digest covers.

The first two `path` lists mirror the table in LICENSE.md. The override blocks mirror THIRD_PARTY_NOTICES.md, which is the authority for vendored material. The Snowball blocks name files individually, because the stemmer directory and the test-vector directory also hold first-party files (`__init__.py`, `MANIFEST.sha256`). The doctest and nanobench directories hold only the vendored header, so a glob covers each. When a file moves, update both REUSE.toml and the document it mirrors.

### Ruff hook from environment

Source: `.pre-commit-config.yaml`.

The ruff and ruff-format hooks are `repo: local` hooks with `language: system`. They run `python -m ruff` from the installed environment instead of using the `astral-sh/ruff-pre-commit` mirror. A mirror carries its own `rev`, which is a second version statement that can differ from the exact `ruff==` pin in requirements-dev.txt. The hook and CI would then enforce different linters, and a newer ruff that stabilises a rule (as ruff 0.16 did with PLR0917) can fail the lint job on a commit that changes no Python. With a local hook, requirements-dev.txt is the only place that states the version. In exchange, pre-commit does not install ruff itself, so the hook needs the dev requirements installed. The other local hooks also run on the environment's Python with `language: system`, but they need only the standard library and the repository's own code, so they need no extra install.

### Sanitizer choices

Source: `.github/workflows/sanitizers.yml`.

UBSan runs with `float-divide-by-zero` (cpp/cmake/Sanitizers.cmake), because in this project a division by a zero vector norm is a logic error. README section 2.3 fixes cosine to 0 for a zero vector. That convention must apply before the division, not be detected afterwards as an Inf.

ASan matters because of the ranking layer. A NaN in a sort key breaks the strict weak ordering that `std::sort` requires, and the ranker sorts its keys with `std::sort` for some operators. The final insertion pass of the libstdc++ `std::sort` then moves past the start of the array, which is an out-of-bounds write and not only a wrong answer. The finiteness guards (`all_finite` in `cpp/include/tfidf/ranking/sort_keys.hpp`) prevent this, and ASan checks that they do.

### Sdist vendored cpp check

Source: `.github/workflows/release.yml`.

The step runs `pip download --no-deps --no-binary :all:` on the built sdist, which unpacks the archive and prepares its metadata through the build backend. `[build-system] requires` names scikit-build-core and nanobind, and pip gets both from PyPI. The step therefore needs network access, and a `--no-index` run fails before it reaches any C++. The C++ third-party sources are vendored under cpp/third_party/. The Python build backend is a declared dependency, not a vendored one.

### Signed zeros

Source: `cpp/cmake/NumericsFlags.cmake`.

The GCC/Clang flag list contains -fsigned-zeros, never -fno-signed-zeros. The negative form is a fast-math sub-option that lets the compiler ignore the sign of zero. In a list of -fno-* disables it looks like one more disable, but it relaxes IEEE 754. Under -fno-signed-zeros, (-0.0) + 0.0 can compile to -0.0 (bit pattern 8000000000000000). Under -fsigned-zeros it gives +0.0, as IEEE 754 requires. Every score is compared on its raw bit pattern (`bits_of` and `same_bits` in `utils/numerics.py`), because float `==` treats -0.0 and +0.0 as equal, so the sign of zero is part of a result. The flag in NumericsFlags.cmake is the main control. GCC and Clang define __NO_SIGNED_ZEROS__ for the relaxed form, and fp_guard.hpp refuses to compile when that macro is set. This guard catches a build that does not use NumericsFlags.cmake.

### Stability certificate demo

Source: `examples/stability_certificate_demo.py`.

Section 4.4 states one implication: `|ds_i| <= eps` for every document together with `eps < m_k / 2` guarantees that the top-k set is invariant. The condition is sufficient only, and readers take it as one of two stronger claims, in opposite directions, from the same sentence.

Read as a prediction, "not certified" becomes "will break" and the certificate becomes a false alarm: perturbations of many times the radius usually leave the top-k unchanged, because a flip needs a direction that scattered noise almost never takes. Read as a safety margin, "sufficient" becomes "conservative" and the radius is reduced; but `m_k / 2` is exact, and a perturbation that breaks the set at `m_k / 2 + delta` can be written down.

Section 7.2 reports certified radii as empirical certificates of stability, and the transition curve in `analysis/stability_profile.py` is interpretable only once worst case and average case are separated. `examples/stability_certificate_demo.py` therefore shows three cases on one hand-built ranking of six documents at k = 3, where m_k = 0.25 and the radius is 0.125:

1. below the radius (every score moved by 0.0625), in the worst direction: unchanged, by proof;
2. four times the radius, scattered: unchanged anyway, so the converse of the theorem is false;
3. just above the radius, aimed: flipped, so the bound is tight.

Every score, perturbation and intermediate value is an integer multiple of 2^-20 with magnitude below 2, so each needs at most 21 significant bits of the 53-bit binary64 significand and no operation rounds. A demonstration about the decision boundary that also carried rounding error would not let a reader separate the two effects. The construction is the dyadic one used in `tests/test_margins_and_flip_radii.py`, in the terms of the tightness note under section 4.4 in `docs/spec_addenda.md`.

### Testpypi rehearsal

Source: `.github/workflows/release.yml`.

A PyPI filename is permanent. A wheel published with the wrong contents cannot be replaced; it can only be yanked and replaced by a new version. The `testpypi` job uploads the same artefacts to TestPyPI first, so a metadata or naming fault appears while the version number can still be reused. `skip-existing` is set because TestPyPI rejects a filename it already holds, and a re-run of a release must not fail for that reason alone. Any other failure in this job blocks `publish`, which depends on it.

### Tie break discontinuity demo

Source: `examples/tie_break_discontinuity_demo.py`.

Research question A2 asks whether deterministic tie-breaking induces decision-level discontinuities independent of numerical error. The corpus-scale measurements (the disagreement rates of section 7.3, the case study of section 7.4) are rates over thousands of queries, and a rate invites the objection that part of it is arithmetic noise. `examples/tie_break_discontinuity_demo.py` removes that objection on a corpus small enough to read in full.

Of six documents, three share one bit pattern. Under that exact tie the three ranking operators of section 4.5 give three different answers, and the example checks that all three consumed byte-identical scores. No arithmetic runs between the operators, so numerical error is zero and the disagreement has one cause. The scores are written down, not computed from text: under G22, `tf = count / L` makes the finest text edit a `1/L` relative perturbation, so no corpus can be authored onto a chosen separation. Dyadic values make each score exact in binary64, so the three at 0.5 are one bit pattern rather than three nearby ones.

Two separate discontinuities appear. At k = 3 the tie straddles the boundary and the operators return three different top-k sets. At k = 4 the boundary margin is 0.25 and all three return the same set in three different orders. The note under section 4.4 in `docs/spec_addenda.md` explains why neither bounds the other: `m_k` constrains the boundary, `m_min^top` the interior, and the two gap index sets are disjoint.

The second half addresses G1. The tie group of section 2.3.3 is a ball, and the paper calls its members indistinguishable, but the relation is intransitive, so the implementation carries three separately named objects: ball, chain and clique. The dyadic ladder `s_i = i * 2^-20` at `tau = 2^-20` is the standard witness: every score, gap and comparison in it is exact in binary64, so no observation can be attributed to rounding. On six rungs, one chain covers the whole ladder while the largest clique has two members, so the ratio rho of largest chain to largest clique is 3.

### Wheel force include assets

Source: `pyproject.toml`.

The hash-verified stopword list (`data/assets/stopwords_en_v1.txt`) and the normative config (`configs/default.yaml`) sit at the repository root, outside the package directory, so `wheel.packages` alone does not put them in the wheel. Without the stopword list, an installed distribution cannot fix its vocabulary, on which every published number depends. Without the config, it has no default configuration. The files stay at the root rather than under `src/` because `data/README.md` states their licence position beside them, and `scripts/check_vendored.py`, `data/assets/MANIFEST.sha256` and the docs address them by that path.

`wheel.force-include` copies `data/assets` to `tfidf_stability/data/assets` and `configs` to `tfidf_stability/configs`, which is the location the loaders try first. `_resolve_asset_dir` in `preprocessing/stopwords.py` and `_resolve_default_config` in `cli/commands.py` try the packaged location and fall back to the checkout. `tests/test_packaged_assets.py` tests both layouts on temporary directory trees. build-wheels.yml runs the suite against each installed wheel, where only the packaged location exists.
