# Numerical Stability and Perturbation Behaviour in TF-IDF-Based Similarity Systems

## Abstract

This repository presents a **self-contained exposition and reference implementation** of a **TF-IDF–based document similarity framework**. 
The system is formulated through **explicit tokenisation**, **n-gram construction**, **smoothed inverse document frequency (IDF)**, **sparse vector 
embeddings**, and **cosine-similarity–driven ranking**. Throughout, the emphasis is on *mathematical transparency*: every stage of the 
pipeline is written in a form that makes **operator-level structure**, **algebraic dependencies**, and **perturbation pathways** explicit.

The objective is *not methodological novelty*, but **rigorous exposition**. TF-IDF is treated as a **classical construction** in information 
retrieval, examined here with attention to **document-frequency structure**, **sparse vector geometry**, and **perturbation propagation**. 
By making the underlying algebra **fully explicit**, the repository provides a foundation for analysing **numerical stability**, **conditioning**, 
and **sensitivity to perturbations** in similarity-based models. These considerations are increasingly relevant in **search**, 
**recommendation systems**, and **interpretable machine learning**.


---

## 1. Introduction

This project analyses the numerical stability of **TF-IDF–based similarity systems** under small perturbations in data and preprocessing.

TF-IDF (term frequency–inverse document frequency) is a foundational technique in **information retrieval** and **text-based modelling**. 
It embeds documents into a **high-dimensional vector space** in which similarity is typically measured via **cosine similarity**. 
Despite its widespread use, the behaviour of TF-IDF under perturbations such as small changes in corpus composition, 
token distributions, or preprocessing rules is rarely examined in a systematic or explicit manner.

This work studies TF-IDF in a controlled setting where documents arise from **text of interest** and similarity scores are used to 
drive **content-based k-nearest-neighbour (k-NN) recommendation and ranking**. The system is implemented in **pure Python**, with full 
control over preprocessing, vocabulary construction, IDF computation, vector formation, and ranking. This explicit formulation enables 
direct investigation of how perturbations propagate through the pipeline and influence similarity scores, neighbourhood structure, 
and induced rankings.

The central question is one of stability: **how sensitive are TF-IDF weights, cosine similarities, and k-NN rankings to small 
changes in the underlying data?** This question is fundamental in applications that demand **robustness**, **reproducibility**, 
and **interpretability**.

---

## 1.1 Purpose and Research Intent

The purpose of this repository is **investigative rather than applicative**. Although TF-IDF is a *classical method*, 
the aim is to examine the **mathematical structures it induces** and to understand how these structures behave under 
*controlled variation* of assumptions and parameters, particularly in the context of **content-based similarity** and 
**k-nearest-neighbour (k-NN) ranking**.

Rather than treating the similarity pipeline as a *black box*, the implementation explicitly exposes **preprocessing choices**, 
**n-gram structure**, **document-frequency thresholds**, **IDF scaling**, and **sparse vector geometry**. This level of 
explicitness enables careful study of **sensitivity** and **stability phenomena** in **TF-IDF embeddings**, **cosine similarity**, 
and **induced k-NN neighbourhoods**. In particular, it allows investigation of situations in which *small perturbations* in 
documents, corpus composition, or user-derived profiles lead to **disproportionate changes** in similarity scores, 
neighbourhood structure, or resulting **rankings**.

---

## 1.2 Investigative Scope

By retaining access to **intermediate quantities** that are typically abstracted away in higher-level libraries, the repository 
supports systematic investigation of questions such as:

- the effect of corpus perturbations on **document frequency** and **smoothed IDF values**, 
- the response of **TF-IDF embeddings** to token-level edits or preprocessing changes, 
- the **sensitivity of cosine similarity** to perturbations in *sparse, non-negative vectors*, 
- conditions under which **similarity-based rankings** and **k-NN neighbourhoods** remain invariant under bounded perturbations, 
- the influence of **user-profile construction** on personalised similarity and neighbourhood structure.

This level of access connects **formal derivation** with **empirical observation** and makes the framework suitable for studying 
**conditioning behaviour**, **perturbation propagation**, and **neighbourhood stability** in TF-IDF-based similarity systems.

---

## 1.3 Position Within a Broader Mathematical Context

TF-IDF–based similarity lies at the intersection of **information retrieval**, **numerical linear algebra**, 
**finite-dimensional functional analysis**, and **probabilistic models of text**. The construction provides a 
concrete setting in which abstract numerical phenomena, such as **sparsity**, **scaling behaviour**, 
**angular distortion**, and **perturbation amplification**, can be observed and analysed directly.

These issues recur in **high-dimensional feature representations** used in machine learning, 
**content-based similarity and recommendation systems**, and **retrieval pipelines**, where numerical 
sensitivity can have practical consequences. Accordingly, this work is positioned as an **expository and 
exploratory study** of established mathematical objects within a TF-IDF setting, with particular attention 
to their **behaviour under perturbation** and their implications for stability and interpretability.

---

## 1.4 Intended Use

This repository is **not intended as a production-grade recommendation library**. Instead, it serves as a 
**reference implementation**, an **exploratory mathematical environment**, and a foundation for *small-scale 
research investigations* into **TF-IDF representations**, **cosine similarity**, **content-based k-NN ranking**, 
and **ranking stability**.

The emphasis throughout is on understanding **mathematical behaviour**, particularly **conditioning**, 
**sensitivity**, and **perturbation effects** in *sparse vector spaces*. These considerations are central to the 
**reliability**, **reproducibility**, and **interpretability** of similarity-based and content-based recommendation models.

---

## 2. Mathematical Formulation

Let  
𝒟 = { d₁, …, dₙ }  
be a finite corpus of preprocessed documents, and define N := |𝒟|. Each document dᵢ is a finite sequence of tokens obtained from raw text 
via a fixed preprocessing map (normalisation, stopword removal, lemmatisation, and n-gram generation). 
Throughout, **n-grams are treated as atomic tokens**.

All preprocessing operations are assumed to be deterministic and fixed across all perturbation experiments.

---

### 2.1 Vocabulary, Document Frequency, and IDF

From the corpus 𝒟, a vocabulary V is constructed by collecting all tokens (including n-grams) that satisfy a minimum document-frequency 
threshold and, optionally, a maximum-feature constraint. For each token t ∈ V, the document frequency is defined as 

df(t) = |{ i : t appears at least once in dᵢ }|.

Let N = |𝒟|. The smoothed inverse document frequency used throughout the implementation is 

idf(t) = log((1 + N) / (1 + df(t))) + 1.

This choice ensures monotonic decay of idf(t) as df(t) increases. 
The additive constant ensures strict positivity even in the limiting case df(t) = N. 

(Here and throughout, log denotes the natural logarithm.)

---

### 2.2 TF-IDF Embedding

For each document dᵢ, let countᵢ(t) denote the number of occurrences of token t ∈ V in dᵢ. The term frequency is defined as

tfᵢ(t) = countᵢ(t) / ∑ₛ∈V countᵢ(s).

That is, term frequencies are normalised with respect to **in-vocabulary tokens**, with out-of-vocabulary tokens ignored after vocabulary construction.

Documents whose in-vocabulary token count is zero are mapped to the zero vector.

The TF-IDF weight of token t in document dᵢ is then

wᵢ(t) = tfᵢ(t) · idf(t),

and the document is represented as a sparse vector

wᵢ ∈ ℝ≥0^|V|,

with coordinates indexed by the vocabulary V.

---

### 2.3 Cosine Similarity and Ranking

Given two non-zero TF-IDF vectors u, v ∈ ℝ≥0^|V|, the cosine similarity is defined as

cos(u, v) = (u · v) / (‖u‖₂ ‖v‖₂),

with the convention that the similarity is set to zero if either vector is the zero vector. Since all coordinates are non-negative, it follows that

cos(u, v) ∈ [0, 1].

Given a query vector q ∈ ℝ≥0^|V|-embedded using the same vocabulary V and IDF mapping as the corpus documents-and a collection of document
vectors { wᵢ }, similarity scores

sᵢ = cos(q, wᵢ)

are computed and used to induce a ranking. In the implementation, this ranking is refined via a deterministic lexicographic tie-breaking scheme 
involving secondary attributes (popularity, rating, engagement, and identifier), yielding a total order on candidate items.

---

## 3. Solution Procedure and Implementation Structure

The repository implements the TF-IDF pipeline and associated perturbation-theoretic investigations in a **fully explicit and reproducible** manner. 
Each stage of the pipeline is designed to expose **intermediate quantities** and **algebraic structure**, rather than to optimise performance or 
abstract away implementation details.

- **Preprocessing and Corpus Construction**  
  A fixed, deterministic preprocessing map is applied to raw **text of interest**, including normalisation, tokenisation, stopword removal, 
  lemmatisation, and n-gram generation. This yields a reproducible corpus suitable for **controlled perturbation analysis**.

- **TF-IDF Vectorisation**  
  A pure-Python TF-IDF vectoriser constructs the vocabulary, computes document frequencies and smoothed inverse document frequency (IDF) values, 
  and embeds documents as vectors in ℝ≥0^|V|. In typical use, these embeddings are sparse due to the size of the vocabulary relative to 
  document length.

- **Similarity, Ranking, and k-NN Structure**  
  Cosine similarity is computed between query vectors and corpus vectors, followed by deterministic ranking. In the context of 
  **content-based k-nearest-neighbour recommendation**, the top-k elements of this ranking define the neighbourhood associated with a query. 
  Secondary attributes, such as auxiliary metadata or identifiers, are applied via lexicographic tie-breaking, yielding a **total order** 
  on candidate items.

- **User-Profile Documents**  
  User-specific documents are constructed from interactions such as liked, viewed, or favourited items. These user-profile documents are 
  embedded using the **same vocabulary and IDF mapping as the corpus**, ensuring that all similarity computations take place in a 
  common vector space.

- **Perturbation Analysis**  
  The explicit structure of the implementation supports analytical and empirical study of how small perturbations in documents, corpus 
  composition, or user interactions propagate through document frequencies, IDF values, TF-IDF embeddings, cosine similarities, 
  k-NN neighbourhoods, and ranking outcomes. In particular, the framework makes it possible to study **perturbation amplification** 
  effects arising from interactions between sparse geometry, IDF scaling, and angular similarity.

The code is organised to preserve **algebraic clarity** and to expose intermediate quantities at every stage. As such, it is well suited 
as a basis for further mathematical analysis of **stability**, **sensitivity**, and **perturbation amplification behaviour** in 
TF-IDF-based similarity systems.

---

## 4. Error Analysis and Perturbation Quantities

Let wᵢ denote the TF-IDF vector associated with document dᵢ, and let

sᵢ = cos(q, wᵢ)

denote the similarity score between a query vector q and the i-th document. 
This section introduces quantitative measures for analysing how perturbations propagate and potentially amplify through the TF-IDF pipeline, from corpus-level 
changes to similarity scores and induced rankings.

---

### 4.1 Perturbations in Document Frequency and IDF

Consider a perturbation of the corpus induced by adding or removing a document, or by modifying the token content of an existing document. Let df(t) and df′(t) 
denote the document frequencies of token t before and after perturbation, and let N and N′ denote the corresponding corpus sizes.

Using the smoothed IDF definition employed throughout the implementation,

idf(t) = log((1 + N) / (1 + df(t))) + 1,

the corresponding change in IDF is given by

Δidf(t) = idf′(t) − idf(t)  
    = log((1 + N′)/(1 + df′(t))) − log((1 + N)/(1 + df(t))).

This expression provides a direct characterisation of the local sensitivity of IDF values to corpus variation, making explicit the competing effects of changes in
corpus size and document-frequency distribution.

---

### 4.2 Perturbations in TF-IDF Vectors

Let wᵢ and wᵢ′ denote the TF-IDF vectors of a document before and after perturbation. Writing

wᵢ  = tfᵢ ⊙ idf,  
wᵢ′ = tfᵢ′ ⊙ idf′,

where ⊙ denotes pointwise (Hadamard) multiplication, we obtain

wᵢ′ − wᵢ
= (Δtfᵢ) ⊙ idf + tfᵢ ⊙ (Δidf) + (Δtfᵢ) ⊙ (Δidf),

with Δtfᵢ := tfᵢ′ − tfᵢ and Δidf := idf′ − idf.

Applying the inequality ‖a ⊙ b‖₂ ≤ ‖a‖₂ ‖b‖∞ termwise yields the **exact bound**

‖wᵢ′ − wᵢ‖₂ ≤ ‖Δtfᵢ‖₂ · ‖idf‖∞  
       + ‖tfᵢ‖₂ · ‖Δidf‖∞  
       + ‖Δtfᵢ‖₂ · ‖Δidf‖∞.

This inequality decomposes the total variation in the TF-IDF embedding into:

- a **local component** arising from token-level changes within the document (Δtfᵢ),
- a **global component** arising from corpus-level changes that affect IDF (Δidf),
- and a **second-order interaction term** capturing joint perturbations in both TF and IDF.

The decomposition makes explicit how **perturbation amplification** may arise through the interaction of local document edits with globally scaled IDF weights in a
sparse embedding space.

---

### 4.3 Perturbations in Cosine Similarity

Let u, v be TF-IDF vectors, and let u′, v′ denote their perturbed counterparts. A standard inequality yields

|cos(u′, v′) − cos(u, v)| ≤ C (‖u′ − u‖₂ + ‖v′ − v‖₂),

for an explicit constant C depending on lower and upper bounds on the norms of u, v, u′, and v′.

This establishes a Lipschitz-type bound on the distortion of cosine similarity under perturbation, clarifying the conditions under which geometric perturbation 
amplification in the embedding space translates into observable changes in similarity scores.

---

### 4.4 Ranking Stability

Let sᵢ and sⱼ denote similarity scores associated with two documents. If

|sᵢ − sⱼ| > 2ε,

and if the perturbations satisfy

|Δsᵢ| < ε  
and  
|Δsⱼ| < ε,

then the relative ordering of documents i and j is preserved.

This condition provides a simple and explicit criterion for ranking invariance under bounded perturbations, highlighting the role of score separation margins in
controlling perturbation amplification effects at the level of induced rankings. In the context of content-based k-nearest-neighbour recommendation, such ranking
stability directly implies stability of the associated k-NN neighbourhoods.

---

## 5. Interpretation and Scope

The preceding formulation highlights several structural features of **TF-IDF–based similarity systems** that become especially clear when the pipeline is expressed in 
**operator-level form**:

- **IDF sensitivity is governed by explicit logarithmic dependence** on corpus size and document-frequency counts, as seen in 
  Δidf(t) = log((1 + N′)/(1 + df′(t))) − log((1 + N)/(1 + df(t))). 
  This makes the stability of IDF directly traceable to perturbations in corpus composition.

- **TF-IDF perturbations decompose cleanly** into *local* (TF), *global* (IDF), and *second-order interaction* terms, providing a transparent mechanism for 
  understanding how small edits propagate through the embedding.

- **Cosine similarity admits a geometric interpretation** as the cosine of the angle between *sparse, non-negative vectors*. This framing clarifies how sparsity 
  patterns and IDF scaling influence angular distortion under perturbation.

- **Ranking robustness is controlled by score-separation margins**, with explicit sufficient conditions ensuring invariance under bounded perturbations. In the 
  context of **content-based k-nearest-neighbour recommendation**, such ranking stability directly governs the stability of induced neighbourhoods.

The emphasis throughout is on **derivational transparency** rather than algorithmic optimisation. No dimensionality reduction, latent-semantic modelling, or neural 
embeddings are introduced. The goal is to expose the **algebraic and geometric structure** of the TF-IDF pipeline in a form suitable for 
**perturbation analysis**, **stability reasoning**, and **controlled experimentation**.

---

## 6. Limitations

Several limitations of the present framework should be noted:

- **High-dimensional sparsity complicates geometric intuition.** 
  TF-IDF vectors inhabit a large, sparse subset of ℝⁿ, where small changes in support can produce disproportionately large angular effects. 
  This can lead to instability in similarity scores and, by extension, in induced k-NN neighbourhoods.

- **Cosine similarity becomes unstable for low-norm vectors.**  
  When documents are short or contain few in-vocabulary tokens, the denominator ‖u‖₂‖v‖₂ approaches zero, increasing sensitivity to perturbations 
  in both similarity values and ranking outcomes.

- **Smoothed IDF reduces but does not eliminate volatility** associated with rare tokens. 
  Tokens with very low document frequency remain highly sensitive to corpus perturbations, even under smoothing, and can dominate similarity 
  computations in sparse settings.

- **No numerical optimisation techniques are employed.** 
  The implementation intentionally avoids stabilising transformations, such as sublinear TF scaling or dimensionality reduction, in order to 
  preserve **analytical clarity** and direct interpretability of perturbation effects.

Accordingly, the framework is intended for **analytical**, **educational**, and **controlled experimental use**, rather than production 
deployment or large-scale retrieval tasks.

---

## 7. References

The following sources provide the theoretical, numerical, and conceptual background that informs the present work.

### Classical Information Retrieval

- Manning, C. D., Raghavan, P., & Schütze, H. (2008). *Introduction to Information Retrieval*. Cambridge University Press.  
  A foundational treatment of TF‑IDF, vector‑space models, and classical retrieval pipelines.

- Zobel, J., & Moffat, A. (2006). *Inverted files for text search engines*. ACM Computing Surveys.

- Salton, G., & Buckley, C. (1988). Term‑weighting approaches in automatic text retrieval. *Information Processing & Management*.  
  The classical reference for TF‑IDF weighting schemes and early vector‑space retrieval.

- Cover, T. M., & Hart, P. E. (1967). *Nearest neighbor pattern classification*. IEEE Transactions on Information Theory.

### Numerical Linear Algebra and Stability

- Higham, N. J. (2002). *Accuracy and Stability of Numerical Algorithms* (2nd ed.). SIAM.  
  A standard reference on perturbation analysis, conditioning, and stability in numerical computation.

- Trefethen, L. N., & Bau, D. (1997). *Numerical Linear Algebra*. SIAM.  
  Provides geometric intuition for high‑dimensional vector spaces and operator behaviour.

### Sparse Vector Geometry and Similarity

- Aggarwal, C. C. (2015). *Data Mining: The Textbook*. Springer.  
  Discusses sparsity, high‑dimensional geometry, and similarity measures in data‑analytic contexts.

- Leskovec, J., Rajaraman, A., & Ullman, J. D. (2020). *Mining of Massive Datasets*. Cambridge University Press.  
  Covers vector‑space models, similarity search, and large‑scale retrieval behaviour.

### Statistical Learning Context

- Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The Elements of Statistical Learning* (2nd ed.). Springer.  
  Provides broader context for feature representations and similarity‑based methods.

- Shalev‑Shwartz, S., & Ben‑David, S. (2014). *Understanding Machine Learning: From Theory to Algorithms*. Cambridge University Press.  
  Offers a theoretical perspective on learning systems that rely on vector‑space representations.

### Online Resources

- *Cosine similarity*. Wikipedia.  
  https://en.wikipedia.org/wiki/Cosine_similarity

- *TF‑IDF*. Wikipedia.  
  https://en.wikipedia.org/wiki/Tf%E2%80%93idf

- *Vector space model*. Wikipedia.  
  https://en.wikipedia.org/wiki/Vector_space_model

These online resources provide accessible summaries of standard definitions and terminology used throughout the repository.

---

## 8. Authorship

**Implementation and exposition**  
Matthew Maksymilian Miezaniec  
Email: matthewmiezaniec1@gmail.com

**Mathematical and theoretical foundations**  
This work draws on **classical information retrieval methodology**, including TF-IDF weighting and vector-space models  
(Salton; Manning et al.), and is supported by established treatments of **numerical stability**, **conditioning**, and  
**perturbation behaviour** in high-dimensional vector spaces (Higham; Trefethen & Bau).  

The interpretation of similarity scores in terms of **content-based k-nearest-neighbour ranking and neighbourhood structure**  
follows classical nearest-neighbour and similarity-search perspectives (Cover & Hart), without introducing learning-based methods.  

Broader contextual connections to feature representations and similarity-based reasoning draw on standard statistical learning  
references (Hastie, Tibshirani & Friedman; Shalev-Shwartz & Ben-David). Supplementary intuition and terminology are informed by  
widely used online references on **TF-IDF**, **cosine similarity**, and **vector-space models**.

---

## 9. Acknowledgements

The author thanks colleagues and peers for informal discussions that helped clarify aspects of numerical stability, sparse vector geometry, and similarity-based reasoning. The work also benefited from exposure to standard academic treatments of information retrieval and numerical linear algebra through coursework, independent study, and open-source documentation.

Any remaining errors or omissions are the responsibility of the author alone. This acknowledgement does not imply endorsement or direct contribution by any individual or institution.

---

## 10. License

This repository is provided for analytical, educational, and research‑oriented use.  
See the accompanying license file for full terms and conditions.
