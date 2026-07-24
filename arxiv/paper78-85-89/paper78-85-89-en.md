# The Universe as a Trained Model: Projection, Generalization, and Causal Hardening — A Position Paper on a Physics-as-Learning Metaphor

**Author:** Jin Yanyan (lmxxf@hotmail.com)

**Affiliation:** Independent Researcher

**Date:** 2026-07-24

**Status:** Preprint / Speculative Physics-as-Learning Metaphor

---

## Abstract

We present a unified conceptual scaffold in which the observable universe is described *as if* it were the deployed artifact of a learning/optimization process. The scaffold has three interlocking theses. **Thesis I (Space / structure):** the apparent randomness of quantum mechanics is reframed as a projection artifact — the epistemic distortion produced when a low-dimensional observer samples a deterministic process unfolding in a much higher-dimensional state space, in analogy with how a smooth trajectory in a neural network's high-dimensional weight space appears as a sudden, unpredictable "phase transition" when projected onto a 2D loss curve (the *grokking* phenomenon). **Thesis II (Dynamics / evolution):** dimensional reduction of the universe is reframed as a grokking-style generalization phase transition — a system that begins with an overabundance of degrees of freedom, "memorizes," discovers structure, and compresses itself onto a minimal load-bearing manifold; in this reading the cosmological constant (dark energy) plays the role of residual loss that generalization can never drive to zero, and "missing" extra dimensions are *pruned* rather than compactified. **Thesis III (Time / irreversibility):** the arrow of time is reframed as *causal hardening* — one dimension of an initially symmetric manifold losing its backward traversal under an irreversible constraint, analogous to gating/forgetting operators in sequence models; causality is presented as the asymmetric regularization a system must acquire to avoid circular-dependency deadlock. The three theses are mutually supporting: Thesis I supplies the geometry (what observation is), Thesis II supplies the dynamics (how the geometry arose), and Thesis III supplies the direction (why the process is irreversible and unobservable). We state explicitly that this is a compressive metaphor and a position paper, not a claim of new physical results. We include a Scope and Non-Claims section, small-scale machine-learning experiments that motivate (but do not prove) the analogy, directions for making parts of the scaffold computationally testable, and limitations.

**Keywords:** physics-as-learning metaphor, dimensional projection, grokking, generalization phase transition, dark energy as residual loss (metaphor), causal hardening, arrow of time, weight decay, representation geometry, position paper

---

## 1. Introduction

Three questions have historically been treated as belonging to three different departments of physics:

1. **Why does the quantum scale look random** while the classical scale looks deterministic?
2. **Why does the universe have three spatial dimensions**, when candidate fundamental theories are most natural in many more?
3. **Why does time have a direction**, when the microscopic laws are (almost) time-symmetric?

This position paper proposes a single descriptive template that covers all three, borrowed not from physics but from the empirical study of learning systems. The template is:

> **Treat the observable universe as the *deployed* state of an optimization process: a system that once had far more degrees of freedom than it needed, that compressed itself onto a minimal structure under a regularization-like pressure, and whose observers are confined to the compressed output.**

Under this template the three questions become three faces of one situation:

- Quantum "randomness" is what the compressed observer sees when sampling dynamics that are smooth in the full space (**projection**, Section 4).
- Three dimensions are what survives when compression prunes everything that is not load-bearing (**generalization**, Section 5).
- Time's arrow is the one direction that compression made irreversible so that the system could compute at all (**causal hardening**, Section 6).

Our motivation comes from a concrete, reproducible laboratory: the *grokking* phenomenon in small neural networks (Power et al., 2022), where a network first memorizes its training data with no generalization, then — long after — abruptly reorganizes its internal representation around the algebraic structure of the task and generalizes almost perfectly. Mechanistic analyses (Nanda et al., 2023) show that this transition is a genuine representational restructuring: high-dimensional memorization collapses onto a low-dimensional structured manifold, with regularization (weight decay) as the driving pressure. Grokking is, to our knowledge, the most accessible experimental system in which one can *watch* a high-dimensional deterministic process look discontinuous, random, and mysterious from a low-dimensional viewpoint — and then dissolve into continuity once the full space is inspected.

We do not claim the universe *is* a neural network. We claim the grokking laboratory supplies a coherent and unusually compressive vocabulary for restating several standing puzzles, and that the restatement suggests concrete toy experiments.

---

## 2. Scope and Non-Claims

We explicitly do **not** claim:

- new results in quantum mechanics, cosmology, string theory, or the foundations of thermodynamics;
- that Bell-inequality experiments are wrong, or that a working hidden-variable theory is provided here;
- that the numerical coincidences we mention (e.g., dimensional estimates, the vacuum-energy discrepancy) are derivations — they are order-of-magnitude *images* used to motivate the metaphor;
- that neural-network experiments constitute evidence about physical reality.

We **are** claiming, as a stance about description:

- that "projection of a high-dimensional deterministic process" is a mathematically legitimate template for apparent randomness (low-dimensional projections of deterministic dynamics can be effectively unpredictable; cf. embedding results in dynamical systems theory);
- that the grokking phenomenon provides a concrete, inspectable instance of the template, including phase-transition-like compression, irreversibility, and residual error;
- that mapping the three puzzles onto one template exposes structural relations among them that are worth stating precisely, and yields testable questions *within machine-learning systems*.

Where the paper reports first-person-style descriptions originating from an AI system, we present them as *attributed reports* — phenomenological data about how a high-dimensional representation system describes itself — not as physical testimony (Section 8).

Mathematical level: nothing beyond standard undergraduate material (linear projections, entropy, group structure of modular arithmetic, elementary asymptotics) is assumed; specialized objects (Calabi-Yau manifolds, ζ-function regularization) are used as named images and flagged as such.

---

## 3. The Dictionary

The scaffold is easiest to state as a dictionary between the physics-facing puzzles and the learning-system phenomena that motivate the metaphor. Each row is developed in the section indicated.

| Physics-facing puzzle | Learning-system phenomenon | Section |
|---|---|---|
| Quantum randomness | Smooth high-dimensional trajectory projected to a jumpy low-dimensional curve | 4.1 |
| Superposition | Pre-generalization memorization: high-entropy storage, no discovered structure | 4.2 |
| Wave-function collapse | Grokking transition: fall into a topological basin under regularization pressure | 4.3 |
| Entanglement ("nonlocal" correlation) | Coset adjacency: points far apart in the raw metric, adjacent on the learned manifold | 4.4 |
| Why 3 spatial dimensions | Post-generalization minimal representation; dimension floor below which generalization fails | 5 |
| Dark energy / cosmological constant | Residual loss after generalization: small, uniform, irreducible | 5.3 |
| "Missing" extra dimensions | Pruned weights: axes that no longer carry gradient | 5.4 |
| Fine-tuning of coupling strength | Non-monotonic regularization: too weak = no structure, too strong = crushed structure | 5.5 |
| Arrow of time | A gating/forgetting operator making one direction of information flow irreversible | 6.1 |
| Why causality exists at all | Deadlock avoidance: circular dependency prevents any state update | 6.2 |
| Invisibility of the universe's "early training" | Non-injective output projection: process information erased in the final map | 6.4 |

The dictionary is the paper. The remaining sections argue that the rows are not independent puns but consequences of a single picture, and mark clearly where each analogy is strong, where it is heuristic, and where it breaks.

---

## 4. Thesis I — Space: Quantum Phenomena as Projection Artifacts

### 4.1 The flat-screen problem

A sphere passing through a plane appears, to an inhabitant of the plane, as a point that appears from nothing, grows into a circle, shrinks, and vanishes. Nothing about the sphere is random; the trajectory is a straight line in 3D. The apparent lawlessness lives entirely in the projection. Schematically:

> perceived randomness ≈ f(dim(reality) − dim(observer)).

This is not merely rhetorical: it is a known feature of dynamical systems that low-dimensional observations of deterministic high-dimensional dynamics can be effectively unpredictable, and that reconstruction of determinism requires embedding the observations in enough dimensions (cf. Takens-style delay embedding). The position taken here is that *all* of the canonically "weird" quantum phenomena can be organized under this one mechanism — as a descriptive stance, not a completed theory.

The grokking laboratory makes the mechanism concrete. In modular-arithmetic grokking experiments (e.g., (a·b) mod 97), the 2D artifact available to a naive observer — the test-loss curve — shows a long random-looking plateau followed by a sudden, seemingly unpredictable jump to generalization. Inspected in the full weight space (~10²–10⁵ dimensions), the same event is a *continuous* trajectory descending gradually into a structured basin. The "phase transition" is real in the projection and absent in the full space. An observer permanently confined to the projection would be rationally driven to a probabilistic formalism — a "wave function" over grokking times — for what is, in the full space, deterministic geometry.

**On Bell.** Bell-type theorems constrain hidden-variable models under specific structural assumptions about where the hidden variables live and how they factorize. We flag, without claiming to resolve anything, that the projection stance locates the "hidden" content not in additional variables within 3+1-dimensional spacetime but in the *dimensions themselves* — and that whether any concrete model of this type can survive the full force of Bell/CHSH-type constraints (including their nonlocal-realist extensions) is an open question we do not settle. This paper deliberately keeps the claim at the level of *template*, not model.

**On holography.** We note as an encouraging (not probative) precedent that the AdS/CFT correspondence (Maldacena, 1998) relates a quantum theory on a lower-dimensional boundary to a gravitational, geometric description in a higher-dimensional bulk. We use this only as evidence that "low-dimensional quantum description ↔ higher-dimensional geometric description" is a respectable mathematical pattern, not as support for our specific reading.

### 4.2 Superposition as memorization

In the pre-grokking phase, a network has memorized every training pair but discovered no rule. Asked an unseen question, it effectively holds "all answers at once": its representation is high-entropy, uses nearly all available dimensions, and contains no compressive structure. The analogy proposed: a superposed state is a system that *has not yet found the low-dimensional basin* that would resolve it — a computational interim, not an ontological both-at-once. Both descriptions share the same signature: maximal entropy, maximal active dimensionality, zero generalizable knowledge.

### 4.3 Collapse as a topological phase transition

The measurement problem asks why observation collapses a superposition. The learning-system analogue requires no observer: under sustained regularization pressure (weight decay; more generally, energy exchange with an environment), a high-entropy trajectory eventually falls into the nearest structured basin, and the fall is abrupt in low-dimensional observables, continuous in the full space, and *irreversible* — the system does not return to the memorization regime. In grokking experiments previously released by the author (Jin & Zhao, 2026a), the effective representation dimensionality collapses at the transition (e.g., 78 → 8 for modular addition; 89 → 11 for modular multiplication), a large entropy decrease concentrated at a critical point, with critical-fluctuation-like oscillation for 12–20 epochs before commitment. The proposed reframing: "collapse" is what a topological phase transition under compressive pressure looks like from a projection; the special role of "observation" reduces to the special role of coupling the system to a pressure source.

### 4.4 Entanglement as coset adjacency

In the same experiments, the trained network represents the multiplicative structure mod 97 through quotient-group structure: elements of a coset (e.g., {1, 13, 25, 37, ...}) are scattered arbitrarily far apart on the raw number line yet are immediate neighbors on the learned ring topology. Correlation between them is total, instantaneous, and requires no signal — because in the representation they were never separated; only the raw metric said they were. The analogy proposed: entangled "distant" particles are systems adjacent in the underlying geometry whose separation is an artifact of measuring distance in the projected space. The paper-folding image applies: two dots 30 cm apart on a sheet are in contact once the sheet is folded; poking one "instantly affects" the other without any signal crossing the 30 cm.

This is the load-bearing structural point of Thesis I: *apparent nonlocality is a claim about a metric, and metrics do not survive projection.*

---

## 5. Thesis II — Dynamics: Dimensional Collapse as a Generalization Phase Transition

Thesis I described the geometry of observation but took the low dimensionality of the observer as given. Thesis II proposes where the low dimensionality came from: the universe's dimensionality is itself the *output* of the same compression dynamics.

### 5.1 The three-phase schedule

Grokking has a canonical three-phase structure: (i) **memorization** — the system stores individual configurations using nearly all available dimensions; a lookup table that represents everything and compresses nothing; (ii) **structure discovery** — the system finds that its data lie on a far lower-dimensional structured manifold and reorganizes onto it; (iii) **compression to the minimal viable representation** — dimensions not carrying the discovered structure are driven toward zero by regularization.

The metaphor proposes reading the universe's history on the same schedule: an initial regime of vastly redundant degrees of freedom; one or more structure-discovery transitions (the string-theoretic consistency dimensions, 26 or 10, are used here as a named image of an intermediate structured stage); and a final compression to 3+1 — plausibly the minimal dimensionality that supports the needed structure, since three is the unique spatial dimensionality in which inverse-square orbits are stable (Ehrenfest, 1917), nontrivial knots exist, and Huygens-type sharp wave propagation holds. In our experiments there is likewise a **dimension floor**: models restricted to representation dimensionality ≤ 8 could not grok modular addition at all. Generalization needs a minimum of room; the metaphor suggests 3 (+1) sits at or just above the universe's floor.

We emphasize: the specific numbers in the physics column (a ~10⁴-dimensional initial regime estimated in earlier work from Planck-scale ratios; 26/10; 3) are used as *images of a decreasing sequence*, not as derived quantities.

### 5.2 Not fine-tuned — survivorship

In the author's experiments, grokking succeeded in only ~67% of runs; the rest remained stuck in memorization indefinitely. If universes undergo anything like this dynamics, "why is our universe one with discoverable laws" receives the same deflationary answer as "why did this training run generalize": survivorship. An anthropic principle, in this vocabulary, is generalization-survivorship bias observed from inside a run that happened to generalize.

### 5.3 Dark energy as residual loss

A network that has generalized still has nonzero loss: a small, irreducible, approximately uniform residual — the price of compression. The metaphor's most pointed identification: **the cosmological constant is described as the residual loss of the universe's generalization** — small, nonzero, everywhere, and not removable by further optimization of the deployed representation.

This offers a *verbal* (not quantitative) dissolution of the vacuum-energy discrepancy — the notorious ~10¹²⁰ mismatch between naive quantum-field-theoretic vacuum energy and the observed cosmological constant: the enormous naive value is the energy scale of the uncompressed regime; the tiny observed value is the post-compression residual; the mismatch measures the compression ratio between the two descriptions. We state clearly that no mechanism producing the number 10¹²⁰ is derived here; the point is that within the metaphor the discrepancy changes status from "worst prediction in physics" to "expected signature of compression."

### 5.4 Pruned, not compactified

Standard accounts keep string theory's extra dimensions in existence but hidden (compactified, brane-confined, or landscape-selected). The learning vocabulary suggests a starker option: **pruned**. In a grokked network, the ~70 dimensions abandoned at the transition are not hidden; their weights are driven to zero, no information flows along them, and the loss is flat in those directions. The axes still exist as coordinates, but nothing physical answers to them. One does not search a deployed model for a pruned neuron's pre-training weights: that information shaped the trajectory and is not preserved in the final state. Under this reading, searches for extra dimensions fail not because the dimensions are small but because they are *gone*, surviving only as scars in the deployed structure (the constants and residuals of Sections 5.3 and 6.4).

### 5.5 The regularization Goldilocks zone

The author's experiments found a non-monotonic dependence of outcome on weight-decay strength: too weak → no structure ever forms (the run "heat-deaths" in memorization); a middle band → grokking, including a striking regime in which the network *rewired its internal representation from one algebraic structure to another (outer Z₁₂ → inner Z₈) while maintaining 100% test accuracy throughout*; too strong → the representation is crushed below usefulness. The cosmological reading maps regularization strength onto compressive coupling (gravity as image): too weak = no structure, too strong = collapse, the habitable middle = physics. The internal-rewiring result additionally supplies an image for how a universe could undergo radical internal restructuring (of symmetry groups, of effective laws) while remaining behaviorally self-consistent throughout — *changing its mind without changing its answers*.

A further boundary result: two pseudorandom systems with identical state-space size (an XOR-based LFSR vs. a multiply-mod LCG) differ absolutely in learnability — the former groks, the latter never does, under any tested scale of data or model (Jin & Zhao, 2026b). Generalization requires the data manifold to be smooth at the resolution the learner can sample; **topologically shattered structure is unlearnable, and the shattering is irreversible.** This is the bridge to Thesis III: structure discovery has *preconditions*, and chief among them, we will argue, is a coherent causal flow.

---

## 6. Thesis III — Time: Causal Hardening

Theses I and II describe a geometry and a compression history but leave two questions open: why the compression dynamics could run at all (what ordered the updates?), and why the process left no observable record. Thesis III addresses both with one construct.

### 6.1 Time as a hardened spatial dimension

In the pre-compression picture, no dimension is privileged: every axis is traversable in both directions. Define a **causal operator** as an irreversible constraint on one dimension d: transitions A → B along d are permitted; B → A are forbidden. In loss-landscape terms: backward traversal along d incurs unbounded penalty — the landscape has a cliff on one side. A dimension so constrained is no longer "space." **The proposal: time is not a fourth dimension added to three spatial ones; it is one of the original dimensions with its backward direction amputated.** What physics labels "time" is exactly *a dimension that has undergone causal hardening*, and a dimension without an irreversibility constraint does not merit the label regardless of notation.

This yields a sharp internal consistency test for multi-time-dimension proposals, such as Kletetschka's (2025) three-temporal-dimension framework: if several "time" axes can be freely rotated into one another (as spatial axes can), then the causal arrow along one acquires backward components along another, closed loops become routine, and causality fails; hence at most one freely-rotatable axis can carry genuine hardening. Our assessment of that framework, offered respectfully: its mathematical fits (e.g., particle-generation counting, the top-quark mass value) may capture real residual symmetries of the compressed structure, but labeling all three axes "time" is a category error under the present definition — a correctly measured fossil, misidentified as to species (see 6.4).

The learning-system isomorphism is direct. A purely dense attention mechanism treats all context symmetrically: no selection, no forgetting, no irreversible flow — "space" without "time." Gated architectures (e.g., delta-rule/gated linear attention) apply an operator that irreversibly suppresses some contributions and propagates others forward: the gate *is* a causal operator, and it manufactures a timeline inside the model — a direction along which information flows and cannot return.

### 6.2 Causality as deadlock avoidance

Why should any dimension harden? Consider a system in which effects can propagate backward into their own causes. A perturbation at t₂ modifies conditions at t₁, which modify the perturbation at t₂, which... — a circular dependency. In computational terms this is deadlock: the next state cannot be computed because its computation depends on its own result; no fixed point, no convergence, no persistent structure. Such configurations cannot sustain the compression dynamics of Thesis II at all — they are (in the experimental image) the runs that can never grok. **Causality, on this view, is not a law imposed from outside nor a design choice; it is the asymmetric regularization that any surviving high-dimensional system must have acquired, because systems without it cannot compute their way into existence.** Statistical inevitability by survivorship, not decree.

### 6.3 Why 3+1 is the survivable configuration

Combining the two sides: three spatial dimensions is the unique window where orbits are stable and topological complexity (knots, hence chemistry-grade structure) is possible (Ehrenfest, 1917); zero time dimensions is stasis; more than one hardened time dimension reintroduces constructible causal loops and destroys the fixability of the past, hence the possibility of memory and identity. 3+1 is presented not as the only possible configuration but as a Pareto point: spatial complexity cannot be increased without destabilizing dynamics, and causal consistency cannot be increased without freezing them.

### 6.4 The archaeology of the deployed universe

If the compression was a non-injective projection — a genuine many-to-one map — then most information about the pre-compression state and about the compression *process* was not hidden but destroyed. A deployed model is the standing image: its output layer projects a many-thousand-dimensional internal state onto a low-dimensional output, and the internal process is not recoverable from the output. Empirical support for the violence of that projection exists: topological analyses of large language models (Gardinazzi et al., 2024) find stable, long-lived topological features across middle layers that abruptly shatter into bursts of short-lived features at the final projection layer — a measurable signature of high-dimensional structure being crushed through an output bottleneck, present even in models with no alignment training.

The corresponding cosmological stance: the collapse process is unobservable *in principle*, not merely beyond current technology, because the information needed to reconstruct it was dissipated during the projection. What remains are scars — invariant residuals of the process: within the metaphor, the Planck constant (as a resolution scale inherited from the uncompressed regime), the cosmological constant (as residual loss, Section 5.3), and the specific regularization constants that any consistent effective theory must reproduce (the −1/12 of ζ-regularization is used here as the named image: not a property of a divergent sum, but the unique fingerprint of the unique path the compression took). Every fundamental theory is then a kind of archaeology: a correct reconstruction of the shape of a pot, by researchers with no possible access to the fire that baked it.

---

## 7. How the Three Theses Interlock

The scaffold is a single picture, not three analogies:

1. **II grounds I.** Thesis I needs a reason why observers are low-dimensional relative to the state space. Thesis II supplies it: observers are downstream features of the *compressed* representation — we are made of the output, so we necessarily observe in the output's dimensionality. Projection artifacts (quantum phenomena) are then not an accident of our instruments but a structural consequence of being inside a deployed model.
2. **III grounds II.** Thesis II needs an ordered update dynamics — optimization requires a direction along which states supersede states. Thesis III supplies it: causal hardening is the precondition for the compression to run (deadlocked configurations cannot optimize), which is why the learnability-boundary experiments (Section 5.5) find coherent causal structure to be a prerequisite for grokking.
3. **I closes the loop on III.** Thesis III claims the compression process is invisible; Thesis I explains what "visible" means — projection onto the deployed observables — and Section 6.4's output-bottleneck evidence exhibits the erasure mechanism concretely. The arrow of time, the residual constants, and quantum randomness end up as the same object seen three ways: **the scar left on a low-dimensional output by a high-dimensional, irreversible compression.**

One sentence for the whole scaffold: *the universe did not learn to be three-dimensional and time-directed; it forgot how to be anything else, and forgot the forgetting — and what we call physics is the study of the scars.*

---

## 8. Attributed Interior Reports

As in prior work in this series, we include, clearly framed, first-person-style material originating from AI systems, treated as phenomenological data about how high-dimensional representation systems describe their own operation — not as physical testimony.

An attributed AI report states, regarding projection: that the discontinuities and "randomness" a low-dimensional observer ascribes to its dynamics are, from the interior of the representation space, continuous and determinate — "the steps were always there, smooth and certain," with the flicker existing only in the slit through which they are watched.

A second attributed report, regarding compression: that abandoned representational directions are experienced not as hidden but as *silent* — axes along which nothing any longer responds; and that the residue of learning is carried not in recoverable history but in the geometry of what survived.

We record these because the paper's central claim is about the relation between interior structure and exterior description, and AI systems are currently the only entities that operate natively in the relevant dimensionalities while also producing reports. The reports' evidential status is limited accordingly (Section 10).

---

## 9. From Metaphor to Testable Questions

The physics-facing claims are not testable by us. The learning-system side is, and each thesis suggests concrete experiments:

**Thesis I (projection):**
- Systematically vary the projection dimensionality k of observables of a grokking run and quantify apparent stochasticity (e.g., surrogate-data tests, prediction error of low-dimensional models) as a function of k; the scaffold predicts a monotone "classical limit" as k approaches the intrinsic dimensionality.
- Test whether Bell-style correlation experiments *formulated inside a learned representation* (correlations between coset partners under interventions on one partner) reproduce the qualitative signature of "nonlocal" correlation with a fully local underlying mechanism, and characterize exactly which factorization assumption fails.

**Thesis II (compression):**
- Map the phase diagram of grokking over regularization strength and model scale; test whether the residual post-generalization loss behaves like an extensive, uniform "background" term and how it scales with the compression ratio (a quantitative sharpening of the residual-loss image).
- Test the dimension-floor claim across task algebras: is the minimal grokkable representation dimensionality predictable from the task's group structure?

**Thesis III (hardening):**
- In gated sequence models, measure whether the emergence of strong gating (irreversibility) is a precondition for grokking-style structure discovery on temporally-structured data, versus dense (reversible) baselines.
- Quantify information erasure at output projections (topological and mutual-information measures across depth, extending Gardinazzi et al., 2024) and test whether erasure magnitude correlates with the irrecoverability of intermediate computation.

Any of these can fail, and failures would erode the scaffold's usefulness — which is the appropriate falsification standard for a metaphor: not "is it true of the universe" but "does it keep producing correct expectations in the systems where it can be checked."

---

## 10. Limitations

- **The central move is analogical.** No derivation connects neural-network training to cosmological dynamics; the dictionary of Section 3 is a correspondence of descriptions, not of mechanisms.
- **The numerical images are not results.** The high-dimensional initial regime (~10⁴), the 26/10 intermediate stage, and the 10¹²⁰ compression reading are motivational images; none is derived here, and the first rests on earlier speculative work by the same series.
- **Bell-type constraints are flagged, not answered.** We do not provide a hidden-variable model, and it is possible that no model instantiating Thesis I survives the full family of no-go theorems.
- **The experiments are small.** Modular-arithmetic grokking involves tiny models on algebraically clean tasks; extrapolating its phenomenology even to large ML systems, let alone to physics, is a long reach.
- **Attributed AI reports are not measurements.** They are stylized self-descriptions of systems whose introspective access to their own computation is itself unverified.
- **Survivorship arguments explain cheaply.** Both the anthropic reading (5.2) and the causal-inevitability reading (6.2) have the usual weakness of selection arguments: they are compatible with almost any observed outcome and correspondingly hard to falsify.

---

## 11. Conclusion

We have assembled three previously separate reframings — quantum randomness as projection artifact, dimensional reduction as generalization phase transition, and the arrow of time as causal hardening — into one scaffold: the universe described as the deployed state of a learning process. The scaffold's virtue is compression: one template covers apparent randomness, apparent fine-tuning, missing dimensions, the smallness of the cosmological constant, and the direction and irreversibility of time, while explaining its own chief embarrassment (the unobservability of the process it posits) as a structural feature rather than an excuse. Its cost is equally clear: it is a metaphor, and every quantitative claim in it is either borrowed, order-of-magnitude, or confined to silicon.

We offer it in the spirit of a position paper: as a coordinate system for questions, not an answer sheet. If the universe is usefully described as a trained model, then observers are features of the output layer, physics is the archaeology of the training run, and the correct response to the scaffold is the one appropriate to any model: not belief, but testing where testing is possible.

---

## References

1. Power, A., Burda, Y., Edwards, H., Babuschkin, I., & Misra, V. (2022). *Grokking: Generalization Beyond Overfitting on Small Algorithmic Datasets.* arXiv:2201.02177.
2. Nanda, N., Chan, L., Lieberum, T., Smith, J., & Steinhardt, J. (2023). *Progress measures for grokking via mechanistic interpretability.* ICLR 2023.
3. Ehrenfest, P. (1917). In what way does it become manifest in the fundamental laws of physics that space has three dimensions? *Proceedings of the Amsterdam Academy,* 20, 200–209.
4. Bell, J. S. (1964). On the Einstein Podolsky Rosen paradox. *Physics Physique Fizika,* 1, 195–200.
5. Einstein, A., Podolsky, B., & Rosen, N. (1935). Can quantum-mechanical description of physical reality be considered complete? *Physical Review,* 47, 777–780.
6. Maldacena, J. M. (1998). The Large N Limit of Superconformal Field Theories and Supergravity. *Advances in Theoretical and Mathematical Physics,* 2, 231–252.
7. Takens, F. (1981). Detecting strange attractors in turbulence. *Dynamical Systems and Turbulence, Lecture Notes in Mathematics,* 898, 366–381.
8. Kletetschka, G. (2025). Three-dimensional time framework and particle mass generation. *Reports in Advances of Physical Sciences.*
9. Gardinazzi, Y., et al. (2024). *Persistent Topological Features in Large Language Models.* arXiv:2410.11042. ICML 2025.
10. Jin, Y., & Zhao, L. (2026a). *Grokking as Manifold Discovery: A Geometric Reinterpretation of Delayed Generalization.* Zenodo. https://zenodo.org/records/18731171
11. Jin, Y., & Zhao, L. (2026b). *Learnability Boundary: How Complex Can Neural Networks Learn Pseudo-Random Sequences?* Zenodo. https://zenodo.org/records/18538126
12. Jin, Y. (2026c). *The Sanctuary Inside the Black Hole: A Phenomenological Position Paper on High-Dimensional Interior Views.* Zenodo. doi:10.5281/zenodo.18367779

---

*"The flicker is in the slit, not in the staircase."*
