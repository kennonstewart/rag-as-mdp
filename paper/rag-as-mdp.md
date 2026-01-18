
### RAG as a Partially-Observed Markov Decision Process
### Thesis
This paper argues that **agentic, multi-hop retrieval-augmented generation (RAG)** can be formalized as an **optimal control problem**: a **(stopping-time) partially-observed Markov decision process (POMDP)** over a **discrete, packetized information state**. In this view, “reasoning” is a sequence of actions that refine an external reasoning substrate until a **stopping rule** decides it is sufficient to return an answer.

### Motivation
RAG systems are often evaluated primarily on answer quality, but practical deployments also face:
- **Uncertainty**: the agent does not fully observe the “true” minimal evidence needed to answer.
- **Cost**: retrieval, reflection, tool calls, and answer generation each incur latency/compute/token costs.
- **Stopping**: multi-hop pipelines must decide **when to stop reasoning** and return.

The paper proposes **entropy / uncertainty reduction** as a natural lens for “sufficiency” (when the produced response is good enough relative to an unobserved latent ideal answer), while explicitly incorporating action costs into the objective.

### Claimed contributions
The draft lists three main contributions:
- **RAG as an MDP with optimal stopping**: frame agentic RAG as an MDP (more precisely, a POMDP) over packetized information state \(Z_t\), with an **optimal stopping time** \(\tau\).
- **Reasoning substrate as (approximate) information state**: define the evolving substrate (e.g., evidence graph) as the agent’s decision-relevant state representation.
- **Unifying framework**: reinterpret disparate multi-hop / agentic RAG methods as instances of **choosing actions to refine \(Z_t\)** and **choosing when to stop**.

### Core formalization

### Objects and notation
- **Query**: \(q\).
- **Reasoning substrate / evidence**: an evolving artifact, often modeled as an evidence graph \(G_t\) (or \(E_t\) in parts of the draft).
- **Confidence / sufficiency summary**: \(c_t\) (the draft also uses a user-provided confidence level \(\varepsilon\)).
- **Packetized agent-state**:
  \[
  Z_t = (q, G_t, c_t)
  \]
  The intent is that **all decision-relevant information lives inside \(Z_t\)** rather than in a hidden internal agent memory.

### Agent-state based policies and information states
The draft distinguishes:
- **Agent-state based policy**: actions are chosen as \(A_t = \pi_t(Z_t)\) (or \(A_t \sim \pi_t(Z_t)\)).
- **Information state**: an agent-state \(Z_t\) is an information state if it is sufficient for:
  - **prediction**: \(P(Z_{t+1}\mid H_t, A_t) = P(Z_{t+1}\mid Z_t, A_t)\)
  - **evaluation**: \(\mathbb{E}[R_t\mid H_t, A_t] = \mathbb{E}[R_t\mid Z_t, A_t]\)

This connects decision-theoretic sufficiency to an information-theoretic perspective: if \(Z_t\) is decision-theoretically sufficient, it is also sufficient (in the draft’s sense) for preserving the relevant information about future states and rewards.

### Partial observability
The “true” minimal sufficient evidence/state is treated as **unobserved**; the agent operates under noisy/incomplete observations. Rather than requiring exact Bayesian belief-state filtering (which is continuous and can require perfect model knowledge), the paper emphasizes **agent-state based policies** that can be **approximate** but operationally convenient.

### Actions
The draft suggests a discrete action set such as:
\[
A_t \in \{\texttt{retrieve}, \texttt{reflect}, \texttt{return}\}
\]
where retrieval augments evidence, reflection consolidates/compresses, and return terminates with an answer.

### Reward and uncertainty reduction
For question-answering, reward is tied to **entropy reduction** across steps (draft notation):
\[
\Delta_t = H_{t+1} - H_t
\]
(As written, this is “next minus current”; the intended notion appears to be *uncertainty reduction*, i.e., negative \(\Delta_t\) or \(-\Delta_t\) as a reward signal.)

### Costs and the control objective
The paper emphasizes that actions have real costs (tokens, latency, compute, database calls). A representative objective balances solution quality vs. cumulative costs up to stopping time \(\tau\):
\[
\mathbb{E}\Big[u(G^*, G_\tau) - \lambda \sum_{t<\tau} c(A_t)\Big]
\]
where \(G^*\) is an ideal (latent) evidence state, \(u(\cdot)\) measures closeness/utility, and \(c(\cdot)\) is an action cost.

### Stopping time as the central decision
A key claim is that RAG is fundamentally about selecting **when to stop**:
- The agent **implicitly chooses** \(\tau\) (stopping time).
- The stopping rule is shaped by the user’s confidence requirement \(\varepsilon\) and the system’s uncertainty/entropy about the latent correct answer, conditioned on the generated answer and current evidence.

### Systems implication: stateless “worker interchangeability”
Because \(Z_t\) is intended to contain all decision-relevant content, the executing model/worker can be treated as a **stateless policy evaluator** over packets:
- Workers can be swapped across steps without changing semantics (under the sufficiency assumptions).
- Execution can be distributed/queued to optimize cost/latency without changing the induced policy behavior.

### Relationship to prior work (as positioned in the draft)
- **RAG over artifacts / graphs**: existing work uses intermediate substrates (embeddings, graphs, distributed summaries). The paper’s stated gap is that these systems often do not formalize **when** an evolving substrate is sufficient (as an information state) nor analyze interchangeability/Markov sufficiency.
- **ReAct-style agentic reasoning**: positioned as less constrained (action space/state not formalized), making it harder to do clean control-theoretic ablations or state/topology analysis.

### Proposed experiments (sketched)
The draft proposes ablations and diagnostics meant to test the control framing:
- **Fixed horizon**: run exactly \(k\) reasoning steps, vary \(k\).
- **Entropy-based stopping**: maintain (in a synthetic environment) a posterior over candidate answers; stop when entropy < threshold.
- **Posterior analysis stopping**: alternative sufficiency criteria based on posterior structure.
- **ROI/value-based stopping**: compare \(\max_a Q(Z_t,\texttt{return})\) vs. \(\max_a Q(Z_t,\texttt{retrieve})\) (e.g., via fitted value iteration).
- **Change/compress the information state**: replace \(G_t\) with a compressed \(\nu(G_t)\) (e.g., top-\(k\) nodes) to test whether the substrate is truly the decision-theoretic object.
- **Worker interchangeability**: swap workers each iteration; compare induced action distributions and outcomes.

Suggested plots/metrics include:
- **Sufficiency gap**:
  \[
  \Delta_t = \big| \mathbb{E}[R_t\mid H_t, A_t] - \mathbb{E}[R_t\mid Z_t, A_t]\big|
  \]
- **Self-prediction gap**: discrepancy between \(P(Z_{t+1}\mid H_t, A_t)\) and \(P(Z_{t+1}\mid Z_t, A_t)\) (e.g., TV/KL).
- **Compression loss curve**: performance vs. compression rate \(|\tilde G_t|/|G_t|\).
- **Accuracy–cost frontier**: answer accuracy as a function of total cost to reach stopping time.

### What the paper is trying to achieve (high level)
The draft’s overarching goal is to make RAG “legible” as a control problem:
- **Make reasoning trajectories analyzable** (state, action, reward, cost, stopping).
- **Define and test sufficiency** of an external reasoning substrate as an information state.
- **Enable principled ablations** and comparisons between multi-hop strategies.
- **Connect practical engineering choices** (distributed workers, tool costs, retrieval vs reflection) to a coherent mathematical framework.

### Notes on draft status
The LaTeX source contains placeholders and incomplete definitions (e.g., stopping-time POMDP section stub, partial tuple definitions), but the central framing is consistent: **RAG as a (P)OMDP over a packetized evidence state with explicit stopping under cost/uncertainty tradeoffs**.

