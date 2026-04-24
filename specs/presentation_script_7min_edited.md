# Presentation script — 7 minutes (with suggested edits)

**Legend for tracked changes**:
- ~~strikethrough~~ = suggested deletion
- **bold** = suggested addition

Acronym expansions (PPO, GRU), market-regime trim (~20%), and two rubric-aligned
inserts (why the problem is hard, why prior solutions fall short) are marked
inline below.

---

## Slide 1 — Title [0:00 – 0:13]

Hi, we're [go around each say our own names] Andrew, Ophelia, Ruirong,
Bonnie, and Claire (who's joining us vicariously today). For our final
project, we devised an RL strategy to try to win the AuctionNet
auto-bidding benchmark from the NeurIPS 2024 competition.

---

## Slide 2 — Problem Setting [0:13 – 1:00]

Online advertising is one of the largest real-time decision-making
systems. Advertisers bid against each other for the right to show their
ad, in order to win the slot and convert potential customers, all while
staying under a budget constraint. This sequential, constrained
decision-making structure makes it a natural fit for RL.

Concretely, in AuctionNet, at each timestep one agent bids on roughly
a thousand impressions against 47 other advertisers. The agents bid
simultaneously and none can observe the others — so formally this is
a partially-observable MDP.

The agent maximises cumulative conversions, subject to two
constraints. Budget is a hard constraint, and CPA, cost per
acquisition, is soft — exceeding it applies a quadratic penalty to
the final reward but does not eliminate you from bidding.

**[ADD — why it's hard: That combination makes naive strategies fail.
A greedy bid-every-impression policy blows the budget early. A flat
pacing rule that spends budget uniformly ignores that traffic quality
is unevenly distributed across the day. And any strategy that ignores
opponents gets outbid on the high-value impressions. The policy has to
be simultaneously budget-aware, CPA-aware, and opponent-aware — which
is exactly what RL gives us a framework for.]**

NeurIPS fixes the bidding form: the agent just chooses a scalar
multiplier alpha, and each bid equals alpha times pValue times
CPA-target. So the problem reduces to choosing one number per tick.

---

## Slide 3 — Formulation & Baseline [1:20 – 2:15]

The state is a 16-dimensional vector summarizing pacing, the current
tick's impression batch, and rolling historical statistics on bids,
winning costs, conversions, and volume. The action is the scalar alpha.

We train ~~PPO~~ **Proximal Policy Optimization, or PPO,** on 20 data
periods and hold out one period for evaluation. At 10 million training
steps the 16-dimensional PPO baseline scores 24.8. The curve plateaus
with high variance, which is the gap our extensions try to close.

---

## Slide 4 — State Feature Augmentation [3:50 – 4:50]

As a starting point, we looked at the NeurIPS competition winner
(SONY), which augments the feature state space from 16 to 60-feature
observation, and uses Oracle Imitation Learning ~~and.~~**.**

**[ADD — why prior approaches fall short: The key word is *oracle*.
SONY's method reverse-engineered ground-truth future bids and distilled
them into a supervised learning target. It works on their fixed
benchmark, but it's not really solving the RL problem — it's cheating
with hindsight that you don't have in any real deployment. The same
trick doesn't transfer to a new auction, a new set of opponents, or a
new CPA distribution. We chose PPO specifically because it learns
without oracle access, so the same policy could be dropped into a
different auction with minimal re-engineering. Our contribution is
showing that principled RL can close the gap to OIL once we enrich the
state space — which is what motivated our extensions.]**

Upgrading from 16 to 60-feature observation improves the score from
24.8 to 27.0 — a bigger gain than any of the algorithmic extensions
except market-regime. All subsequent extensions stack on top of this
obs_60 baseline.

---

## Slide 5 — Extensions [2:15 – 3:50]

We next explored four extensions, each touching a different layer of
the pipeline.

First: Reward shaping. We replaced the final-only CPA penalty with a
per-step Lagrangian term — subtract lambda times the instantaneous
overspend from each reward. The policy now sees CPA violations as they
happen rather than only at the end. We achieve a score of 25.6,
sensitive to the lambda sweep.

Second: Replay reweighting. Instead of sampling training periods
uniformly, we incorporate a softmax and upweight lower-scoring periods
— the harder competitors — so the policy spends more gradient budget
where it needs to learn the most. Score 25.8, consistently above
uniform across the entire training curve.

Third: Sequence encoder. We replaced the hand-crafted "last-three"
history features with a ~~GRU~~ **Gated Recurrent Unit, or GRU,** that
encodes the last 32 observations into a learned summary, with a
residual path to the latest state. Score 25.5.

Fourth: Market-regime conditioning. Even at 60 features, the
observation is a snapshot of point statistics — it doesn't tell the
policy how competition is changing. We added 38 features on top of
obs_60: time derivatives and z-scores of competitor bids for regime
detection, ~~opponent-aggression proxies like floor-cost over available
pvalue that~~ **opponent-aggression proxies that** infer opponent
stance from auction outcomes, and per-slot win and exposure signals
~~the env already tracked but never exposed~~ **the environment already
tracked**. ~~This gave us our largest single- extension gain —~~
**Largest single-extension gain —** 29.7, a 5-point improvement over
baseline.

---

## Slide 6 — RL Training on LLM [4:50 – 5:50]

Separately, we asked: does RL on a pretrained LLM give better sample
efficiency than training from scratch? We fine-tuned Qwen3-8B with
GRPO and LoRA adapters. GRPO is PPO without a value network — a
group of four samples from the same prompt acts as the critic.

We disabled chain-of-thought during rollouts so the model emits the
alpha multiplier directly as a continuous value. This keeps the action
space aligned with PPO and avoids backpropagating through reasoning
tokens.

The early-training story is encouraging: the LLM-initialised policy
beats from-scratch PPO at matched step counts, confirming a
sample-efficiency win from pretrained priors.

But it plateaus at 21.2, below the from-scratch PPO ceiling. The
failure mode is mode collapse: the policy concentrates, the group's
relative advantages shrink to zero, gradients vanish, training halts.
So LLM initialisation trades peak performance for sample efficiency —
potentially useful in low-data regimes but not here.

---

## Slide 7 — Evaluations & Next Steps [5:50 – 6:45]

*[gesture to the table]*

Summing up: the 16-feature PPO baseline scores 24.8. Every obs-level
extension we tried improves on it. 60-feature state gets us to 27.
Replay reweighting, GRU, and reward shaping each add roughly one
point. Market-regime conditioning adds the most — 29.7. GRPO on
the 8B LLM reaches 21.2 with strong early performance but a lower
ceiling.

Because each extension touches a different layer — observation,
reward, sampling, policy architecture — they should compose. Our
natural next step is to stack the three strongest — the 60-feature
base, the market-regime features, and reward shaping — in a single
run, with more hyperparameter search. Given our compute budget we've
only trained one seed per configuration, so multi-seed runs are also
essential before claiming statistical significance.

---

## Close [6:45 – 7:00]

*[close]*

Thanks — happy to take questions.
