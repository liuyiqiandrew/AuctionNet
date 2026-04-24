# Presentation script — 7 minutes

Matches the 7-slide deck `COS435_FinalProject_Presentation (2).pdf`. Delivery
at ~130 wpm. Italicised lines are stage directions, not spoken. Bracketed
times are cumulative.

---

## Slide 1 — Title [0:00 – 0:15]

> Hi, we're Ruirong, Ophelia, Claire, Bonnie, and Andrew. For our final
> project, we devised an RL strategy to try to win the AuctionNet
> auto-bidding benchmark from the NeurIPS 2024 competition.

---

## Slide 2 — Problem Setting [0:15 – 1:20]

> Online advertising is one of the largest real-time decision-making
> systems in the world. Every time you load a webpage, advertisers bid
> against each other in a split-second auction for the right to show you
> their ad. For the advertiser, this is a hard optimisation problem:
> spend a fixed budget across a whole day, win the impressions that
> actually convert into customers, and stay under a cost-per-acquisition
> ceiling. That sequential, constrained decision-making structure is a
> natural fit for reinforcement learning.
>
> Concretely, in AuctionNet, at each timestep one agent bids on roughly
> a thousand impressions against 47 other advertisers. 48 agents bid
> simultaneously and none can observe the others — so formally this is
> a partially-observable MDP.
>
> The agent maximises cumulative conversions, subject to two
> constraints. **Budget is a hard constraint**, and **CPA, cost per
> acquisition, is soft** — exceeding it applies a quadratic penalty to
> the final reward.
>
> NeurIPS fixes the bidding form: the agent just chooses a scalar
> multiplier `alpha`, and each bid equals `alpha times pValue times
> CPA-target`. So the RL problem reduces to choosing one number per tick.

---

## Slide 3 — Formulation & Baseline [1:20 – 2:15]

> The state is a 16-dimensional vector summarising pacing — time left,
> budget left — the current tick's impression batch, and rolling
> historical statistics on bids, winning costs, conversions, and
> volume. The action is the scalar `alpha`.
>
> We train PPO on 20 data periods and hold out one period for
> evaluation. At 10 million training steps the 16-dimensional PPO
> baseline scores **24.8**. The curve plateaus with high variance,
> which is the gap our extensions try to close.

---

## Slide 4 — Extensions [2:15 – 3:50]

> We explored four extensions, each touching a different layer of the
> pipeline.
>
> **One: Reward shaping.** We replaced the final-only CPA penalty with a
> per-step Lagrangian term — subtract lambda times the instantaneous
> overspend from each reward. The policy now sees CPA violations as
> they happen rather than only at the end. Score **25.6**, sensitive to
> the lambda sweep.
>
> **Two: Replay reweighting.** Instead of sampling training periods
> uniformly, we upweight lower-scoring periods — the harder advertisers —
> so the policy spends more gradient budget where it needs to learn.
> Score **25.8**, consistently above uniform across the entire training
> curve.
>
> **Three: Sequence encoder.** We replaced the hand-crafted "last-three"
> history features with a GRU that encodes the last 32 observations into
> a learned summary, with a residual path to the latest state. Score
> **25.5**.
>
> **Four: Market-regime conditioning.** Even at 60 features, the
> observation is a snapshot of point statistics — it doesn't tell the
> policy how competition is *changing*. We added 38 features on top of
> obs_60: time derivatives and z-scores of competitor bids for regime
> detection, opponent-aggression proxies like floor-cost over available
> pvalue that infer opponent stance from auction outcomes, and per-slot
> win and exposure signals the env already tracked but never exposed.
> No change to auction dynamics. This gave us our largest single-
> extension gain — **29.7**, a 5-point improvement over baseline.

---

## Slide 5 — State Feature Augmentation & Comparison to OIL [3:50 – 4:50]

> The underlying theme here is that the default 16-feature observation is
> just too small — it doesn't even include the CPA target, so the policy
> literally doesn't know its own constraint.
>
> The NeurIPS competition winner, **OIL — Oracle Imitation Learning** —
> noticed the same thing. They used a 60-feature observation and
> reverse-engineered ground-truth to distil future-state information
> into the policy. That works in their fixed setting but doesn't
> transfer — it's tailored to one known bidding scenario. **We chose PPO
> instead** because it's a generalisable RL approach: it doesn't rely on
> oracle access to ground truth, so the same learned policy could be
> dropped into a different auction with minimal re-engineering. RL
> barely outperforms OIL on their own benchmark, but the gap closes as
> we enrich the state space — which is what motivated our extensions.
>
> Upgrading to the 60-feature observation is a pure config change — the
> environment already computes all 60 values; we just add them to the
> JSON key list. Score goes from **24.8 to 27.0** — a bigger gain than
> any of the algorithmic extensions except market-regime. All subsequent
> extensions stack on top of this obs_60 baseline.

---

## Slide 6 — RL Training on LLM [4:50 – 5:50]

> Separately, we asked: does RL on a pretrained LLM give better sample
> efficiency than training from scratch? We fine-tuned **Qwen3-8B** with
> **GRPO and LoRA adapters**. GRPO is PPO without a value network — a
> group of four samples from the same prompt acts as the critic.
>
> We disabled chain-of-thought during rollouts so the model emits the
> alpha multiplier directly as a continuous value. This keeps the action
> space aligned with PPO and avoids backpropagating through reasoning
> tokens.
>
> The early-training story is encouraging: the LLM-initialised policy
> beats from-scratch PPO at matched step counts, confirming a sample-
> efficiency win from pretrained priors.
>
> But it plateaus at **21.2**, below the from-scratch PPO ceiling. The
> failure mode is mode collapse: the policy concentrates, the group's
> relative advantages shrink to zero, gradients vanish, training halts.
> So LLM initialisation trades peak performance for sample efficiency —
> potentially useful in low-data regimes but not here.

---

## Slide 7 — Evaluations & Next Steps [5:50 – 6:45]

> *[gesture to the table]*
>
> Summing up: the 16-feature PPO baseline scores 24.8. Every obs-level
> extension we tried improves on it. 60-feature state gets us to 27.
> Replay reweighting, GRU, and reward shaping each add roughly one
> point. Market-regime conditioning adds the most — **29.7**. GRPO on
> the 8B LLM reaches 21.2 with strong early performance but a lower
> ceiling.
>
> Because each extension touches a different layer — observation,
> reward, sampling, policy architecture — they should compose. Our
> natural next step is to stack the three strongest — the 60-feature
> base, the market-regime features, and reward shaping — in a single
> run, with more hyperparameter search. Given our compute budget we've
> only trained one seed per configuration, so multi-seed runs are also
> essential before claiming statistical significance.

---

## Close [6:45 – 7:00]

> *[close]*
>
> Thanks — happy to take questions.

---

## Pacing table

| Slide | Minutes | Cumulative |
|---|---|---|
| 1 Title | 0:15 | 0:15 |
| 2 Problem | 1:05 | 1:20 |
| 3 Baseline | 0:55 | 2:15 |
| 4 Extensions | 1:35 | 3:50 |
| 5 State aug / OIL | 1:00 | 4:50 |
| 6 LLM | 1:00 | 5:50 |
| 7 Eval | 0:55 | 6:45 |
| close | 0:15 | **7:00** |
