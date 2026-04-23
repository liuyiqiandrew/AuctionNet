# Presentation script — RL for Large-Scale Auto-Bidding Challenge

**Total target: ~15 minutes.** Speaker notes follow each slide. Time budget per
slide is in brackets — use as a pacing guide, not a hard constraint. Lines in
*italics* are stage directions. Bracketed names like *[Bonnie]* mark suggested
speakers when a slide spans multiple authors; rearrange to fit your team's
division of labour.

---

## Slide 1 — Title [0:30]

*[any speaker — opens]*

> Hi everyone. We're presenting our COS 435 final project: Reinforcement
> Learning for Large-Scale Auto-Bidding. Our team is Ruirong, Ophelia, Claire,
> Bonnie, and Andrew. We tackled the AuctionNet competition from NeurIPS — a
> benchmark where you build an RL agent that bids on ad slots in a 48-step
> auction game against 47 other advertisers. Over the next 15 minutes we'll
> walk you through the problem setup, the baseline, our five RL extensions,
> and a separate experiment fine-tuning an 8B language model to bid
> end-to-end.

---

## Slide 2 — Motivation & Background [2:00]

*[any speaker]*

> AuctionNet is a multi-agent ad-auction benchmark. There's an advertiser
> with a budget — say ten thousand dollars — and a CPA cap, "cost per
> acquisition" — say twelve dollars per conversion. They want to spend their
> budget across a delivery period of 48 timesteps and maximise the number of
> conversions while staying under the CPA cap.
>
> At each of those 48 steps, the agent sees a batch of impression
> opportunities. Each impression comes with a predicted conversion
> probability — `pValue` — and an uncertainty estimate. The agent picks a
> single scalar `alpha`, and bids `alpha times pValue` for every impression
> in the batch. Those bids enter a generalised second-price auction with the
> 47 other advertisers. Top three bidders win the three ad slots.
>
> *[gesture to the right column]*
>
> Two hard constraints. Budget — total spend can't exceed your wallet. CPA —
> total cost over total conversions can't exceed your target. Violate the
> CPA, and your score gets multiplied by a heavy penalty: the ratio of your
> target CPA to actual CPA, squared. So a 50% over-spend slashes your score
> by 75%.
>
> The hard part is that the reward function couples bidding aggression,
> pacing across time, and the unknowable behaviour of 47 opponents — all
> under a final-period score that you don't see until the end.

---

## Slide 3 — Problem Formulation & Baseline [2:00]

*[any speaker]*

> The standard AuctionNet benchmark gives you a 16-dimensional observation
> vector: time fraction, budget fraction, average bid, average winning
> cost, conversion rates, win rates, traffic volume — each as historical
> averages and last-three-tick views.
>
> The action is a single non-negative scalar — the bid coefficient
> `alpha`. The bid for each impression is `alpha times pValue`.
>
> *[click to highlight the orange "Key issue" line]*
>
> The catch: with only a 16-dim summary of history, the policy doesn't see
> enough state to be Markovian. So we're really solving a partially
> observable MDP, but pretending it's an MDP.
>
> Our baseline is a two-stage PPO trained on periods 7 through 26: a five-
> epoch behaviour-cloning warm-start, then ten million online-PPO steps in
> the simulator. We hold out period 27 for evaluation. The 16-dim baseline
> reaches a score of about 24 to 25, but the training curve plateaus and
> has high run-to-run variance.

---

## Slide 4 — Extension: Observation Augmentation (16 → 60) [2:00]

*[any speaker, ideally [Andrew] or [Ruirong] who built the env]*

> Our first extension — and the foundation everything else builds on —
> upgrades the observation vector from 16 to 60 features. The recipe is
> from a paper called OIL: instead of summarising the auction with a
> handful of statistics, expose the full set of constraint-aware features
> the env already tracks. CPA constraint, absolute budget, per-step cost
> and conversion history, market signals — bid-over-winning-cost ratios,
> per-slot cost statistics, distribution percentiles of competitor bids,
> and so on.
>
> Three reasons this should help. One, the policy now sees the CPA target
> directly, so it can learn constraint-aware bidding. Two, more history
> features bring the POMDP closer to an MDP. Three, the per-step pacing
> signals let the policy learn budget velocity.
>
> *[gesture to the chart]*
>
> The chart shows obs_60 in red versus obs_16 in blue, both trained for
> ten million steps. obs_60 climbs higher and the loss curves look
> healthier. The PPO loss is more negative — the policy is making bigger
> updates — and the entropy stays higher, indicating better exploration.
>
> So 60-dim obs is our new baseline going forward. Every subsequent
> extension is layered on top of obs_60.

---

## Slide 5 — Extensions, Part One [3:30]

*[four ~50-second blocks, one speaker per quadrant]*

### Quadrant 1 — Budget-aware pacing + action gating [0:50]

*[Ray / speaker]*

> First extension: a hard action gate that scales the bid coefficient down
> when the remaining budget fraction is low. Concretely, if you've burned
> through 90% of your budget but only 50% of your time, the gate clips
> alpha so you can't keep over-spending. It's a deterministic safety
> wrapper that prevents the over-spend tail of episodes that ruin the
> CPA score. Training curves show a much tighter spread of cost-over-
> budget around the target.

### Quadrant 2 — Replay reweighting [0:50]

*[Claire / speaker]*

> Second: when we sample which advertiser to train on, instead of uniform
> sampling we use a softmax over each advertiser's historical NeurIPS
> score. So the policy sees the harder advertisers more often — those
> with tighter CPA constraints, weirder budget profiles, or different
> categories. The mixing weight `alpha` blends the score-weighted
> distribution with uniform.
>
> *[gesture to the chart]*
>
> Blue is weighted with T=30, alpha=0.9. Grey is uniform. Across the
> full 10M steps, the weighted run consistently outperforms by about
> two score points, and learns faster in the first 4M steps.

### Quadrant 3 — Sequence encoder (GRU) [0:50]

*[Andrew / speaker]*

> Third: replace the hand-crafted "last-three" features with a learned
> sequence encoder. We feed the per-step observation through a GRU and
> let the policy condition on the resulting hidden state. The GRU
> compresses arbitrary-length history into a fixed-dim vector, so the
> policy stops being limited to a 3-step lookback.

### Quadrant 4 — Market-regime / opponent-aware conditioning [1:00]

*[Bonnie — main extension; spend slightly longer here if you're the lead presenter]*

> Fourth — and this is mine. Even at 60 features, the obs vector is just a
> *snapshot*: per-tick averages and historical means. It doesn't tell the
> policy how the *regime is changing*. Is competition heating up? Are
> opponents pulling away in the right tail of the bid distribution? Are we
> currently in an unusually high-traffic tick?
>
> So I added 38 new features on top of obs_60, in two tiers. *Tier one* —
> 20 features that the simulator already tracks but the obs vector never
> exposed: per-slot win counts, per-slot exposure rates, CPA exceedence
> rate, and episode cumulative spend. Pure JSON change, no code.
>
> *Tier two* — 18 derived features I added to the env's state-builder.
> These are time derivatives, second moments, and opponent proxies:
> trend slopes of competitor bids over the last five ticks, standard
> deviation and z-score of competitor bids — that's the regime detector —
> and ratios that infer opponent aggression like floor-cost over
> available pvalue. Total: a 98-dim observation.
>
> The intuition: a good human bidder doesn't just react to the *level* of
> competition — they react to whether it's rising or falling, whether it's
> stable or volatile, and how unusual the current moment is. The policy
> should see the same.

---

## Slide 6 — Extensions, Part Two & Final Evaluation [2:30]

### Quadrant 1 — Reward shaping [0:40]

*[Ophelia / speaker]*

> Last extension: reward shaping. We tried two flavours. *Potential-based*
> shaping adds a `gamma * Phi(s') - Phi(s)` term — provably preserves the
> optimal policy while giving denser learning signal. *Direct CPA
> penalty* subtracts a Lagrangian term `lambda * max(0, CPA_t - c)` per
> step, so the policy is punished proportionally for current overspend.

### Final Evaluation table [1:50]

*[any speaker — present the table]*

> *[gesture to the table]*
>
> Here are the headline results, all on the held-out period 27. The
> 16-dim PPO baseline scores around 23.
>
> *[walk through each row briefly]*
>
> Adding the 60-dim obs gives us [X]. Stacking budget-aware pacing on
> top brings us to [Y]. Replay reweighting adds another [Z]. The GRU
> encoder gives [W]. My market-regime extension gives [V] and notably
> tightens the CPA-violation distribution.
>
> Reward shaping with the right `lambda` adds [U]. And our best
> *combination* — stacking the strongest extensions together — reaches
> [final score]. CPA violation rate drops from [X%] in the baseline to
> [Y%] in the best combo.
>
> The take-away: each extension targets a distinct failure mode of the
> baseline, and they compose constructively because they live in
> different layers of the pipeline — observation, action, sampling,
> policy architecture, and reward.

*[The numbers above are placeholders. Update the script after you have the eval JSONs from each run, and pick three numbers to highlight verbally — listeners can read the rest from the slide.]*

---

## Slide 7 — LLM Fine-Tuning: GRPO + LoRA on Qwen3-8B [2:00]

*[likely [Andrew] or whoever ran the LLM experiment]*

> Separately, we ran an experiment fine-tuning an 8-billion-parameter LLM
> — Qwen-3-8B — to bid directly. We used GRPO, the variant of PPO that
> dropped the value network in favour of group-relative advantage
> estimation, with LoRA adapters so we could fit it on one GPU.
>
> The training hierarchy: a *trajectory* is one (advertiser, period) pair
> over 48 ticks. At each rollout step, we sample 4 trajectories from the
> same prompt — that's the "group" GRPO uses as a critic. Each training
> *step* is a batch of 4 prompts times 4 samples, so 16 episodes per
> gradient update.
>
> *[gesture to the score plot, top left]*
>
> Score climbs early — train score goes from 20 to 30 in the first 100
> steps. Validation on period 27 sits flat around 17, just below the
> base model.
>
> *[gesture to the entropy plot, top right]*
>
> But around step 120, the policy entropy collapses by two orders of
> magnitude, and gradients go to zero. Classic mode-collapse failure: the
> model latches onto a few bid trajectories and stops exploring.
> Training stalls completely.
>
> The takeaway is that GRPO on an 8B model is fragile in this setting —
> the reward is sparse and high-variance, so once the group's relative
> advantage signal weakens, the LoRA gradients vanish before the policy
> generalises.

---

## Slide 8 — Future Directions [1:00]

*[any speaker — closes]*

> Looking ahead, the most direct next step is methodological: combine the
> extensions that worked best — for example, the 60-dim observation plus
> my market-regime features, plus the GRU encoder, plus reward shaping —
> in a single run. Each extension targets a different layer, so we expect
> the combination to outperform any single one.
>
> Beyond that: bigger sequence encoders to actually learn temporal
> patterns from raw history, multi-agent training to stop treating
> opponents as a fixed environment, and offline regime clustering to
> give the policy an explicit cluster ID for which market regime it's in
> at each tick.
>
> *[pause]*
>
> Thanks — happy to take questions.

---

## Backup notes for Q&A

Likely questions and short answers:

- **"How do you get a Markov state out of an auction?"** — You don't, fully. The 60-dim and 98-dim observation vectors approximate it by including enough history aggregates to make most policy-relevant past info recoverable.
- **"Why second-price not first-price?"** — Standard AuctionNet setup. Second-price gives a clean dominant-strategy interpretation per impression but the multi-impression budget-CPA setup makes the global problem still non-trivial.
- **"How does Bonnie's extension differ from the GRU?"** — Different layer. The GRU compresses the *temporal sequence* of obs vectors. Mine adds *new feature dimensions* (slopes, z-scores, opponent proxies) that the GRU would otherwise have to learn from scratch. They're complementary — combining them is on the future-work list.
- **"Why did the LLM fail?"** — Mode collapse from GRPO's group-advantage signal vanishing as the policy concentrates. Mitigations would be: KL penalty against the base model, larger groups, or adaptive entropy bonus.
- **"What's the best baseline you could've used?"** — IQL / DT on the offline data is in the AuctionNet repo but we focused on online PPO since that's what the team chose to specialise in.

---

## Pacing checklist

| Slide | Target time | Cumulative |
|---|---|---|
| 1. Title | 0:30 | 0:30 |
| 2. Motivation | 2:00 | 2:30 |
| 3. Baseline | 2:00 | 4:30 |
| 4. Obs 16→60 | 2:00 | 6:30 |
| 5. Extensions I (4 quadrants) | 3:30 | 10:00 |
| 6. Extensions II + Eval table | 2:30 | 12:30 |
| 7. LLM GRPO | 2:00 | 14:30 |
| 8. Future + Q&A intro | 1:00 | **15:30** |

If you're running long, tighten Slides 2, 3, and 7 first — the audience usually grasps the setup and the LLM-failure narrative quickly. Don't compress the eval table; that's where the contribution lands.
