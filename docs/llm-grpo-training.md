# LLM Auto-Bidder: GRPO Training (Technical Summary)

This document summarises the technical details behind the LLM auto-bidding agent
trained with GRPO inside `strategy_train_env/bidding_train_env/llm/`. It is
intended as a reference for the methods section of the report.

External references:

- DeepSeekMath / GRPO paper (Shao et al., 2024): <https://arxiv.org/pdf/2402.03300>
- VeRL paper (Sheng et al., 2024, *HybridFlow*): <https://arxiv.org/pdf/2409.19256v2>
- VeRL implementation: <https://github.com/verl-project/verl>

## 1. Training objective

We optimise the policy parameters $\theta$ of an LLM bidding agent
$\pi_\theta(\,\cdot\mid x)$ with **Group Relative Policy Optimization (GRPO)**
of Shao et al. (2024). GRPO inherits the clipped PPO surrogate but eliminates
the learned value function: the per-sample advantage is replaced by a within
group $z$-score over rewards from $G$ rollouts of the same prompt $x$, and a
KL term to a frozen reference policy $\pi_{\mathrm{ref}}$ regularises drift.

For each prompt $x \sim \mathcal{D}$ we sample $G$ trajectories
$\{\tau_i\}_{i=1}^G \sim \pi_{\theta_{\mathrm{old}}}(\cdot\mid x)$ and observe
their scalar episode returns $R_i \in \mathbb{R}$. The **group-relative
advantage** assigned to every token of $\tau_i$ is

$$
\hat A_i \;=\; \frac{R_i - \mu_R}{\sigma_R + \varepsilon},
\qquad
\mu_R = \frac{1}{G}\sum_{j=1}^{G} R_j,
\qquad
\sigma_R^{2} = \frac{1}{G}\sum_{j=1}^{G}(R_j - \mu_R)^{2}.
$$

Letting $\rho_{i,t}(\theta) = \pi_\theta(\tau_{i,t}\mid x,\tau_{i,<t}) /
\pi_{\theta_{\mathrm{old}}}(\tau_{i,t}\mid x,\tau_{i,<t})$ be the per-token
importance ratio and $|\tau_i|$ the number of *trained* tokens in trajectory
$i$, the GRPO objective optimised here is

$$
\mathcal{L}_{\mathrm{GRPO}}(\theta)
=
\mathbb{E}_{x,\{\tau_i\}}\!\left[
\frac{1}{G}\sum_{i=1}^{G}\frac{1}{|\tau_i|}\sum_{t=1}^{|\tau_i|}
\Big(
\min\!\big(\rho_{i,t}\hat A_i,\;
\mathrm{clip}(\rho_{i,t},1-\epsilon_{\mathrm{c}},1+\epsilon_{\mathrm{c}})\hat A_i\big)
\;-\;\beta\,D_{\mathrm{KL}}^{\mathrm{lv}}\!\big(\pi_\theta\,\Vert\,\pi_{\mathrm{ref}}\big)
\Big)
\right].
$$

Concrete settings used in
`bidding_train_env/llm/verl/config/qwen3_8b_grpo_lora.yaml`:

| Symbol | Meaning | Value |
| --- | --- | --- |
| $G$ | group size per prompt | `actor_rollout_ref.rollout.n = 4` |
| $\epsilon_{\mathrm{c}}$ | PPO clip range | VeRL default ($0.2$) |
| $\beta$ | KL coefficient | `actor.kl_loss_coef = 1.0\text{e-}3` |
| $D_{\mathrm{KL}}^{\mathrm{lv}}$ | low-variance KL estimator | `actor.kl_loss_type = low_var_kl` |
| advantage estimator | within-group $z$-score | `algorithm.adv_estimator = grpo` |

The KL term is enabled via `actor.use_kl_loss = true` and folded directly into
the loss, so the trajectory-level KL controller is disabled
(`algorithm.kl_ctrl.kl_coef = 0`) and the reward signal is left un-shaped. The
entropy bonus is set to zero (`actor.entropy_coeff = 0`).

## 2. How the LLM and the auctioneer correspond to the formula

### 2.1 What is $\pi_\theta(\cdot\mid x)$?

The policy is the chat-formatted generative distribution of **Qwen3-8B** with
LoRA adapters injected on the transformer-block linear layers (see §4 for the
exact set). It runs in two roles during training:

1. **Rollout role** ($\pi_{\theta_{\mathrm{old}}}$): served by an asynchronous
   vLLM engine (`rollout.mode = async`, `tensor_model_parallel_size = 1`).
2. **Training role** ($\pi_\theta$): an FSDP-sharded copy of the same weights
   used for log-prob recomputation and the gradient step. Adapter deltas are
   pushed back to the rollout engine via VeRL's
   async weight-sync `checkpoint_engine` (bucket size raised to 4096 MB so
   Qwen3-8B's $151\,936\times 4096$ embed_tokens tensor fits in one bucket).

Chain-of-thought is **disabled** at both call sites. Qwen3 emits `<think>...</think>`
by default; we set `data.apply_chat_template_kwargs.enable_thinking = false`,
which `AgentLoopBase.apply_chat_template` forwards to the tokenizer for the
initial prompt and every incremental chat-template call. Without this,
per-turn responses balloon to ~2900 tokens and a 48-tick episode clips at
~5 turns before the budget is exhausted, leaving `episode_score = 0`
indefinitely.

### 2.2 What is the prompt $x$?

One row of the train/val parquet (`build_rl_dataset.py`). Each row
corresponds to a single $(\text{period}, \text{advertiser})$ pair and stores
the initial chat used to seed the rollout:

- `system`: the bidding system prompt from `prompt.SYSTEM_PROMPT` (action
  contract `bid_i = exp(α) · pvalue_i · target_cpa`, $\alpha\in[-10,10]$,
  output format `<alpha>X</alpha>`).
- `user`: a `<auction_meta>{json}</auction_meta>` header (period, advertiser,
  budget, target_cpa, fixed seed) followed by the rendered tick-0 observation
  produced by `prompt.build_user_message(state, tick=0)`.

The `<auction_meta>` header is what the rollout's `BiddingAgentLoop` parses
to reconstruct the AuctionNet replay environment for that specific
$(\text{period}, \text{advertiser})$.

### 2.3 What is a trajectory $\tau_i$?

A single $\tau_i$ is **one full 48-tick episode** for one advertiser in one
period (`EPISODE_LENGTH = 48` in `online/definitions.py`). Inside a rollout
(see `bidding_agent_loop.py::BiddingAgentLoop.run`), each tick performs:

1. Fetch env state via `BiddingEnv._get_state_dict`.
2. Render the observation with `prompt.build_user_message` and append it as a
   `user` turn. Its tokens are added to `response_ids` with
   **`response_mask = 0`** — visible to the policy at the next step but
   excluded from the gradient.
3. Call the async vLLM server for an `assistant` turn. Its tokens are
   appended with **`response_mask = 1`** — these are the only tokens that
   contribute to $\rho_{i,t}$ and the loss.
4. Parse $\alpha$ from `<alpha>X</alpha>` via `prompt.parse_alpha` (clamped
   to $[-10,10]$, fallback $0.0$ on malformed output) and call
   `env.step(np.array([α]))`.
5. Stop when the env terminates or truncates, or when `len(response_ids) >=
   response_length = 28672` tokens.

The entire 48-turn conversation forms one trajectory $\tau_i$; the per-turn
$\rho_{i,t}$ is therefore a per-token ratio over the assistant tokens only,
matching the masked sum in the objective above.

### 2.4 What is the return $R_i$?

The terminal AuctionNet score, computed inside the env on the last step:

$$
R_i \;=\; \min\!\Big(1,\,\big(\text{target\_cpa} / \text{realized\_cpa}\big)^{2}\Big)
\cdot \text{conversions}.
$$

`BiddingAgentLoop` pulls this from `step_info["score"]` and emits it as
`AgentLoopOutput.reward_score`. VeRL's `AgentLoopWorker` writes that scalar
into `batch["rm_scores"]`, placed on the final assistant token; because the
reward is purely terminal and `algorithm.gamma = 1.0`, the per-token return
is constant across the trajectory and equals $R_i$. There is no separate
reward model — the optional `auction_reward_manager.py` is only used to
surface auxiliary metrics (CPA, conversions, malformed_count) for logging.

### 2.5 How a step of GRPO is assembled

Per training step:

- `data.train_batch_size = 4` prompts are drawn from `train.parquet`.
- For each prompt the rollout engine produces $G = 4$ trajectories
  (`rollout.n = 4`), giving 16 episodes per step.
- The **per-prompt seed is fixed** in the dataset (`seed = period*1000 +
  advertiser`), so the four rollouts in a group share environment realisations
  — within-group variation in $R_i$ reflects policy variation rather than
  auction noise. This is essential for the within-group $z$-score to be a
  meaningful baseline.
- The 16 trajectories are processed under FSDP with
  `ppo_mini_batch_size = 4` and `ppo_micro_batch_size_per_gpu = 1`, with
  param + optimizer offloading enabled (`fsdp_config.param_offload = true`,
  `optimizer_offload = true`).

### 2.6 Sampling and optimisation hyperparameters

Consolidated from `qwen3_8b_grpo_lora.yaml`. These are the knobs that shape
both the on-policy distribution and the gradient step.

| Group | Field | Value |
| --- | --- | --- |
| Rollout sampling | `rollout.temperature` | $1.0$ |
|                  | `rollout.top_p`       | $0.95$ |
|                  | `rollout.n` (= $G$)   | $4$ |
|                  | `rollout.max_model_len` | $32{,}768$ tokens |
|                  | `rollout.max_num_batched_tokens` | $40{,}960$ |
|                  | `rollout.gpu_memory_utilization` | $0.5$ |
|                  | `rollout.tensor_model_parallel_size` | $1$ |
| Episode shape    | `data.max_prompt_length` | $1{,}024$ |
|                  | `data.max_response_length` | $28{,}672$ (≈ $48 \times 500$ + headroom) |
|                  | `EPISODE_LENGTH` | $48$ ticks |
| Optimiser        | `actor.optim.lr` | $1.0\text{e-}5$ |
|                  | `actor.entropy_coeff` | $0$ |
|                  | `actor.use_kl_loss` / `kl_loss_coef` / `kl_loss_type` | `true` / $10^{-3}$ / `low_var_kl` |
|                  | `algorithm.gamma`, `algorithm.lam` | $1.0$, $1.0$ |
| Batching         | `data.train_batch_size` | $4$ prompts/step ($\times G = 16$ rollouts) |
|                  | `actor.ppo_mini_batch_size` | $4$ |
|                  | `actor.ppo_micro_batch_size_per_gpu` | $1$ |
| Trainer          | `total_epochs` | $1$ (≈ 60 update steps over ~960 train rows) |
|                  | `save_freq` / `test_freq` | $20$ / $20$ |
|                  | `nnodes` / `n_gpus_per_node` | $1$ / $1$ |

Two notes worth flagging for the report:

- Temperature $1.0$ with `top_p = 0.95` is the **only** source of within-group
  trajectory variation, since each group of $G = 4$ rollouts shares a fixed
  env seed (§2.5). When the policy sharpens, sampler stochasticity is the
  whole budget for keeping $\sigma_R > 0$.
- `entropy_coeff = 0` means there is no explicit exploration bonus; entropy
  is shaped purely by the KL term to $\pi_{\mathrm{ref}}$ and by the sampling
  temperature.

## 3. Software stack: VeRL + GRPO

The training stack uses **VeRL 0.8.x** (Sheng et al., 2024, *HybridFlow*,
arXiv:2409.19256v2; <https://github.com/verl-project/verl>). Relevant pieces
(see `verl/launch_train.py`, `verl/bidding_agent_loop.py`,
`verl/auction_reward_manager.py`):

- **PPO/GRPO trainer**: we compose against VeRL's bundled
  `verl.trainer.config.ppo_trainer` and override only the GRPO/LoRA-specific
  fields. `algorithm.adv_estimator = grpo` selects the within-group
  $z$-score advantage; the rest of the clipped-ratio update reuses the PPO
  trainer code path.
- **Async rollout engine**: `actor_rollout_ref.rollout = vllm` with
  `mode = async`, `gpu_memory_utilization = 0.5`, `max_model_len = 32768`,
  `max_num_batched_tokens = 40960`, `temperature = 1.0`, `top_p = 0.95`.
  HybridFlow's hybrid-controller design (Sheng et al., 2024 §3) is what lets
  us co-locate the FSDP actor and the vLLM rollout on the same H200.
- **Custom AgentLoop**: `BiddingAgentLoop` is registered as
  `default_agent_loop = bidding_agent`. VeRL invokes it once per dataset row;
  it produces the multi-turn `(prompt_ids, response_ids, response_mask,
  reward_score)` quadruple that the trainer expects.
- **Worker setup hook**: because Ray rollout workers run in fresh Python
  interpreters, `launch_train.py` registers a
  `worker_process_setup_hook = bidding_train_env.llm.verl._worker_setup.setup`
  that re-imports `bidding_agent_loop` inside every worker so the
  `@register("bidding_agent")` decorator populates the worker-local
  registry.
- **No reward model**: when `reward_model.enable` is absent, VeRL skips
  RewardManager instantiation and uses `rm_scores` directly as the per-episode
  reward (see `verl/trainer/ppo/ray_trainer.py`).

## 4. LoRA: why and how

### Why

Full-parameter fine-tuning of Qwen3-8B is **not feasible on a single H200
(141 GB)**: the FSDP actor + vLLM rollout engine + reference policy + KL
regulariser already saturate the device, and a separate value head for an 8B
policy (as standard PPO would require) would push it well over. Training is
constrained to **1 H200, 1 node** (`trainer.nnodes = 1`,
`trainer.n_gpus_per_node = 1`, see `main_verl_train.sh`'s
`--gres=gpu:1`). LoRA (Hu et al., 2021) is therefore mandatory: only adapter
weights are stored in optimiser state, and the frozen base weights serve
double duty as the GRPO reference $\pi_{\mathrm{ref}}$ for the KL term.
Choosing GRPO over PPO further removes the value head, freeing additional
memory.

### How

From `qwen3_8b_grpo_lora.yaml` under `actor_rollout_ref.model`:

```yaml
lora_rank: 32
lora_alpha: 32
target_modules: all-linear
enable_gradient_checkpointing: true
```

Under HuggingFace PEFT semantics, `target_modules = all-linear` matches every
`nn.Linear` in the model **except the output layer**. For Qwen3-8B that means
adapters are inserted on the attention projections (`q_proj`, `k_proj`,
`v_proj`, `o_proj`) and the MLP projections (`gate_proj`, `up_proj`,
`down_proj`) inside every transformer block, while `embed_tokens` (an
`nn.Embedding`, not `nn.Linear`), the RMSNorm layers, and `lm_head` (excluded
by the `all-linear` rule) remain frozen and adapter-free. Gradient
checkpointing trims activation memory during the backward pass. `VLLM_ALLOW_RUNTIME_LORA_UPDATING = true` is set in
`launch_train.py`'s Ray runtime env so the rollout engine can hot-swap
adapter deltas without restarting.

After training, the trained adapter under
`output/llm/training/qwen3_8b_grpo_lora/` is merged into a fresh copy of the
base weights, and `main_eval_llm.py --model <merged_dir>` evaluates it against
the same protocol used by the PPO baseline.

## 5. Periods used and dataset construction

### 5.1 Period split

Source data lives at
`strategy_train_env/data/traffic/online_rl_data/period-{p}_{pvalues|bids|constraints}.parquet`
(`RL_DATA_DIR` in `online/definitions.py`). The period split is hard-coded in
`build_rl_dataset.py`:

| Split | Periods | Purpose |
| --- | --- | --- |
| Train | $7,8,\dots,26$ (20 periods) | GRPO updates |
| Val   | $27$                       | `val_before_train = true`, `test_freq = 20` |

The eval pipeline (`main_eval_llm.py`) is also pinned to period 27, so
training-time validation and end-to-end LLM evaluation use byte-identical
prompts at tick 0.

### 5.2 Per-row processing

For every period $p$ in the split, `build_rl_dataset.py` does the
following:

1. Loads `period-{p}_constraints.parquet`, filters to rows with
   `deliveryPeriodIndex == p`. Each row is one advertiser's campaign
   contract: $(\text{advertiser}, \text{budget}, \text{target\_cpa})$.
2. Constructs a `BiddingEnv` via `EnvironmentFactory.create` using the same
   $(\text{pvalues}, \text{bids}, \text{constraints})$ parquet triple, with
   `obs_keys = obs_16_keys`, `act_keys = act_1_key` (single scalar action,
   the same configuration the PPO baseline uses).
3. For each advertiser:
   - calls `env.reset()` then
     `env.unwrapped.set_campaign(advertiser, budget, target_cpa, period)` to
     pin the campaign context;
   - extracts the tick-0 state via `BiddingEnv._get_state_dict` and renders
     it with `prompt.build_user_message(state, tick=0, episode_length=48)`;
   - assembles a metadata header
     `<auction_meta>{period, advertiser, budget, target_cpa, seed}</auction_meta>`
     where `seed = period*1000 + advertiser` is **fixed per row** so all $G$
     GRPO group samples for that prompt see the same env realisation;
   - emits one parquet row of the form

     ```text
     data_source = "auctionnet_bidding"
     prompt = [{system, SYSTEM_PROMPT},
               {user,   <auction_meta>...</auction_meta>\n<initial obs>}]
     ability = "auto_bidding"
     reward_model = {style: "env", ground_truth: ""}
     extra_info = {period, advertiser, budget, target_cpa, seed}
     ```

This yields ~960 train rows (≈48 advertisers × 20 periods) and ~48 val rows.
With `total_epochs = 1`, `train_batch_size = 4`, `rollout.n = 4`, that maps
to roughly 60 GRPO update steps per training run — small by LLM standards
but sufficient to move `episode_score` measurably above the frozen-base
LLM baseline on this task.

### 5.3 Online (rollout-time) data flow

At rollout time, `BiddingAgentLoop` reads `extra_info` from the dataset row
and rebuilds an env identical to the one used at dataset-construction time
(same parquet triple, same fixed seed). Because the env is reconstructed
per rollout from on-disk data, training is fully **offline-data / online-env**:
no live auctions, but a fresh 48-tick replay each time. This is what makes
`R_i` reproducible across the $G$ samples in a group up to the policy's
sampling stochasticity, which in turn is what makes the within-group
$z$-score baseline a low-variance estimator of the centred return.

## 6. Observed training dynamics

A representative training run (`slurm_out/llm_rl_p7_26-7226239.out`,
visualised by `output/llm/results/plot_training_7226239.py`) exhibits a
co-occurring pattern that, given the task structure, is essentially
expected rather than pathological:

- **`actor/entropy` collapses by orders of magnitude** over the first dozen
  steps, reaching $\sim 10^{-3}$ nats/token within ~30 updates.
- **`actor/grad_norm` decays** alongside it, approaching numerical floor as
  training proceeds.
- **`actor/pg_loss` flattens near zero**, with a slight negative bias
  consistent with within-group advantage normalisation.
- **`val-core/auctionnet_bidding/reward/mean@1`** (greedy, period 27)
  nonetheless rises clearly above the frozen-base-model baseline (≈ 17.08)
  and continues to improve at every checkpoint.

### Caption for the training-trajectory figure

> **Figure: GRPO + LoRA fine-tune of Qwen3-8B on AuctionNet periods 7–26
> (validation on period 27).** *Left:* per-step training score (rolling-10
> mean and min/max band over each batch of 16 rollouts), greedy validation
> reward at every checkpoint, and the frozen-base-model validation baseline
> (gray dotted line at 17.08). *Middle:* actor token entropy on a log scale,
> tracking exploration collapse over training. *Right:* policy-gradient
> surrogate loss (raw and rolling-10) on the left axis and gradient norm
> (log scale) on the right axis, jointly indicating optimisation activity
> and stability.

### Why entropy and grad-norm vanish while validation keeps rising

Four mechanisms — three from the task structure and one from the
chain-of-thought-disabled regime — converge to produce this signature.

**(i) With chain-of-thought disabled, token entropy is a direct measure of
α-spread.** The action contract forces every assistant turn to be
`<alpha>X</alpha>` — a fixed three-token wrapper plus a short decimal. After
a few format-locking updates, the only token positions that *can* carry
non-trivial entropy are the digits of $X$. Per-token entropy therefore
becomes an almost direct proxy for "how spread is the per-tick $\alpha$
distribution." With CoT enabled, entropy could live in a long reasoning
prefix even after $\alpha$ is decided — the
`apply_chat_template_kwargs.enable_thinking = false` setting deliberately
removes that channel (§2.1), so entropy collapse in this run is structural,
not a sign of mode failure.

**(ii) GRPO's group baseline starves its own gradient as the group
converges.** The advantage
$\hat A_i = (R_i - \mu_R)/(\sigma_R + \varepsilon)$ is computed across $G =
4$ rollouts that share an env seed (§2.5). Across-group variation in $R_i$
therefore comes entirely from policy stochasticity. As the digit
distribution sharpens (mechanism i), the four rollouts in a group produce
near-identical $\alpha$-trajectories and hence near-identical returns,
$\sigma_R \to 0$, and every $\hat A_i$ becomes near-zero. The clipped
surrogate then has no signal to backpropagate, so `pg_loss` flatlines and
`grad_norm` decays. This is a known GRPO failure mode (the DeepSeekMath
paper, arXiv:2402.03300, observes the analogous behaviour on problems that
have become easy for the model).

**(iii) The reward landscape rewards a *small* repertoire of α-trajectories,
not diversity.** The terminal score $\min(1,(\text{target\_cpa}/
\text{realized\_cpa})^{2})\cdot\text{conversions}$ is roughly unimodal in
pacing aggressiveness for a given $(\text{budget}, \text{target\_cpa})$
pair: the optimal $\alpha$-profile is a smooth, slowly-varying schedule.
The policy does not need to retain multiple modes to do well — committing
to one schedule *is* the right answer. Entropy collapse onto a peaked
digit distribution is therefore aligned with task optimality rather than
symptomatic of premature convergence.

**(iv) Validation is greedy, so it tracks the *mode*, not the spread.**
`main_eval_llm.py` performs a single deterministic rollout per advertiser,
so the metric reports the quality of the policy's argmax decision. Even
when the on-policy entropy has collapsed and `grad_norm` is small, the
*location* of the mode can keep migrating — the residual gradient signal
acts on whichever within-group differences survive, and validation
captures that movement at every `test_freq = 20` checkpoint.

Together: with CoT off the policy has nowhere to put entropy except in the
$\alpha$-digit distribution; that distribution sharpens because the task
is unimodal and the within-group baseline rewards consistency; and the
greedy validation metric continues to rise because the *mode* of the
sharpened distribution keeps inching toward better pacing schedules. The
same plot, rerun with `enable_thinking = true`, would almost certainly
show entropy stalling at a much higher floor (CoT carries it) but with
substantially worse `episode_score` because per-turn responses would
clip the 48-tick episode after only a handful of turns
(`max_response_length = 28672` is dimensioned for thinking-off; with
thinking on, ~5 turns saturate the budget — see the explicit YAML comment
on `data.apply_chat_template_kwargs`).

### Caveat for the report

This signature is also consistent with the policy committing prematurely to
a *locally* optimal pacing schedule. The greedy val curve climbing past
the frozen-base baseline is evidence that the local optimum is non-trivial,
but the dwindling gradient means further training will not improve on it.
Two natural follow-ups, if more compute is available:

1. Raise `rollout.temperature` (currently $1.0$) or reintroduce a small
   `actor.entropy_coeff` (currently $0$) to keep $\sigma_R$ alive longer.
2. Inject a small action-space perturbation into each group rather than
   relying entirely on sampler stochasticity to provide within-group
   variation.
