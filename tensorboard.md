## 1 Add TensorBoard logging to your training script

```python
from stable_baselines3 import PPO
from flappy_game.flappy_env import FlappyEnv

# 1️⃣  choose a log directory
LOGDIR = "./tb_logs/flappy"

env   = FlappyEnv(render_mode="none")
model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    tensorboard_log=LOGDIR,      # 👈  enables TB summaries
)

model.learn(total_timesteps=500_000)
model.save("flappy_ppo_model")
env.close()
```

*SB3 writes TensorBoard event files to* `./tb_logs/flappy/PPO_<yyyy_mm_dd-hh_mm_ss>/*`.

---

## 2 Start TensorBoard

```bash
tensorboard --logdir ./tb_logs/flappy
```

Then open [http://localhost:6006](http://localhost:6006) in your browser.

*Tip (remote servers)* forward port 6006 (SSH tunnel or VS Code’s “Port Forwarding”).

---

## 3 Folders & tags you will see

| TensorBoard tab | SB3 tag prefix | meaning                                                                                       |
| --------------- | -------------- | --------------------------------------------------------------------------------------------- |
| **Scalars**     | `rollout/…`    | episode-level numbers gathered while *running* the policy: reward, length, success rate, etc. |
|                 | `train/…`      | losses & diagnostics recorded each gradient update.                                           |
|                 | `time/…`       | wall-clock timing stats (steps/s, fps).                                                       |
| **Graphs**      | n/a            | computation graph (rarely needed).                                                            |

---

## 4 Key curves to watch

| tag                          | why it matters                                             | what to look for                                                                                                                                           |
| ---------------------------- | ---------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `rollout/ep_rew_mean`        | **Main score** – average total reward per episode          | should trend **upward** or stabilise at a high value.                                                                                                      |
| `rollout/ep_len_mean`        | average episode length                                     | for Flappy Bird this grows as the bird survives more pipes; should rise together with reward.                                                              |
| `train/value_loss`           | critic MSE                                                 | a spike at the start, then plateaus; extremely high or exploding values ⇒ instability.                                                                     |
| `train/policy_loss`          | objective being minimised                                  | should settle near 0; wild oscillations imply too large a learning rate / poor normalisation.                                                              |
| `train/approx_kl`            | how far the new policy diverges from the old (PPO only)    | should hover near the target (\~0.01 by default). If it keeps hitting the clip range or drops to \~0, adjust `target_kl` / learning rate.                  |
| `train/clip_fraction`        | fraction of updates that hit the clip                      | 0.1 – 0.2 is typical. If always ≈0 → learning may stall; if always ≈1 → policy steps too big.                                                              |
| `train/entropy_loss`         | exploration measure (neg entropy)                          | should **decrease slowly** as the policy converges (less randomness). A flat line near 0 too early means premature convergence / insufficient exploration. |
| `rollout/explained_variance` | how well the value function predicts returns (1 = perfect) | rising toward 0.8 – 1.0 is a good sign; dropping negative ⇒ critic collapse.                                                                               |
| `time/fps`                   | simulation speed                                           | purely diagnostic.                                                                                                                                         |

*(Exact tag names may differ slightly; open the Scalars tab and expand the namespaces.)*

---

## 5 Typical “is it learning?” checklist

1. **Mean episode reward ↑** over time and levels off → good.
   *If it’s flat or noisy for tens of thousands of steps, try a smaller learning
   rate, reward shaping, or observation normalisation.*

2. **Episode length ↑** in sync with reward.

3. **Value loss & explained variance** settle (critic is neither diverging nor
   useless).

4. **Entropy gradually declines** – exploration giving way to exploitation.

5. **Approx KL inside target** (not exploding, not stuck at 0).

If all of the above look healthy, the bird is almost certainly improving; confirm by running your **`play_agent.py`** script and watching it survive more pipes.

---

### Extra tips

* **Smooth the curves** using TensorBoard’s slider (e.g. 0.9 smoothing) – SB3
  rewards are noisy frame-to-frame.
* **Multiple runs** SB3 creates a new sub-folder each execution. TensorBoard
  overlays them automatically so you can compare different hyper-parameters.
* **Long training** log directory can get large; prune old event files or use
  `tensorboard --samples_per_plugin scalars=1000` to limit memory.
