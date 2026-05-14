"""
Stage 3: DQN for Gridworld Random Mode
機制: S1~S5 全部 + 穩定化技巧（梯度裁剪 / Soft Target Update / 獎勵裁剪 / LR Schedule）
框架: TensorFlow / tf.keras (禁止使用 model.fit)

分析說明：
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. 失敗症狀 (Failure Symptom)
   → Stage 2 (S1~S4) 在 Random Mode 訓練時出現：
     • 梯度爆炸（Loss 突然跳至 100+），訓練崩潰
     • 稀疏獎勵導致 Win Rate 幾乎為 0（目標、陷阱隨機，
       大多數 episode 超時未到達終止狀態）
     • Q 值估計嚴重震盪，無法穩定收斂

2. 為何失敗 (Why it fails)
   → 隨機模式中每局棋盤完全不同，Agent 難以從均勻取樣的
     Replay Buffer 中找到有效的「成功經驗」。
   → 絕大多數 transition 的 reward = -1（一般步），
     只有極少數有 ±10（終止），樣本嚴重失衡。
   → 硬更新 (hard sync) Target Network 在高方差環境下引入更大
     目標震盪；梯度未裁剪導致偶發性的梯度爆炸。

3. 解法 (Solution)
   → S5 Prioritized Experience Replay（PER）：
     根據 TD 誤差大小賦予樣本優先權，使 Agent 更頻繁
     地學習「高資訊量」transition（終止 / 大錯誤），
     搭配 Importance Sampling (IS) 權重修正偏差。
   → 穩定化技巧：
     • 梯度裁剪 (clip_norm=1.0)：防止梯度爆炸
     • Soft Target Update (τ=0.01)：每步混合更新，
       比週期性硬更新更平滑
     • 獎勵裁剪 (clip to [-1,1])：縮小 TD 誤差尺度，
       減少 Loss 震盪
     • ExponentialDecay LR Schedule：訓練後期降低學習率

4. 無跳過機制（全部啟用）
   → S1~S5 全部啟用，Random Mode 是最高難度挑戰，
     需要所有機制協同作用才能穩定收斂。
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
import os, sys, urllib.request
import numpy as np
import tensorflow as tf
from collections import deque
import random

# ── 自動下載 Gridworld 環境 ──────────────────────────────────────────────────
_DIR = os.path.dirname(os.path.abspath(__file__))
_URLS = {
    "Gridworld.py": "https://github.com/DeepReinforcementLearning/"
                    "DeepReinforcementLearningInAction/raw/master/Errata/Gridworld.py",
    "GridBoard.py": "https://github.com/DeepReinforcementLearning/"
                    "DeepReinforcementLearningInAction/raw/master/Errata/GridBoard.py",
}
for fname, url in _URLS.items():
    fpath = os.path.join(_DIR, fname)
    if not os.path.exists(fpath):
        print(f"Downloading {fname} ...")
        urllib.request.urlretrieve(url, fpath)

if _DIR not in sys.path:
    sys.path.insert(0, _DIR)

from Gridworld import Gridworld  # noqa: E402

# ── 常數 ─────────────────────────────────────────────────────────────────────
ACTION_SET   = {0: "u", 1: "d", 2: "l", 3: "r"}
ACTION_ARROW = {0: "↑", 1: "↓", 2: "←", 3: "→"}
CELL_COLOR   = {"+": "#4CAF50", "-": "#F44336",
                "W": "#9E9E9E", "P": "#2196F3", " ": "#F5F5F5"}

NUM_ACTIONS = 4
INPUT_DIM   = 64


# ── S5: Prioritized Experience Replay (PER) ──────────────────────────────────
class PrioritizedReplayBuffer:
    """
    基於 TD 誤差優先取樣的 Replay Buffer。
    alpha: 優先權指數（0=均勻, 1=完全優先）
    beta:  IS 修正指數（從 beta_start 線性退火至 1.0）
    """
    def __init__(self, max_size: int = 10000, alpha: float = 0.6,
                 beta_start: float = 0.4, beta_frames: int = 2000):
        self.max_size    = max_size
        self.alpha       = alpha
        self.beta_start  = beta_start
        self.beta_frames = beta_frames
        self.frame       = 1          # 用於 beta 退火

        self.buf        = []
        self.priorities = np.zeros(max_size, dtype=np.float32)
        self.pos        = 0           # 環形指標

    @property
    def beta(self) -> float:
        """beta 從 beta_start 線性退火至 1.0。"""
        frac = min(1.0, self.frame / self.beta_frames)
        return self.beta_start + frac * (1.0 - self.beta_start)

    def add(self, s, a, r, ns, done):
        max_p = self.priorities[:len(self.buf)].max() if self.buf else 1.0
        if len(self.buf) < self.max_size:
            self.buf.append((s, int(a), float(r), ns, float(done)))
        else:
            self.buf[self.pos] = (s, int(a), float(r), ns, float(done))
        self.priorities[self.pos] = max_p
        self.pos = (self.pos + 1) % self.max_size

    def sample(self, batch_size: int):
        n       = len(self.buf)
        priors  = self.priorities[:n] ** self.alpha
        probs   = priors / priors.sum()

        indices = np.random.choice(n, batch_size, replace=False, p=probs)
        # Importance Sampling 權重，修正優先取樣引入的偏差
        weights = (n * probs[indices]) ** (-self.beta)
        weights /= weights.max()         # 正規化

        batch = [self.buf[i] for i in indices]
        s, a, r, ns, d = zip(*batch)
        self.frame += 1
        return (
            np.array(s,  dtype=np.float32),
            np.array(a,  dtype=np.int32),
            np.array(r,  dtype=np.float32),
            np.array(ns, dtype=np.float32),
            np.array(d,  dtype=np.float32),
            indices,
            np.array(weights, dtype=np.float32),
        )

    def update_priorities(self, indices, td_errors):
        """用最新 TD 誤差更新優先權（加小常數 ε 避免優先權為 0）。"""
        for idx, err in zip(indices, td_errors):
            self.priorities[idx] = abs(err) + 1e-6

    def __len__(self):
        return len(self.buf)


# ── S4: Dueling Network（Functional API，繼承自 Stage 2）────────────────────
def build_dueling_model(input_dim=INPUT_DIM, num_actions=NUM_ACTIONS):
    inputs = tf.keras.Input(shape=(input_dim,), name="state_input")
    x = tf.keras.layers.Dense(150, activation="relu", name="shared_fc1")(inputs)
    x = tf.keras.layers.Dense(100, activation="relu", name="shared_fc2")(x)

    v = tf.keras.layers.Dense(64, activation="relu", name="value_fc")(x)
    v = tf.keras.layers.Dense(1,  name="value_out")(v)

    a = tf.keras.layers.Dense(64, activation="relu", name="adv_fc")(x)
    a = tf.keras.layers.Dense(num_actions, name="adv_out")(a)

    q = tf.keras.layers.Lambda(
        lambda va: va[0] + va[1] - tf.reduce_mean(va[1], axis=1, keepdims=True),
        name="q_output"
    )([v, a])
    return tf.keras.Model(inputs=inputs, outputs=q, name="DuelingDQN_S3")


# ── @tf.function 訓練步驟（S3 Double DQN + S5 PER IS 加權）──────────────────
@tf.function
def _train_step(model, target_model, optimizer, s, a, r, ns, d, gamma, weights):
    """
    S3 Double DQN + PER Importance Sampling 加權 Loss。
    loss = mean( IS_weight * (TD_target - Q_pred)^2 )
    """
    with tf.GradientTape() as tape:
        q_pred  = model(s, training=True)
        q_taken = tf.reduce_sum(q_pred * tf.one_hot(a, NUM_ACTIONS), axis=1)

        # S3: online 選動作，target 評估 Q
        best_a  = tf.argmax(model(ns, training=False), axis=1, output_type=tf.int32)
        q_next  = tf.reduce_sum(
            target_model(ns, training=False) * tf.one_hot(best_a, NUM_ACTIONS), axis=1
        )
        td_target = tf.stop_gradient(r + gamma * q_next * (1.0 - d))

        td_errors = td_target - q_taken
        # S5: IS 加權 MSE
        loss = tf.reduce_mean(weights * tf.square(td_errors))

    grads = tape.gradient(loss, model.trainable_variables)
    # 穩定化：梯度裁剪（clip norm=1.0）
    grads, _ = tf.clip_by_global_norm(grads, 1.0)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return loss, td_errors


# ── DQN Agent（S1~S5 + 穩定化技巧）─────────────────────────────────────────
class DQNAgent:
    def __init__(self,
                 gamma=0.9, lr=1e-3,
                 epsilon_start=1.0, epsilon_min=0.05,
                 buffer_size=10000, batch_size=64,
                 tau=0.01,
                 lr_decay_steps=2000, lr_decay_rate=0.96,
                 per_alpha=0.6, per_beta_start=0.4, per_beta_frames=2000,
                 reward_clip=True):
        self.gamma        = gamma
        self.epsilon      = epsilon_start
        self.epsilon_min  = epsilon_min
        self.batch_size   = batch_size
        self.tau          = tau           # Soft Target Update 係數
        self.reward_clip  = reward_clip

        # S4: Dueling Network
        self.model        = build_dueling_model()
        self.target_model = build_dueling_model()
        self.sync_target_hard()

        # 穩定化：ExponentialDecay LR Schedule
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=lr,
            decay_steps=lr_decay_steps,
            decay_rate=lr_decay_rate,
            staircase=True,
        )
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

        # S5: PER
        self.replay_buf = PrioritizedReplayBuffer(
            max_size=buffer_size,
            alpha=per_alpha,
            beta_start=per_beta_start,
            beta_frames=per_beta_frames,
        )
        self._step = 0

    def sync_target_hard(self):
        """初始化時完整同步。"""
        self.target_model.set_weights(self.model.get_weights())

    def sync_target_soft(self):
        """穩定化：Soft Update θ_target = τ·θ_online + (1-τ)·θ_target"""
        online_w = self.model.get_weights()
        target_w = self.target_model.get_weights()
        new_w = [self.tau * ow + (1 - self.tau) * tw
                 for ow, tw in zip(online_w, target_w)]
        self.target_model.set_weights(new_w)

    @staticmethod
    def get_state(game) -> np.ndarray:
        return (game.board.render_np().reshape(64)
                + np.random.rand(64) / 10.0).astype(np.float32)

    def select_action(self, state: np.ndarray, greedy=False) -> int:
        if not greedy and np.random.random() < self.epsilon:
            return np.random.randint(NUM_ACTIONS)
        q = self.model(state.reshape(1, -1), training=False).numpy()[0]
        return int(np.argmax(q))

    def learn(self) -> float | None:
        if len(self.replay_buf) < self.batch_size:
            return None

        s, a, r, ns, d, indices, weights = self.replay_buf.sample(self.batch_size)

        # 穩定化：獎勵裁剪
        if self.reward_clip:
            r = np.clip(r, -1.0, 1.0)

        loss, td_errors = _train_step(
            self.model, self.target_model, self.optimizer,
            tf.constant(s), tf.constant(a), tf.constant(r),
            tf.constant(ns), tf.constant(d),
            self.gamma, tf.constant(weights),
        )
        # S5: 更新 PER 優先權
        self.replay_buf.update_priorities(indices, td_errors.numpy())

        # 穩定化：Soft Target Update（每步執行）
        self.sync_target_soft()
        self._step += 1

        return float(loss.numpy())

    def decay_epsilon(self, total_epochs: int):
        step_size = (1.0 - self.epsilon_min) / total_epochs
        self.epsilon = max(self.epsilon_min, self.epsilon - step_size)


# ── 執行單一 Episode ──────────────────────────────────────────────────────────
def run_episode(agent: DQNAgent,
                mode="random",
                max_steps=50,
                training=True) -> dict:
    game  = Gridworld(size=4, mode=mode)
    state = agent.get_state(game)

    total_reward = 0.0
    losses = []
    trajectory = [game.board.render_np().copy()] if not training else None

    for _ in range(max_steps):
        action = agent.select_action(state, greedy=not training)
        game.makeMove(ACTION_SET[action])
        next_state = agent.get_state(game)
        reward     = game.reward()
        done       = abs(reward) == 10

        if training:
            agent.replay_buf.add(state, action, reward, next_state, done)
            loss = agent.learn()
            if loss is not None:
                losses.append(loss)
        else:
            trajectory.append(game.board.render_np().copy())

        state         = next_state
        total_reward += reward
        if done:
            break

    return {
        "reward":     total_reward,
        "avg_loss":   float(np.mean(losses)) if losses else 0.0,
        "win":        total_reward > 0,
        "trajectory": trajectory,
    }


# ── 批次測試 ──────────────────────────────────────────────────────────────────
def evaluate(agent: DQNAgent, n=100, mode="random") -> float:
    wins = sum(run_episode(agent, mode=mode, training=False)["win"]
               for _ in range(n))
    return wins / n


# ── 計算 Policy Grid（隨機棋盤，取樣一局作為視覺化參考）────────────────────
def compute_policy(agent: DQNAgent, mode="random"):
    ref   = Gridworld(size=4, mode=mode)
    board = ref.display()

    policy = np.full((4, 4), -1, dtype=int)
    q_grid = np.zeros((4, 4, NUM_ACTIONS))

    goal_pos = [(r, c) for r in range(4) for c in range(4) if board[r, c] == "+"]
    trap_pos = [(r, c) for r in range(4) for c in range(4) if board[r, c] == "-"]
    wall_pos = [(r, c) for r in range(4) for c in range(4) if board[r, c] == "W"]

    for row in range(4):
        for col in range(4):
            if board[row, col] in ["+", "-", "W"]:
                continue
            s = np.zeros(64, dtype=np.float32)
            s[0 * 16 + row * 4 + col] = 1.0
            for r, c in goal_pos:
                s[1 * 16 + r * 4 + c] = 1.0
            for r, c in trap_pos:
                s[2 * 16 + r * 4 + c] = 1.0
            for r, c in wall_pos:
                s[3 * 16 + r * 4 + c] = 1.0

            q = agent.model(s.reshape(1, -1), training=False).numpy()[0]
            q_grid[row, col] = q
            policy[row, col] = int(np.argmax(q))

    return policy, q_grid, board


# ── 主程式 ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import json

    EPOCHS = 2000
    agent  = DQNAgent()
    log    = {"losses": [], "rewards": [], "wins": []}

    for ep in range(EPOCHS):
        result = run_episode(agent, mode="random")
        agent.decay_epsilon(EPOCHS)

        log["losses"].append(result["avg_loss"])
        log["rewards"].append(result["reward"])
        log["wins"].append(int(result["win"]))

        if (ep + 1) % 100 == 0:
            wr = np.mean(log["wins"][-100:])
            print(f"[{ep+1:4d}/{EPOCHS}] "
                  f"loss={result['avg_loss']:.4f}  "
                  f"reward={result['reward']:+.0f}  "
                  f"win_rate={wr:.0%}  "
                  f"ε={agent.epsilon:.3f}  "
                  f"β={agent.replay_buf.beta:.3f}")

    # 先儲存，再評估（避免 emoji 崩潰導致儲存失敗）
    with open(os.path.join(_DIR, "training_log_stage3.json"), "w") as f:
        json.dump(log, f)
    agent.model.save_weights(os.path.join(_DIR, "dqn_random_weights.weights.h5"))
    print("Weights & log saved.")

    wr_final = evaluate(agent, n=200)
    print(f"[OK] Final Win Rate (200 games): {wr_final:.1%}")
