"""
Stage 2: DQN for Gridworld Player Mode
機制: S1 Replay Buffer + S2 Target Network + S3 Double DQN + S4 Dueling Network
框架: TensorFlow / tf.keras (禁止使用 model.fit)

分析說明：
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. 失敗症狀 (Failure Symptom)
   → Stage 1 的基本 DQN (S1+S2) 在 Player Mode 出現 Q 值系統性過估
     (Overestimation)，Loss 居高不下，Win Rate 停滯在 40~60%。

2. 為何失敗 (Why it fails)
   → Vanilla DQN 使用 max(Q_target) 同時做「選動作」與「評估 Q 值」，
     在隨機起始位置下狀態分佈更廣，導致過估誤差累積更嚴重。
   → 單一 Q 網路對不同起始狀態的泛化能力不足（所有格子共享同一估值空間）。

3. 解法 (Solution)
   → S3 Double DQN：主網路選動作，目標網路評估 Q 值，分離兩個職責，
     消除最大化偏差 (Maximization Bias)。
   → S4 Dueling Network：分解 Q = V(s) + A(s,a) - mean(A)，
     使網路能獨立學習「狀態價值」與「動作優勢」，
     對起始位置不同但目標相同的狀態更有效率。

4. 為何跳過 S5 (Why skip S5)
   → Player Mode 的獎勵訊號仍然充足（目標位置固定），
     不像 Random Mode 那樣極度稀疏，暫不需要 PER 的優先取樣。
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


# ── S1: Replay Buffer ─────────────────────────────────────────────────────────
class ReplayBuffer:
    def __init__(self, max_size: int = 5000):
        self.buf = deque(maxlen=max_size)

    def add(self, s, a, r, ns, done):
        self.buf.append((s, int(a), float(r), ns, float(done)))

    def sample(self, batch_size: int):
        batch = random.sample(self.buf, min(len(self.buf), batch_size))
        s, a, r, ns, d = zip(*batch)
        return (np.array(s,  dtype=np.float32),
                np.array(a,  dtype=np.int32),
                np.array(r,  dtype=np.float32),
                np.array(ns, dtype=np.float32),
                np.array(d,  dtype=np.float32))

    def __len__(self):
        return len(self.buf)


# ── S4: Dueling Network（Functional API）────────────────────────────────────
def build_dueling_model(input_dim=INPUT_DIM, num_actions=NUM_ACTIONS):
    """
    Dueling DQN 架構：
        Q(s,a) = V(s) + A(s,a) - mean_a(A(s,·))
    分成 Value stream 與 Advantage stream，最終合併為 Q 值。
    """
    inputs = tf.keras.Input(shape=(input_dim,), name="state_input")

    # 共享特徵層
    x = tf.keras.layers.Dense(150, activation="relu", name="shared_fc1")(inputs)
    x = tf.keras.layers.Dense(100, activation="relu", name="shared_fc2")(x)

    # Value stream：估計狀態價值 V(s)
    v = tf.keras.layers.Dense(64,  activation="relu", name="value_fc")(x)
    v = tf.keras.layers.Dense(1,   name="value_out")(v)             # shape: (batch, 1)

    # Advantage stream：估計動作優勢 A(s,a)
    a = tf.keras.layers.Dense(64,  activation="relu", name="adv_fc")(x)
    a = tf.keras.layers.Dense(num_actions, name="adv_out")(a)       # shape: (batch, 4)

    # 合併：Q = V + (A - mean(A))
    # 減去均值消除不可識別性 (identifiability issue)
    q = tf.keras.layers.Lambda(
        lambda va: va[0] + va[1] - tf.reduce_mean(va[1], axis=1, keepdims=True),
        name="q_output"
    )([v, a])

    return tf.keras.Model(inputs=inputs, outputs=q, name="DuelingDQN")


# ── @tf.function 訓練步驟（S3 Double DQN + S4 Dueling）──────────────────────
@tf.function
def _train_step(model, target_model, optimizer, s, a, r, ns, d, gamma):
    """
    S3 Double DQN 更新規則：
        target = r + γ * Q_target(s', argmax_a Q_online(s', a)) * (1 - done)
    主網路選動作，目標網路評估 Q 值，避免 overestimation bias。
    """
    with tf.GradientTape() as tape:
        # 主網路預測當前 Q 值
        q_pred  = model(s, training=True)
        q_taken = tf.reduce_sum(q_pred * tf.one_hot(a, NUM_ACTIONS), axis=1)

        # S3: 主網路選出下一步最優動作
        q_online_next   = model(ns, training=False)
        best_actions    = tf.argmax(q_online_next, axis=1, output_type=tf.int32)

        # S3: 目標網路評估選出動作的 Q 值
        q_target_next   = target_model(ns, training=False)
        q_next_selected = tf.reduce_sum(
            q_target_next * tf.one_hot(best_actions, NUM_ACTIONS), axis=1
        )

        td_target = tf.stop_gradient(r + gamma * q_next_selected * (1.0 - d))
        loss      = tf.reduce_mean(tf.square(td_target - q_taken))

    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return loss


# ── DQN Agent（含 S1~S4）────────────────────────────────────────────────────
class DQNAgent:
    def __init__(self,
                 gamma=0.9, lr=1e-3,
                 epsilon_start=1.0, epsilon_min=0.1,
                 buffer_size=5000, batch_size=64,
                 target_update_freq=50):
        self.gamma              = gamma
        self.epsilon            = epsilon_start
        self.epsilon_min        = epsilon_min
        self.batch_size         = batch_size
        self.target_update_freq = target_update_freq

        # S4: 使用 Dueling Network
        self.model        = build_dueling_model()
        self.target_model = build_dueling_model()
        self.sync_target()

        self.optimizer  = tf.keras.optimizers.Adam(learning_rate=lr)
        self.replay_buf = ReplayBuffer(max_size=buffer_size)
        self._step      = 0

    # S2: 同步 Target Network（硬更新）
    def sync_target(self):
        self.target_model.set_weights(self.model.get_weights())

    @staticmethod
    def get_state(game) -> np.ndarray:
        """將 4×4×4 棋盤攤平為 64 維向量，並加微小雜訊。"""
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
        s, a, r, ns, d = self.replay_buf.sample(self.batch_size)

        loss = _train_step(
            self.model, self.target_model, self.optimizer,
            tf.constant(s), tf.constant(a), tf.constant(r),
            tf.constant(ns), tf.constant(d),
            self.gamma,
        )

        self._step += 1
        if self._step % self.target_update_freq == 0:
            self.sync_target()

        return float(loss.numpy())

    def decay_epsilon(self, total_epochs: int):
        step_size = (1.0 - self.epsilon_min) / total_epochs
        self.epsilon = max(self.epsilon_min, self.epsilon - step_size)


# ── 執行單一 Episode ──────────────────────────────────────────────────────────
def run_episode(agent: DQNAgent,
                mode="player",
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
        next_state   = agent.get_state(game)
        reward       = game.reward()
        done         = abs(reward) == 10

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
def evaluate(agent: DQNAgent, n=100, mode="player") -> float:
    wins = sum(run_episode(agent, mode=mode, training=False)["win"]
               for _ in range(n))
    return wins / n


# ── 計算 Policy Grid（供視覺化用）────────────────────────────────────────────
def compute_policy(agent: DQNAgent, mode="player"):
    """回傳每個格子最優動作與 Q 值矩陣（以固定初始棋盤為參考）。"""
    ref   = Gridworld(size=4, mode=mode)
    board = ref.display()   # shape (4,4) 字元陣列

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
            s[0 * 16 + row * 4 + col] = 1.0   # Player
            for r, c in goal_pos:
                s[1 * 16 + r * 4 + c] = 1.0   # Goal
            for r, c in trap_pos:
                s[2 * 16 + r * 4 + c] = 1.0   # Trap
            for r, c in wall_pos:
                s[3 * 16 + r * 4 + c] = 1.0   # Wall

            q = agent.model(s.reshape(1, -1), training=False).numpy()[0]
            q_grid[row, col] = q
            policy[row, col] = int(np.argmax(q))

    return policy, q_grid, board


# ── 主程式（單獨執行訓練）────────────────────────────────────────────────────
if __name__ == "__main__":
    import json

    EPOCHS = 1500
    agent  = DQNAgent()
    log    = {"losses": [], "rewards": [], "wins": []}

    for ep in range(EPOCHS):
        result = run_episode(agent, mode="player")
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
                  f"ε={agent.epsilon:.3f}")

    # 先儲存，再評估（避免 print emoji 崩潰導致儲存失敗）
    with open(os.path.join(_DIR, "training_log_stage2.json"), "w") as f:
        json.dump(log, f)
    agent.model.save_weights(os.path.join(_DIR, "dqn_player_weights.weights.h5"))
    print("Weights & log saved.")

    wr_final = evaluate(agent, n=200)
    print(f"[OK] Final Win Rate (200 games): {wr_final:.1%}")
