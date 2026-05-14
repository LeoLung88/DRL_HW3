"""
Stage 1: DQN for Gridworld Static Mode
機制: S1 Replay Buffer + S2 Target Network
框架: TensorFlow / tf.keras (禁止使用 model.fit)
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


# ── S1: Replay Buffer ─────────────────────────────────────────────────────────
class ReplayBuffer:
    def __init__(self, max_size: int = 2000):
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


# ── DQN 模型建構 ──────────────────────────────────────────────────────────────
def build_model(input_dim=64, output_dim=4, h1=150, h2=100):
    return tf.keras.Sequential([
        tf.keras.Input(shape=(input_dim,)),
        tf.keras.layers.Dense(h1, activation="relu"),
        tf.keras.layers.Dense(h2, activation="relu"),
        tf.keras.layers.Dense(output_dim),
    ])


# ── DQN Agent ─────────────────────────────────────────────────────────────────
class DQNAgent:
    def __init__(self,
                 gamma=0.9, lr=1e-3,
                 epsilon_start=1.0, epsilon_min=0.1,
                 buffer_size=2000, batch_size=64,
                 target_update_freq=50):
        self.gamma              = gamma
        self.epsilon            = epsilon_start
        self.epsilon_min        = epsilon_min
        self.batch_size         = batch_size
        self.target_update_freq = target_update_freq

        self.model        = build_model()
        self.target_model = build_model()
        self.sync_target()

        self.optimizer    = tf.keras.optimizers.Adam(learning_rate=lr)
        self.replay_buf   = ReplayBuffer(max_size=buffer_size)
        self._step        = 0

    # S2: 同步 Target Network
    def sync_target(self):
        self.target_model.set_weights(self.model.get_weights())

    @staticmethod
    def get_state(game) -> np.ndarray:
        """將 4×4×4 棋盤攤平為 64 維向量，並加微小雜訊。"""
        return (game.board.render_np().reshape(64)
                + np.random.rand(64) / 10.0).astype(np.float32)

    def select_action(self, state: np.ndarray, greedy=False) -> int:
        if not greedy and np.random.random() < self.epsilon:
            return np.random.randint(4)
        q = self.model(state.reshape(1, -1), training=False).numpy()[0]
        return int(np.argmax(q))

    def learn(self) -> float | None:
        if len(self.replay_buf) < self.batch_size:
            return None
        s, a, r, ns, d = self.replay_buf.sample(self.batch_size)

        # ── @tf.function 編譯成計算圖，大幅提升訓練速度 ──────────────
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


# ── @tf.function 訓練步驟（模組層級，避免每次 trace）────────────────────────
@tf.function
def _train_step(model, target_model, optimizer,
                s, a, r, ns, d, gamma):
    """以 tf.function 編譯成靜態計算圖，比 eager mode 快 5~10 倍。"""
    with tf.GradientTape() as tape:
        q_pred    = model(s, training=True)
        q_taken   = tf.reduce_sum(q_pred * tf.one_hot(a, 4), axis=1)
        q_next    = tf.reduce_max(target_model(ns, training=False), axis=1)
        td_target = tf.stop_gradient(r + gamma * q_next * (1.0 - d))
        loss      = tf.reduce_mean(tf.square(td_target - q_taken))
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return loss


# ── 執行單一 Episode ──────────────────────────────────────────────────────────
def run_episode(agent: DQNAgent,
                mode="static",
                max_steps=50,
                training=True) -> dict:
    game  = Gridworld(size=4, mode=mode)
    state = agent.get_state(game)

    total_reward = 0.0
    losses = []
    # 只在測試時記錄軌跡，訓練時不需要，省去大量 numpy copy
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
            # 測試模式：記錄軌跡供視覺化
            trajectory.append(game.board.render_np().copy())

        state         = next_state
        total_reward += reward
        if done:
            break

    return {
        "reward":     total_reward,
        "avg_loss":   float(np.mean(losses)) if losses else 0.0,
        "win":        total_reward > 0,
        "trajectory": trajectory,   # 訓練時為 None
    }


# ── 批次測試 ──────────────────────────────────────────────────────────────────
def evaluate(agent: DQNAgent, n=100, mode="static") -> float:
    wins = sum(run_episode(agent, mode=mode, training=False)["win"]
               for _ in range(n))
    return wins / n


# ── 計算 Policy Grid（供視覺化用）────────────────────────────────────────────
def compute_policy(agent: DQNAgent, mode="static"):
    """回傳每個格子最優動作與 Q 值矩陣。"""
    ref = Gridworld(size=4, mode=mode)
    board = ref.display()              # shape (4,4) 字元陣列

    policy  = np.full((4, 4), -1, dtype=int)
    q_grid  = np.zeros((4, 4, 4))

    # 找各 entity 位置
    goal_pos  = [(r, c) for r in range(4) for c in range(4) if board[r, c] == "+"]
    trap_pos  = [(r, c) for r in range(4) for c in range(4) if board[r, c] == "-"]
    wall_pos  = [(r, c) for r in range(4) for c in range(4) if board[r, c] == "W"]

    for row in range(4):
        for col in range(4):
            if board[row, col] in ["+", "-", "W"]:
                continue
            # 建構合成狀態（不加雜訊，方便分析）
            s = np.zeros(64, dtype=np.float32)
            s[0 * 16 + row * 4 + col] = 1.0          # Player
            for r, c in goal_pos:
                s[1 * 16 + r * 4 + c] = 1.0          # Goal
            for r, c in trap_pos:
                s[2 * 16 + r * 4 + c] = 1.0          # Trap
            for r, c in wall_pos:
                s[3 * 16 + r * 4 + c] = 1.0          # Wall

            q = agent.model(s.reshape(1, -1), training=False).numpy()[0]
            q_grid[row, col] = q
            policy[row, col] = int(np.argmax(q))

    return policy, q_grid, board


# ── 主程式（單獨執行訓練）────────────────────────────────────────────────────
if __name__ == "__main__":
    import json

    EPOCHS = 1000
    agent  = DQNAgent()
    log    = {"losses": [], "rewards": [], "wins": []}

    for ep in range(EPOCHS):
        result = run_episode(agent, mode="static")
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

    # 先儲存，再評估（避免 emoji 崩潰導致儲存失敗）
    with open(os.path.join(_DIR, "training_log.json"), "w") as f:
        json.dump(log, f)
    agent.model.save_weights(os.path.join(_DIR, "dqn_static_weights.weights.h5"))
    print("Weights & log saved.")

    wr_final = evaluate(agent, n=200)
    print(f"[OK] Final Win Rate (200 games): {wr_final:.1%}")
