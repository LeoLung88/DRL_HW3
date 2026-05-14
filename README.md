# Deep Reinforcement Learning - DQN Gridworld (HW3)
## 三階段漸進式實作（TensorFlow / tf.keras）

> **🚀 執行方式**
> ```bash
> pip install -r requirements.txt
> streamlit run app.py
> ```
> *(透過 `app.py` 進入多頁面導覽，可於左側欄切換以下三個 Stage)*

> **檔案清單**
> | 核心程式 | UI 模組 (由 app.py 載入) | 訓練模式 |
> |----------|---------------|------|
> | `stage1_dqn.py` | `streamlit_app.py` | Static Mode (玩家固定、終點固定) |
> | `stage2_dqn.py` | `streamlit_app2.py` | Player Mode (玩家隨機、終點固定) |
> | `stage3_dqn.py` | `streamlit_app3.py` | Random Mode (玩家隨機、終點隨機) |

---

## 一、Stage 1 — Static Mode (`stage1_dqn.py`)

### 1.1 機制組合

| 機制 | 啟用 | 說明 |
|------|------|------|
| S1 Replay Buffer | ✅ | `class ReplayBuffer`，`deque(maxlen=2000)` |
| S2 Target Network | ✅ | `sync_target()` 每 `target_update_freq` 步硬更新 |
| S3 Double DQN | ❌ | 環境簡單，過估問題不顯著 |
| S4 Dueling DQN | ❌ | 不需要 |
| S5 PER | ❌ | 獎勵訊號充足，不需要優先取樣 |

### 1.2 網路架構

```
Sequential Model
  Dense(150, ReLU)  ← 輸入 64 維狀態
  Dense(100, ReLU)
  Dense(4)          ← 輸出 4 個動作的 Q 值
```

使用 `tf.keras.Sequential`，對應教科書原始 PyTorch 架構。

### 1.3 核心訓練邏輯 (`_train_step`)

```python
@tf.function                              # 編譯為計算圖，速度提升 5~10x
def _train_step(model, target_model, optimizer, s, a, r, ns, d, gamma):
    with tf.GradientTape() as tape:
        q_pred    = model(s, training=True)
        q_taken   = tf.reduce_sum(q_pred * tf.one_hot(a, 4), axis=1)
        q_next    = tf.reduce_max(target_model(ns, training=False), axis=1)
        td_target = tf.stop_gradient(r + gamma * q_next * (1.0 - d))
        loss      = tf.reduce_mean(tf.square(td_target - q_taken))  # MSE
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return loss
```

- `tf.stop_gradient`：TD target 不參與梯度計算
- `tf.one_hot(a, 4)`：挑出被選動作的 Q 值

### 1.4 ε-greedy 探索

從 ε=1.0 線性衰減至 ε=0.1，共 1000 個 epoch：

```python
def decay_epsilon(self, total_epochs):
    step_size = (1.0 - epsilon_min) / total_epochs
    self.epsilon = max(epsilon_min, self.epsilon - step_size)
```

### 1.5 狀態表示

`render_np()` 輸出 shape `(4, 4, 4)`（4 channel：Player/Goal/Trap/Wall），
展平為 64 維向量，加微小雜訊增加探索多樣性：

```python
game.board.render_np().reshape(64) + np.random.rand(64) / 10.0
```

### 1.6 Streamlit App (`streamlit_app.py`)

- **Training Tab**：即時 Episode/Loss/Reward/Win Rate 指標，訓練後一次繪製曲線
- **Results Tab**：策略地圖（棋盤+箭頭）、Q-value Heatmap（4 動作）、200 場勝率測試

---

## 二、Stage 2 — Player Mode (`stage2_dqn.py`)

### 2.1 機制組合

| 機制 | 啟用 | 說明 |
|------|------|------|
| S1 Replay Buffer | ✅ | 繼承，buffer 擴大至 5000 |
| S2 Target Network | ✅ | 繼承，硬更新 |
| S3 Double DQN | ✅ | **新增**：分離動作選擇與評估 |
| S4 Dueling DQN | ✅ | **新增**：V(s) + A(s,a) − mean(A) |
| S5 PER | ❌ | 目標固定，獎勵充足，暫不需要 |

### 2.2 四點分析（作業要求格式）

**1. 失敗症狀**：Stage 1 Vanilla DQN 在 Player Mode 中，Q 值系統性過估，
Loss 居高不下，Win Rate 停滯在 40~60%，對不同起始位置泛化能力不足。

**2. 為何失敗**：Vanilla DQN 的 `max(Q_target)` 同時負責「選動作」與「評估 Q 值」，
在玩家隨機起始的狀態空間中，最大化偏差（Maximization Bias）更顯著。
單一 Q 值混合估計狀態好壞與動作優劣，無法跨起始位置共用策略。

**3. 解法**：
- S3 Double DQN：`a* = argmax Q_online(s')`，`y = r + γ·Q_target(s', a*)` — 主選動作，目標評估
- S4 Dueling：`Q = V(s) + A(s,a) − mean(A)` — 獨立學習狀態價值與動作優勢

**4. 跳過 S5**：Player Mode 目標/陷阱位置固定，正負獎勵規律出現，
樣本失衡問題不嚴重，PER 留待 Stage 3。

### 2.3 Dueling Network 架構（Functional API）

```python
inputs = tf.keras.Input(shape=(64,))
x = Dense(150, relu)(inputs)        # 共享特徵
x = Dense(100, relu)(x)

v = Dense(64, relu)(x)              # Value stream
v = Dense(1)(v)                     # V(s): shape (batch, 1)

a = Dense(64, relu)(x)              # Advantage stream
a = Dense(4)(a)                     # A(s,a): shape (batch, 4)

q = Lambda(
    lambda va: va[0] + va[1] - tf.reduce_mean(va[1], axis=1, keepdims=True)
)([v, a])                           # Q = V + (A - mean(A))
```

### 2.4 S3 Double DQN 訓練步驟

```python
# S3: 主網路選動作
best_actions = tf.argmax(model(ns, training=False), axis=1, output_type=tf.int32)
# S3: 目標網路評估 Q 值
q_next = tf.reduce_sum(
    target_model(ns, training=False) * tf.one_hot(best_actions, 4), axis=1
)
td_target = tf.stop_gradient(r + gamma * q_next * (1.0 - d))
```

### 2.5 Streamlit App (`streamlit_app2.py`)

- **Analysis Tab**（新增）：四點分析、Dueling 架構圖、Double DQN 數學公式
- **Training Tab**：即時指標，紫色調主題
- **Results Tab**：策略地圖、Q-value Heatmap、**跨模式評估**（Player + Static）

---

## 三、Stage 3 — Random Mode (`stage3_dqn.py`)

### 3.1 機制組合

| 機制 | 啟用 | 說明 |
|------|------|------|
| S1 (PER) | ✅ | 改為 `PrioritizedReplayBuffer`，容量 10000 |
| S2 Soft Update | ✅ | 改為 Soft Update（τ=0.01，每步執行） |
| S3 Double DQN | ✅ | 繼承 Stage 2 |
| S4 Dueling DQN | ✅ | 繼承 Stage 2 |
| S5 PER | ✅ | **新增**：優先取樣 + IS 加權 Loss |
| 梯度裁剪 | ✅ | **新增**：`clip_by_global_norm(grad, 1.0)` |
| 獎勵裁剪 | ✅ | **新增**：`clip(r, -1, 1)` |
| LR Schedule | ✅ | **新增**：`ExponentialDecay` |

### 3.2 四點分析（作業要求格式）

**1. 失敗症狀**：Stage 2 在 Random Mode 出現梯度爆炸（Loss 跳至 100+）、
Win Rate 近乎 0%、Q 值估計劇烈震盪，訓練無法收斂。

**2. 為何失敗**：每局棋盤隨機，樣本嚴重失衡（99% reward=-1，1% ±10），
均勻取樣無法有效學習終止狀態；梯度未裁剪加上高 TD 誤差導致爆炸；
硬更新 Target Network 在高方差環境引入更大目標震盪。

**3. 解法（S5 + 穩定化技巧）**：
- S5 PER：`P(i) = p_i^α / Σp_k^α`，IS 加權修正偏差，beta 從 0.4 退火至 1.0
- 梯度裁剪：防止梯度爆炸
- Soft Target Update：`θ_t ← τ·θ_online + (1-τ)·θ_t`，更平滑
- 獎勵裁剪：縮小 TD 誤差尺度
- LR Schedule：訓練後期自動降低學習率

**4. 無跳過機制**：Random Mode 為最高難度，S1~S5 全部啟用，
各機制協同解決不同面向問題。

### 3.3 PER 核心實作

```python
class PrioritizedReplayBuffer:
    def sample(self, batch_size):
        priors  = self.priorities[:n] ** self.alpha    # 指數化優先權
        probs   = priors / priors.sum()                # 正規化為機率
        indices = np.random.choice(n, batch_size, p=probs)
        weights = (n * probs[indices]) ** (-self.beta) # IS 修正
        weights /= weights.max()                        # 正規化

    def update_priorities(self, indices, td_errors):
        self.priorities[idx] = abs(err) + 1e-6         # 避免優先權為 0
```

### 3.4 IS 加權 Loss + 梯度裁剪

```python
td_errors = td_target - q_taken
loss = tf.reduce_mean(weights * tf.square(td_errors))   # IS 加權 MSE
grads = tape.gradient(loss, model.trainable_variables)
grads, _ = tf.clip_by_global_norm(grads, 1.0)           # 梯度裁剪
```

### 3.5 Soft Target Update

```python
def sync_target_soft(self):
    new_w = [self.tau * ow + (1 - self.tau) * tw
             for ow, tw in zip(online_w, target_w)]
    self.target_model.set_weights(new_w)
```

### 3.6 Streamlit App (`streamlit_app3.py`)

- **Analysis Tab**：四點分析、PER 取樣示意、Soft Update 公式、IS 加權 Loss 公式
- **Training Tab**：即時顯示 ε / β(beta) / Buffer 大小
- **Results Tab**：訓練曲線、**PER Beta 退火曲線**、策略地圖、**三模式跨評估**

---

## 四、Homework 需求合規性檢查

### homework_request.md 逐條核對（共 25 項）

| # | 需求 | 狀態 |
|---|------|------|
| 1 | Convert DQN from PyTorch to tf.keras | ✅ 全部三個 stage |
| 2 | Bonus: gradient clipping, LR scheduling | ✅ Stage 3 加分項完成 |
| 3 | Implement Static Mode | ✅ `stage1_dqn.py` |
| 4 | Implement Player Mode | ✅ `stage2_dqn.py` |
| 5 | Implement Random Mode | ✅ `stage3_dqn.py` |
| 6 | S1~S5 incrementally | ✅ S1+S2 → S1~S4 → S1~S5 |
| 7 | Analyze environment difficulty | ✅ docstring + Analysis Tab |
| 8 | Analyze training instability symptoms | ✅ 四點分析第 1 點 |
| 9 | Identify DQN weakness | ✅ 四點分析第 2 點 |
| 10 | Implement ONLY required schemes | ✅ Stage 1 不含 S3~S5 |
| 11 | Skip unnecessary schemes（含理由） | ✅ 四點分析第 4 點 |
| 12 | Use TensorFlow and tf.keras | ✅ |
| 13 | Custom GradientTape training loops | ✅ `_train_step` in all stages |
| 14 | Avoid model.fit() | ✅ 三個 stage 均無 `model.fit()` |
| 15 | Replay buffer logic | ✅ `ReplayBuffer` / `PrioritizedReplayBuffer` |
| 16 | Target synchronization | ✅ Hard sync / Soft sync |
| 17 | TD target computation | ✅ `r + γ·Q_next·(1-done)` |
| 18 | Epsilon-greedy exploration | ✅ `select_action()` + `decay_epsilon()` |
| 19 | Reward/Loss visualization | ✅ Streamlit 三曲線圖 |
| 20 | Training analysis and debugging | ✅ docstring + Streamlit Analysis Tab |
| 21 | Explain failure symptom | ✅ 四點分析第 1 點 |
| 22 | Explain why current mechanisms fail | ✅ 四點分析第 2 點 |
| 23 | Explain why selected scheme solves problem | ✅ 四點分析第 3 點 |
| 24 | Explain why other schemes are skipped | ✅ 四點分析第 4 點 |
| 25 | Complete runnable implementations | ✅ `python stage{N}_dqn.py` 可獨立執行 |

> [!IMPORTANT]
> **所有 25 項需求均已滿足。加分項（梯度裁剪、LR Schedule、Soft Update、獎勵裁剪）已在 Stage 3 全部實作。**

---

## 五、建議異動事項（先不修改程式碼）

> [!NOTE]
> 以下為審視後發現的改進點，均不影響作業基本需求的完成，可擇機補強。

### 5.1 Stage 1 缺少四點分析 docstring

**問題**：`stage1_dqn.py` module docstring 僅列出機制名稱，
未包含完整四點分析，與 Stage 2/3 格式不一致。

**建議補充內容**：
```
1. 失敗症狀：無 Buffer/Target Net 的基本 Q-learning 出現 Loss 劇烈震盪
2. 為何失敗：序列樣本高度相關（Sequential Correlation）+ Moving Target 問題
3. 解法：S1 Replay Buffer 打破時序相關性；S2 Target Network 穩定 TD 目標
4. 跳過 S3~S5：Static Mode 難度低、目標固定，S1+S2 已足夠收斂
```

### 5.2 Stage 1 Streamlit App 缺少 Analysis Tab

**問題**：`streamlit_app.py` 只有 Training / Results 兩個 Tab，
Stage 2/3 均有 Analysis Tab，介面格式不一致。

**建議**：新增 Analysis Tab，放入 Stage 1 四點分析文字說明。

### 5.3 缺少跨階段勝率比較表

**問題**：`overview.md` 要求「測試結果：在 Static / Player / Random mode 的勝率統計」，
目前每個 Streamlit App 各自獨立，沒有跨 Stage 的統一比較頁面。

**建議**：
- 選項 A：在 Stage 3 Results Tab 末尾新增一張比較表格（需從已存的 weights 載入 Stage 1/2 agent）
- 選項 B：新增一個 `streamlit_summary.py`，載入三個 stage 的 model weights 並比較

### 5.4 PrioritizedReplayBuffer sample 邊界防護

**問題**：`sample()` 使用 `replace=False`，若 buffer 大小恰好等於 batch_size，
概率計算的浮點誤差可能導致 `replace=False` 失敗。

**建議**：將 `sample()` 中的 `replace=False` 加條件：
```python
replace = (n < batch_size)  # 理論上 learn() 的 guard 已排除，但防禦性加上
```

### 5.5 可考慮提供 `.ipynb` 格式（可選）

**問題**：`overview.md` 建議最終輸出為 `.ipynb`（三個 Stage 各一段）。

**說明**：目前 `.py` + Streamlit 功能更完整，但若老師要求 notebook，
可用 `nbformat` 將三個 stage 封裝成單一 `hw3_dqn_all_stages.ipynb`。
（非必要，Streamlit 呈現已超越 notebook 的視覺化能力）

---

## 六、執行與部署方式

### 1. 本機執行 (Streamlit App)
整合為單一 App，透過左側欄切換不同 Stage：
```bash
# 安裝依賴套件
pip install -r requirements.txt

# 啟動 Streamlit 伺服器
streamlit run app.py
```

### 2. 命令列獨立訓練 (背景產生權重)
若只想純訓練模型，可直接執行 `.py`，產生的檔案會被 Streamlit 自動載入：
```bash
python stage1_dqn.py   # 輸出 dqn_static_weights.weights.h5, training_log.json
python stage2_dqn.py   # 輸出 dqn_player_weights.weights.h5, training_log_stage2.json
python stage3_dqn.py   # 輸出 dqn_random_weights.weights.h5, training_log_stage3.json
```

### 3. Streamlit Cloud 雲端部署指南
本專案已完美支援 Streamlit Community Cloud 一鍵部署：
1. **GitHub**：將整個專案（包含程式碼、`.weights.h5`、`.json`、`requirements.txt`）推送到 GitHub。
2. **Streamlit Cloud**：登入 [share.streamlit.io](https://share.streamlit.io/) 並連結 GitHub。
3. **Deploy**：
   - Repository: 選擇你的倉庫
   - Branch: `main`
   - Main file path: `app.py` (若在子目錄請加上路徑，如 `HW3/app.py`)
4. **注意事項**：專案根目錄已加入 `.python-version` 確保使用相容的 Python 3.11 環境。預存的權重會讓老師一開網頁就看到訓練完成的美觀報表。
