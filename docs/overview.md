# HW3-3：以 TensorFlow/Keras 漸進式實作 DQN（含訓練穩定技巧）

## 一、作業需求摘要

### 核心目標

將原本以 **PyTorch** 撰寫的 DQN（Deep Q-Network）模型，轉換為 **TensorFlow `tf.keras`** 實作，並針對三種 Gridworld 遊戲模式（Static → Player → Random）**漸進式地**加入 DQN 機制（S1～S5）。

### 技術規格（必須遵守）

| 項目 | 要求 |
|------|------|
| 框架 | `TensorFlow` + `tf.keras` |
| 訓練方式 | 自定義 `GradientTape` 訓練迴圈，**禁止**使用 `model.fit()` |
| 核心元件 | Replay Buffer、Target Network 同步、TD 目標計算、ε-greedy 探索 |
| 視覺化 | Reward / Loss 曲線圖 |
| 分析說明 | 每階段需說明失敗症狀、原因、解法、為何跳過其他機制 |

---

## 二、Gridworld 環境說明（來自 `第3章程式_ALL_IN_ONE.ipynb`）

### 環境基本設定

- **棋盤大小**：4×4 格
- **元素**：
  - `P`：玩家（Player）
  - `+`：目標（達到得 +10 分）
  - `-`：陷阱（踩到得 -10 分）
  - `W`：牆壁（不可進入）
  - ` `：空格
- **動作空間**：上（`u`）、下（`d`）、左（`l`）、右（`r`），共 4 個離散動作
- **狀態表示**：`render_np()` → shape `(4, 4, 4)`，展平後為長度 **64** 的向量

### 三種遊戲模式

| 模式 | 說明 | 難度 |
|------|------|------|
| **Static** | 目標、陷阱、牆壁位置固定 | 低 |
| **Player** | 玩家起始位置隨機 | 中 |
| **Random** | 所有元素（包括目標、陷阱）隨機放置 | 高 |

### 原始 PyTorch DQN 架構（教科書版本）

```
輸入層 (64) → 隱藏層1 (150, ReLU) → 隱藏層2 (100, ReLU) → 輸出層 (4)
```

- **損失函數**：MSE
- **優化器**：Adam（lr = 1e-3）
- **折扣因子 γ**：0.9
- **ε-greedy**：從 1.0 線性衰減至 0.1（共 1000 個 epoch）

### 靜態模式可成功，隨機模式失敗

原始程式的實驗結果顯示：
- **Static mode**：訓練後可穩定找到最佳路徑（7 步到達目標）
- **Random mode**：訓練後仍然失敗（在牆壁旁來回振盪，超過 15 步上限）

這驗證了基本 DQN 在高難度環境中的不足，是本作業要解決的核心問題。

---

## 三、DQN 五大機制（S1～S5）

### S1：Replay Buffer（經驗回放）

- **作用**：儲存過去的 `(s, a, r, s', done)` 轉換，隨機取樣 mini-batch 進行訓練
- **解決**：時間序列資料的高相關性（sequential correlation），提升樣本效率

### S2：Target Network（目標網路）

- **作用**：建立一個獨立的目標網路（週期性從主網路複製權重），用於計算 TD 目標
- **解決**：訓練目標不穩定（Moving Target）的問題

### S3：Double DQN

- **作用**：主網路選擇動作，目標網路評估 Q 值，分離動作選擇與評估
- **解決**：Q 值過估（Overestimation Bias）

### S4：Dueling DQN

- **作用**：將 Q 網路分解為 **Value Stream V(s)** 與 **Advantage Stream A(s,a)**
  ```
  Q(s,a) = V(s) + (A(s,a) - mean(A(s,·)))
  ```
- **解決**：在動作效果差異小的狀態下提升學習效率

### S5：Prioritized Experience Replay（PER）

- **作用**：根據 TD 誤差大小給予樣本優先權，高誤差樣本被更頻繁地取樣
- **解決**：稀疏獎勵環境中的學習效率問題

---

## 四、漸進式實作計畫（三階段）

### 階段一：Static Mode → 基本 DQN + S1 + S2

**目標**：驗證 Keras 架構可學習固定配置的 Gridworld

**分析**：
- 環境難度低（目標位置固定），基本 DQN 理應可收斂
- 可能症狀：訓練震盪、Q 值不穩定

**實作機制**：
- ✅ S1：Replay Buffer（解決樣本相關性）
- ✅ S2：Target Network（穩定 TD 目標）
- ❌ S3、S4、S5：Static mode 不需要

**程式重點**：
```python
# GradientTape 訓練迴圈骨架
with tf.GradientTape() as tape:
    q_values = model(states, training=True)
    action_q = tf.reduce_sum(q_values * one_hot_actions, axis=1)
    td_target = rewards + gamma * tf.reduce_max(target_model(next_states), axis=1) * (1 - dones)
    loss = tf.reduce_mean(tf.square(td_target - action_q))
grads = tape.gradient(loss, model.trainable_variables)
optimizer.apply_gradients(zip(grads, model.trainable_variables))
```

---

### 階段二：Player Mode → Stage 1 + S3 + S4

**目標**：玩家起始位置隨機，需要更強的泛化能力

**分析**：
- 可能症狀：Q 值系統性過估，泛化能力不足
- S1+S2 不夠，出現 Overestimation Bias

**實作機制**：
- ✅ S1、S2（繼承自 Stage 1）
- ✅ S3：Double DQN（解決過估）
- ✅ S4：Dueling Network（提升對不同起始狀態的學習效率）
- ❌ S5：Player mode 獎勵訊號尚充足，暫不需要 PER

**Dueling Network 架構（Keras Functional API）**：
```python
inputs = tf.keras.Input(shape=(64,))
x = Dense(128, activation='relu')(inputs)
# Value stream
v = Dense(64, activation='relu')(x)
v = Dense(1)(v)
# Advantage stream
a = Dense(64, activation='relu')(x)
a = Dense(num_actions)(a)
# 合併
q = v + (a - tf.reduce_mean(a, axis=1, keepdims=True))
model = tf.keras.Model(inputs=inputs, outputs=q)
```

---

### 階段三：Random Mode → Stage 2 + S5 + 穩定化技巧

**目標**：解決最難的全隨機配置，使訓練穩定收斂

**分析**：
- 可能症狀：梯度爆炸、稀疏獎勵導致學習停滯、震盪嚴重
- S3+S4 仍不足，需要對稀疏獎勵更有效的取樣策略

**實作機制**：
- ✅ S1～S4（繼承自 Stage 2）
- ✅ S5：Prioritized Experience Replay（聚焦高 TD 誤差樣本）
- ✅ **穩定化技巧（加分項）**：

| 技巧 | 說明 |
|------|------|
| **梯度裁剪** | `optimizer.apply_gradients` 前裁剪梯度（clip norm = 1.0） |
| **學習率排程** | 使用 `ExponentialDecay` 或 `ReduceLROnPlateau` |
| **Soft Target Update** | 每步用 τ 混合更新目標網路（而非週期性複製） |
| **獎勵裁剪** | 將獎勵裁剪至 [-1, 1] 範圍 |

**PER 核心邏輯**：
```python
class PrioritizedReplayBuffer:
    def sample(self, batch_size, alpha=0.6, beta=0.4):
        priorities = np.array(self.priorities) ** alpha
        probs = priorities / priorities.sum()
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        # 計算 Importance Sampling 權重
        weights = (len(self.buffer) * probs[indices]) ** (-beta)
        weights /= weights.max()
        return samples, indices, weights
```

---

## 五、每階段說明文件要求

作業要求每個模式的實作必須附上**清楚的分析說明**，格式如下：

```
1. 失敗症狀（Failure Symptom）
   → 觀察到什麼問題？（e.g., 損失震盪、reward 停滯）

2. 為何失敗（Why it fails）
   → 理論原因說明（e.g., moving target, overestimation bias）

3. 解法（Solution）
   → 選用哪個 DQN 機制？為何它能解決這個問題？

4. 為何跳過其他機制（Why skip others）
   → 說明不加入 S3/S4/S5 的理由（環境尚不需要）
```

---

## 六、最終輸出格式建議

| 項目 | 內容 |
|------|------|
| 程式碼 | 完整可執行的 `.ipynb`（三個 Stage 各一段） |
| 訓練曲線 | 每個 Stage 的 Loss 與 Reward 圖 |
| 分析文字 | 每個 Stage 的四點說明（症狀、原因、解法、跳過理由） |
| 測試結果 | 在 Static / Player / Random mode 的勝率統計 |

---

## 七、PyTorch vs Keras API 對應表

| 原始（PyTorch）| Keras 對應實作 |
|----------------|---------------|
| `torch.nn.Sequential` | `tf.keras.Sequential` 或 Functional API |
| `torch.optim.Adam` | `tf.keras.optimizers.Adam` |
| `torch.nn.MSELoss` | `tf.reduce_mean(tf.square(...))` |
| `model(state)` | `model(state, training=True/False)` |
| `loss.backward()` + `optimizer.step()` | `GradientTape` + `apply_gradients` |
| `torch.no_grad()` | `training=False` 或 `tf.stop_gradient()` |
| `model.parameters()` | `model.trainable_variables` |
| `tensor.detach()` | `tf.stop_gradient(tensor)` |

---

> **總結**：本作業的關鍵學習目標是理解 DQN 各機制「為什麼必要」，而非盲目堆疊所有技巧。
> 漸進式的實作方式讓你能清楚地觀察每個機制在對應環境難度下所帶來的改善效果。
