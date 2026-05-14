"""
Streamlit 視覺化介面 — Stage 2 DQN Player Mode
執行方式: streamlit run streamlit_app2.py
機制: S1 Replay Buffer + S2 Target Network + S3 Double DQN + S4 Dueling Network
"""
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pandas as pd
import os, sys, json

_DIR = os.path.dirname(os.path.abspath(__file__))
if _DIR not in sys.path:
    sys.path.insert(0, _DIR)

from stage2_dqn import (
    DQNAgent, run_episode, evaluate, compute_policy,
    ACTION_ARROW, CELL_COLOR
)

# ─────────────────────────────────────────────────────────────────────────────
# 頁面設定
# ─────────────────────────────────────────────────────────────────────────────
# (st.set_page_config 移至 app.py 統一設定)

st.markdown("""
<style>
  .metric-box{background:#1a1f35;border-radius:10px;padding:12px;text-align:center;
              border:1px solid #2d3a5e;}
  .metric-val{font-size:2rem;font-weight:700;color:#a78bfa;}
  .metric-lbl{font-size:.8rem;color:#94a3b8;margin-top:4px;}
  .badge-s3{background:#7c3aed;color:white;padding:2px 8px;border-radius:12px;
             font-size:.75rem;font-weight:600;}
  .badge-s4{background:#0e7490;color:white;padding:2px 8px;border-radius:12px;
             font-size:.75rem;font-weight:600;}
  .analysis-box{background:#1e293b;border-left:4px solid #a78bfa;padding:14px 18px;
                border-radius:6px;margin-bottom:12px;}
</style>
""", unsafe_allow_html=True)

st.title("🧠 DQN Gridworld — Stage 2: Player Mode")
st.caption("S1 Replay Buffer + S2 Target Network + **S3 Double DQN** + **S4 Dueling Network**  |  框架：TensorFlow / tf.keras")

# ─────────────────────────────────────────────────────────────────────────────
# 側欄
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ 超參數設定")
    epochs       = st.slider("Training Episodes",            300,  3000, 1500, 100)
    gamma        = st.slider("折扣因子 γ",              0.50, 0.99, 0.90, 0.01)
    lr           = st.select_slider("學習率 (lr)",
                                    options=[1e-4, 5e-4, 1e-3, 5e-3, 1e-2],
                                    value=1e-3,
                                    format_func=lambda x: f"{x:.0e}")
    batch_size   = st.select_slider("Batch Size",       options=[32, 64, 128], value=64)
    buf_size     = st.select_slider("Buffer Size",      options=[1000, 2000, 5000, 10000], value=5000)
    tuf          = st.slider("Target Update Freq",       10,  200,  50,  10)
    update_every = st.slider("圖表更新頻率 (每 N 回合)", 10,  100,  20,  10)
    max_steps    = st.slider("每回合最大步數",            20,  100,  50,  10)

    st.divider()
    st.markdown("**S1** ✅ Replay Buffer")
    st.markdown("**S2** ✅ Target Network")
    st.markdown("**S3** ✅ Double DQN")
    st.markdown("**S4** ✅ Dueling Network")
    st.markdown("**S5** ❌ PER（階段三）")

    st.divider()
    st.markdown("**模式說明**")
    st.info("🎮 **Player Mode**\n玩家起始位置隨機，目標與陷阱固定。\n比 Static Mode 需要更強的泛化能力。")

# ─────────────────────────────────────────────────────────────────────────────
# Session State
# ─────────────────────────────────────────────────────────────────────────────
for key, default in [
    ("agent2",    None),
    ("losses2",   []),
    ("rewards2",  []),
    ("wins2",     []),
    ("trained2",  False),
    ("preloaded2", False),
]:
    if key not in st.session_state:
        st.session_state[key] = default

# ─────────────────────────────────────────────────────────────────────────────
# 啟動自動載入：若已有预存模型檔案，直接還原訓練完成狀態
# ─────────────────────────────────────────────────────────────────────────────
_WEIGHTS_PATH = os.path.join(_DIR, "models/dqn_player_weights.weights.h5")
_LOG_PATH     = os.path.join(_DIR, "logs/training_log_stage2.json")

if not st.session_state.trained2:
    if os.path.exists(_WEIGHTS_PATH) and os.path.exists(_LOG_PATH):
        _agent = DQNAgent()                               # 建構相同架構
        _agent.model.load_weights(_WEIGHTS_PATH)          # 載入预存樊重
        _agent.target_model.set_weights(                  # 同步 target
            _agent.model.get_weights()
        )
        _agent.epsilon = _agent.epsilon_min               # 設為訓練完成狀態
        with open(_LOG_PATH, "r") as _f:
            _log_data = json.load(_f)
        st.session_state.update({
            "agent2":        _agent,
            "losses2":       _log_data["losses"],
            "rewards2":      _log_data["rewards"],
            "wins2":         _log_data["wins"],
            "trained2":      True,
            "preloaded2":    True,   # 標記為預載入，供 UI 區分
        })


# ─────────────────────────────────────────────────────────────────────────────
def draw_board(board_np, policy=None, title="", figsize=(4, 4)):
    char_map = {0: "P", 1: "+", 2: "-", 3: "W"}
    grid = np.full((4, 4), " ", dtype="<U2")
    for ch, char in char_map.items():
        pos = np.argwhere(board_np[ch] == 1)
        for r, c in pos:
            grid[r, c] = char

    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor("#0f172a")
    ax.set_facecolor("#0f172a")

    for r in range(4):
        for c in range(4):
            cell  = grid[r, c]
            color = CELL_COLOR.get(cell, "#F5F5F5")
            rect  = patches.FancyBboxPatch(
                (c + 0.05, 3 - r + 0.05), 0.90, 0.90,
                boxstyle="round,pad=0.05",
                linewidth=0, facecolor=color, alpha=0.92)
            ax.add_patch(rect)

            label = {"P": "P", "+": "GOAL", "-": "TRAP", "W": "WALL", " ": ""}
            ax.text(c + 0.5, 3 - r + 0.55, label.get(cell, cell),
                    ha="center", va="center", fontsize=11,
                    color="white", fontweight="bold")

            if policy is not None and policy[r, c] >= 0:
                ax.text(c + 0.5, 3 - r + 0.18,
                        ACTION_ARROW[policy[r, c]],
                        ha="center", va="center",
                        fontsize=14, color="white", alpha=0.90)

    ax.set_xlim(0, 4); ax.set_ylim(0, 4)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(title, color="white", fontsize=11, pad=6)
    for spine in ax.spines.values():
        spine.set_visible(False)
    plt.tight_layout()
    return fig


def plot_curves(losses, rewards, wins):
    smooth = lambda x, w=20: pd.Series(x).rolling(w, min_periods=1).mean().tolist()

    fig, axes = plt.subplots(1, 3, figsize=(13, 3))
    fig.patch.set_facecolor("#0f172a")

    datasets = [
        (axes[0], smooth(losses),  "#c084fc", "Loss (Smoothed, w=20)",   "MSE Loss"),
        (axes[1], smooth(rewards), "#34d399", "Reward (Smoothed, w=20)", "Episode Reward"),
        (axes[2], [np.mean(wins[max(0, i-50):i+1]) for i in range(len(wins))],
                               "#60a5fa", "Win Rate (last 50 eps)",   "Win Rate"),
    ]

    for ax, data, color, title, ylabel in datasets:
        ax.plot(data, color=color, linewidth=1.5)
        ax.set_facecolor("#1e293b")
        ax.set_title(title, color="white", fontsize=10)
        ax.set_ylabel(ylabel, color="#94a3b8", fontsize=8)
        ax.set_xlabel("Episode", color="#94a3b8", fontsize=8)
        ax.tick_params(colors="#94a3b8", labelsize=7)
        for spine in ax.spines.values():
            spine.set_color("#334155")

    plt.tight_layout()
    return fig


def plot_value_advantage(q_grid):
    """繪製 Value / Advantage 分解的 Q-value heatmap（4 個動作）。"""
    action_names = ["Up", "Down", "Left", "Right"]
    fig, axs = plt.subplots(2, 2, figsize=(6, 5))
    fig.patch.set_facecolor("#0f172a")
    for i, ax in enumerate(axs.flat):
        im = ax.imshow(q_grid[:, :, i], cmap="RdYlGn", vmin=-3, vmax=6)
        ax.set_title(f"{ACTION_ARROW[i]} {action_names[i]}", color="white", fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_facecolor("#1e293b")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────────────────────────
tab_train, tab_result, tab_analysis = st.tabs(["Training", "Results", "Analysis"])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1: 訓練
# ══════════════════════════════════════════════════════════════════════════════
with tab_train:
    # 預載入提示橫幅
    if st.session_state.get("preloaded2"):
        st.info("📦 目前顯示的是預存訓練結果。"
                "可按 **▶ 開始訓練** 並重新訓練以得到新結果。", icon="📦")
    col_btn1, col_btn2, _ = st.columns([1, 1, 5])
    start_btn = col_btn1.button("▶ 開始訓練", type="primary", use_container_width=True)
    reset_btn = col_btn2.button("🔄 重置",    use_container_width=True)

    if reset_btn:
        for k in ("agent2", "losses2", "rewards2", "wins2", "trained2", "preloaded2"):
            st.session_state[k] = (None if k == "agent2"
                                   else [] if k in ("losses2", "rewards2", "wins2")
                                   else False)
        st.rerun()

    m1, m2, m3, m4 = st.columns(4)
    ep_ph   = m1.empty()
    loss_ph = m2.empty()
    rwd_ph  = m3.empty()
    wr_ph   = m4.empty()

    def render_metrics(ep, loss, rwd, wr, eps):
        for ph, val, lbl in [
            (ep_ph,   f"{ep}/{epochs}",  "Episode"),
            (loss_ph, f"{loss:.4f}",     "Avg Loss"),
            (rwd_ph,  f"{rwd:+.0f}",     "Last Reward"),
            (wr_ph,   f"{wr:.0%}",       f"Win Rate | e={eps:.2f}"),
        ]:
            ph.markdown(f"""
            <div class="metric-box">
              <div class="metric-val">{val}</div>
              <div class="metric-lbl">{lbl}</div>
            </div>""", unsafe_allow_html=True)

    chart_ph    = st.empty()
    progress_ph = st.progress(0)
    status_ph   = st.empty()

    if start_btn:
        agent = DQNAgent(
            gamma=gamma, lr=lr,
            buffer_size=buf_size, batch_size=batch_size,
            target_update_freq=tuf,
        )
        losses, rewards, wins = [], [], []

        for ep in range(1, epochs + 1):
            result = run_episode(agent, mode="player", max_steps=max_steps)
            agent.decay_epsilon(epochs)

            losses.append(result["avg_loss"])
            rewards.append(result["reward"])
            wins.append(int(result["win"]))

            if ep % update_every == 0 or ep == epochs:
                wr = np.mean(wins[-50:]) if len(wins) >= 50 else np.mean(wins)
                render_metrics(ep, result["avg_loss"], result["reward"], wr, agent.epsilon)
                progress_ph.progress(ep / epochs)
                status_ph.caption(
                    f"🔄 Episode {ep}/{epochs}  |  "
                    f"Win Rate={wr:.0%}  |  ε={agent.epsilon:.3f}"
                )

        with chart_ph.container():
            st.pyplot(plot_curves(losses, rewards, wins), use_container_width=True)
            plt.close("all")
        status_ph.empty()

        st.session_state.update({
            "agent2": agent, "losses2": losses,
            "rewards2": rewards, "wins2": wins, "trained2": True,
        })
        st.success(f"✅ 訓練完成！最終勝率（近 100 回合）= {np.mean(wins[-100:]):.1%}")

    elif st.session_state.trained2:
        render_metrics(
            epochs,
            np.mean(st.session_state.losses2[-20:]),
            st.session_state.rewards2[-1],
            np.mean(st.session_state.wins2[-50:]),
            st.session_state.agent2.epsilon,
        )
        with chart_ph.container():
            st.pyplot(
                plot_curves(st.session_state.losses2,
                            st.session_state.rewards2,
                            st.session_state.wins2),
                use_container_width=True,
            )
        progress_ph.progress(1.0)
    else:
        st.info("👈 設定超參數後，按 **▶ 開始訓練** 啟動。")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2: 結果分析
# ══════════════════════════════════════════════════════════════════════════════
with tab_result:
    if not st.session_state.trained2:
        st.info("請先完成訓練。")
    else:
        agent   = st.session_state.agent2
        losses  = st.session_state.losses2
        rewards = st.session_state.rewards2
        wins    = st.session_state.wins2

        # ── 統計摘要
        st.subheader("Training Summary")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Win Rate (last 100 eps)",   f"{np.mean(wins[-100:]):.1%}")
        c2.metric("Avg Loss (last 100 eps)",   f"{np.mean(losses[-100:]):.4f}")
        c3.metric("Avg Reward (last 100 eps)", f"{np.mean(rewards[-100:]):+.2f}")
        c4.metric("Total Episodes",            f"{len(losses)}")

        # ── 曲線圖
        st.subheader("Training Curves")
        st.pyplot(plot_curves(losses, rewards, wins), use_container_width=True)
        plt.close("all")

        # ── 策略地圖 + Q-value Heatmap
        st.subheader("Policy Grid — Learned Strategy (Player Mode)")
        policy, q_grid, board = compute_policy(agent, mode="player")

        col_board, col_q = st.columns(2)
        with col_board:
            demo = run_episode(agent, mode="player", max_steps=30, training=False)
            fig  = draw_board(demo["trajectory"][0], policy=policy,
                              title="Sample Board + Policy Arrows")
            st.pyplot(fig, use_container_width=True)
            plt.close("all")
            steps = len(demo["trajectory"]) - 1
            outcome = "✅ Win" if demo["win"] else "❌ Loss / Timeout"
            st.caption(f"Demo result: **{outcome}** in {steps} steps  |  "
                       f"Total Reward: {demo['reward']:+.0f}")

        with col_q:
            st.markdown("**Q-value Heatmap (per action, Dueling output)**")
            fig2 = plot_value_advantage(q_grid)
            st.pyplot(fig2, use_container_width=True)
            plt.close("all")

        # ── Win-rate 對比（使用 Static vs Player 雙模式測試）
        st.subheader("Cross-Mode Evaluation")
        st.caption("以 Stage 2 模型在兩種模式下各跑 100 場，比較泛化能力。")

        if st.button("Run 100-game Evaluation (Player + Static)"):
            with st.spinner("Evaluating Player Mode..."):
                wr_player = evaluate(agent, n=100, mode="player")
            with st.spinner("Evaluating Static Mode..."):
                wr_static = evaluate(agent, n=100, mode="static")

            col_a, col_b = st.columns(2)
            col_a.metric("Player Mode Win Rate", f"{wr_player:.1%}",
                         help="訓練模式，預期較高")
            col_b.metric("Static Mode Win Rate", f"{wr_static:.1%}",
                         help="跨模式測試，觀察泛化能力")
            st.success("Evaluation complete!")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 0: 分析說明
# ══════════════════════════════════════════════════════════════════════════════
with tab_analysis:
    st.subheader("Stage 2 — Player Mode 機制分析")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### 1. 失敗症狀 (Failure Symptom)")
        st.markdown("""
<div class="analysis-box">
Stage 1 (S1+S2) 應用於 Player Mode 時出現：<br>
• Q 值系統性<b>過估 (Overestimation)</b><br>
• Loss 收斂緩慢，震盪大<br>
• Win Rate 停滯在 40~60%，無法突破<br>
• 對不同起始位置泛化能力不足
</div>
""", unsafe_allow_html=True)

        st.markdown("#### 2. 為何失敗 (Why it fails)")
        st.markdown("""
<div class="analysis-box">
<b>Overestimation Bias：</b><br>
Vanilla DQN 使用 <code>max(Q_target)</code> 同時「選動作」與「評估 Q 值」，
max 操作的系統性偏差在 Player Mode 隨機起始狀態分佈下放大。<br><br>
<b>泛化不足：</b><br>
單一 Q 值對「狀態好壞」與「動作優劣」混合估計，
無法在不同起始位置間有效共用學到的策略。
</div>
""", unsafe_allow_html=True)

    with col2:
        st.markdown("#### 3. 解法 (Solution)")
        st.markdown("""
<div class="analysis-box">
<span class="badge-s3">S3 Double DQN</span><br>
主網路選動作，目標網路評估 Q 值：<br>
<code>target = r + γ · Q_target(s', argmax_a Q_online(s', a))</code><br>
→ 分離選擇與評估，消除最大化偏差<br><br>
<span class="badge-s4">S4 Dueling Network</span><br>
<code>Q(s,a) = V(s) + A(s,a) − mean(A(s,·))</code><br>
→ 獨立學習「狀態價值 V(s)」與「動作優勢 A(s,a)」<br>
→ 對相同目標/陷阱位置但不同起始點的狀態更有效率
</div>
""", unsafe_allow_html=True)

        st.markdown("#### 4. 為何跳過 S5 (Why skip S5)")
        st.markdown("""
<div class="analysis-box">
Player Mode 中目標與陷阱位置<b>固定</b>，獎勵訊號密度充足，
Agent 能在合理 episode 數內收到正負回饋。<br><br>
PER (Prioritized Experience Replay) 主要解決<b>極度稀疏獎勵</b>，
留待 Stage 3 (Random Mode) 時再引入。
</div>
""", unsafe_allow_html=True)

    st.divider()
    st.subheader("Dueling Network 架構圖")
    st.code("""
Input (64,)
    │
    ├─ Dense(150, ReLU)
    │       │
    │   Dense(100, ReLU)
    │       │
    │   ┌───┴───────────────┐
    │   │                   │
    │ Dense(64, ReLU)   Dense(64, ReLU)
    │ Dense(1)          Dense(4)
    │   V(s)            A(s, a)
    │   │                   │
    └───┴──── Q = V + (A - mean(A)) ────► Output (4,)
""", language="text")

    st.subheader("S3 Double DQN 更新公式")
    st.latex(r"""
a^* = \arg\max_a Q_{\text{online}}(s', a)
\qquad
y = r + \gamma \cdot Q_{\text{target}}(s', a^*) \cdot (1 - \text{done})
""")
