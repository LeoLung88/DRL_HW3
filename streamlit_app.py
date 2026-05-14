"""
Streamlit 視覺化介面 — Stage 1 DQN Static Mode
執行方式: streamlit run streamlit_app.py
"""
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pandas as pd
import os, sys, json

# 確保能 import stage1_dqn
_DIR = os.path.dirname(os.path.abspath(__file__))
if _DIR not in sys.path:
    sys.path.insert(0, _DIR)

from stage1_dqn import (
    DQNAgent, run_episode, evaluate, compute_policy,
    ACTION_ARROW, CELL_COLOR
)

# ─────────────────────────────────────────────────────────────────────────────
# 頁面設定
# ─────────────────────────────────────────────────────────────────────────────
# (st.set_page_config 移至 app.py 統一設定)

st.markdown("""
<style>
  .metric-box{background:#1e293b;border-radius:10px;padding:12px;text-align:center;}
  .metric-val{font-size:2rem;font-weight:700;color:#38bdf8;}
  .metric-lbl{font-size:.8rem;color:#94a3b8;margin-top:4px;}
</style>
""", unsafe_allow_html=True)

st.title("🤖 DQN Gridworld — Stage 1: Static Mode")
st.caption("基本 DQN + S1 Replay Buffer + S2 Target Network  |  框架：TensorFlow / tf.keras")

# ─────────────────────────────────────────────────────────────────────────────
# 側欄：超參數設定
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ 超參數設定")
    epochs       = st.slider("Training Episodes",            200,  2000, 1000, 100)
    gamma        = st.slider("折扣因子 γ",              0.50, 0.99, 0.90, 0.01)
    lr           = st.select_slider("學習率 (lr)",
                                    options=[1e-4, 5e-4, 1e-3, 5e-3, 1e-2],
                                    value=1e-3,
                                    format_func=lambda x: f"{x:.0e}")
    batch_size   = st.select_slider("Batch Size",       options=[32, 64, 128], value=64)
    buf_size     = st.select_slider("Buffer Size",      options=[500, 1000, 2000, 5000], value=2000)
    tuf          = st.slider("Target Update Freq",       10,  200,  50,  10)
    update_every = st.slider("圖表更新頻率 (每 N 回合)", 10,  100,  20,  10)
    max_steps    = st.slider("每回合最大步數",            20,  100,  50,  10)

    st.divider()
    st.markdown("**S1** ✅ Replay Buffer")
    st.markdown("**S2** ✅ Target Network")
    st.markdown("**S3** ❌ Double DQN（階段二）")
    st.markdown("**S4** ❌ Dueling DQN（階段二）")
    st.markdown("**S5** ❌ PER（階段三）")

# ─────────────────────────────────────────────────────────────────────────────
# Session State 初始化
# ─────────────────────────────────────────────────────────────────────────────
for key, default in [
    ("agent",    None),
    ("losses",   []),
    ("rewards",  []),
    ("wins",     []),
    ("trained",  False),
    ("preloaded", False),
]:
    if key not in st.session_state:
        st.session_state[key] = default

# ─────────────────────────────────────────────────────────────────────────────
# 啟動自動載入：若已有预存模型檔案，直接還原訓練完成狀態
# ─────────────────────────────────────────────────────────────────────────────
_WEIGHTS_PATH = os.path.join(_DIR, "dqn_static_weights.weights.h5")
_LOG_PATH     = os.path.join(_DIR, "training_log.json")

if not st.session_state.trained:
    if os.path.exists(_WEIGHTS_PATH) and os.path.exists(_LOG_PATH):
        _agent = DQNAgent()                               # 建構相同架構
        _agent.model.load_weights(_WEIGHTS_PATH)          # 載入预存權重
        _agent.target_model.set_weights(                  # 同步 target
            _agent.model.get_weights()
        )
        _agent.epsilon = _agent.epsilon_min               # 設為訓練完成狀態
        with open(_LOG_PATH, "r") as _f:
            _log_data = json.load(_f)
        st.session_state.update({
            "agent":        _agent,
            "losses":       _log_data["losses"],
            "rewards":      _log_data["rewards"],
            "wins":         _log_data["wins"],
            "trained":      True,
            "preloaded":    True,   # 標記為預載入，供 UI 區分
        })



# ─────────────────────────────────────────────────────────────────────────────
# 工具函式：繪製 Gridworld 棋盤
# ─────────────────────────────────────────────────────────────────────────────
def draw_board(board_np, policy=None, title="", figsize=(4, 4)):
    """board_np: shape (4,4,4) numpy 陣列（render_np 輸出）"""
    # 解碼各 channel
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
            cell = grid[r, c]
            color = CELL_COLOR.get(cell, "#F5F5F5")
            rect = patches.FancyBboxPatch(
                (c + 0.05, 3 - r + 0.05), 0.90, 0.90,
                boxstyle="round,pad=0.05",
                linewidth=0, facecolor=color, alpha=0.92)
            ax.add_patch(rect)

            # Cell label: use plain ASCII (emoji/CJK not supported by default mpl font)
            label = {"P": "P", "+": "GOAL", "-": "TRAP", "W": "WALL", " ": ""}
            ax.text(c + 0.5, 3 - r + 0.55, label.get(cell, cell),
                    ha="center", va="center", fontsize=11,
                    color="white", fontweight="bold")

            # Policy arrow (ASCII arrows already safe)
            if policy is not None and policy[r, c] >= 0:
                ax.text(c + 0.5, 3 - r + 0.18,
                        ACTION_ARROW[policy[r, c]],
                        ha="center", va="center",
                        fontsize=14, color="white", alpha=0.90)

    ax.set_xlim(0, 4)
    ax.set_ylim(0, 4)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, color="white", fontsize=11, pad=6)
    for spine in ax.spines.values():
        spine.set_visible(False)
    plt.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 工具函式：繪製訓練曲線
# ─────────────────────────────────────────────────────────────────────────────
def plot_curves(losses, rewards, wins):
    smooth = lambda x, w=20: pd.Series(x).rolling(w, min_periods=1).mean().tolist()

    fig, axes = plt.subplots(1, 3, figsize=(13, 3))
    fig.patch.set_facecolor("#0f172a")

    datasets = [
        (axes[0], smooth(losses),  "#f472b6", "Loss (Smoothed, w=20)",    "MSE Loss"),
        (axes[1], smooth(rewards), "#34d399", "Reward (Smoothed, w=20)",  "Episode Reward"),
        (axes[2], [np.mean(wins[max(0,i-50):i+1]) for i in range(len(wins))],
                               "#60a5fa", "Win Rate (last 50 eps)",    "Win Rate"),
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


# ─────────────────────────────────────────────────────────────────────────────
# TAB 定義
# ─────────────────────────────────────────────────────────────────────────────
tab_train, tab_result = st.tabs(["Training", "Results"])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1: 訓練
# ══════════════════════════════════════════════════════════════════════════════
with tab_train:
    # 預載入提示橫幅
    if st.session_state.get("preloaded"):
        st.info("📦 目前顯示的是預存訓練結果。"
                "可按 **▶ 開始訓練** 並重新訓練以得到新結果。", icon="📦")
    col_btn1, col_btn2, _ = st.columns([1, 1, 5])
    start_btn = col_btn1.button("▶ 開始訓練", type="primary", use_container_width=True)
    reset_btn = col_btn2.button("🔄 重置",    use_container_width=True)

    if reset_btn:
        for k in ("agent", "losses", "rewards", "wins", "trained", "preloaded"):
            st.session_state[k] = (None if k == "agent"
                                   else [] if k in ("losses", "rewards", "wins")
                                   else False)
        st.rerun()

    # 指標列
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
    # 輕量狀態列（不觸發 matplotlib，速度快）
    status_ph   = st.empty()

    if start_btn:
        # 建立新 Agent
        agent = DQNAgent(
            gamma=gamma, lr=lr,
            buffer_size=buf_size, batch_size=batch_size,
            target_update_freq=tuf,
        )
        losses, rewards, wins = [], [], []

        for ep in range(1, epochs + 1):
            result = run_episode(agent, mode="static", max_steps=max_steps)
            agent.decay_epsilon(epochs)

            losses.append(result["avg_loss"])
            rewards.append(result["reward"])
            wins.append(int(result["win"]))

            if ep % update_every == 0 or ep == epochs:
                wr = np.mean(wins[-50:]) if len(wins) >= 50 else np.mean(wins)
                # ── 只更新文字指標（不重繪 matplotlib，避免瓶頸）──────
                render_metrics(ep, result["avg_loss"], result["reward"], wr, agent.epsilon)
                progress_ph.progress(ep / epochs)
                status_ph.caption(
                    f"🔄 Episode {ep}/{epochs}  |  "
                    f"Win Rate={wr:.0%}  |  ε={agent.epsilon:.3f}"
                )

        # ── 訓練結束後，才一次性繪製曲線圖（省去訓練中大量 pyplot 呼叫）──
        with chart_ph.container():
            st.pyplot(plot_curves(losses, rewards, wins), use_container_width=True)
            plt.close("all")   # 釋放 matplotlib 記憶體
        status_ph.empty()

        # 儲存到 session state
        st.session_state.update({
            "agent": agent, "losses": losses,
            "rewards": rewards, "wins": wins, "trained": True,
        })
        st.success(f"✅ 訓練完成！最終勝率（近 100 回合）= {np.mean(wins[-100:]):.1%}")

    elif st.session_state.trained:
        render_metrics(
            epochs,
            np.mean(st.session_state.losses[-20:]),
            st.session_state.rewards[-1],
            np.mean(st.session_state.wins[-50:]),
            st.session_state.agent.epsilon,
        )
        with chart_ph.container():
            st.pyplot(
                plot_curves(st.session_state.losses,
                            st.session_state.rewards,
                            st.session_state.wins),
                use_container_width=True,
            )
        progress_ph.progress(1.0)
    else:
        st.info("👈 設定超參數後，按 **▶ 開始訓練** 啟動。")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2: 結果分析
# ══════════════════════════════════════════════════════════════════════════════
with tab_result:
    if not st.session_state.trained:
        st.info("請先完成訓練。")
    else:
        agent   = st.session_state.agent
        losses  = st.session_state.losses
        rewards = st.session_state.rewards
        wins    = st.session_state.wins

        # ── 統計摘要 ─────────────────────────────────────────────────
        st.subheader("Training Summary")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Win Rate (last 100 eps)",    f"{np.mean(wins[-100:]):.1%}")
        c2.metric("Avg Loss (last 100 eps)",    f"{np.mean(losses[-100:]):.4f}")
        c3.metric("Avg Reward (last 100 eps)",  f"{np.mean(rewards[-100:]):+.2f}")
        c4.metric("Total Episodes",             f"{len(losses)}")

        # ── 曲線圖 ──────────────────────────────────────────────────
        st.subheader("Training Curves")
        st.pyplot(plot_curves(losses, rewards, wins), use_container_width=True)

        # ── 策略地圖 ─────────────────────────────────────────────────
        st.subheader("Policy Grid — Learned Strategy")
        policy, q_grid, board = compute_policy(agent, mode="static")

        col_board, col_q = st.columns(2)
        with col_board:
            from stage1_dqn import run_episode as _re  # noqa
            demo = _re(agent, mode="static", max_steps=30, training=False)
            fig = draw_board(demo["trajectory"][0], policy=policy,
                             title="Initial Board + Policy Arrows")
            st.pyplot(fig, use_container_width=True)

        with col_q:
            st.markdown("**Q-value Heatmap (per action)**")
            action_names = ["↑ Up", "↓ Down", "← Left", "→ Right"]
            fig2, axs = plt.subplots(2, 2, figsize=(6, 5))
            fig2.patch.set_facecolor("#0f172a")
            for i, ax in enumerate(axs.flat):
                im = ax.imshow(q_grid[:, :, i], cmap="RdYlGn", vmin=-2, vmax=5)
                ax.set_title(action_names[i], color="white", fontsize=9)
                ax.set_xticks([]); ax.set_yticks([])
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            plt.tight_layout()
            st.pyplot(fig2, use_container_width=True)

        # ── 200 場測試 ───────────────────────────────────────────────
        if st.button("Run 200-game Evaluation"):
            with st.spinner("Evaluating..."):
                wr = evaluate(agent, n=200, mode="static")
            st.success(f"Win rate over 200 games: **{wr:.1%}**")



