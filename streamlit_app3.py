"""
Streamlit 視覺化介面 — Stage 3 DQN Random Mode
執行方式: streamlit run streamlit_app3.py
機制: S1~S5 全部 + 梯度裁剪 / Soft Target Update / 獎勵裁剪 / LR Schedule
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

from stage3_dqn import (
    DQNAgent, run_episode, evaluate, compute_policy,
    ACTION_ARROW, CELL_COLOR
)

# ─────────────────────────────────────────────────────────────────────────────
# 頁面設定
# ─────────────────────────────────────────────────────────────────────────────
# (st.set_page_config 移至 app.py 統一設定)

st.markdown("""
<style>
  .metric-box{background:#1a0a2e;border-radius:10px;padding:12px;text-align:center;
              border:1px solid #4a1d8a;}
  .metric-val{font-size:2rem;font-weight:700;color:#f59e0b;}
  .metric-lbl{font-size:.8rem;color:#94a3b8;margin-top:4px;}
  .badge{padding:2px 9px;border-radius:12px;font-size:.75rem;font-weight:600;display:inline-block;margin:2px;}
  .badge-s5{background:#b45309;color:white;}
  .badge-stab{background:#065f46;color:white;}
  .analysis-box{background:#1e293b;border-left:4px solid #f59e0b;padding:14px 18px;
                border-radius:6px;margin-bottom:12px;font-size:.9rem;}
</style>
""", unsafe_allow_html=True)

st.title("🔥 DQN Gridworld — Stage 3: Random Mode")
st.caption("S1~S4 (繼承) + **S5 PER** + **梯度裁剪** + **Soft Target Update** + **獎勵裁剪** + **LR Decay**  |  TensorFlow / tf.keras")

# ─────────────────────────────────────────────────────────────────────────────
# 側欄
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ 超參數設定")

    st.subheader("基礎訓練")
    epochs     = st.slider("Training Episodes",  500, 5000, 2000, 100)
    gamma      = st.slider("折扣因子 γ",    0.50, 0.99, 0.90, 0.01)
    lr         = st.select_slider("初始學習率 (lr)",
                                  options=[1e-4, 5e-4, 1e-3, 5e-3],
                                  value=1e-3,
                                  format_func=lambda x: f"{x:.0e}")
    batch_size = st.select_slider("Batch Size", options=[32, 64, 128], value=64)
    buf_size   = st.select_slider("Buffer Size",
                                  options=[2000, 5000, 10000, 20000], value=10000)
    max_steps  = st.slider("每回合最大步數",   20, 100, 50, 10)
    update_every = st.slider("圖表更新頻率 (每 N 回合)", 10, 100, 25, 5)

    st.subheader("S5 PER 設定")
    per_alpha      = st.slider("PER alpha (優先權指數)", 0.0, 1.0, 0.6, 0.1)
    per_beta_start = st.slider("PER beta start (IS 修正)", 0.1, 0.9, 0.4, 0.1)
    per_beta_frames = st.slider("Beta 退火幀數", 500, 5000, 2000, 500)

    st.subheader("穩定化技巧")
    tau          = st.select_slider("Soft Update τ",
                                    options=[0.001, 0.005, 0.01, 0.05, 0.1],
                                    value=0.01,
                                    format_func=lambda x: f"{x:.3f}")
    reward_clip  = st.checkbox("獎勵裁剪 [-1, 1]", value=True)
    lr_decay_rate = st.slider("LR Decay Rate", 0.90, 1.00, 0.96, 0.01)
    lr_decay_steps = st.select_slider("LR Decay Steps",
                                      options=[500, 1000, 2000, 5000], value=2000)

    st.divider()
    for s, enabled, color in [
        ("S1 Replay Buffer", True, "green"),
        ("S2 Target Network", True, "green"),
        ("S3 Double DQN", True, "green"),
        ("S4 Dueling Network", True, "green"),
        ("S5 PER", True, "orange"),
        ("梯度裁剪", True, "orange"),
        ("Soft Target Update", True, "orange"),
        ("獎勵裁剪", reward_clip, "orange"),
        ("LR Schedule", True, "orange"),
    ]:
        icon = "✅" if enabled else "❌"
        st.markdown(f"**{s}** {icon}")

# ─────────────────────────────────────────────────────────────────────────────
# Session State
# ─────────────────────────────────────────────────────────────────────────────
for key, default in [
    ("agent3",    None),
    ("losses3",   []),
    ("rewards3",  []),
    ("wins3",     []),
    ("trained3",  False),
    ("preloaded3", False),
]:
    if key not in st.session_state:
        st.session_state[key] = default

# ─────────────────────────────────────────────────────────────────────────────
# 啟動自動載入：若已有预存模型檔案，直接還原訓練完成狀態
# ─────────────────────────────────────────────────────────────────────────────
_WEIGHTS_PATH = os.path.join(_DIR, "dqn_random_weights.weights.h5")
_LOG_PATH     = os.path.join(_DIR, "training_log_stage3.json")

if not st.session_state.trained3:
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
            "agent3":        _agent,
            "losses3":       _log_data["losses"],
            "rewards3":      _log_data["rewards"],
            "wins3":         _log_data["wins"],
            "trained3":      True,
            "preloaded3":    True,   # 標記為預載入，供 UI 區分
        })



# ─────────────────────────────────────────────────────────────────────────────
# 工具函式
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
                ax.text(c + 0.5, 3 - r + 0.18, ACTION_ARROW[policy[r, c]],
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
    smooth = lambda x, w=30: pd.Series(x).rolling(w, min_periods=1).mean().tolist()
    fig, axes = plt.subplots(1, 3, figsize=(13, 3))
    fig.patch.set_facecolor("#0f172a")

    datasets = [
        (axes[0], smooth(losses),  "#fb923c", "Loss (Smoothed, w=30)",   "MSE Loss"),
        (axes[1], smooth(rewards), "#34d399", "Reward (Smoothed, w=30)", "Episode Reward"),
        (axes[2], [np.mean(wins[max(0, i-100):i+1]) for i in range(len(wins))],
                               "#60a5fa", "Win Rate (last 100 eps)", "Win Rate"),
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


def plot_per_beta(buf, total_eps):
    """顯示 PER beta 退火曲線（理論值）。"""
    frames = np.arange(1, total_eps + 1)
    betas  = [buf.beta_start + min(1.0, f / buf.beta_frames) *
              (1.0 - buf.beta_start) for f in frames]
    fig, ax = plt.subplots(figsize=(6, 2.5))
    fig.patch.set_facecolor("#0f172a")
    ax.set_facecolor("#1e293b")
    ax.plot(frames, betas, color="#fb923c", linewidth=2)
    ax.axhline(1.0, color="#94a3b8", linewidth=0.8, linestyle="--")
    ax.set_title("PER Beta Annealing Schedule", color="white", fontsize=10)
    ax.set_xlabel("Episode", color="#94a3b8", fontsize=8)
    ax.set_ylabel("Beta (IS weight)", color="#94a3b8", fontsize=8)
    ax.tick_params(colors="#94a3b8", labelsize=7)
    for spine in ax.spines.values():
        spine.set_color("#334155")
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
    if st.session_state.get("preloaded3"):
        st.info("📦 目前顯示的是預存訓練結果。"
                "可按 **▶ 開始訓練** 並重新訓練以得到新結果。", icon="📦")
    col_btn1, col_btn2, _ = st.columns([1, 1, 5])
    start_btn = col_btn1.button("▶ 開始訓練", type="primary", use_container_width=True)
    reset_btn = col_btn2.button("🔄 重置",    use_container_width=True)

    if reset_btn:
        for k in ("agent3", "losses3", "rewards3", "wins3", "trained3", "preloaded3"):
            st.session_state[k] = (None if k == "agent3"
                                   else [] if k in ("losses3", "rewards3", "wins3")
                                   else False)
        st.rerun()

    m1, m2, m3, m4 = st.columns(4)
    ep_ph   = m1.empty()
    loss_ph = m2.empty()
    rwd_ph  = m3.empty()
    wr_ph   = m4.empty()

    def render_metrics(ep, loss, rwd, wr, eps):
        for ph, val, lbl in [
            (ep_ph,   f"{ep}/{epochs}", "Episode"),
            (loss_ph, f"{loss:.4f}",    "Avg Loss"),
            (rwd_ph,  f"{rwd:+.0f}",    "Last Reward"),
            (wr_ph,   f"{wr:.0%}",      f"Win Rate | ε={eps:.2f}"),
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
            tau=tau, reward_clip=reward_clip,
            lr_decay_steps=lr_decay_steps,
            lr_decay_rate=lr_decay_rate,
            per_alpha=per_alpha,
            per_beta_start=per_beta_start,
            per_beta_frames=per_beta_frames,
        )
        losses, rewards, wins = [], [], []

        for ep in range(1, epochs + 1):
            result = run_episode(agent, mode="random", max_steps=max_steps)
            agent.decay_epsilon(epochs)

            losses.append(result["avg_loss"])
            rewards.append(result["reward"])
            wins.append(int(result["win"]))

            if ep % update_every == 0 or ep == epochs:
                wr = np.mean(wins[-100:]) if len(wins) >= 100 else np.mean(wins)
                render_metrics(ep, result["avg_loss"], result["reward"], wr, agent.epsilon)
                progress_ph.progress(ep / epochs)
                status_ph.caption(
                    f"🔄 Episode {ep}/{epochs}  |  "
                    f"Win Rate={wr:.0%}  |  ε={agent.epsilon:.3f}  |  "
                    f"β={agent.replay_buf.beta:.3f}  |  "
                    f"Buffer={len(agent.replay_buf)}"
                )

        with chart_ph.container():
            st.pyplot(plot_curves(losses, rewards, wins), use_container_width=True)
            plt.close("all")
        status_ph.empty()

        st.session_state.update({
            "agent3": agent, "losses3": losses,
            "rewards3": rewards, "wins3": wins, "trained3": True,
        })
        st.success(f"✅ 訓練完成！最終勝率（近 100 回合）= {np.mean(wins[-100:]):.1%}")

    elif st.session_state.trained3:
        render_metrics(
            epochs,
            np.mean(st.session_state.losses3[-20:]),
            st.session_state.rewards3[-1],
            np.mean(st.session_state.wins3[-100:]),
            st.session_state.agent3.epsilon,
        )
        with chart_ph.container():
            st.pyplot(
                plot_curves(st.session_state.losses3,
                            st.session_state.rewards3,
                            st.session_state.wins3),
                use_container_width=True,
            )
        progress_ph.progress(1.0)
    else:
        st.info("👈 設定超參數後，按 **▶ 開始訓練** 啟動。")
        st.warning("⚠️ Random Mode 訓練較慢，建議至少 2000 個 episode。")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2: 結果分析
# ══════════════════════════════════════════════════════════════════════════════
with tab_result:
    if not st.session_state.trained3:
        st.info("請先完成訓練。")
    else:
        agent   = st.session_state.agent3
        losses  = st.session_state.losses3
        rewards = st.session_state.rewards3
        wins    = st.session_state.wins3

        # ── 統計摘要
        st.subheader("Training Summary")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Win Rate (last 100 eps)",   f"{np.mean(wins[-100:]):.1%}")
        c2.metric("Avg Loss (last 100 eps)",   f"{np.mean(losses[-100:]):.4f}")
        c3.metric("Avg Reward (last 100 eps)", f"{np.mean(rewards[-100:]):+.2f}")
        c4.metric("Total Episodes",            f"{len(losses)}")

        # ── 訓練曲線
        st.subheader("Training Curves")
        st.pyplot(plot_curves(losses, rewards, wins), use_container_width=True)
        plt.close("all")

        # ── PER Beta 退火曲線
        st.subheader("PER Beta Annealing")
        st.pyplot(plot_per_beta(agent.replay_buf, len(losses)), use_container_width=True)
        plt.close("all")

        # ── 策略地圖
        st.subheader("Policy Grid — Sample Random Board")
        policy, q_grid, board = compute_policy(agent, mode="random")

        col_board, col_q = st.columns(2)
        with col_board:
            demo = run_episode(agent, mode="random", max_steps=30, training=False)
            fig  = draw_board(demo["trajectory"][0], policy=policy,
                              title="Sample Board + Policy Arrows")
            st.pyplot(fig, use_container_width=True)
            plt.close("all")
            steps = len(demo["trajectory"]) - 1
            outcome = "✅ Win" if demo["win"] else "❌ Loss / Timeout"
            st.caption(f"Demo: **{outcome}** in {steps} steps  |  "
                       f"Total Reward: {demo['reward']:+.0f}")

        with col_q:
            st.markdown("**Q-value Heatmap (per action)**")
            action_names = ["Up", "Down", "Left", "Right"]
            fig2, axs = plt.subplots(2, 2, figsize=(6, 5))
            fig2.patch.set_facecolor("#0f172a")
            for i, ax in enumerate(axs.flat):
                im = ax.imshow(q_grid[:, :, i], cmap="RdYlGn", vmin=-3, vmax=6)
                ax.set_title(f"{ACTION_ARROW[i]} {action_names[i]}",
                             color="white", fontsize=9)
                ax.set_xticks([]); ax.set_yticks([])
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            plt.tight_layout()
            st.pyplot(fig2, use_container_width=True)
            plt.close("all")

        # ── 跨模式評估
        st.subheader("Cross-Mode Evaluation")
        st.caption("以 Stage 3 模型在三種難度模式下各跑 100 場，驗證泛化能力。")
        if st.button("Run 100-game Evaluation (All Modes)"):
            results = {}
            for mode in ["static", "player", "random"]:
                with st.spinner(f"Evaluating {mode} mode..."):
                    results[mode] = evaluate(agent, n=100, mode=mode)

            ca, cb, cc = st.columns(3)
            ca.metric("Static Mode",  f"{results['static']:.1%}")
            cb.metric("Player Mode",  f"{results['player']:.1%}")
            cc.metric("Random Mode",  f"{results['random']:.1%}",
                      help="訓練模式，預期最高")
            st.success("Evaluation complete!")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 0: 分析說明
# ══════════════════════════════════════════════════════════════════════════════
with tab_analysis:
    st.subheader("Stage 3 — Random Mode 機制分析")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### 1. 失敗症狀 (Failure Symptom)")
        st.markdown("""
<div class="analysis-box">
Stage 2 (S1~S4) 應用於 Random Mode 時出現：<br>
• <b>梯度爆炸</b>：Loss 突然跳至 100+，訓練崩潰<br>
• <b>稀疏獎勵停滯</b>：大多數 episode 超時，Win Rate ≈ 0%<br>
• <b>Q 值震盪</b>：每局棋盤完全不同，估值無法穩定
</div>
""", unsafe_allow_html=True)

        st.markdown("#### 2. 為何失敗 (Why it fails)")
        st.markdown("""
<div class="analysis-box">
<b>獎勵極度稀疏：</b><br>
隨機模式每局配置不同，Agent 難以從均勻取樣
Replay Buffer 中找到有效的「成功 transition」。<br><br>
<b>梯度不穩定：</b><br>
Q 值估計誤差大 → TD 誤差大 → 梯度劇烈震盪，
硬更新 Target Network 進一步放大不穩定性。<br><br>
<b>樣本失衡：</b><br>
99%+ 的樣本 reward = -1（普通步），
只有極少數 ±10（終止），均勻取樣效率極低。
</div>
""", unsafe_allow_html=True)

    with col2:
        st.markdown("#### 3. 解法 (Solution)")
        st.markdown("""
<div class="analysis-box">
<span class="badge badge-s5">S5 PER</span>
根據 TD 誤差優先取樣，聚焦高資訊量 transition：<br>
<code>P(i) = p_i^α / Σ p_k^α</code><br>
IS 權重修正取樣偏差：<code>w_i = (N·P(i))^{-β}</code><br>
beta 從 β₀ 線性退火至 1.0<br><br>
<span class="badge badge-stab">梯度裁剪</span>
clip_by_global_norm(grad, 1.0) 防止爆炸<br><br>
<span class="badge badge-stab">Soft Target Update</span>
θ_target ← τ·θ_online + (1-τ)·θ_target，
比週期性硬更新更平滑<br><br>
<span class="badge badge-stab">獎勵裁剪</span>
clip(r, -1, 1) 縮小 TD 誤差尺度<br><br>
<span class="badge badge-stab">LR Schedule</span>
ExponentialDecay，訓練後期降低學習率
</div>
""", unsafe_allow_html=True)

        st.markdown("#### 4. 全部機制啟用")
        st.markdown("""
<div class="analysis-box">
Random Mode 為最高難度挑戰，S1~S5 全部啟用，
各機制協同解決不同面向的問題：<br>
S1+S2 解決樣本相關性與目標穩定性，<br>
S3+S4 解決過估偏差與泛化能力，<br>
S5 解決稀疏獎勵的學習效率，<br>
穩定化技巧解決梯度爆炸與震盪。
</div>
""", unsafe_allow_html=True)

    st.divider()
    c_arch, c_per = st.columns(2)
    with c_arch:
        st.subheader("PER 取樣示意")
        st.code("""
Buffer 中的 transitions（依 TD 誤差排序）：
  ┌──────────────────────────────────┐
  │ [高 TD 誤差] ← 被頻繁取樣       │  ← α 控制集中程度
  │ [中 TD 誤差]                    │
  │ [低 TD 誤差] ← 較少被取樣       │
  └──────────────────────────────────┘
  IS 權重 w_i = (N·P(i))^{-β} 補償偏差
  beta 退火：β₀ → 1.0（訓練過程中逐漸矯正）
""", language="text")

    with c_per:
        st.subheader("Soft Target Update")
        st.latex(r"\theta_{target} \leftarrow \tau \cdot \theta_{online} + (1-\tau) \cdot \theta_{target}")
        st.markdown("每個訓練步驟都執行，τ 很小（如 0.01）使目標網路緩慢跟蹤主網路，比週期性硬更新更平滑穩定。")
        st.subheader("完整 Loss 公式")
        st.latex(r"\mathcal{L} = \frac{1}{N}\sum_i w_i \cdot \left(y_i - Q(s_i, a_i)\right)^2")
