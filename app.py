import streamlit as st

# 1. 統一的 Page Config (必須是第一個 Streamlit 呼叫)
st.set_page_config(
    page_title="DQN Gridworld Project",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. 定義各個階段的頁面
stage1_page = st.Page(
    "streamlit_app.py",
    title="Stage 1: Static Mode",
    icon="🤖",
    default=True  # 預設首頁
)

stage2_page = st.Page(
    "streamlit_app2.py",
    title="Stage 2: Player Mode",
    icon="🧠"
)

stage3_page = st.Page(
    "streamlit_app3.py",
    title="Stage 3: Random Mode",
    icon="🔥"
)

# 3. 設定側欄導覽選單 (這會在左上角產生完美的模式切換列表)
pg = st.navigation({
    "DQN HW3 - Learning Stages": [stage1_page, stage2_page, stage3_page]
})

# 4. 加上簡單的側邊欄頁腳說明
with st.sidebar:
    st.caption("---")
    st.caption("NCHU 1142 DRL HW3")

# 5. 執行選中的頁面
pg.run()
