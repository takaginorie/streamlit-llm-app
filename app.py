# app.py
# ------------------------------------------------------------
# LangChain × Streamlit の最小構成デモ
# ・入力フォーム1つ
# ・ラジオボタンで専門家A/Bを切替
# ・選択に応じてSystemメッセージを変更し、LLMへプロンプト
# ・回答を画面に表示
# ※Streamlit Community Cloud での Python は 3.11 を想定
# ------------------------------------------------------------
from dotenv import load_dotenv
load_dotenv()

import os
import streamlit as st

# --- LangChain (Lesson8 想定: 最新のRunnable構成) ---
# pip 例：
#   pip install "langchain>=0.2,<0.4" "langchain-openai>=0.2" openai>=1.0 streamlit
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


# ============ LLM 実行関数（要件：引数を受け取り文字列を返す） ============
def run_expert_llm(input_text: str, expert_choice: str) -> str:
    """
    入力テキストとラジオボタンの選択値（"A" もしくは "B"）を受け取り、
    LangChain 経由で LLM に問い合わせ、回答テキストを返します。
    """

    # 専門家A/Bの振る舞い（ご自身で設計）
    # ここでは A=「不動産投資アドバイザー」, B=「栄養学エキスパート」を例示
    expert_system_messages = {
        "A": (
            "あなたは経験豊富な不動産投資アドバイザーです。"
            "市場調査、収益性評価（表面・実質利回り）、資金計画、融資、出口戦略、"
            "税務上の留意点などを、わかりやすく段階的に助言してください。"
            "数値が必要な場合は、仮定条件を明示し簡便な試算も提案してください。"
        ),
        "B": (
            "あなたは科学的根拠を重視する栄養学エキスパートです。"
            "目的（減量・増量・健康維持等）に応じ、摂取カロリー、PFCバランス、"
            "食事例、買い物リスト、注意点を、具体的かつ実行可能な形で提示してください。"
            "一般向けの説明で専門用語は短く補足してください。"
        ),
    }

    system_text = expert_system_messages.get(
        expert_choice,
        "あなたは有能なアシスタントです。丁寧かつ具体的に回答してください。",
    )

    # LangChain の Prompt（Lesson8 で紹介されることが多い ChatPromptTemplate を採用）
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_text),
            (
                "human",
                # 必要に応じてフォーマットを拡張
                "以下のユーザー入力に基づき、専門家として最適なアドバイスを提示してください。\n\nユーザー入力: {user_input}",
            ),
        ]
    )

    # OpenAI APIキーの取得（Streamlit Cloud では st.secrets 推奨）
    openai_api_key = st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY"))
    if not openai_api_key:
        return (
            "⚠️ OpenAI APIキーが設定されていません。\n"
            "・ローカル: 環境変数 OPENAI_API_KEY を設定してください。\n"
            "・Streamlit Community Cloud: Secrets に OPENAI_API_KEY を追加してください。"
        )

    # LLM 定義（gpt-4o-mini 例。お好みで変更可）
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.4,
        api_key=openai_api_key,
    )

    # Runnable でチェーンを作成（Prompt → LLM → 文字列出力）
    chain = prompt | llm | StrOutputParser()

    # 実行
    answer: str = chain.invoke({"user_input": input_text})
    return answer.strip()


# ========================= Streamlit UI =========================
st.set_page_config(page_title="LangChain Lesson8 参考：専門家切替デモ", page_icon="🧭", layout="centered")

st.title("LangChain × Streamlit：専門家A/B切替デモ")
st.caption("Python 3.11 / LangChain Runnable 構成（Lesson8想定）")

with st.expander("アプリの概要・操作方法（必読）", expanded=True):
    st.markdown(
        """
**このアプリでできること**  
- テキストを1つ入力し、その内容を **LangChain** 経由で **LLM** に送って回答を表示します。  
- 画面の **ラジオボタン** で専門家の種類（A/B）を切り替えると、LLMの**システムメッセージ**が変化し、回答の観点が変わります。

**使い方**  
1. 上部のラジオボタンで **A（不動産投資）** または **B（栄養学）** を選びます。  
2. 入力欄に相談したい内容を記入します。  
3. **「送信」** ボタンを押すと、LLMの回答が表示されます。

**デプロイのポイント（Streamlit Community Cloud）**  
- **Python のバージョンは 3.11** を使用してください。  
  - 例：`[tool.poetry]` / `runtime.txt` / `packages` など環境に応じた指定方法で 3.11 を選択  
- **Secrets** に `OPENAI_API_KEY` を登録してください（ローカルの場合は環境変数でも可）。
        """
    )

# 専門家の選択（A/B）
expert_choice = st.radio(
    "LLMに振る舞わせる専門家を選択してください：",
    options=["A", "B"],
    index=0,
    captions=["A：不動産投資アドバイザー", "B：栄養学エキスパート"],
    horizontal=True,
)

# 入力フォーム（単一）
user_input = st.text_area(
    "入力フォーム",
    placeholder="相談したい内容を入力してください（例：都心中古ワンルーム投資の注意点 / 体脂肪を落とすための食事例など）",
    height=140,
)

# 送信ボタン
if st.button("送信", type="primary", use_container_width=True):
    if not user_input.strip():
        st.warning("入力フォームが空です。テキストを入力してください。")
    else:
        with st.spinner("LLM に問い合わせ中..."):
            result_text = run_expert_llm(user_input.strip(), expert_choice)
        st.subheader("回答")
        st.write(result_text)

# フッター
st.markdown("---")
st.caption(
    "Powered by LangChain & OpenAI · 環境変数または Streamlit Secrets に OPENAI_API_KEY を設定してください · Python 3.11"
)
