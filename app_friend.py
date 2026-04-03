import fitz
import re
import streamlit as st
from openai import OpenAI

st.set_page_config(page_title="レポート生成アプリ", layout="wide")

st.title("複数資料対応 レポート生成アプリ")
st.caption("複数PDFを読み込み、テーマに応じて関連部分を抽出し、1本のレポートを生成します。")

# ==============================
# APIキー設定
# Streamlit Cloudでは Secrets を使う
# ローカルで試すときは sidebar から入力
# ==============================
secrets_key = st.secrets.get("OPENAI_API_KEY", None)
api_key_input = st.sidebar.text_input("OpenAI APIキー（ローカル確認用）", type="password")
api_key = secrets_key or api_key_input

# ==============================
# 入力欄
# ==============================
uploaded_files = st.file_uploader(
    "PDFを複数アップロードしてください",
    type=["pdf"],
    accept_multiple_files=True,
)

theme = st.text_input(
    "レポートのテーマ",
    placeholder="例：この患者の問題点、これからの治療計画"
)

target_length = st.number_input(
    "目標文字数",
    min_value=200,
    max_value=5000,
    value=1200,
    step=100,
)

score_threshold = st.sidebar.selectbox(
    "採用する関連度の最低点",
    [1, 2, 3],
    index=1,
)

max_sources = st.sidebar.slider(
    "最大採用ページ数",
    min_value=3,
    max_value=15,
    value=8,
    step=1,
)

run_button = st.button("レポート生成")


# ==============================
# 関数群
# ==============================
def get_client(key: str) -> OpenAI:
    return OpenAI(api_key=key.strip())


def extract_pages(files):
    all_pages = []
    for uploaded_file in files:
        file_bytes = uploaded_file.read()
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        for i, page in enumerate(doc):
            text = page.get_text("text").strip()
            if text and text != "（テキストなし）":
                all_pages.append({
                    "file": uploaded_file.name,
                    "page": i + 1,
                    "text": text,
                })
    return all_pages


def summarize_page(client: OpenAI, text: str) -> str:
    prompt = f"""
以下は講義スライド1ページ分の内容です。
重要点だけを日本語で80〜120字程度で簡潔に要約してください。
余計な説明は禁止です。

{text[:3000]}
"""
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "あなたは要点整理が得意なアシスタントです。簡潔に要約してください。"},
            {"role": "user", "content": prompt},
        ],
    )
    return response.choices[0].message.content.strip()


def score_page_for_theme(client: OpenAI, theme: str, summary: str):
    prompt = f"""
あなたの仕事は、スライド要約がレポートテーマにどれだけ使えるかを評価することです。

テーマ:
{theme}

スライド要約:
{summary}

次の形式だけで答えてください。
score: 0/1/2/3
reason: 20字以内

評価基準:
0 = 無関係
1 = 少し関係ある
2 = かなり関係ある
3 = レポートの核になる
"""
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "厳密に評価してください。指定形式以外で答えないでください。"},
            {"role": "user", "content": prompt},
        ],
    )

    content = response.choices[0].message.content.strip()
    score_match = re.search(r"score:\s*([0-3])", content)
    reason_match = re.search(r"reason:\s*(.*)", content)

    score = int(score_match.group(1)) if score_match else 0
    reason = reason_match.group(1).strip() if reason_match else ""
    return score, reason


def build_source_text(selected_pages, max_items=12):
    chosen = selected_pages[:max_items]
    blocks = []
    for p in chosen:
        blocks.append(
            f"資料: {p['file']} / Page {p['page']}\n"
            f"関連度: {p['score']}\n"
            f"要約: {p['summary']}\n"
        )
    return "\n".join(blocks)


def generate_report(client: OpenAI, theme: str, target_length: int, source_text: str) -> str:
    prompt = f"""
以下は複数の講義資料から、テーマに関連する部分だけを抽出・要約したものです。

{source_text}

この材料をもとに、「{theme}」というテーマで約{target_length}字のレポートを書いてください。

条件:
・大学生の提出レポートとして自然な日本語にする
・本文のみ出力する
・見出しは禁止
・箇条書きは禁止
・Markdown記法は禁止
・#、##、###、*、- は使わない
・導入→本論→まとめの流れを自然に含める
・資料にないことを大きく膨らませない
・資料に書かれていない治療や薬剤は極力追加しない
・複数資料を統合した内容にする
・抽象論だけでなく、適度に具体性を持たせる
"""
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": "あなたは優秀な大学生です。提出用レポート本文のみをプレーンテキストで出力してください。Markdownは禁止です。",
            },
            {"role": "user", "content": prompt},
        ],
    )
    return response.choices[0].message.content.strip()


def add_human_touch(client: OpenAI, report: str, theme: str) -> str:
    prompt = f"""
以下のレポートを、提出用として自然で人間らしい文章に改善してください。

テーマ:
{theme}

レポート:
{report}

条件:
・本文のみ出力
・見出し禁止
・箇条書き禁止
・Markdown禁止
・#、##、###、*、- は使わない
・不自然なAIっぽさを減らす
・1〜2文だけ、学びや気づきが感じられる表現を入れる
・感想文にはしすぎない
・元の論旨は保つ
"""
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": "あなたは自然な日本語を書く大学生です。本文のみを出力してください。",
            },
            {"role": "user", "content": prompt},
        ],
    )
    return response.choices[0].message.content.strip()


def adjust_length(client: OpenAI, text: str, target_length: int) -> str:
    prompt = f"""
以下のレポート本文を、内容をできるだけ保ったまま約{target_length}字に調整してください。

条件:
・本文のみ出力
・見出し禁止
・箇条書き禁止
・Markdown禁止
・#、##、###、*、- は使わない
・不自然な言い換えは避ける
・短すぎず長すぎず、目標字数に近づける

本文:
{text}
"""
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": "文字数調整が得意な大学生として、自然な本文だけを返してください。",
            },
            {"role": "user", "content": prompt},
        ],
    )
    return response.choices[0].message.content.strip()


def clean_text(text: str) -> str:
    bad_patterns = ["###", "##", "#", "* ", "- "]
    for b in bad_patterns:
        text = text.replace(b, "")
    return text.strip()


# ==============================
# 実行
# ==============================
if run_button:
    if not api_key:
        st.error("APIキーを入力してください。")
        st.stop()

    if not uploaded_files:
        st.error("PDFを1つ以上アップロードしてください。")
        st.stop()

    if not theme.strip():
        st.error("テーマを入力してください。")
        st.stop()

    client = get_client(api_key)

    with st.spinner("PDFを解析中..."):
        all_pages = extract_pages(uploaded_files)

    st.success(f"文字ありページ数: {len(all_pages)}")

    with st.spinner("ページごとに要約中..."):
        summaries = []
        progress_bar = st.progress(0)
        for idx, p in enumerate(all_pages):
            s = summarize_page(client, p["text"])
            summaries.append({
                "file": p["file"],
                "page": p["page"],
                "text": p["text"],
                "summary": s,
            })
            progress_bar.progress((idx + 1) / len(all_pages))

    with st.spinner("テーマとの関連度を採点中..."):
        scored_pages = []
        progress_bar = st.progress(0)
        for idx, s in enumerate(summaries):
            score, reason = score_page_for_theme(client, theme, s["summary"])
            scored_pages.append({
                **s,
                "score": score,
                "reason": reason,
            })
            progress_bar.progress((idx + 1) / len(summaries))

    scored_pages = sorted(scored_pages, key=lambda x: x["score"], reverse=True)

    selected_pages = [p for p in scored_pages if p["score"] >= score_threshold]
    if len(selected_pages) < 3:
        selected_pages = scored_pages[:min(max_sources, len(scored_pages))]
    else:
        selected_pages = selected_pages[:max_sources]

    source_text = build_source_text(selected_pages, max_items=max_sources)

    with st.spinner("レポートを生成中..."):
        raw_report = generate_report(client, theme, target_length, source_text)
        raw_report = clean_text(raw_report)

        humanized_report = add_human_touch(client, raw_report, theme)
        humanized_report = clean_text(humanized_report)

        final_report = adjust_length(client, humanized_report, target_length)
        final_report = clean_text(final_report)

    st.subheader("採用されたページ")
    for p in selected_pages:
        with st.expander(f"{p['score']}点 | {p['file']} Page {p['page']} | {p['reason']}"):
            st.write(p["summary"])

    st.subheader("最終レポート")
    st.text_area("生成結果", final_report, height=500)
    st.write(f"文字数: {len(final_report)}")
