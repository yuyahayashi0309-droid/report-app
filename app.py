import fitz
import re
import streamlit as st
from openai import OpenAI

st.set_page_config(page_title="ReportFlow", page_icon="📝", layout="wide")

st.markdown(
    """
    <style>
    .block-container {
        max-width: 1120px;
        padding-top: 1.6rem;
        padding-bottom: 4rem;
    }
    .hero {
        padding: 1.4rem 1.5rem;
        border: 1px solid rgba(120,120,120,0.18);
        border-radius: 22px;
        margin-bottom: 1rem;
        background: linear-gradient(180deg, rgba(250,250,250,0.04), rgba(250,250,250,0.015));
    }
    .card {
        padding: 1rem 1.1rem;
        border: 1px solid rgba(120,120,120,0.16);
        border-radius: 18px;
        background: rgba(250,250,250,0.02);
        height: 100%;
    }
    .small-muted {
        font-size: 0.92rem;
        opacity: 0.78;
    }
    .pill {
        display: inline-block;
        padding: 0.3rem 0.65rem;
        border-radius: 999px;
        border: 1px solid rgba(120,120,120,0.16);
        margin-right: 0.45rem;
        margin-top: 0.35rem;
        font-size: 0.86rem;
    }
    .section-title {
        margin-bottom: 0.35rem;
    }
    div[data-testid="stButton"] > button {
        border-radius: 14px;
        min-height: 2.9rem;
        font-weight: 600;
    }
    div[data-testid="stDownloadButton"] > button {
        border-radius: 14px;
        min-height: 2.9rem;
        font-weight: 600;
    }
    .spacer-sm {
        height: 0.5rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero">
        <h1 style="margin-bottom:0.2rem;">📝 ReportFlow</h1>
        <div class="small-muted">
            複数PDFをまとめて読み込み、課題文に関係するページだけを自動選別して、
            レポート本文を生成します。速さと精度の両立を狙った2段階選抜方式です。
        </div>
        <div style="margin-top:0.55rem;">
            <span class="pill">複数PDF対応</span>
            <span class="pill">ノイズ除去</span>
            <span class="pill">長文対応</span>
            <span class="pill">コピーボタン付き</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ==============================
# APIキー設定
# ==============================
secrets_key = st.secrets.get("OPENAI_API_KEY", None)
api_key_input = st.sidebar.text_input("OpenAI APIキー（ローカル確認用）", type="password")
api_key = secrets_key or api_key_input

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
                    "text_short": text[:1200],
                })
    return all_pages


def quick_score_page(client: OpenAI, theme: str, text_short: str):
    prompt = f"""
以下の文章が、課題文に対してどれくらい重要かをざっくり判定してください。

課題文:
{theme}

文章:
{text_short}

以下の形式だけで答えてください。
score: 0/1/2/3
reason: 15字以内

基準:
0 = 無関係
1 = 少し関係ある
2 = かなり関係ある
3 = 核になりそう
"""
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "速度重視でざっくり評価してください。指定形式以外で答えないでください。"},
            {"role": "user", "content": prompt},
        ],
    )
    content = response.choices[0].message.content.strip()
    score_match = re.search(r"score:\s*([0-3])", content)
    reason_match = re.search(r"reason:\s*(.*)", content)
    score = int(score_match.group(1)) if score_match else 0
    reason = reason_match.group(1).strip() if reason_match else ""
    return score, reason


def summarize_page(client: OpenAI, text: str) -> str:
    prompt = f"""
以下は講義スライド1ページ分の内容です。
重要点だけを日本語で80〜120字程度で簡潔に要約してください。
余計な説明は禁止です。

{text[:2500]}
"""
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "あなたは要点整理が得意なアシスタントです。簡潔に要約してください。"},
            {"role": "user", "content": prompt},
        ],
    )
    return response.choices[0].message.content.strip()


def precise_score_page(client: OpenAI, theme: str, summary: str):
    prompt = f"""
あなたの仕事は、スライド要約がレポート課題にどれだけ使えるかを丁寧に評価することです。

課題文:
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


def build_source_text(selected_pages, max_items=10):
    chosen = selected_pages[:max_items]
    blocks = []
    for p in chosen:
        blocks.append(
            f"資料: {p['file']} / Page {p['page']}\n"
            f"関連度: {p['score']}\n"
            f"要約: {p['summary']}\n"
        )
    return "\n".join(blocks)


def generate_report(client: OpenAI, theme: str, target_length: int, source_text: str, abstraction_mode: str) -> str:
    abstraction_instruction = {
        "標準": "抽象と具体のバランスを標準にしてください。",
        "抽象度高め": "個別事実を少し整理し、より上位概念でまとめるようにしてください。",
        "抽象度低め": "一般論に逃げず、具体的な症状・所見・資料内容をやや多めに含めてください。",
    }[abstraction_mode]

    prompt = f"""
以下は複数の講義資料から、課題文に関連する部分だけを抽出・要約したものです。

{source_text}

この材料をもとに、「{theme}」という課題に対して約{target_length}字のレポートを書いてください。

条件:
・大学生の提出レポートとして自然な日本語にする
・本文のみ出力する
・見出しは禁止
・箇条書きは禁止
・Markdown記法は禁止
・#、##、###、*、- は使わない
・導入→本論→まとめの流れを自然に含める
・資料にないことを大きく膨らませない
・資料に書かれていない内容は極力追加しない
・複数資料を統合した内容にする
・{abstraction_instruction}
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


def build_long_report_plan(client: OpenAI, theme: str, source_text: str, target_length: int) -> str:
    prompt = f"""
以下の資料要約をもとに、「{theme}」という課題の長文レポート構成案を作成してください。
目標文字数は約{target_length}字です。

資料要約:
{source_text}

条件:
・本文はまだ書かない
・3〜5個の章立てを提案する
・各章で何を書くべきかを簡潔に示す
・プレーンテキストのみ
・Markdown記法は禁止
"""
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "あなたは長文レポートの構成設計が得意なアシスタントです。"},
            {"role": "user", "content": prompt},
        ],
    )
    return response.choices[0].message.content.strip()


def generate_section(client: OpenAI, theme: str, source_text: str, plan: str, section_name: str, section_length: int, previous_text: str = "", abstraction_mode: str = "標準") -> str:
    abstraction_instruction = {
        "標準": "抽象と具体のバランスを標準にしてください。",
        "抽象度高め": "やや抽象化して整理してください。",
        "抽象度低め": "やや具体的に書いてください。",
    }[abstraction_mode]

    prompt = f"""
以下の資料要約と全体構成案をもとに、「{theme}」のレポートのうち「{section_name}」に対応する本文だけを書いてください。
目標文字数は約{section_length}字です。

全体構成案:
{plan}

資料要約:
{source_text}

すでに書かれている前半部分:
{previous_text[-1500:] if previous_text else "なし"}

条件:
・今回書く節の本文だけ出力する
・見出しは禁止
・箇条書き禁止
・Markdown禁止
・#、##、###、*、- は使わない
・資料にないことを大きく膨らませない
・前半と自然につながる文体にする
・{abstraction_instruction}
"""
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "あなたは自然な長文レポートを書く大学生です。本文のみを返してください。"},
            {"role": "user", "content": prompt},
        ],
    )
    return response.choices[0].message.content.strip()


def generate_long_report(client: OpenAI, theme: str, target_length: int, source_text: str, abstraction_mode: str) -> str:
    plan = build_long_report_plan(client, theme, source_text, target_length)
    section_names = ["導入", "本論前半", "本論後半", "まとめ"]
    base_lengths = [int(target_length * 0.15), int(target_length * 0.35), int(target_length * 0.35), int(target_length * 0.15)]

    parts = []
    previous_text = ""
    for section_name, section_length in zip(section_names, base_lengths):
        part = generate_section(
            client=client,
            theme=theme,
            source_text=source_text,
            plan=plan,
            section_name=section_name,
            section_length=section_length,
            previous_text=previous_text,
            abstraction_mode=abstraction_mode,
        )
        parts.append(part)
        previous_text += "\n" + part

    return "\n\n".join(parts).strip()


def add_human_touch(client: OpenAI, report: str, theme: str, style_mode: str) -> str:
    style_instruction = {
        "標準": "自然で読みやすい文体にしてください。",
        "やや硬め": "少しフォーマルで論文寄りの文体にしてください。",
        "やや柔らかめ": "少し柔らかく、読みやすい文体にしてください。",
    }[style_mode]

    prompt = f"""
以下のレポートを、提出用として自然で人間らしい文章に改善してください。

課題文:
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
・{style_instruction}
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


def refine_style(client: OpenAI, text: str, style_mode: str) -> str:
    style_instruction = {
        "標準": "自然で読みやすい文体に整えてください。",
        "やや硬め": "少しフォーマルで簡潔な文体に整えてください。",
        "やや柔らかめ": "少し柔らかく読みやすい文体に整えてください。",
    }[style_mode]
    prompt = f"""
以下の本文を、内容を保ったまま文体だけ整えてください。

条件:
・本文のみ出力
・見出し禁止
・箇条書き禁止
・Markdown禁止
・#、##、###、*、- は使わない
・{style_instruction}

本文:
{text}
"""
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "文章校正が得意なアシスタントです。本文のみ返してください。"},
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


def abstraction_transform(client: OpenAI, text: str, mode: str) -> str:
    if mode == "標準":
        return text
    instruction = "より抽象化して、上位概念で整理してください。" if mode == "抽象度高め" else "より具体化して、資料上の事実や記述に近づけてください。"
    prompt = f"""
以下の本文を、内容を大きく変えずに調整してください。

条件:
・本文のみ出力
・見出し禁止
・箇条書き禁止
・Markdown禁止
・#、##、###、*、- は使わない
・{instruction}

本文:
{text}
"""
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "文章の抽象度調整が得意なアシスタントです。本文のみ返してください。"},
            {"role": "user", "content": prompt},
        ],
    )
    return response.choices[0].message.content.strip()


# ==============================
# 状態管理
# ==============================
if "generated_report" not in st.session_state:
    st.session_state.generated_report = ""
if "selected_pages" not in st.session_state:
    st.session_state.selected_pages = []
if "all_pages_count" not in st.session_state:
    st.session_state.all_pages_count = 0
if "coarse_pages_count" not in st.session_state:
    st.session_state.coarse_pages_count = 0
if "theme_snapshot" not in st.session_state:
    st.session_state.theme_snapshot = ""


# ==============================
# レイアウト
# ==============================
left, right = st.columns([1.15, 0.85], gap="large")

with left:
    st.subheader("入力")
    st.write("課題文を入れて、関連するPDFを複数アップロードしてください。")

    uploaded_files = st.file_uploader(
        "PDFを複数アップロード",
        type=["pdf"],
        accept_multiple_files=True,
        help="講義スライド、配布資料、補足PDFなどをまとめて入れられます。",
    )

    theme = st.text_area(
        "課題文 / テーマ",
        placeholder="例：この患者の問題点と今後の治療計画について、講義資料を踏まえて述べなさい。",
        height=120,
    )

    target_length = st.number_input(
        "目標文字数",
        min_value=200,
        max_value=10000,
        value=3000,
        step=100,
        help="実用上は3000〜5000字がもっとも使いやすいレンジです。",
    )

with right:
    st.subheader("設定")
    coarse_keep = st.slider(
        "第1段階で残すページ数",
        min_value=10,
        max_value=30,
        value=20,
        step=1,
        help="まず全ページをざっくり採点し、上位だけ残します。速さに効きます。",
    )

    final_keep = st.slider(
        "最終的に使うページ数",
        min_value=5,
        max_value=10,
        value=8,
        step=1,
        help="最後のレポートに使う精鋭ページ数です。少ないほど深くなります。",
    )

    style_mode = st.selectbox(
        "仕上げの文体",
        ["標準", "やや硬め", "やや柔らかめ"],
        index=0,
    )

    abstraction_mode = st.selectbox(
        "抽象度",
        ["標準", "抽象度高め", "抽象度低め"],
        index=0,
        help="高めると概念整理寄り、低めると具体記述寄りになります。",
    )

    st.markdown(
        """
        <div class="card">
            <b>このアプリの流れ</b><br><br>
            1. 全ページをざっくり採点<br>
            2. 上位だけ要約<br>
            3. 要約を丁寧に再採点<br>
            4. 精鋭ページだけでレポート生成<br>
            5. 文体と文字数を整える
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("<div class='spacer-sm'></div>", unsafe_allow_html=True)

confirm_col, action_col = st.columns([1.35, 0.65])
with confirm_col:
    confirm_generate = st.checkbox("この設定で本当に生成する", value=False)
with action_col:
    run_button = st.button("レポートを生成", use_container_width=True, type="primary")

st.divider()

# ==============================
# 実行
# ==============================
if run_button:
    if not confirm_generate:
        st.warning("生成前に『この設定で本当に生成する』にチェックを入れてください。")
        st.stop()

    if not api_key:
        st.error("APIキーを入力してください。")
        st.stop()

    if not uploaded_files:
        st.error("PDFを1つ以上アップロードしてください。")
        st.stop()

    if not theme.strip():
        st.error("課題文またはテーマを入力してください。")
        st.stop()

    client = get_client(api_key)

    with st.spinner("PDFを解析中..."):
        all_pages = extract_pages(uploaded_files)
    st.session_state.all_pages_count = len(all_pages)

    top_metrics = st.columns(4)
    top_metrics[0].metric("文字ありページ数", len(all_pages))
    top_metrics[1].metric("アップロードPDF数", len(uploaded_files))
    top_metrics[2].metric("第1段階残し", coarse_keep)
    top_metrics[3].metric("最終採用", final_keep)

    with st.spinner("第1段階：全ページをざっくり採点中..."):
        coarse_scored = []
        progress_bar = st.progress(0, text="粗い採点をしています")
        for idx, p in enumerate(all_pages):
            score, reason = quick_score_page(client, theme, p["text_short"])
            coarse_scored.append({
                **p,
                "coarse_score": score,
                "coarse_reason": reason,
            })
            progress_bar.progress((idx + 1) / len(all_pages), text=f"粗い採点 {idx + 1}/{len(all_pages)}")

    coarse_scored = sorted(coarse_scored, key=lambda x: x["coarse_score"], reverse=True)
    coarse_selected = coarse_scored[:min(coarse_keep, len(coarse_scored))]
    st.session_state.coarse_pages_count = len(coarse_selected)

    with st.spinner("第2段階：残したページだけ要約中..."):
        summaries = []
        progress_bar = st.progress(0, text="要約を作成しています")
        for idx, p in enumerate(coarse_selected):
            s = summarize_page(client, p["text"])
            summaries.append({
                "file": p["file"],
                "page": p["page"],
                "text": p["text"],
                "summary": s,
                "coarse_score": p["coarse_score"],
            })
            progress_bar.progress((idx + 1) / len(coarse_selected), text=f"要約中 {idx + 1}/{len(coarse_selected)}")

    with st.spinner("第2段階：要約を丁寧に再採点中..."):
        scored_pages = []
        progress_bar = st.progress(0, text="精密採点をしています")
        for idx, s in enumerate(summaries):
            score, reason = precise_score_page(client, theme, s["summary"])
            scored_pages.append({
                **s,
                "score": score,
                "reason": reason,
            })
            progress_bar.progress((idx + 1) / len(summaries), text=f"精密採点 {idx + 1}/{len(summaries)}")

    scored_pages = sorted(scored_pages, key=lambda x: (x["score"], x["coarse_score"]), reverse=True)
    selected_pages = scored_pages[:min(final_keep, len(scored_pages))]
    source_text = build_source_text(selected_pages, max_items=final_keep)

    with st.spinner("レポートを生成中..."):
        if target_length <= 2500:
            raw_report = generate_report(client, theme, target_length, source_text, abstraction_mode)
        else:
            raw_report = generate_long_report(client, theme, target_length, source_text, abstraction_mode)
        raw_report = clean_text(raw_report)

        humanized_report = add_human_touch(client, raw_report, theme, style_mode)
        humanized_report = clean_text(humanized_report)

        final_report = refine_style(client, humanized_report, style_mode)
        final_report = clean_text(final_report)

        final_report = adjust_length(client, final_report, target_length)
        final_report = clean_text(final_report)

    st.session_state.generated_report = final_report
    st.session_state.selected_pages = selected_pages
    st.session_state.theme_snapshot = theme


# ==============================
# 結果表示
# ==============================
if st.session_state.generated_report:
    summary_tab, source_tab, tools_tab = st.tabs(["最終レポート", "採用ページ", "微調整"])

    with summary_tab:
        st.subheader("生成結果")
        st.text_area("レポート本文", st.session_state.generated_report, height=520)
        stat1, stat2, stat3 = st.columns(3)
        stat1.metric("最終文字数", len(st.session_state.generated_report))
        stat2.metric("第1段階通過", st.session_state.coarse_pages_count)
        stat3.metric("最終採用ページ数", len(st.session_state.selected_pages))

        copy_col1, copy_col2 = st.columns([0.55, 0.45])
        with copy_col1:
            st.download_button(
                "本文をテキストで保存",
                data=st.session_state.generated_report,
                file_name="report.txt",
                mime="text/plain",
                use_container_width=True,
            )
        with copy_col2:
            st.code(st.session_state.generated_report, language=None)

    with source_tab:
        st.subheader("採用されたページ")
        for p in st.session_state.selected_pages:
            with st.expander(f"{p['score']}点 | {p['file']} Page {p['page']} | {p['reason']}"):
                st.write(p["summary"])

    with tools_tab:
        st.subheader("微調整")
        st.write("抽象度の上下や文体の整え直しを、現在の結果に対して追加で行えます。")
        tool_col1, tool_col2, tool_col3 = st.columns(3)
        abstract_up = tool_col1.button("抽象度を上げる", use_container_width=True)
        abstract_down = tool_col2.button("抽象度を下げる", use_container_width=True)
        restyle = tool_col3.button("文体をもう一度整える", use_container_width=True)

        if abstract_up or abstract_down or restyle:
            if not api_key:
                st.error("APIキーを入力してください。")
            else:
                client = get_client(api_key)
                current_text = st.session_state.generated_report
                with st.spinner("微調整中..."):
                    if abstract_up:
                        current_text = abstraction_transform(client, current_text, "抽象度高め")
                    if abstract_down:
                        current_text = abstraction_transform(client, current_text, "抽象度低め")
                    if restyle:
                        current_text = refine_style(client, current_text, style_mode)
                    current_text = clean_text(current_text)
                st.session_state.generated_report = current_text
                st.success("微調整が完了しました。『最終レポート』タブを確認してください。")
