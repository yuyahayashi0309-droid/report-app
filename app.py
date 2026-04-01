import json
import math
import html
import re
import textwrap
from collections import Counter, defaultdict
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Tuple

import fitz
import streamlit as st
from openai import OpenAI

# ============================================================
# Page config / style
# ============================================================
st.set_page_config(page_title="ReportFlow Pro v2", page_icon="🧠", layout="wide")

st.markdown(
    """
    <style>
    .block-container {
        max-width: 1200px;
        padding-top: 1.2rem;
        padding-bottom: 4rem;
    }
    .hero {
        padding: 1.2rem 1.35rem;
        border: 1px solid rgba(120,120,120,0.16);
        border-radius: 22px;
        margin-bottom: 1rem;
        background: linear-gradient(180deg, rgba(250,250,250,0.045), rgba(250,250,250,0.015));
    }
    .card {
        padding: 1rem 1.05rem;
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
        padding: 0.28rem 0.65rem;
        border-radius: 999px;
        border: 1px solid rgba(120,120,120,0.16);
        margin-right: 0.42rem;
        margin-top: 0.35rem;
        font-size: 0.84rem;
    }
    .tiny {
        font-size: 0.85rem;
        opacity: 0.83;
    }
    div[data-testid="stButton"] > button,
    div[data-testid="stDownloadButton"] > button {
        border-radius: 14px;
        min-height: 2.9rem;
        font-weight: 650;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero">
        <h1 style="margin-bottom:0.18rem;">🧠 ReportFlow Pro v2</h1>
        <div class="small-muted">
            複数PDFからページではなく意味チャンク単位で根拠を抽出し、重複を生成前に減らし、
            課題文ごとに節構成と証拠束を割り当てて、資料密着型のレポートを一発で出す設計です。
        </div>
        <div style="margin-top:0.55rem;">
            <span class="pill">意味チャンク抽出</span>
            <span class="pill">重複クラスタ除去</span>
            <span class="pill">資料固有語優先</span>
            <span class="pill">節ごとの証拠束</span>
            <span class="pill">後処理は条件分岐</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# Constants
# ============================================================
STOPWORDS_JA = {
    "する", "ある", "いる", "こと", "これ", "それ", "ため", "よう", "もの", "また", "さらに", "できる",
    "なる", "いる", "おける", "および", "より", "など", "その", "この", "あの", "いう", "して",
    "した", "している", "について", "による", "として", "れる", "られる", "及び", "各", "本", "的",
    "的な", "である", "ます", "ました", "です", "でした", "ない", "なり", "一方", "場合", "では",
    "ので", "から", "まで", "など", "へ", "を", "に", "が", "は", "も", "と", "の",
}

DEFAULT_CONFIG = {
    "model_fast": "gpt-4.1-mini",
    "model_writer": "gpt-4.1-mini",
    "chunk_char_min": 220,
    "chunk_char_max": 1200,
    "stage1_keep": 28,
    "stage2_keep": 14,
    "final_evidence_keep": 10,
    "min_specific_terms": 4,
}


# ============================================================
# Dataclasses
# ============================================================
@dataclass
class Chunk:
    chunk_id: str
    file: str
    page: int
    block_range: str
    text: str
    short: str
    char_count: int
    lexical_terms: List[str]
    specificity_hint: float


@dataclass
class Evidence:
    chunk_id: str
    file: str
    page: int
    block_range: str
    text: str
    topic: str
    proposition: str
    evidence: str
    example: str
    terminology: List[str]
    contrast: str
    cause_effect: str
    role: str
    assignment_relevance: str
    specificity_score: int
    usefulness_score: int
    precise_reason: str
    coarse_score: int
    precise_score: int
    final_score: float
    duplicate_group: int = -1


# ============================================================
# OpenAI helpers
# ============================================================
def get_api_key() -> str:
    secrets_key = st.secrets.get("OPENAI_API_KEY", None)
    api_key_input = st.sidebar.text_input("OpenAI APIキー（ローカル確認用）", type="password")
    return (secrets_key or api_key_input or "").strip()


def get_client(api_key: str) -> OpenAI:
    return OpenAI(api_key=api_key)


def call_json(
    client: OpenAI,
    model: str,
    system: str,
    user_prompt: str,
    temperature: float = 0.2,
    max_output_tokens: int = 1800,
) -> Dict[str, Any]:
    response = client.responses.create(
        model=model,
        temperature=temperature,
        max_output_tokens=max_output_tokens,
        text={"format": {"type": "json_object"}},
        input=[
            {"role": "system", "content": [{"type": "input_text", "text": system}]},
            {"role": "user", "content": [{"type": "input_text", "text": user_prompt}]},
        ],
    )
    raw = response.output_text.strip()
    return json.loads(raw)


def call_text(
    client: OpenAI,
    model: str,
    system: str,
    user_prompt: str,
    temperature: float = 0.35,
    max_output_tokens: int = 2600,
) -> str:
    response = client.responses.create(
        model=model,
        temperature=temperature,
        max_output_tokens=max_output_tokens,
        input=[
            {"role": "system", "content": [{"type": "input_text", "text": system}]},
            {"role": "user", "content": [{"type": "input_text", "text": user_prompt}]},
        ],
    )
    return response.output_text.strip()


# ============================================================
# Generic text utilities
# ============================================================
def normalize_space(text: str) -> str:
    text = html.unescape(text or "")
    text = text.replace("\u3000", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def strip_markdownish(text: str) -> str:
    if not text:
        return ""
    for bad in ["###", "##", "#", "* ", "- "]:
        text = text.replace(bad, "")
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def approx_char_delta(text: str, target: int) -> int:
    return abs(len(text) - target)


def split_lines(text: str) -> List[str]:
    lines = [normalize_space(x) for x in text.splitlines()]
    return [x for x in lines if x]


def sentenceish_split(text: str) -> List[str]:
    parts = re.split(r"(?<=[。！？])\s+|\n+", normalize_space(text))
    return [p.strip() for p in parts if p.strip()]


def lexical_terms(text: str, top_k: int = 12) -> List[str]:
    text = normalize_space(text)
    cands = re.findall(r"[A-Za-z][A-Za-z0-9_\-]{2,}|[一-龥ァ-ヶー]{2,}", text)
    cands = [c for c in cands if c not in STOPWORDS_JA and not re.fullmatch(r"\d+", c)]
    freq = Counter(cands)
    ranked = [w for w, _ in freq.most_common(top_k * 3)]
    unique = []
    seen = set()
    for w in ranked:
        lw = w.lower()
        if lw in seen:
            continue
        seen.add(lw)
        unique.append(w)
        if len(unique) >= top_k:
            break
    return unique


def specificity_hint_score(text: str) -> float:
    text = normalize_space(text)
    if not text:
        return 0.0
    caps = len(re.findall(r"\b[A-Z][A-Za-z0-9_\-]+\b", text))
    digits = len(re.findall(r"\d+", text))
    keywords = len(lexical_terms(text, top_k=10))
    punct = len(re.findall(r"[:：（）()『』「」]", text))
    length_factor = min(len(text) / 700, 1.0)
    return round(min(10.0, 1.2 * caps + 0.5 * digits + 0.45 * keywords + 0.25 * punct + 2.5 * length_factor), 2)


def jaccard_similarity(a_terms: List[str], b_terms: List[str]) -> float:
    a = {x.lower() for x in a_terms}
    b = {x.lower() for x in b_terms}
    if not a or not b:
        return 0.0
    return len(a & b) / max(len(a | b), 1)


def safe_json_list(value: Any) -> List[str]:
    if isinstance(value, list):
        return [str(x).strip() for x in value if str(x).strip()]
    if isinstance(value, str):
        parts = re.split(r"[,、/\n]", value)
        return [p.strip() for p in parts if p.strip()]
    return []


# ============================================================
# PDF chunk extraction
# ============================================================
def block_texts_from_page(page) -> List[Tuple[int, str]]:
    blocks = page.get_text("blocks")
    cleaned = []
    for idx, block in enumerate(blocks):
        if len(block) < 5:
            continue
        text = normalize_space(block[4])
        if not text:
            continue
        if len(text) <= 2:
            continue
        cleaned.append((idx, text))
    return cleaned


def merge_blocks_semantically(blocks: List[Tuple[int, str]], min_chars: int, max_chars: int) -> List[Tuple[str, str]]:
    chunks: List[Tuple[str, str]] = []
    current_indices: List[int] = []
    current_texts: List[str] = []

    def flush():
        nonlocal current_indices, current_texts
        if not current_texts:
            return
        text = "\n".join(current_texts).strip()
        if text:
            idx_label = f"{current_indices[0]}-{current_indices[-1]}"
            chunks.append((idx_label, text))
        current_indices = []
        current_texts = []

    for idx, text in blocks:
        text = normalize_space(text)
        if not text:
            continue

        is_headerish = (len(text) <= 42 and not text.endswith("。")) or re.fullmatch(r"[0-9０-９IVXivx一二三四五六七八九十]+[\.．\-：:]?.*", text)
        is_bulletish = bool(re.match(r"^[・•●◦▪■□▶▸\-–—]+", text))

        candidate = ("\n".join(current_texts + [text])).strip() if current_texts else text
        too_long = len(candidate) > max_chars

        if current_texts and (is_headerish or too_long):
            flush()

        current_indices.append(idx)
        current_texts.append(text)

        if len("\n".join(current_texts)) >= min_chars:
            if not is_bulletish and text.endswith(("。", ".", "!", "?", "）", ")")):
                flush()

    flush()
    return chunks


def extract_semantic_chunks(files, chunk_char_min: int, chunk_char_max: int) -> List[Chunk]:
    all_chunks: List[Chunk] = []
    for uploaded_file in files:
        file_bytes = uploaded_file.read()
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        for page_no, page in enumerate(doc, start=1):
            blocks = block_texts_from_page(page)
            if not blocks:
                continue
            merged = merge_blocks_semantically(blocks, min_chars=chunk_char_min, max_chars=chunk_char_max)
            if not merged:
                continue
            for local_idx, (block_range, text) in enumerate(merged, start=1):
                short = normalize_space(text)[:900]
                terms = lexical_terms(text, top_k=12)
                chunk = Chunk(
                    chunk_id=f"{uploaded_file.name}::p{page_no}::c{local_idx}",
                    file=uploaded_file.name,
                    page=page_no,
                    block_range=block_range,
                    text=text,
                    short=short,
                    char_count=len(text),
                    lexical_terms=terms,
                    specificity_hint=specificity_hint_score(text),
                )
                all_chunks.append(chunk)
    return all_chunks


# ============================================================
# LLM scoring / extraction
# ============================================================
def coarse_score_chunk(client: OpenAI, model: str, theme: str, chunk: Chunk) -> Tuple[int, int, str]:
    prompt = f"""
あなたは大学レポート用の資料選抜者です。
課題文に対して、この資料断片がどれくらい使えるかを高速判定してください。

返答はJSONのみ。
必要キー:
- relevance_score: 0-3
- usefulness_score: 0-3
- reason: 18字以内

評価基準:
relevance_score
0 = 無関係
1 = 少し関係
2 = かなり関係
3 = 核候補

usefulness_score
0 = 書く材料にならない
1 = 補足程度
2 = 段落の材料になる
3 = 主張や定義の核になる

課題文:
{theme}

資料断片:
{chunk.short}

補助情報:
- 推定固有性ヒント: {chunk.specificity_hint}
- 候補語: {', '.join(chunk.lexical_terms[:8])}
"""
    data = call_json(
        client=client,
        model=model,
        system="速度重視。ただし一般論より資料固有性をやや優先して判定する。JSONのみ返す。",
        user_prompt=prompt,
        temperature=0.0,
        max_output_tokens=220,
    )
    relevance = int(max(0, min(3, data.get("relevance_score", 0))))
    usefulness = int(max(0, min(3, data.get("usefulness_score", 0))))
    reason = str(data.get("reason", "")).strip()[:18]
    return relevance, usefulness, reason


def build_evidence_record(client: OpenAI, model: str, theme: str, chunk: Chunk) -> Dict[str, Any]:
    prompt = f"""
以下の資料断片を、大学レポート生成に使いやすいよう厳密に構造化してください。
返答はJSONのみ。

必要キー:
- topic: 1文
- proposition: レポートに使える主張を1文
- evidence: この断片が根拠として与える内容を1文
- example: 具体例があれば1文、なければ "なし"
- terminology: 重要概念の配列（3〜8個）
- contrast: 対比があれば1文、なければ "なし"
- cause_effect: 因果があれば1文、なければ "なし"
- role: 次のどれか一つ [definition, claim, evidence, example, mechanism, comparison, implication]
- assignment_relevance: 課題文のどの論点に使えるか1文
- specificity_score: 0-10

課題文:
{theme}

資料断片:
{chunk.text[:3000]}
"""
    data = call_json(
        client=client,
        model=model,
        system="講義資料の断片を、レポート用の中間表現に変換する。資料にないことは足さない。JSONのみ返す。",
        user_prompt=prompt,
        temperature=0.1,
        max_output_tokens=900,
    )
    data["terminology"] = safe_json_list(data.get("terminology", []))
    data["specificity_score"] = int(max(0, min(10, data.get("specificity_score", 0))))
    return data


def precise_score_evidence(client: OpenAI, model: str, theme: str, evidence_data: Dict[str, Any]) -> Tuple[int, str]:
    compact = json.dumps(evidence_data, ensure_ascii=False)
    prompt = f"""
以下の構造化情報が、課題文に対してどれだけ中核的に使えるかを厳密評価してください。
返答はJSONのみ。

必要キー:
- precise_score: 0-3
- precise_reason: 24字以内

評価基準:
0 = 採用不要
1 = 補足なら可
2 = 段落に使える
3 = 核材料

課題文:
{theme}

構造化情報:
{compact}
"""
    data = call_json(
        client=client,
        model=model,
        system="厳密採点者。一般論ではなく、課題適合性と資料固有性を重視する。JSONのみ返す。",
        user_prompt=prompt,
        temperature=0.0,
        max_output_tokens=180,
    )
    precise_score = int(max(0, min(3, data.get("precise_score", 0))))
    precise_reason = str(data.get("precise_reason", "")).strip()[:24]
    return precise_score, precise_reason


# ============================================================
# Dedupe / ranking
# ============================================================
def final_score_formula(coarse_score: int, precise_score: int, usefulness_score: int, specificity_score: int) -> float:
    rel = (coarse_score + precise_score) / 6
    useful = usefulness_score / 3
    spec = specificity_score / 10
    score = 0.50 * rel + 0.20 * useful + 0.30 * spec
    return round(score * 100, 2)


def cluster_duplicates(evidences: List[Evidence], threshold: float = 0.42) -> List[Evidence]:
    group_id = 0
    for i in range(len(evidences)):
        if evidences[i].duplicate_group != -1:
            continue
        evidences[i].duplicate_group = group_id
        base_terms = lexical_terms(
            " ".join([
                evidences[i].topic,
                evidences[i].proposition,
                evidences[i].evidence,
                " ".join(evidences[i].terminology),
            ]),
            top_k=14,
        )
        for j in range(i + 1, len(evidences)):
            if evidences[j].duplicate_group != -1:
                continue
            cand_terms = lexical_terms(
                " ".join([
                    evidences[j].topic,
                    evidences[j].proposition,
                    evidences[j].evidence,
                    " ".join(evidences[j].terminology),
                ]),
                top_k=14,
            )
            sim = jaccard_similarity(base_terms, cand_terms)
            same_page = evidences[i].file == evidences[j].file and evidences[i].page == evidences[j].page
            if sim >= threshold or (same_page and sim >= 0.28):
                evidences[j].duplicate_group = group_id
        group_id += 1
    return evidences


def select_representatives(evidences: List[Evidence], limit: int) -> List[Evidence]:
    grouped: Dict[int, List[Evidence]] = defaultdict(list)
    for ev in evidences:
        grouped[ev.duplicate_group].append(ev)

    reps: List[Evidence] = []
    for _, items in grouped.items():
        items = sorted(
            items,
            key=lambda x: (x.final_score, x.precise_score, x.specificity_score, len(x.terminology)),
            reverse=True,
        )
        reps.append(items[0])

    reps = sorted(
        reps,
        key=lambda x: (x.final_score, x.precise_score, x.usefulness_score, x.specificity_score),
        reverse=True,
    )
    return reps[:limit]


def important_terms_from_evidences(evidences: List[Evidence], top_k: int = 14) -> List[str]:
    counter = Counter()
    for ev in evidences:
        for t in ev.terminology:
            if len(t) >= 2:
                counter[t] += 1 + ev.specificity_score / 10
        for t in lexical_terms(" ".join([ev.topic, ev.proposition, ev.evidence]), top_k=6):
            counter[t] += 0.6
    return [w for w, _ in counter.most_common(top_k)]


# ============================================================
# Planning
# ============================================================
def render_evidence_brief(ev: Evidence) -> str:
    return textwrap.dedent(
        f"""
        [Evidence {ev.chunk_id}]
        source: {ev.file} p.{ev.page} blocks {ev.block_range}
        topic: {ev.topic}
        proposition: {ev.proposition}
        evidence: {ev.evidence}
        example: {ev.example}
        terminology: {', '.join(ev.terminology)}
        contrast: {ev.contrast}
        cause_effect: {ev.cause_effect}
        role: {ev.role}
        assignment_relevance: {ev.assignment_relevance}
        specificity_score: {ev.specificity_score}
        final_score: {ev.final_score}
        """
    ).strip()


def build_report_blueprint(client: OpenAI, model: str, theme: str, evidences: List[Evidence], target_length: int) -> Dict[str, Any]:
    evidence_text = "\n\n".join(render_evidence_brief(ev) for ev in evidences[:12])
    section_count = 4 if target_length < 3500 else 5

    prompt = f"""
あなたは大学レポートの設計者です。
次の資料根拠だけを使って、課題文に対するレポート設計図を作ってください。
返答はJSONのみ。

必要キー:
- thesis: レポート全体の中心主張 1文
- sections: 配列
  各要素に必要なキー:
    - name
    - role
    - objective
    - key_claim
    - evidence_ids: 使用するEvidenceのchunk_id配列（2〜4個）
    - must_use_terms: 必ず入れる概念配列（2〜5個）
    - avoid_overlap_with: 重複を避けたい節名配列
    - target_chars: 目安文字数

条件:
- 節数は {section_count} 個
- 同じ根拠を全節に使い回しすぎない
- 各節は別の役割を持つ
- 一般論で水増ししない
- 資料固有語を優先
- 少なくとも1節は資料間の接続や対比を担当させる
- 導入と結論だけで逃げず、本論で根拠を使う

課題文:
{theme}

目標文字数:
{target_length}

根拠群:
{evidence_text}
"""
    data = call_json(
        client=client,
        model=model,
        system="大学レポートの論点設計に強い設計者。JSONのみ返す。",
        user_prompt=prompt,
        temperature=0.2,
        max_output_tokens=2200,
    )
    return data


def evidence_lookup(evidences: List[Evidence]) -> Dict[str, Evidence]:
    return {ev.chunk_id: ev for ev in evidences}


def section_evidence_bundle(section: Dict[str, Any], lookup: Dict[str, Evidence]) -> str:
    lines = []
    for eid in section.get("evidence_ids", []):
        ev = lookup.get(eid)
        if not ev:
            continue
        lines.append(render_evidence_brief(ev))
    return "\n\n".join(lines)


# ============================================================
# Generation
# ============================================================
def generate_section_text(
    client: OpenAI,
    model: str,
    theme: str,
    thesis: str,
    section: Dict[str, Any],
    evidence_bundle: str,
    previous_text: str,
    used_terms: List[str],
    style_mode: str,
    abstraction_mode: str,
) -> str:
    style_instruction = {
        "標準": "自然で読みやすい大学レポート文体にする。",
        "やや硬め": "少しフォーマルで簡潔な文体にする。",
        "やや柔らかめ": "少し柔らかいが幼くしない文体にする。",
    }[style_mode]
    abstraction_instruction = {
        "標準": "抽象と具体のバランスを標準にする。",
        "抽象度高め": "やや抽象化し、上位概念で整理する。",
        "抽象度低め": "やや具体的にし、資料上の概念や事例を厚く使う。",
    }[abstraction_mode]

    must_terms = section.get("must_use_terms", [])
    used_terms_text = ", ".join(used_terms[-18:]) if used_terms else "なし"
    previous_tail = previous_text[-1800:] if previous_text else "なし"

    prompt = f"""
課題文に対するレポート本文のうち、この節だけを書いてください。
本文のみ出力してください。

課題文:
{theme}

全体の中心主張:
{thesis}

今回の節:
name: {section.get('name', '')}
role: {section.get('role', '')}
objective: {section.get('objective', '')}
key_claim: {section.get('key_claim', '')}
target_chars: {section.get('target_chars', 0)}
must_use_terms: {', '.join(must_terms)}
avoid_overlap_with: {', '.join(section.get('avoid_overlap_with', []))}

この節で使う根拠:
{evidence_bundle}

すでに本文で使った概念:
{used_terms_text}

すでに書かれている前半:
{previous_tail}

厳守条件:
- 本文のみ
- 見出し禁止
- 箇条書き禁止
- Markdown禁止
- 同じ主張を言い換えて繰り返さない
- この節では、この節のkey_claimだけに集中する
- 必ず must_use_terms を2個以上入れる
- evidence_bundleの内容にないことを大きく膨らませない
- 可能なら因果、対比、具体例のどれかを最低1つ入れる
- 前半と自然につながる
- 「重要である」で止めず、なぜそう言えるかまで書く
- {style_instruction}
- {abstraction_instruction}
"""
    text = call_text(
        client=client,
        model=model,
        system="資料密着型の大学レポートを書く。与えられた根拠から逸脱しない。本文のみ返す。",
        user_prompt=prompt,
        temperature=0.45,
        max_output_tokens=2200,
    )
    return strip_markdownish(text)


def verify_and_patch_missing_terms(
    client: OpenAI,
    model: str,
    theme: str,
    text: str,
    evidences: List[Evidence],
    min_terms: int,
) -> str:
    important_terms = important_terms_from_evidences(evidences, top_k=12)
    missing = [t for t in important_terms if t not in text][:max(0, min_terms)]
    if not missing:
        return text

    evidence_snippets = []
    for ev in evidences[:6]:
        overlap = [t for t in missing if t in ev.terminology or t in ev.topic or t in ev.proposition or t in ev.evidence]
        if overlap:
            evidence_snippets.append(render_evidence_brief(ev))

    prompt = f"""
以下のレポート本文を、論旨を保ったまま必要最小限だけ書き直してください。
目的は、未使用の重要概念を自然に織り込むことです。
本文のみ出力してください。

課題文:
{theme}

未使用の重要概念:
{', '.join(missing)}

参考根拠:
{'\n\n'.join(evidence_snippets[:4])}

本文:
{text}

条件:
- 本文のみ
- 構成を壊さない
- 無理に全部入れなくてよいが、できれば2個以上自然に入れる
- 一般論で水増ししない
- 資料にないことを追加しない
"""
    patched = call_text(
        client=client,
        model=model,
        system="最小修正で概念不足を補う。本文のみ返す。",
        user_prompt=prompt,
        temperature=0.25,
        max_output_tokens=2400,
    )
    return strip_markdownish(patched)


# ============================================================
# Critique / conditional revision
# ============================================================
def critique_report(client: OpenAI, model: str, theme: str, report: str, evidences: List[Evidence]) -> Dict[str, Any]:
    reference_terms = important_terms_from_evidences(evidences, top_k=12)
    prompt = f"""
以下のレポートを厳しく評価してください。返答はJSONのみ。

必要キー:
- overall_score: 0-100
- specificity_ok: true/false
- repetition_ok: true/false
- coverage_ok: true/false
- ai_stiffness: 0-3
- length_fit: 0-3
- weaknesses: 配列（0〜5件）
- revision_needed: true/false
- revision_focus: 配列（0〜4件）

参考重要概念:
{', '.join(reference_terms)}

課題文:
{theme}

本文:
{report}
"""
    return call_json(
        client=client,
        model=model,
        system="厳格な添削者。資料固有性、重複、論点カバー、AIっぽさを判定する。JSONのみ返す。",
        user_prompt=prompt,
        temperature=0.0,
        max_output_tokens=700,
    )


def conditional_rewrite(
    client: OpenAI,
    model: str,
    theme: str,
    report: str,
    critique: Dict[str, Any],
    evidences: List[Evidence],
    style_mode: str,
) -> str:
    if not critique.get("revision_needed", False) and critique.get("overall_score", 0) >= 86:
        return report

    style_instruction = {
        "標準": "自然で読みやすくする。",
        "やや硬め": "少しフォーマルで締まった文体にする。",
        "やや柔らかめ": "やや柔らかくするが、幼くしない。",
    }[style_mode]

    evidence_text = "\n\n".join(render_evidence_brief(ev) for ev in evidences[:6])
    focus = ", ".join(critique.get("revision_focus", [])) or "資料固有性と重複"
    weaknesses = "\n".join(f"- {w}" for w in critique.get("weaknesses", [])) or "- なし"

    prompt = f"""
以下のレポートを必要最小限の修正で改善してください。
本文のみ出力してください。

課題文:
{theme}

修正の重点:
{focus}

弱点:
{weaknesses}

参考根拠:
{evidence_text}

元の本文:
{report}

条件:
- 本文のみ
- なるべく構成を維持
- 資料固有語を消さない
- 重複を減らす
- 抽象的すぎる箇所だけ具体化
- {style_instruction}
"""
    rewritten = call_text(
        client=client,
        model=model,
        system="最小限の改稿で品質を上げる。本文のみ返す。",
        user_prompt=prompt,
        temperature=0.3,
        max_output_tokens=2600,
    )
    return strip_markdownish(rewritten)


def conditional_humanize(client: OpenAI, model: str, theme: str, report: str, critique: Dict[str, Any]) -> str:
    if int(critique.get("ai_stiffness", 0)) < 2:
        return report

    prompt = f"""
以下のレポート本文を、内容を変えすぎずに少しだけ自然な日本語へ整えてください。
本文のみ出力してください。

課題文:
{theme}

本文:
{report}

条件:
- 本文のみ
- 論旨維持
- 資料固有語を残す
- 感想文にしない
- 1〜2文だけ、人が読んでも不自然でない流れに整える
"""
    text = call_text(
        client=client,
        model=model,
        system="AIっぽさだけを軽く減らす。本文のみ返す。",
        user_prompt=prompt,
        temperature=0.35,
        max_output_tokens=2400,
    )
    return strip_markdownish(text)


def conditional_length_adjust(client: OpenAI, model: str, report: str, target_length: int) -> str:
    if approx_char_delta(report, target_length) <= max(120, int(target_length * 0.08)):
        return report

    prompt = f"""
以下の本文を、内容をなるべく保ったまま約{target_length}字に調整してください。
本文のみ出力してください。

条件:
- 本文のみ
- 見出し禁止
- 箇条書き禁止
- 不自然な圧縮をしない
- 資料固有語を残す

本文:
{report}
"""
    text = call_text(
        client=client,
        model=model,
        system="文字数だけを自然に調整する。本文のみ返す。",
        user_prompt=prompt,
        temperature=0.2,
        max_output_tokens=2600,
    )
    return strip_markdownish(text)


# ============================================================
# Orchestration
# ============================================================
def run_pipeline(
    client: OpenAI,
    theme: str,
    uploaded_files,
    target_length: int,
    style_mode: str,
    abstraction_mode: str,
    cfg: Dict[str, Any],
) -> Dict[str, Any]:
    chunk_char_min = cfg["chunk_char_min"]
    chunk_char_max = cfg["chunk_char_max"]
    stage1_keep = cfg["stage1_keep"]
    stage2_keep = cfg["stage2_keep"]
    final_evidence_keep = cfg["final_evidence_keep"]

    status = st.empty()

    with st.spinner("PDFから意味チャンクを抽出中..."):
        status.info("ページではなく、見出しや段落まとまりに近い単位へ分割しています。")
        chunks = extract_semantic_chunks(uploaded_files, chunk_char_min=chunk_char_min, chunk_char_max=chunk_char_max)

    if not chunks:
        raise ValueError("有効なテキストチャンクを抽出できませんでした。PDFの文字層を確認してください。")

    metrics = st.columns(4)
    metrics[0].metric("抽出チャンク数", len(chunks))
    metrics[1].metric("PDF数", len(uploaded_files))
    metrics[2].metric("Stage1残し", min(stage1_keep, len(chunks)))
    metrics[3].metric("最終根拠候補", min(final_evidence_keep, len(chunks)))

    with st.spinner("Stage 1: 粗スコアリング中..."):
        coarse_rows = []
        bar = st.progress(0, text="課題との関連性を高速判定しています")
        for i, chunk in enumerate(chunks):
            relevance, usefulness, reason = coarse_score_chunk(client, cfg["model_fast"], theme, chunk)
            coarse_rows.append({
                "chunk": chunk,
                "coarse_score": relevance,
                "usefulness_score": usefulness,
                "coarse_reason": reason,
            })
            bar.progress((i + 1) / len(chunks), text=f"粗スコア {i + 1}/{len(chunks)}")

    coarse_rows.sort(
        key=lambda x: (x["coarse_score"], x["usefulness_score"], x["chunk"].specificity_hint, x["chunk"].char_count),
        reverse=True,
    )
    stage1 = coarse_rows[: min(stage1_keep, len(coarse_rows))]

    with st.spinner("Stage 2: 構造化抽出と精密採点中..."):
        evidences: List[Evidence] = []
        bar = st.progress(0, text="チャンクをレポート用の証拠データへ変換しています")
        for i, row in enumerate(stage1):
            chunk: Chunk = row["chunk"]
            structured = build_evidence_record(client, cfg["model_fast"], theme, chunk)
            precise_score, precise_reason = precise_score_evidence(client, cfg["model_fast"], theme, structured)
            final_score = final_score_formula(
                coarse_score=row["coarse_score"],
                precise_score=precise_score,
                usefulness_score=row["usefulness_score"],
                specificity_score=int(structured.get("specificity_score", 0)),
            )
            evidences.append(
                Evidence(
                    chunk_id=chunk.chunk_id,
                    file=chunk.file,
                    page=chunk.page,
                    block_range=chunk.block_range,
                    text=chunk.text,
                    topic=str(structured.get("topic", "")).strip(),
                    proposition=str(structured.get("proposition", "")).strip(),
                    evidence=str(structured.get("evidence", "")).strip(),
                    example=str(structured.get("example", "")).strip(),
                    terminology=safe_json_list(structured.get("terminology", [])),
                    contrast=str(structured.get("contrast", "")).strip(),
                    cause_effect=str(structured.get("cause_effect", "")).strip(),
                    role=str(structured.get("role", "evidence")).strip(),
                    assignment_relevance=str(structured.get("assignment_relevance", "")).strip(),
                    specificity_score=int(structured.get("specificity_score", 0)),
                    usefulness_score=row["usefulness_score"],
                    precise_reason=precise_reason,
                    coarse_score=row["coarse_score"],
                    precise_score=precise_score,
                    final_score=final_score,
                )
            )
            bar.progress((i + 1) / len(stage1), text=f"精密化 {i + 1}/{len(stage1)}")

    evidences.sort(
        key=lambda x: (x.final_score, x.precise_score, x.specificity_score, x.usefulness_score),
        reverse=True,
    )
    stage2 = evidences[: min(stage2_keep, len(evidences))]
    stage2 = cluster_duplicates(stage2, threshold=0.42)
    selected = select_representatives(stage2, limit=min(final_evidence_keep, len(stage2)))

    with st.spinner("論点設計中..."):
        status.info("節ごとに使う証拠束を割り当てています。")
        blueprint = build_report_blueprint(client, cfg["model_writer"], theme, selected, target_length)

    thesis = str(blueprint.get("thesis", ""))
    sections = blueprint.get("sections", []) if isinstance(blueprint.get("sections", []), list) else []
    if not sections:
        raise ValueError("節設計に失敗しました。もう一度実行してください。")

    lookup = evidence_lookup(selected)
    report_parts = []
    used_terms: List[str] = []
    previous_text = ""

    with st.spinner("節ごとに本文生成中..."):
        bar = st.progress(0, text="証拠束を使って各節を書いています")
        for i, section in enumerate(sections):
            bundle = section_evidence_bundle(section, lookup)
            part = generate_section_text(
                client=client,
                model=cfg["model_writer"],
                theme=theme,
                thesis=thesis,
                section=section,
                evidence_bundle=bundle,
                previous_text=previous_text,
                used_terms=used_terms,
                style_mode=style_mode,
                abstraction_mode=abstraction_mode,
            )
            report_parts.append(part)
            previous_text = "\n\n".join(report_parts)
            used_terms.extend(lexical_terms(part, top_k=8))
            bar.progress((i + 1) / len(sections), text=f"節生成 {i + 1}/{len(sections)}")

    raw_report = strip_markdownish("\n\n".join(report_parts))

    with st.spinner("重要概念の未使用チェック中..."):
        patched = verify_and_patch_missing_terms(
            client=client,
            model=cfg["model_writer"],
            theme=theme,
            text=raw_report,
            evidences=selected,
            min_terms=cfg["min_specific_terms"],
        )

    with st.spinner("条件付き品質改善中..."):
        critique1 = critique_report(client, cfg["model_fast"], theme, patched, selected)
        revised = conditional_rewrite(client, cfg["model_writer"], theme, patched, critique1, selected, style_mode)
        critique2 = critique_report(client, cfg["model_fast"], theme, revised, selected)
        humanized = conditional_humanize(client, cfg["model_writer"], theme, revised, critique2)
        final_report = conditional_length_adjust(client, cfg["model_writer"], humanized, target_length)
        final_report = strip_markdownish(final_report)
        final_critique = critique_report(client, cfg["model_fast"], theme, final_report, selected)

    return {
        "chunks": chunks,
        "stage1": stage1,
        "stage2": stage2,
        "selected_evidences": selected,
        "blueprint": blueprint,
        "report": final_report,
        "critique": final_critique,
        "initial_critique": critique1,
    }


# ============================================================
# Session state
# ============================================================
for key, default in {
    "result": None,
    "theme_snapshot": "",
    "debug_open": False,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default


# ============================================================
# UI
# ============================================================
left, right = st.columns([1.2, 0.8], gap="large")

with left:
    st.subheader("入力")
    uploaded_files = st.file_uploader(
        "PDFを複数アップロード",
        type=["pdf"],
        accept_multiple_files=True,
        help="講義スライド、配布資料、補足PDFなどを一括投入できます。",
    )
    theme = st.text_area(
        "課題文 / テーマ",
        placeholder="例：マーケティングとブランド戦略について、講義資料の内容を関連づけて論じなさい。",
        height=130,
    )
    target_length = st.number_input(
        "目標文字数",
        min_value=300,
        max_value=10000,
        value=3000,
        step=100,
    )

with right:
    st.subheader("設定")
    style_mode = st.selectbox("文体", ["標準", "やや硬め", "やや柔らかめ"], index=0)
    abstraction_mode = st.selectbox("抽象度", ["標準", "抽象度高め", "抽象度低め"], index=0)
    model_fast = st.text_input("高速モデル", value=DEFAULT_CONFIG["model_fast"])
    model_writer = st.text_input("本文生成モデル", value=DEFAULT_CONFIG["model_writer"])
    stage1_keep = st.slider("Stage1で残すチャンク数", min_value=12, max_value=50, value=DEFAULT_CONFIG["stage1_keep"], step=2)
    stage2_keep = st.slider("Stage2で精密化するチャンク数", min_value=8, max_value=24, value=DEFAULT_CONFIG["stage2_keep"], step=1)
    final_keep = st.slider("最終採用する根拠数", min_value=5, max_value=14, value=DEFAULT_CONFIG["final_evidence_keep"], step=1)
    st.markdown(
        """
        <div class="card">
            <b>今回の改善点</b><br><br>
            1. ページではなく意味チャンク単位で抽出<br>
            2. 構造化schemaを強化<br>
            3. 生成前に重複クラスタ除去<br>
            4. 節ごとに証拠束を指定して生成<br>
            5. 後処理は必要時のみ実行
        </div>
        """,
        unsafe_allow_html=True,
    )

api_key = get_api_key()
confirm = st.checkbox("この設定で本当に生成する", value=False)
run_button = st.button("レポートを生成", use_container_width=True, type="primary")


# ============================================================
# Run
# ============================================================
if run_button:
    if not confirm:
        st.warning("生成前に確認チェックを入れてください。")
        st.stop()
    if not api_key:
        st.error("APIキーを入力してください。")
        st.stop()
    if not uploaded_files:
        st.error("PDFを1つ以上アップロードしてください。")
        st.stop()
    if not theme.strip():
        st.error("課題文を入力してください。")
        st.stop()

    cfg = dict(DEFAULT_CONFIG)
    cfg["model_fast"] = model_fast.strip()
    cfg["model_writer"] = model_writer.strip()
    cfg["stage1_keep"] = stage1_keep
    cfg["stage2_keep"] = stage2_keep
    cfg["final_evidence_keep"] = final_keep

    client = get_client(api_key)

    try:
        result = run_pipeline(
            client=client,
            theme=theme,
            uploaded_files=uploaded_files,
            target_length=int(target_length),
            style_mode=style_mode,
            abstraction_mode=abstraction_mode,
            cfg=cfg,
        )
        st.session_state.result = result
        st.session_state.theme_snapshot = theme
        st.success("生成が完了しました。")
    except Exception as e:
        st.exception(e)
        st.stop()


# ============================================================
# Result rendering
# ============================================================
if st.session_state.result:
    result = st.session_state.result
    report = result["report"]
    selected = result["selected_evidences"]
    blueprint = result["blueprint"]
    critique = result["critique"]

    tab1, tab2, tab3, tab4 = st.tabs(["最終レポート", "採用根拠", "節設計", "内部評価"])

    with tab1:
        st.subheader("生成結果")
        st.text_area("レポート本文", report, height=520)
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("最終文字数", len(report))
        m2.metric("採用根拠数", len(selected))
        m3.metric("品質スコア", int(critique.get("overall_score", 0)))
        m4.metric("固有語候補", len(important_terms_from_evidences(selected, top_k=12)))
        c1, c2 = st.columns([0.55, 0.45])
        with c1:
            st.download_button(
                "本文をテキスト保存",
                data=report,
                file_name="reportflow_v2_report.txt",
                mime="text/plain",
                use_container_width=True,
            )
        with c2:
            st.code(report, language=None)

    with tab2:
        st.subheader("採用された根拠")
        for ev in selected:
            title = f"{int(ev.final_score)}点 | {ev.file} p.{ev.page} | {ev.topic[:28]}"
            with st.expander(title):
                st.write(f"chunk_id: {ev.chunk_id}")
                st.write(f"blocks: {ev.block_range}")
                st.write(f"topic: {ev.topic}")
                st.write(f"proposition: {ev.proposition}")
                st.write(f"evidence: {ev.evidence}")
                st.write(f"example: {ev.example}")
                st.write(f"terminology: {', '.join(ev.terminology)}")
                st.write(f"contrast: {ev.contrast}")
                st.write(f"cause_effect: {ev.cause_effect}")
                st.write(f"role: {ev.role}")
                st.write(f"assignment_relevance: {ev.assignment_relevance}")
                st.caption(ev.text[:1800])

    with tab3:
        st.subheader("節設計")
        st.write(f"中心主張: {blueprint.get('thesis', '')}")
        sections = blueprint.get("sections", [])
        for sec in sections:
            with st.expander(sec.get("name", "section")):
                st.write(f"role: {sec.get('role', '')}")
                st.write(f"objective: {sec.get('objective', '')}")
                st.write(f"key_claim: {sec.get('key_claim', '')}")
                st.write(f"target_chars: {sec.get('target_chars', '')}")
                st.write(f"must_use_terms: {', '.join(sec.get('must_use_terms', []))}")
                st.write(f"evidence_ids: {', '.join(sec.get('evidence_ids', []))}")
                st.write(f"avoid_overlap_with: {', '.join(sec.get('avoid_overlap_with', []))}")

    with tab4:
        st.subheader("内部評価")
        st.json(critique)
        st.markdown("<div class='tiny'>初回添削より必要な時だけ改稿する設計です。</div>", unsafe_allow_html=True)
        with st.expander("抽出チャンク数と中間情報"):
            st.write(f"抽出チャンク数: {len(result['chunks'])}")
            st.write(f"Stage1候補数: {len(result['stage1'])}")
            st.write(f"Stage2候補数: {len(result['stage2'])}")
            st.write("重要概念候補:")
            st.write(", ".join(important_terms_from_evidences(selected, top_k=16)))
