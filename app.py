import json
import html
import re
import textwrap
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import fitz
import streamlit as st
from openai import OpenAI

# ============================================================
# UI
# ============================================================
st.set_page_config(page_title="ReportFlow Pro", page_icon="📝", layout="wide")

st.markdown(
    """
    <style>
    .block-container { max-width: 1180px; padding-top: 1.1rem; padding-bottom: 4rem; }
    .hero {
        padding: 1.2rem 1.3rem;
        border: 1px solid rgba(120,120,120,0.16);
        border-radius: 22px;
        margin-bottom: 1rem;
        background: linear-gradient(180deg, rgba(250,250,250,0.045), rgba(250,250,250,0.015));
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
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero">
        <h1 style="margin-bottom:0.18rem;">📝 ReportFlow Pro</h1>
        <div style="opacity:0.8;">
            講義PDFを読み込み、課題文に合わせて資料固有の概念を活かしたレポート本文を生成します。
            高速モードは待てる速度、ハイエンドモードは長文・提出前仕上げ向けです。
        </div>
        <div style="margin-top:0.55rem;">
            <span class="pill">高速モード</span>
            <span class="pill">ハイエンドモード</span>
            <span class="pill">根拠可視化</span>
            <span class="pill">字数補正</span>
            <span class="pill">10000字対応</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# Constants
# ============================================================
FAST_DEFAULT_MODEL = "gpt-4.1-mini"
HIGH_DEFAULT_MODEL = "gpt-4.1-mini"

STOPWORDS_JA = {
    "する", "ある", "いる", "こと", "これ", "それ", "ため", "よう", "もの", "また", "さらに", "できる",
    "なる", "おける", "および", "より", "など", "その", "この", "あの", "いう", "して", "した",
    "している", "について", "による", "として", "れる", "られる", "及び", "各", "本", "的", "的な",
    "である", "ます", "ました", "です", "でした", "ない", "なり", "一方", "場合", "では", "ので",
    "から", "まで", "へ", "を", "に", "が", "は", "も", "と", "の", "年", "月", "日",
}

ABSTRACT_TERMS = {
    "競争優位", "顧客価値", "ブランド価値", "差別化", "信頼", "重要", "必要", "有効", "戦略", "価値",
    "消費者", "顧客", "市場", "競争", "企業", "製品", "サービス",
}

FAST_SETTINGS = {
    "chunk_char_min": 320,
    "chunk_char_max": 1200,
    "local_keep": 18,
    "api_keep": 10,
    "final_keep": 7,
    "min_source_terms": 3,
}

HIGH_SETTINGS = {
    "chunk_char_min": 220,
    "chunk_char_max": 1000,
    "local_keep": 30,
    "api_keep": 18,
    "final_keep": 10,
    "min_source_terms": 5,
}

# ============================================================
# Data classes
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
    local_score: float = 0.0


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
    coarse_score: int
    precise_score: int
    final_score: float
    reason: str
    duplicate_group: int = -1

# ============================================================
# API helpers
# ============================================================
def get_api_key() -> str:
    secrets_key = st.secrets.get("OPENAI_API_KEY", None)
    api_key_input = st.sidebar.text_input("OpenAI APIキー", type="password")
    return (secrets_key or api_key_input or "").strip()


def get_client(api_key: str) -> OpenAI:
    return OpenAI(api_key=api_key)


def _extract_json_object(text: str) -> str:
    s = (text or "").strip()
    s = re.sub(r"^```json\s*", "", s)
    s = re.sub(r"^```\s*", "", s)
    s = re.sub(r"\s*```$", "", s)
    start = s.find("{")
    end = s.rfind("}")
    if start != -1 and end != -1 and end > start:
        return s[start:end + 1]
    return s


def _repair_json_text(text: str) -> str:
    s = _extract_json_object(text)
    s = s.replace("“", '"').replace("”", '"').replace("‘", "'").replace("’", "'")
    s = re.sub(r",(\s*[}\]])", r"\1", s)
    open_braces = s.count("{")
    close_braces = s.count("}")
    if close_braces < open_braces:
        s += "}" * (open_braces - close_braces)
    open_brackets = s.count("[")
    close_brackets = s.count("]")
    if close_brackets < open_brackets:
        s += "]" * (open_brackets - close_brackets)
    return s


def call_json(
    client: OpenAI,
    model: str,
    system: str,
    user_prompt: str,
    temperature: float = 0.2,
    max_output_tokens: int = 2200,
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
    raw = (response.output_text or "").strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        repaired = _repair_json_text(raw)
        return json.loads(repaired)


def call_text(
    client: OpenAI,
    model: str,
    system: str,
    user_prompt: str,
    temperature: float = 0.35,
    max_output_tokens: int = 3200,
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
# Text helpers
# ============================================================
def normalize_space(text: str) -> str:
    text = html.unescape(text or "")
    text = text.replace("\u3000", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def clean_text(text: str) -> str:
    if not text:
        return ""
    text = text.replace("###", "").replace("##", "").replace("#", "")
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def safe_json_list(value: Any) -> List[str]:
    if isinstance(value, list):
        return [str(x).strip() for x in value if str(x).strip()]
    if isinstance(value, str):
        return [p.strip() for p in re.split(r"[,、/\n]", value) if p.strip()]
    return []


def lexical_terms(text: str, top_k: int = 12) -> List[str]:
    text = normalize_space(text)
    cands = re.findall(r"[A-Za-z][A-Za-z0-9_\-]{2,}|[一-龥ァ-ヶー]{2,}", text)
    cands = [c for c in cands if c not in STOPWORDS_JA and not re.fullmatch(r"\d+", c)]
    freq = Counter(cands)
    seen = set()
    out = []
    for word, _ in freq.most_common(top_k * 4):
        lw = word.lower()
        if lw in seen:
            continue
        seen.add(lw)
        out.append(word)
        if len(out) >= top_k:
            break
    return out


def specificity_hint_score(text: str) -> float:
    digits = len(re.findall(r"\d+", text))
    caps = len(re.findall(r"\b[A-Z][A-Za-z0-9_\-]+\b", text))
    punct = len(re.findall(r"[:：（）()『』「」]", text))
    terms = len(lexical_terms(text, top_k=10))
    return round(min(10.0, 0.5 * digits + 1.1 * caps + 0.22 * punct + 0.38 * terms), 2)


def theme_overlap_score(theme_terms: List[str], chunk_terms: List[str], text: str) -> float:
    theme_set = {x.lower() for x in theme_terms}
    chunk_set = {x.lower() for x in chunk_terms}
    overlap = len(theme_set & chunk_set)
    digits = len(re.findall(r"\d+", text))
    return overlap * 2.4 + min(len(text) / 600, 2.3) + min(digits, 3) * 0.2


def jaccard_similarity(a_terms: List[str], b_terms: List[str]) -> float:
    a = {x.lower() for x in a_terms}
    b = {x.lower() for x in b_terms}
    if not a or not b:
        return 0.0
    return len(a & b) / max(len(a | b), 1)


def final_score_formula(coarse_score: int, precise_score: int, usefulness_score: int, specificity_score: int) -> float:
    rel = (coarse_score + precise_score) / 6
    useful = usefulness_score / 3
    spec = specificity_score / 10
    return round((0.50 * rel + 0.20 * useful + 0.30 * spec) * 100, 2)


def section_split_count(target_length: int, mode: str) -> int:
    if mode == "高速":
        if target_length <= 2200:
            return 1
        if target_length <= 4000:
            return 2
        return 3
    if target_length <= 2200:
        return 2
    if target_length <= 4000:
        return 3
    if target_length <= 6000:
        return 4
    if target_length <= 8000:
        return 5
    return 7


def length_band_status(text: str, target_length: int, strict: bool = False) -> str:
    current = len(text)
    low_ratio = 0.95 if strict else 0.92
    high_ratio = 1.07 if strict else 1.10
    low = int(target_length * low_ratio)
    high = int(target_length * high_ratio)
    if current < low:
        return "short"
    if current > high:
        return "long"
    return "ok"


def build_length_targets(target_length: int, strict: bool = False) -> Dict[str, int]:
    low_ratio = 0.95 if strict else 0.92
    high_ratio = 1.07 if strict else 1.10
    return {"min": int(target_length * low_ratio), "ideal": target_length, "max": int(target_length * high_ratio)}


def important_terms_from_evidences(evidences: List[Evidence], top_k: int = 14) -> List[str]:
    counter = Counter()
    for ev in evidences:
        for t in ev.terminology:
            if len(t) >= 2:
                counter[t] += 1 + ev.specificity_score / 10
        for t in lexical_terms(" ".join([ev.topic, ev.proposition, ev.evidence]), top_k=6):
            counter[t] += 0.4
    return [w for w, _ in counter.most_common(top_k)]


def count_abstract_term_hits(text: str) -> int:
    return sum(1 for t in ABSTRACT_TERMS if t in text)


def detect_external_example_risk(text: str, evidences: List[Evidence]) -> int:
    allowed_text = " ".join(ev.topic + " " + ev.proposition + " " + ev.evidence + " " + ev.example + " " + " ".join(ev.terminology) for ev in evidences)
    names = re.findall(r"\b[A-Z][A-Za-z&\-]{2,}(?:\s+[A-Z][A-Za-z&\-]{2,})*\b", text)
    risk = 0
    for n in names[:20]:
        if n not in allowed_text and len(n) >= 3:
            risk += 1
    if risk == 0:
        return 0
    if risk <= 2:
        return 1
    if risk <= 4:
        return 2
    return 3


def is_truncated_text(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return False
    bad_endings = ("は", "が", "を", "に", "で", "と", "、", "・", "（", "(", "の", "も", "や", "より", "し", "する", "いる")
    if stripped.endswith(bad_endings):
        return True
    return not stripped.endswith(("。", "！", "？", ".", "!", "?", "」", "』", ")", "）"))


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


def join_evidence_briefs(evidences: List[Evidence], limit: int) -> str:
    return "\n\n".join(render_evidence_brief(ev) for ev in evidences[:limit])


def evidence_lookup(evidences: List[Evidence]) -> Dict[str, Evidence]:
    return {ev.chunk_id: ev for ev in evidences}


def section_evidence_bundle(section: Dict[str, Any], lookup: Dict[str, Evidence]) -> str:
    blocks = []
    for eid in section.get("evidence_ids", []):
        ev = lookup.get(eid)
        if ev:
            blocks.append(render_evidence_brief(ev))
    return "\n\n".join(blocks)


def strict_source_constraint_text(strict_source_only: bool, hard: bool = False) -> str:
    if strict_source_only:
        if hard:
            return "- 資料にない企業名・ブランド名・事例は削除または資料語へ置換する\n- 自分の一般知識で補った具体例は禁止\n"
        return "- 資料に明示されていない企業名・ブランド名・事例は原則出さない\n- 自分の一般知識で補った具体例は禁止\n"
    return "- 企業例は資料にあるものを優先し、資料外例は原則避ける\n"

# ============================================================
# PDF extraction
# ============================================================
def block_texts_from_page(page) -> List[Tuple[int, str]]:
    blocks = page.get_text("blocks")
    out = []
    for idx, block in enumerate(blocks):
        if len(block) < 5:
            continue
        text = normalize_space(block[4])
        if not text or len(text) <= 2:
            continue
        out.append((idx, text))
    return out


def merge_blocks_semantically(blocks: List[Tuple[int, str]], min_chars: int, max_chars: int) -> List[Tuple[str, str]]:
    chunks = []
    current_idxs: List[int] = []
    current_texts: List[str] = []

    def flush() -> None:
        nonlocal current_idxs, current_texts
        if not current_texts:
            return
        text = "\n".join(current_texts).strip()
        if text:
            chunks.append((f"{current_idxs[0]}-{current_idxs[-1]}", text))
        current_idxs = []
        current_texts = []

    for idx, text in blocks:
        text = normalize_space(text)
        if not text:
            continue
        is_headerish = (len(text) <= 42 and not text.endswith("。")) or bool(re.fullmatch(r"[0-9０-９IVXivx一二三四五六七八九十]+[\.．\-：:]?.*", text))
        candidate = "\n".join(current_texts + [text]) if current_texts else text
        if current_texts and (len(candidate) > max_chars or is_headerish):
            flush()
        current_idxs.append(idx)
        current_texts.append(text)
        if len("\n".join(current_texts)) >= min_chars and text.endswith(("。", ".", "?", "!", "）", ")")):
            flush()

    flush()
    return chunks


def extract_chunks(files, chunk_char_min: int, chunk_char_max: int) -> List[Chunk]:
    chunks: List[Chunk] = []
    for uploaded_file in files:
        file_bytes = uploaded_file.read()
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        for page_no, page in enumerate(doc, start=1):
            blocks = block_texts_from_page(page)
            if not blocks:
                continue
            merged = merge_blocks_semantically(blocks, min_chars=chunk_char_min, max_chars=chunk_char_max)
            for local_idx, (block_range, text) in enumerate(merged, start=1):
                chunks.append(
                    Chunk(
                        chunk_id=f"{uploaded_file.name}::p{page_no}::c{local_idx}",
                        file=uploaded_file.name,
                        page=page_no,
                        block_range=block_range,
                        text=text,
                        short=normalize_space(text)[:1000],
                        char_count=len(text),
                        lexical_terms=lexical_terms(text, top_k=12),
                        specificity_hint=specificity_hint_score(text),
                    )
                )
    return chunks

# ============================================================
# Ranking and evidence
# ============================================================
def local_prefilter(chunks: List[Chunk], theme: str, keep: int) -> List[Chunk]:
    theme_terms = lexical_terms(theme, top_k=12)
    scored = []
    for chunk in chunks:
        score = theme_overlap_score(theme_terms, chunk.lexical_terms, chunk.text) + 0.7 * chunk.specificity_hint
        chunk.local_score = score
        scored.append((score, chunk))
    scored.sort(key=lambda x: (x[0], x[1].char_count), reverse=True)
    return [x[1] for x in scored[: min(keep, len(scored))]]


def build_evidence_one_call(client: OpenAI, model: str, theme: str, chunk: Chunk) -> Evidence:
    prompt = f"""
以下の資料断片を、大学レポート作成用に構造化しつつ採点してください。
返答はJSONのみ。

必要キー:
- topic: 1文
- proposition: レポートに使える主張 1文
- evidence: この断片が根拠として与える内容 1文
- example: 具体例があれば1文、なければ "なし"
- terminology: 重要概念の配列 3〜8個
- contrast: 対比があれば1文、なければ "なし"
- cause_effect: 因果があれば1文、なければ "なし"
- role: [definition, claim, evidence, example, mechanism, comparison, implication]
- assignment_relevance: 課題文のどの論点に使えるか 1文
- coarse_score: 0-3
- precise_score: 0-3
- usefulness_score: 0-3
- specificity_score: 0-10
- reason: 24字以内

課題文:
{theme}

資料断片:
{chunk.text[:3800]}
"""
    data = call_json(client, model, "資料断片を中間表現へ変換しつつ採点する。資料にないことを足さない。JSONのみ返す。", prompt, temperature=0.1, max_output_tokens=1400)
    terminology = safe_json_list(data.get("terminology", []))
    coarse_score = int(max(0, min(3, data.get("coarse_score", 0))))
    precise_score = int(max(0, min(3, data.get("precise_score", 0))))
    usefulness_score = int(max(0, min(3, data.get("usefulness_score", 0))))
    specificity_score = int(max(0, min(10, data.get("specificity_score", 0))))
    final_score = final_score_formula(coarse_score, precise_score, usefulness_score, specificity_score)
    return Evidence(
        chunk_id=chunk.chunk_id,
        file=chunk.file,
        page=chunk.page,
        block_range=chunk.block_range,
        text=chunk.text,
        topic=str(data.get("topic", "")).strip(),
        proposition=str(data.get("proposition", "")).strip(),
        evidence=str(data.get("evidence", "")).strip(),
        example=str(data.get("example", "")).strip(),
        terminology=terminology,
        contrast=str(data.get("contrast", "")).strip(),
        cause_effect=str(data.get("cause_effect", "")).strip(),
        role=str(data.get("role", "evidence")).strip(),
        assignment_relevance=str(data.get("assignment_relevance", "")).strip(),
        specificity_score=specificity_score,
        usefulness_score=usefulness_score,
        coarse_score=coarse_score,
        precise_score=precise_score,
        final_score=final_score,
        reason=str(data.get("reason", "")).strip()[:24],
    )


def cluster_duplicates(evidences: List[Evidence], threshold: float = 0.42) -> List[Evidence]:
    group_id = 0
    for i in range(len(evidences)):
        if evidences[i].duplicate_group != -1:
            continue
        evidences[i].duplicate_group = group_id
        base_terms = lexical_terms(" ".join([evidences[i].topic, evidences[i].proposition, evidences[i].evidence, " ".join(evidences[i].terminology)]), top_k=14)
        for j in range(i + 1, len(evidences)):
            if evidences[j].duplicate_group != -1:
                continue
            cand_terms = lexical_terms(" ".join([evidences[j].topic, evidences[j].proposition, evidences[j].evidence, " ".join(evidences[j].terminology)]), top_k=14)
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
    reps = []
    for _, items in grouped.items():
        reps.append(sorted(items, key=lambda x: (x.final_score, x.precise_score, x.specificity_score), reverse=True)[0])
    return sorted(reps, key=lambda x: (x.final_score, x.precise_score, x.usefulness_score), reverse=True)[:limit]

# ============================================================
# Planning / generation
# ============================================================
def build_argument_map(client: OpenAI, model: str, theme: str, evidences: List[Evidence], points: int = 4) -> str:
    source = join_evidence_briefs(evidences, 10)
    prompt = f"""
以下は課題文と、採用候補の資料根拠です。
この課題に答えるための論点を {points} 個前後で整理してください。
プレーンテキストのみ出力してください。

条件:
- point / source / why_it_matters の形で簡潔にまとめる
- 似た論点は統合する
- 一般論ではなく資料根拠に依拠する
- 資料固有の概念を優先する

課題文:
{theme}

資料根拠:
{source}
"""
    return call_text(client, model, "講義資料から論点設計を行う。プレーンテキストのみ返す。", prompt, temperature=0.2, max_output_tokens=1600)


def build_fallback_blueprint(evidences: List[Evidence], target_length: int, sections_count: int) -> Dict[str, Any]:
    thesis = "科学技術の進展は、マーケティングチャネル設計を直接化・情報化・再編成する方向へ変化させている。"
    section_names = [
        "問題設定と基本概念",
        "ディスインターメディエーションの進展",
        "電子市場と垂直市場の再編",
        "EDIと物流情報管理",
        "流通業者の役割変化",
        "垂直的チャネル・コンフリクト",
        "総合考察と結論",
    ]
    roles = [
        "definition",
        "mechanism",
        "structure_change",
        "evidence",
        "role_change",
        "conflict_adjustment",
        "conclusion",
    ]
    char_base = max(700, target_length // max(sections_count, 1))
    ids = [ev.chunk_id for ev in evidences[: max(2, min(len(evidences), sections_count * 2))]]
    sections: List[Dict[str, Any]] = []
    for i in range(sections_count):
        ev_slice = ids[i:i + 2] if i < len(ids) else ids[:2]
        must_terms: List[str] = []
        if i < len(evidences):
            must_terms.extend(evidences[i].terminology[:3])
        if len(must_terms) < 2:
            must_terms.extend(["マーケティングチャネル", "流通業者"])
        sections.append(
            {
                "name": section_names[i] if i < len(section_names) else f"第{i+1}節",
                "role": roles[i] if i < len(roles) else "analysis",
                "objective": f"{section_names[i] if i < len(section_names) else f'第{i+1}節'}に関する論点を整理する",
                "key_claim": f"{section_names[i] if i < len(section_names) else f'第{i+1}節'}を通じて、科学技術の進展がチャネル設計に与える影響を示す",
                "evidence_ids": ev_slice,
                "must_use_terms": must_terms[:5],
                "avoid_overlap_with": [s["name"] for s in sections[-2:]],
                "forbidden_patterns": ["同じ主張の言い換え", "一般論だけの説明"],
                "must_not_repeat": [s["name"] for s in sections[:-1]],
                "target_chars": char_base,
            }
        )
    return {"thesis": thesis, "sections": sections}


def build_blueprint(client: OpenAI, model: str, theme: str, evidences: List[Evidence], argument_map: str, target_length: int, sections_count: int) -> Dict[str, Any]:
    source = join_evidence_briefs(evidences, 12)
    prompt = f"""
次の課題文、論点整理、資料根拠をもとにレポート設計図を作ってください。
返答はJSONのみ。

必要キー:
- thesis: 全体の中心主張 1文
- sections: 配列
  各要素の必要キー:
    - name
    - role
    - objective
    - key_claim
    - evidence_ids: chunk_id配列（1〜4個）
    - must_use_terms: 概念配列（2〜5個）
    - avoid_overlap_with: 節名配列
    - forbidden_patterns: 配列
    - must_not_repeat: 配列（この節では再説明しない論点）
    - target_chars: 目安文字数

条件:
- 節数は {sections_count}
- 各節は別の役割を持つ
- 各節は前の節で十分に説明した話題を繰り返さない
- 少なくとも1節は資料間のつながりや対比を扱う
- 一般論で水増ししない
- 長さ配分は目標文字数に合わせる

課題文:
{theme}

目標文字数:
{target_length}

論点整理:
{argument_map}

資料根拠:
{source}
"""
    try:
        data = call_json(client, model, "大学レポートの設計図を作る。JSONのみ返す。", prompt, temperature=0.2, max_output_tokens=2600)
        if not isinstance(data, dict):
            raise ValueError("invalid blueprint")
        if "sections" not in data or not isinstance(data["sections"], list) or not data["sections"]:
            raise ValueError("invalid blueprint sections")
        return data
    except Exception:
        return build_fallback_blueprint(evidences, target_length, sections_count)


def generate_single_pass_report(client: OpenAI, model: str, theme: str, target_length: int, argument_map: str, evidences: List[Evidence], style_mode: str, abstraction_mode: str, strict_source_only: bool) -> str:
    style_instruction = {"標準": "自然で読みやすい文体にする。", "やや硬め": "少しフォーマルで簡潔な文体にする。", "やや柔らかめ": "少し柔らかいが幼くしない文体にする。"}[style_mode]
    abstraction_instruction = {"標準": "抽象と具体のバランスを標準にする。", "抽象度高め": "やや抽象化して上位概念で整理する。", "抽象度低め": "やや具体的にし、資料上の概念や事例を厚めに使う。"}[abstraction_mode]
    source = join_evidence_briefs(evidences, 8)
    must_terms = important_terms_from_evidences(evidences, top_k=8)
    source_constraint = strict_source_constraint_text(strict_source_only)
    prompt = f"""
以下の課題文、論点整理、資料根拠をもとに約{target_length}字のレポート本文を書いてください。
本文のみ出力してください。

課題文:
{theme}

論点整理:
{argument_map}

資料根拠:
{source}

必ず使いたい概念:
{', '.join(must_terms)}

条件:
- 本文のみ
- 見出し禁止
- 箇条書き禁止
- Markdown禁止
- 同じ主張の言い換え反復を避ける
- 資料固有の概念を最低4つ入れる
- 各段落で資料由来の概念を最低1つは明示する
- 資料間のつながりや比較を最低1回入れる
- 一般論に逃げすぎない
- 抽象語だけで段落を締めない
{source_constraint}- 最後は短く自分の考察で締める
- {style_instruction}
- {abstraction_instruction}
"""
    return clean_text(call_text(client, model, "資料密着型の大学レポート本文を書く。本文のみ返す。", prompt, temperature=0.42, max_output_tokens=5200))


def generate_section_text(client: OpenAI, model: str, theme: str, thesis: str, section: Dict[str, Any], evidence_bundle: str, previous_text: str, used_terms: List[str], style_mode: str, abstraction_mode: str, strict_source_only: bool) -> str:
    style_instruction = {"標準": "自然で読みやすい大学レポート文体にする。", "やや硬め": "少しフォーマルで締まった文体にする。", "やや柔らかめ": "少し柔らかいが幼くしない文体にする。"}[style_mode]
    abstraction_instruction = {"標準": "抽象と具体のバランスを標準にする。", "抽象度高め": "やや抽象化し、上位概念で整理する。", "抽象度低め": "やや具体化し、資料記述を厚めに使う。"}[abstraction_mode]
    source_constraint = strict_source_constraint_text(strict_source_only)
    prev_tail = previous_text[-1500:] if previous_text else "なし"
    used_terms_text = ", ".join(used_terms[-18:]) if used_terms else "なし"
    must_terms = ", ".join(section.get("must_use_terms", []))
    must_not_repeat = ", ".join(section.get("must_not_repeat", []))
    forbidden_patterns = ", ".join(section.get("forbidden_patterns", []))
    target_chars = int(section.get("target_chars", 1200) or 1200)
    prompt = f"""
課題文に対するレポートのうち、この節だけを書いてください。
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
target_chars: {target_chars}
must_use_terms: {must_terms}
forbidden_patterns: {forbidden_patterns}
must_not_repeat: {must_not_repeat}

この節で使う根拠:
{evidence_bundle}

これまで本文で使った概念:
{used_terms_text}

すでに書かれている前半:
{prev_tail}

条件:
- 本文のみ
- 見出し禁止
- 箇条書き禁止
- 同じ主張の繰り返し禁止
- この節の key_claim に集中する
- must_use_terms を最低2個使う
- must_not_repeat に入っている話題は再説明しない
- 前半ですでに十分述べた論点を言い換えて繰り返さない
- 各段落で evidence_bundle の用語を少なくとも1つは明示する
- 資料にないことを大きく膨らませない
- 因果、対比、具体例のどれかを最低1つ入れる
- 抽象語だけで段落を締めない
{source_constraint}- 前半と自然につながる
- {style_instruction}
- {abstraction_instruction}
- 目安として {int(target_chars * 0.90)} 字以上は書く
"""
    return clean_text(call_text(client, model, "資料根拠に忠実で、前半の再説明を避けた大学レポート本文を書く。本文のみ返す。", prompt, temperature=0.35, max_output_tokens=4200))


def complete_section_if_truncated(client: OpenAI, model: str, theme: str, section_name: str, section_text: str, evidences: List[Evidence], strict_source_only: bool) -> str:
    if not is_truncated_text(section_text):
        return section_text
    source_constraint = strict_source_constraint_text(strict_source_only, hard=True)
    evidence_text = join_evidence_briefs(evidences, 4)
    prompt = f"""
以下の本文は途中で切れている可能性があります。続きを短く補完して、この節を自然に完結させてください。
補完部分のみ出力してください。

課題文:
{theme}

節名:
{section_name}

参考根拠:
{evidence_text}

現在の本文:
{section_text[-3500:]}

条件:
- 補完部分のみ
- 新しい論点を追加しない
- 既存の議論を自然に完結させる
{source_constraint}- 300〜700字程度でまとめる
"""
    addition = clean_text(call_text(client, model, "途中で切れた本文を自然に完結させる補完文を書く。本文のみ返す。", prompt, temperature=0.25, max_output_tokens=1600))
    if not addition:
        return section_text
    return clean_text(section_text + "\n\n" + addition)


def patch_missing_terms(client: OpenAI, model: str, theme: str, text: str, evidences: List[Evidence], min_terms: int, strict_source_only: bool) -> str:
    important = important_terms_from_evidences(evidences, top_k=12)
    missing = [t for t in important if t not in text][:min_terms]
    if not missing:
        return text
    ref_blocks = []
    for ev in evidences[:6]:
        if any(t in ev.terminology or t in ev.topic or t in ev.proposition for t in missing):
            ref_blocks.append(render_evidence_brief(ev))
    ref_text = "\n\n".join(ref_blocks[:4])
    source_constraint = "資料外の企業例・一般知識の補完は禁止。" if strict_source_only else "資料外例は原則避ける。"
    prompt = f"""
以下のレポート本文を、論旨を保ったまま最小限だけ書き直してください。
目的は、未使用の重要概念を自然に織り込むことです。
本文のみ出力してください。

課題文:
{theme}

未使用の重要概念:
{', '.join(missing)}

参考根拠:
{ref_text}

本文:
{text}

条件:
- 本文のみ
- 構成を大きく変えない
- 無理やり全部入れなくてよい
- 一般論で水増ししない
- {source_constraint}
"""
    return clean_text(call_text(client, model, "最小修正で概念不足を補う。本文のみ返す。", prompt, temperature=0.25, max_output_tokens=3200))


def regenerate_conclusion(client: OpenAI, model: str, theme: str, report: str, evidences: List[Evidence], style_mode: str) -> str:
    paragraphs = [p.strip() for p in re.split(r"\n\n+", report) if p.strip()]
    if len(paragraphs) < 2:
        return report
    body = "\n\n".join(paragraphs[:-1])
    last = paragraphs[-1]
    evidence_text = join_evidence_briefs(evidences, 5)
    style_instruction = {"標準": "自然で読みやすい締めにする。", "やや硬め": "少しフォーマルで締まった締めにする。", "やや柔らかめ": "少し柔らかいが幼くしない締めにする。"}[style_mode]
    prompt = f"""
以下のレポート本文の最後の1段落だけ書き直してください。
本文1段落のみ出力してください。

課題文:
{theme}

前半本文:
{body[-3200:]}

現在の結論段落:
{last}

参考根拠:
{evidence_text}

条件:
- 新しい論点を追加しない
- それまでの議論を再統合する
- 抽象語の反復だけで終わらない
- 資料に依拠したまとめにする
- {style_instruction}
"""
    new_last = clean_text(call_text(client, model, "結論段落だけを再統合して強化する。本文1段落のみ返す。", prompt, temperature=0.3, max_output_tokens=1000))
    paragraphs[-1] = new_last
    return "\n\n".join(paragraphs)

# ============================================================
# Critique and length control
# ============================================================
def critique_report(client: OpenAI, model: str, theme: str, report: str, evidences: List[Evidence]) -> Dict[str, Any]:
    ref_terms = important_terms_from_evidences(evidences, top_k=12)
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
{', '.join(ref_terms)}

課題文:
{theme}

本文:
{report}
"""
    data = call_json(client, model, "資料固有性、重複、論点カバー、AIっぽさを厳しく評価する。JSONのみ返す。", prompt, temperature=0.0, max_output_tokens=900)
    data["external_example_risk"] = detect_external_example_risk(report, evidences)
    data["abstract_term_pressure"] = count_abstract_term_hits(report)
    missing_terms = [t for t in important_terms_from_evidences(evidences, top_k=12) if t not in report][:6]
    data["missing_source_terms"] = missing_terms
    if data["external_example_risk"] >= 1 and "資料外例の混入" not in data.get("revision_focus", []):
        data.setdefault("revision_focus", []).append("資料外例の混入")
        data["revision_needed"] = True
    if len(missing_terms) >= 3 and "資料語の不足" not in data.get("revision_focus", []):
        data.setdefault("revision_focus", []).append("資料語の不足")
    return data


def rewrite_once(client: OpenAI, model: str, theme: str, report: str, critique: Dict[str, Any], evidences: List[Evidence], style_mode: str, strict_source_only: bool) -> str:
    style_instruction = {"標準": "自然で読みやすくする。", "やや硬め": "少しフォーマルで締まった文体にする。", "やや柔らかめ": "少し柔らかくするが、幼くしない。"}[style_mode]
    source_constraint = strict_source_constraint_text(strict_source_only, hard=True)
    source = join_evidence_briefs(evidences, 6)
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
{source}

元の本文:
{report}

条件:
- 本文のみ
- なるべく構成を維持
- 資料固有語を消さない
- 重複を減らす
- 抽象的すぎる箇所だけ具体化する
{source_constraint}- {style_instruction}
"""
    return clean_text(call_text(client, model, "最小限の改稿で品質を上げる。本文のみ返す。", prompt, temperature=0.28, max_output_tokens=4200))


def humanize_if_needed(client: OpenAI, model: str, theme: str, report: str, critique: Dict[str, Any]) -> str:
    if int(critique.get("ai_stiffness", 0)) < 2:
        return report
    prompt = f"""
以下のレポートを、内容を変えすぎずに少しだけ自然な日本語へ整えてください。
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
"""
    return clean_text(call_text(client, model, "AIっぽさだけを軽く減らす。本文のみ返す。", prompt, temperature=0.3, max_output_tokens=3200))


def append_report_if_too_short(client: OpenAI, model: str, theme: str, report: str, evidences: List[Evidence], target_length: int, strict_source_only: bool) -> str:
    status = length_band_status(report, target_length, strict=strict_source_only)
    if status != "short":
        return report
    targets = build_length_targets(target_length, strict=strict_source_only)
    shortage = max(0, targets["min"] - len(report))
    if shortage <= 0:
        return report
    source_constraint = strict_source_constraint_text(strict_source_only, hard=True)
    evidence_text = join_evidence_briefs(evidences, 6)
    target_addition = max(700, min(shortage + 250, 1400))
    prompt = f"""
以下の本文は字数が不足しています。不足分を埋める追加段落だけを書いてください。
追加段落のみ出力してください。

課題文:
{theme}

現在の文字数:
{len(report)}
最低必要文字数:
{targets['min']}
理想文字数:
{targets['ideal']}
不足目安:
約{shortage}字
今回追加したい目安:
約{target_addition}字

参考根拠:
{evidence_text}

既存本文:
{report[-5000:]}

条件:
- 追加段落のみ
- 新しい論点を追加しない
- まだ十分に展開していない論点の補足だけを行う
- 既に詳しく説明した論点の言い換え再説明は禁止
- 各段落で資料由来の概念を明示する
{source_constraint}- 約{target_addition}字ぶんの中身を増やす
"""
    addition = clean_text(call_text(client, model, "字数不足を、未展開論点の補足だけで埋める追加段落を書く。本文のみ返す。", prompt, temperature=0.26, max_output_tokens=2800))
    if not addition:
        return report
    return clean_text(report + "\n\n" + addition)


def append_global_continuation_if_needed(client: OpenAI, model: str, theme: str, report: str, evidences: List[Evidence], target_length: int, strict_source_only: bool) -> str:
    if length_band_status(report, target_length, strict=strict_source_only) != "short":
        return report
    source_constraint = strict_source_constraint_text(strict_source_only, hard=True)
    evidence_text = join_evidence_briefs(evidences, 6)
    targets = build_length_targets(target_length, strict=strict_source_only)
    shortage = max(0, targets["min"] - len(report))
    target_addition = max(700, min(shortage + 250, 1300))
    prompt = f"""
以下の本文はまだ字数が不足しています。続きを追加してください。本文のみ出力してください。

課題文:
{theme}

不足目安:
約{shortage}字

参考根拠:
{evidence_text}

現在の本文:
{report[-6000:]}

条件:
- 本文のみ
- 新しい論点を追加しない
- 既存論点のうち、まだ浅い部分の比較・含意・限定条件だけを補う
- 既に説明した内容の言い換え再説明は禁止
- 最後は自然に終える
{source_constraint}- 約{target_addition}字を目安に追記する
"""
    continuation = clean_text(call_text(client, model, "不足字数を、未展開部分の補足だけで埋める続きを書く。本文のみ返す。", prompt, temperature=0.24, max_output_tokens=2600))
    if not continuation:
        return report
    return clean_text(report + "\n\n" + continuation)


def compress_report_if_too_long(client: OpenAI, model: str, report: str, target_length: int, strict_source_only: bool) -> str:
    if length_band_status(report, target_length, strict=strict_source_only) != "long":
        return report
    targets = build_length_targets(target_length, strict=strict_source_only)
    prompt = f"""
以下の本文を、内容をなるべく保ったまま約{targets['ideal']}字に調整してください。
本文のみ出力してください。

条件:
- 本文のみ
- 不自然な圧縮をしない
- 資料固有語を残す
- 明らかに重複している段落や言い換え反復を優先して削る
- できれば {targets['max']} 字以下にする

本文:
{report}
"""
    return clean_text(call_text(client, model, "文字数と重複を同時に自然に圧縮する。本文のみ返す。", prompt, temperature=0.2, max_output_tokens=4200))


def enforce_length_requirements(client: OpenAI, model: str, theme: str, report: str, evidences: List[Evidence], target_length: int, strict_source_only: bool) -> str:
    for _ in range(3):
        status = length_band_status(report, target_length, strict=strict_source_only)
        if status == "short":
            report = append_report_if_too_short(client, model, theme, report, evidences, target_length, strict_source_only)
            report = append_global_continuation_if_needed(client, model, theme, report, evidences, target_length, strict_source_only)
            continue
        if status == "long":
            report = compress_report_if_too_long(client, model, report, target_length, strict_source_only)
        break
    return report


def deduplicate_report_pass(client: OpenAI, model: str, theme: str, report: str, evidences: List[Evidence], target_length: int) -> str:
    evidence_text = join_evidence_briefs(evidences, 6)
    prompt = f"""
以下のレポート本文について、論旨を保ったまま重複箇所だけを整理してください。本文のみ出力してください。

課題文:
{theme}

参考根拠:
{evidence_text}

目標文字数:
{target_length}

条件:
- 本文のみ
- 同じ論点の言い換え反復を統合する
- 後半で前半をなぞっている箇所を削る
- 構成の流れは保つ
- 資料固有語は残す

本文:
{report}
"""
    return clean_text(call_text(client, model, "長文レポートの重複だけを整理して締める。本文のみ返す。", prompt, temperature=0.18, max_output_tokens=4200))

# ============================================================
# Theme shaping
# ============================================================
def build_user_guidance(theme: str, focus_points: str, preferred_concepts: str, excluded_topics: str, source_scope: str) -> str:
    parts = [f"課題文: {theme.strip()}"]
    if focus_points.strip():
        parts.append(f"重視観点: {focus_points.strip()}")
    if preferred_concepts.strip():
        parts.append(f"扱いたい概念: {preferred_concepts.strip()}")
    if excluded_topics.strip():
        parts.append(f"除外したい話題: {excluded_topics.strip()}")
    if source_scope.strip():
        parts.append(f"優先範囲: {source_scope.strip()}")
    return "\n".join(parts)


def broad_theme_warning(theme: str, focus_points: str, preferred_concepts: str, excluded_topics: str, source_scope: str) -> bool:
    broad_markers = ["マーケティング", "ブランド", "消費者", "科学技術", "戦略", "流通", "企業", "影響", "論じ", "重要性", "発展"]
    score = sum(1 for m in broad_markers if m in theme)
    helper_count = sum(bool(x.strip()) for x in [focus_points, preferred_concepts, excluded_topics, source_scope])
    return len(theme.strip()) < 55 or (score >= 3 and helper_count == 0)

# ============================================================
# Pipelines
# ============================================================
def run_fast_pipeline(client: OpenAI, model: str, theme: str, uploaded_files, target_length: int, style_mode: str, abstraction_mode: str) -> Dict[str, Any]:
    cfg = FAST_SETTINGS
    chunks = extract_chunks(uploaded_files, cfg["chunk_char_min"], cfg["chunk_char_max"])
    if not chunks:
        raise ValueError("有効なテキストを抽出できませんでした。PDFが画像だけの可能性があります。")

    prefiltered = local_prefilter(chunks, theme, keep=cfg["local_keep"])
    evidences: List[Evidence] = []
    for chunk in prefiltered[: min(cfg["api_keep"], len(prefiltered))]:
        evidences.append(build_evidence_one_call(client, model, theme, chunk))

    evidences.sort(key=lambda x: (x.final_score, x.precise_score, x.specificity_score), reverse=True)
    evidences = cluster_duplicates(evidences, threshold=0.42)
    selected = select_representatives(evidences, limit=cfg["final_keep"])
    argument_map = build_argument_map(client, model, theme, selected, points=3)

    split_count = section_split_count(target_length, mode="高速")
    if split_count == 1:
        report = generate_single_pass_report(client, model, theme, target_length, argument_map, selected, style_mode, abstraction_mode, strict_source_only=False)
    else:
        blueprint = build_blueprint(client, model, theme, selected, argument_map, target_length, sections_count=split_count)
        lookup = evidence_lookup(selected)
        thesis = str(blueprint.get("thesis", ""))
        sections = blueprint.get("sections", [])
        parts: List[str] = []
        used_terms: List[str] = []
        previous = ""
        for i, section in enumerate(sections):
            bundle = section_evidence_bundle(section, lookup)
            part = generate_section_text(client, model, theme, thesis, section, bundle, previous, used_terms, style_mode, abstraction_mode, strict_source_only=False)
            part = complete_section_if_truncated(client, model, theme, str(section.get("name", f"section_{i+1}")), part, selected, strict_source_only=False)
            parts.append(part)
            previous = "\n\n".join(parts)
            used_terms.extend(lexical_terms(part, top_k=8))
        report = clean_text("\n\n".join(parts))
        report = regenerate_conclusion(client, model, theme, report, selected, style_mode)

    report = patch_missing_terms(client, model, theme, report, selected, min_terms=cfg["min_source_terms"], strict_source_only=False)
    report = enforce_length_requirements(client, model, theme, report, selected, target_length, strict_source_only=False)
    if target_length >= 6000:
        report = deduplicate_report_pass(client, model, theme, report, selected, target_length)
        report = enforce_length_requirements(client, model, theme, report, selected, target_length, strict_source_only=False)
    critique = critique_report(client, model, theme, report, selected)
    if critique.get("revision_needed", False) and int(critique.get("overall_score", 0)) < 84:
        report = rewrite_once(client, model, theme, report, critique, selected, style_mode, strict_source_only=False)
        report = enforce_length_requirements(client, model, theme, report, selected, target_length, strict_source_only=False)
        critique = critique_report(client, model, theme, report, selected)

    return {"mode": "高速", "chunks": chunks, "selected_evidences": selected, "argument_map": argument_map, "report": report, "critique": critique}


def run_high_pipeline(client: OpenAI, model: str, theme: str, uploaded_files, target_length: int, style_mode: str, abstraction_mode: str) -> Dict[str, Any]:
    cfg = HIGH_SETTINGS
    chunks = extract_chunks(uploaded_files, cfg["chunk_char_min"], cfg["chunk_char_max"])
    if not chunks:
        raise ValueError("有効なテキストを抽出できませんでした。PDFが画像だけの可能性があります。")

    prefiltered = local_prefilter(chunks, theme, keep=cfg["local_keep"])
    evidences: List[Evidence] = []
    for chunk in prefiltered[: min(cfg["api_keep"], len(prefiltered))]:
        evidences.append(build_evidence_one_call(client, model, theme, chunk))

    evidences.sort(key=lambda x: (x.final_score, x.precise_score, x.specificity_score), reverse=True)
    evidences = cluster_duplicates(evidences, threshold=0.40)
    selected = select_representatives(evidences, limit=cfg["final_keep"])
    argument_map = build_argument_map(client, model, theme, selected, points=5)
    blueprint = build_blueprint(client, model, theme, selected, argument_map, target_length, sections_count=section_split_count(target_length, mode="ハイエンド"))

    lookup = evidence_lookup(selected)
    thesis = str(blueprint.get("thesis", ""))
    sections = blueprint.get("sections", [])
    if not sections:
        raise ValueError("設計図の sections が空です")

    parts: List[str] = []
    used_terms: List[str] = []
    previous = ""
    for i, section in enumerate(sections):
        bundle = section_evidence_bundle(section, lookup)
        part = generate_section_text(client, model, theme, thesis, section, bundle, previous, used_terms, style_mode, abstraction_mode, strict_source_only=True)
        part = complete_section_if_truncated(client, model, theme, str(section.get("name", f"section_{i+1}")), part, selected, strict_source_only=True)
        parts.append(part)
        previous = "\n\n".join(parts)
        used_terms.extend(lexical_terms(part, top_k=8))

    report = clean_text("\n\n".join(parts))
    report = patch_missing_terms(client, model, theme, report, selected, min_terms=cfg["min_source_terms"], strict_source_only=True)
    report = enforce_length_requirements(client, model, theme, report, selected, target_length, strict_source_only=True)
    report = regenerate_conclusion(client, model, theme, report, selected, style_mode)
    critique1 = critique_report(client, model, theme, report, selected)
    if critique1.get("revision_needed", False) or int(critique1.get("overall_score", 0)) < 88 or critique1.get("external_example_risk", 0) >= 1:
        report = rewrite_once(client, model, theme, report, critique1, selected, style_mode, strict_source_only=True)
    critique2 = critique_report(client, model, theme, report, selected)
    report = humanize_if_needed(client, model, theme, report, critique2)
    if target_length >= 6000:
        report = deduplicate_report_pass(client, model, theme, report, selected, target_length)
    report = enforce_length_requirements(client, model, theme, report, selected, target_length, strict_source_only=True)
    critique_final = critique_report(client, model, theme, report, selected)

    return {"mode": "ハイエンド", "chunks": chunks, "selected_evidences": selected, "argument_map": argument_map, "blueprint": blueprint, "report": report, "critique": critique_final}

# ============================================================
# Session state / UI
# ============================================================
for key, default in {"result": None, "theme_snapshot": ""}.items():
    if key not in st.session_state:
        st.session_state[key] = default

left, right = st.columns([1.18, 0.82], gap="large")
with left:
    st.subheader("入力")
    uploaded_files = st.file_uploader("PDFを複数アップロード", type=["pdf"], accept_multiple_files=True)
    theme = st.text_area("課題文 / テーマ", placeholder="例：科学技術の発展がマーケティング・チャネル設計に与えた影響を、ディスインターメディエーションと電子市場を中心に論じなさい。", height=130)
    focus_points = st.text_input("特に重視したい観点（任意）", placeholder="例：チャネル設計、ディスインターメディエーション、チャネル・コンフリクト")
    preferred_concepts = st.text_input("扱いたい概念（任意）", placeholder="例：電子市場、垂直型システム、EDI、リテール・リンク")
    excluded_topics = st.text_input("除外したい話題（任意）", placeholder="例：ブランド価値、感情分析、新製品開発")
    source_scope = st.text_input("使う範囲・章指定（任意）", placeholder="例：7章中心、7マーケティング・チャネル p.11〜38 を優先")
    target_length = st.number_input("目標文字数", min_value=300, max_value=10000, value=3000, step=100, help="6000〜10000字はハイエンド推奨です。")

with right:
    st.subheader("設定")
    mode = st.radio("生成モード", ["高速", "ハイエンド"], index=0, horizontal=True)
    style_mode = st.selectbox("文体", ["標準", "やや硬め", "やや柔らかめ"], index=0)
    abstraction_mode = st.selectbox("抽象度", ["標準", "抽象度高め", "抽象度低め"], index=0)
    model_name = st.text_input("使用モデル", value=FAST_DEFAULT_MODEL if mode == "高速" else HIGH_DEFAULT_MODEL)
    st.info("高速はローカル粗選別＋軽い改稿、ハイエンドは証拠束つき分割生成＋厳しめ仕上げです。")

api_key = get_api_key()
confirm_generate = st.checkbox("この設定で本当に生成する", value=False)
fast_col, high_col = st.columns(2)
fast_clicked = fast_col.button("高速で生成", use_container_width=True)
high_clicked = high_col.button("ハイエンドで生成", use_container_width=True, type="primary")

if broad_theme_warning(theme, focus_points, preferred_concepts, excluded_topics, source_scope):
    st.info("この課題は範囲が広いため、出力が総花的になる可能性があります。補助欄を入れると精度が上がります。")
if target_length > 4000 and mode == "高速":
    st.warning("高速モードは長文に不向きです。4000字超、とくに6000〜10000字はハイエンドモードを推奨します。")


def validate_inputs(api_key_value: str, uploaded, theme_text: str, confirm_flag: bool) -> None:
    if not confirm_flag:
        st.warning("生成前にチェックを入れてください。")
        st.stop()
    if not api_key_value:
        st.error("APIキーを入力してください。")
        st.stop()
    if not uploaded:
        st.error("PDFを1つ以上アップロードしてください。")
        st.stop()
    if not theme_text.strip():
        st.error("課題文またはテーマを入力してください。")
        st.stop()


if fast_clicked or high_clicked:
    validate_inputs(api_key, uploaded_files, theme, confirm_generate)
    client = get_client(api_key)
    effective_theme = build_user_guidance(theme, focus_points, preferred_concepts, excluded_topics, source_scope)
    try:
        if fast_clicked:
            result = run_fast_pipeline(client, model_name, effective_theme, uploaded_files, int(target_length), style_mode, abstraction_mode)
        else:
            result = run_high_pipeline(client, model_name, effective_theme, uploaded_files, int(target_length), style_mode, abstraction_mode)
        result["effective_theme"] = effective_theme
        st.session_state.result = result
        st.session_state.theme_snapshot = theme
        st.success("生成が完了しました。")
    except Exception as e:
        st.exception(e)
        st.stop()

if st.session_state.result:
    result = st.session_state.result
    report = result["report"]
    selected = result["selected_evidences"]
    critique = result["critique"]
    tab1, tab2, tab3, tab4 = st.tabs(["最終レポート", "採用根拠", "論点設計", "内部評価"])
    with tab1:
        st.subheader("生成結果")
        st.text_area("レポート本文", report, height=520)
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("モード", result.get("mode", ""))
        m2.metric("最終文字数", len(report))
        m3.metric("採用根拠数", len(selected))
        m4.metric("品質スコア", int(critique.get("overall_score", 0)))
        st.download_button("本文をテキスト保存", data=report, file_name="reportflow_report.txt", mime="text/plain", use_container_width=True)
    with tab2:
        st.subheader("採用された根拠")
        for ev in selected:
            title = f"{int(ev.final_score)}点 | {ev.file} p.{ev.page} | {ev.reason}"
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
        st.subheader("論点設計")
        if result.get("effective_theme"):
            st.text_area("実際に使ったテーマ解釈", result.get("effective_theme", ""), height=140)
        st.text_area("argument_map", result.get("argument_map", ""), height=260)
        if result.get("blueprint"):
            blueprint = result.get("blueprint", {})
            st.write(f"中心主張: {blueprint.get('thesis', '')}")
            for sec in blueprint.get("sections", []):
                with st.expander(sec.get("name", "section")):
                    st.write(f"role: {sec.get('role', '')}")
                    st.write(f"objective: {sec.get('objective', '')}")
                    st.write(f"key_claim: {sec.get('key_claim', '')}")
                    st.write(f"target_chars: {sec.get('target_chars', '')}")
                    st.write(f"must_use_terms: {', '.join(sec.get('must_use_terms', []))}")
                    st.write(f"evidence_ids: {', '.join(sec.get('evidence_ids', []))}")
                    st.write(f"avoid_overlap_with: {', '.join(sec.get('avoid_overlap_with', []))}")
                    st.write(f"forbidden_patterns: {', '.join(sec.get('forbidden_patterns', []))}")
                    st.write(f"must_not_repeat: {', '.join(sec.get('must_not_repeat', []))}")
    with tab4:
        st.subheader("内部評価")
        st.json(critique)
        with st.expander("重要概念候補"):
            st.write(", ".join(important_terms_from_evidences(selected, top_k=16)))
