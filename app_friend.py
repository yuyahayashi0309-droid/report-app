import json
import html
import re
import sqlite3
import textwrap
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import date
from typing import Any, Dict, List, Tuple

import fitz
import streamlit as st
from openai import OpenAI

# ============================================================
# 00. 友人テスト版の基本方針
# - 高速版のみ
# - APIキー入力欄なし（st.secretsのみ使用）
# - 共通パスワードあり
# - 1日あたり利用回数制限あり
# - PDF数 / ファイルサイズ / 文字数の上限あり
# - 論点重複を抑えるため、段落ごと生成 + 意味重複整理を採用
# ============================================================

st.set_page_config(page_title="ReportFlow Friend Test", page_icon="📝", layout="wide")

st.markdown(
    """
    <style>
    .block-container { max-width: 1180px; padding-top: 1.1rem; padding-bottom: 4rem; }
    .hero {
        padding: 1.15rem 1.25rem;
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
    .stage-box {
        border: 1px solid rgba(120,120,120,0.14);
        border-radius: 18px;
        padding: 0.8rem 0.9rem;
        margin-top: 0.6rem;
        background: rgba(255,255,255,0.02);
    }
    .stage-title {
        font-weight: 700;
        margin-bottom: 0.25rem;
    }
    .stage-sub {
        opacity: 0.8;
        font-size: 0.94rem;
    }
    .small-muted {
        opacity: 0.75;
        font-size: 0.9rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero">
        <h1 style="margin-bottom:0.18rem;">📝 ReportFlow Friend Test</h1>
        <div style="opacity:0.82;">
            講義PDFを読み込み、課題文に合わせて資料固有の概念を活かしたレポート本文を生成します。
            この版は友人テスト用のため、高速版のみ・利用回数制限ありです。
        </div>
        <div style="margin-top:0.55rem;">
            <span class="pill">高速版のみ</span>
            <span class="pill">資料重視</span>
            <span class="pill">字数補正</span>
            <span class="pill">未完文対策</span>
            <span class="pill">根拠可視化</span>
            <span class="pill">重複抑制</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# 01. secrets / 制限値
# ============================================================
def secret_str(key: str, default: str = "") -> str:
    try:
        return str(st.secrets[key])
    except Exception:
        return default


def secret_int(key: str, default: int) -> int:
    try:
        return int(st.secrets[key])
    except Exception:
        return default


OPENAI_API_KEY = secret_str("OPENAI_API_KEY")
APP_PASSWORD = secret_str("APP_PASSWORD", "friend-test")
MAX_DAILY_RUNS = secret_int("MAX_DAILY_RUNS", 3)
MAX_PDFS = secret_int("MAX_PDFS", 3)
MAX_FILE_MB = secret_int("MAX_FILE_MB", 10)
MAX_TARGET_CHARS = secret_int("MAX_TARGET_CHARS", 3000)

# ============================================================
# 02. アクセス制限
# ============================================================
st.sidebar.subheader("テスト利用")
gate = st.sidebar.text_input("テスト用パスワード", type="password")
if gate != APP_PASSWORD:
    st.info("パスワードを入力すると利用できます。")
    st.stop()

if not OPENAI_API_KEY:
    st.error("サーバー側のAPIキー設定が見つかりません。")
    st.stop()

# ============================================================
# 03. 使用量DB
# ============================================================
def init_usage_db() -> None:
    conn = sqlite3.connect("usage.db")
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS usage_log (
            user_id TEXT,
            day TEXT,
            count INTEGER,
            PRIMARY KEY (user_id, day)
        )
        """
    )
    conn.commit()
    conn.close()


def get_usage_count(user_id: str) -> int:
    today = str(date.today())
    conn = sqlite3.connect("usage.db")
    cur = conn.cursor()
    cur.execute("SELECT count FROM usage_log WHERE user_id=? AND day=?", (user_id, today))
    row = cur.fetchone()
    conn.close()
    return int(row[0]) if row else 0


def check_and_increment_usage(user_id: str, daily_limit: int) -> Tuple[bool, int]:
    today = str(date.today())
    conn = sqlite3.connect("usage.db")
    cur = conn.cursor()
    cur.execute("SELECT count FROM usage_log WHERE user_id=? AND day=?", (user_id, today))
    row = cur.fetchone()

    if row is None:
        cur.execute(
            "INSERT INTO usage_log (user_id, day, count) VALUES (?, ?, ?)",
            (user_id, today, 1),
        )
        conn.commit()
        conn.close()
        return True, 1

    current = int(row[0])
    if current >= daily_limit:
        conn.close()
        return False, current

    cur.execute(
        "UPDATE usage_log SET count=? WHERE user_id=? AND day=?",
        (current + 1, user_id, today),
    )
    conn.commit()
    conn.close()
    return True, current + 1


init_usage_db()

# ============================================================
# 04. 定数 / 設定
# ============================================================
MODEL_NAME = "gpt-4.1-mini"

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

FLOW_STEPS = [
    "PDF抽出",
    "候補選定",
    "根拠精査",
    "論点設計",
    "本文生成",
    "字数補正",
    "重複整理",
    "完成",
]

# ============================================================
# 05. データ構造
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
# 06. 進捗UI
# ============================================================
class ProgressUI:
    def __init__(self):
        self.box = st.container()
        self.progress_bar = self.box.progress(0.0)
        self.status = self.box.empty()
        self.stage = self.box.empty()
        self.detail = self.box.empty()
        self.metric_slot = self.box.empty()
        self.notes = self.box.empty()
        self.current_step_index = 0
        self.render_stage("待機中", "入力を確認しています。", 0.0)

    def render_stage(self, title: str, subtitle: str, progress_value: float, details: str = "") -> None:
        value = max(0.0, min(1.0, progress_value))
        self.progress_bar.progress(value)
        self.status.markdown("**高速モード実行中**")
        self.stage.markdown(
            f"""
            <div class="stage-box">
                <div class="stage-title">{html.escape(title)}</div>
                <div class="stage-sub">{html.escape(subtitle)}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if details:
            self.detail.caption(details)
        else:
            self.detail.empty()

    def set_flow_step(self, step_name: str, subtitle: str = "", details: str = "") -> None:
        try:
            idx = FLOW_STEPS.index(step_name)
        except ValueError:
            idx = self.current_step_index
        self.current_step_index = idx
        base_progress = idx / max(len(FLOW_STEPS), 1)
        self.render_stage(step_name, subtitle or f"{step_name}を実行しています。", base_progress, details)

    def subprogress(self, done: int, total: int, label: str) -> None:
        total = max(total, 1)
        local = done / total
        base = self.current_step_index / len(FLOW_STEPS)
        next_base = (self.current_step_index + 1) / len(FLOW_STEPS)
        overall = base + (next_base - base) * local
        step_name = FLOW_STEPS[self.current_step_index] if self.current_step_index < len(FLOW_STEPS) else "進行中"
        self.render_stage(step_name, label, overall)

    def metrics(self, pairs: List[Tuple[str, Any]]) -> None:
        if not pairs:
            self.metric_slot.empty()
            return
        cols = self.metric_slot.columns(len(pairs))
        for col, (label, value) in zip(cols, pairs):
            col.metric(label, value)

    def note(self, text: str) -> None:
        self.notes.info(text)

    def finish(self, text: str = "生成が完了しました。") -> None:
        self.render_stage("完成", text, 1.0)
        self.notes.success(text)

# ============================================================
# 07. OpenAI API ヘルパー
# ============================================================
def get_client() -> OpenAI:
    return OpenAI(api_key=OPENAI_API_KEY)


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
    return (response.output_text or "").strip()

# ============================================================
# 08. 文字列 / スコアリング補助
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
    return {
        "min": int(target_length * low_ratio),
        "ideal": target_length,
        "max": int(target_length * high_ratio),
    }


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
    allowed_text = " ".join(
        ev.topic + " " + ev.proposition + " " + ev.evidence + " " + ev.example + " " + " ".join(ev.terminology)
        for ev in evidences
    )
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
    stripped = (text or "").strip()
    if not stripped:
        return False

    bad_endings = (
        "は", "が", "を", "に", "で", "と", "へ", "も", "の",
        "、", "・", "（", "(", "「", "『",
        "より", "や", "し", "ため", "について", "として",
        "する", "した", "して", "いる", "なり", "可能", "必要", "柔軟"
    )
    if stripped.endswith(bad_endings):
        return True

    if not stripped.endswith(("。", "！", "？", ".", "!", "?", "」", "』", "）", ")")):
        return True

    pairs = [("(", ")"), ("（", "）"), ("「", "」"), ("『", "』")]
    for left, right in pairs:
        if stripped.count(left) > stripped.count(right):
            return True

    parts = re.split(r"[。.!?！？]", stripped)
    last_clause = parts[-2].strip() if len(parts) >= 2 else stripped
    if len(last_clause) <= 4:
        return True

    return False


def force_close_text(text: str) -> str:
    t = (text or "").rstrip()
    if not t:
        return t

    if t.endswith(("。", "！", "？", ".", "!", "?", "」", "』", "）", ")")):
        return t

    replacements = {
        "柔軟": "柔軟な対応が求められる。",
        "必要": "必要である。",
        "可能": "可能である。",
        "重要": "重要である。",
        "有効": "有効である。",
        "適切": "適切である。",
        "必要が": "必要がある。",
        "ことが": "ことが求められる。",
        "ため": "ためである。",
        "する": "する必要がある。",
        "して": "していく必要がある。",
        "いる": "いる。",
    }
    for k, v in replacements.items():
        if t.endswith(k):
            return t[:-len(k)] + v

    if t.endswith(("は", "が", "を", "に", "で", "と", "も", "の", "へ")):
        return t + "ついて検討する必要がある。"

    return t + "。"


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

# ============================================================
# 09. PDF抽出
# ============================================================
def block_texts_from_page(page) -> List[Tuple[int, str]]:
    blocks = page.get_text("blocks")
    out: List[Tuple[int, str]] = []
    for idx, block in enumerate(blocks):
        if len(block) < 5:
            continue
        text = normalize_space(block[4])
        if not text or len(text) <= 2:
            continue
        out.append((idx, text))
    return out


def merge_blocks_semantically(blocks: List[Tuple[int, str]], min_chars: int, max_chars: int) -> List[Tuple[str, str]]:
    chunks: List[Tuple[str, str]] = []
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
# 10. 候補選定 / 根拠化
# ============================================================
def local_prefilter(chunks: List[Chunk], theme: str, keep: int) -> List[Chunk]:
    theme_terms = lexical_terms(theme, top_k=12)
    scored: List[Tuple[float, Chunk]] = []
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
    data = call_json(
        client,
        model,
        "資料断片を中間表現へ変換しつつ採点する。資料にないことを足さない。JSONのみ返す。",
        prompt,
        temperature=0.1,
        max_output_tokens=1400,
    )
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
        reps.append(sorted(items, key=lambda x: (x.final_score, x.precise_score, x.specificity_score), reverse=True)[0])
    return sorted(reps, key=lambda x: (x.final_score, x.precise_score, x.usefulness_score), reverse=True)[:limit]

# ============================================================
# 11. 要約・設計・本文生成
# ============================================================
def build_argument_map(
    client: OpenAI,
    model: str,
    theme: str,
    evidences: List[Evidence],
    points: int = 4,
) -> Dict[str, Any]:
    source = join_evidence_briefs(evidences, 10)
    prompt = f"""
以下は課題文と、採用候補の資料根拠です。
この課題に答えるための論点設計をJSONで作成してください。
JSONのみ返してください。

必須キー:
- thesis: レポート全体の中心主張 1文
- paragraphs: 配列（{points}個前後）
  - paragraph_id: "P1" 形式
  - role: [intro, structure_change, intermediary_role, efficiency_mechanism, conflict_adjustment, conclusion]
  - main_claim: その段落で述べる中心主張 1文
  - evidence_ids: 使用するEvidenceのchunk_id配列（1〜3個）
  - must_terms: その段落で必ず使う概念 2〜4個
  - forbidden_topics: この段落では主役として再説明してはいけない内容 1〜3個
  - relation_to_prev: 前段落との接続 1文
- notes:
  - avoid_repetition: 重複回避の要点を2〜4個
  - conclusion_focus: 結論で強調すべき点 1文

条件:
- 似た論点は統合する
- 同じ具体例を複数段落の主役にしない
- 各段落は固有の役割を持たせる
- 一般論より資料根拠を優先する
- 資料固有概念を優先する
- 「字数稼ぎの言い換え」を避ける設計にする

課題文:
{theme}

資料根拠:
{source}
"""
    data = call_json(
        client,
        model,
        "講義資料の根拠を重複の少ない段落設計JSONへ変換する。JSONのみ返す。",
        prompt,
        temperature=0.15,
        max_output_tokens=2400,
    )

    if not isinstance(data, dict):
        raise ValueError("論点設計JSONの生成に失敗しました。")

    paragraphs = data.get("paragraphs", [])
    if not isinstance(paragraphs, list) or not paragraphs:
        raise ValueError("paragraphs が取得できませんでした。")

    return data


def evidence_map_by_id(evidences: List[Evidence]) -> Dict[str, Evidence]:
    return {ev.chunk_id: ev for ev in evidences}


def render_evidence_min(ev: Evidence) -> str:
    return textwrap.dedent(
        f"""
        id: {ev.chunk_id}
        topic: {ev.topic}
        proposition: {ev.proposition}
        evidence: {ev.evidence}
        example: {ev.example}
        terminology: {', '.join(ev.terminology)}
        contrast: {ev.contrast}
        cause_effect: {ev.cause_effect}
        """
    ).strip()


def gather_outline_evidence_text(outline_item: Dict[str, Any], ev_map: Dict[str, Evidence]) -> str:
    ids = outline_item.get("evidence_ids", []) or []
    picked = []
    for eid in ids:
        if eid in ev_map:
            picked.append(render_evidence_min(ev_map[eid]))
    return "\n\n".join(picked[:3])


def normalize_outline_item(item: Dict[str, Any], idx: int) -> Dict[str, Any]:
    return {
        "paragraph_id": str(item.get("paragraph_id", f"P{idx+1}")),
        "role": str(item.get("role", "body")),
        "main_claim": str(item.get("main_claim", "")).strip(),
        "evidence_ids": safe_json_list(item.get("evidence_ids", [])),
        "must_terms": safe_json_list(item.get("must_terms", [])),
        "forbidden_topics": safe_json_list(item.get("forbidden_topics", [])),
        "relation_to_prev": str(item.get("relation_to_prev", "")).strip(),
    }


def generate_paragraph(
    client: OpenAI,
    model: str,
    theme: str,
    outline_item: Dict[str, Any],
    ev_map: Dict[str, Evidence],
    thesis: str,
    used_claims: List[str],
    used_examples: List[str],
    style_mode: str,
    abstraction_mode: str,
    paragraph_target: int,
) -> str:
    style_instruction = {
        "標準": "自然で読みやすい文体にする。",
        "やや硬め": "少しフォーマルで簡潔な文体にする。",
        "やや柔らかめ": "少し柔らかいが幼くしない文体にする。",
    }[style_mode]
    abstraction_instruction = {
        "標準": "抽象と具体のバランスを標準にする。",
        "抽象度高め": "やや抽象化して上位概念で整理する。",
        "抽象度低め": "やや具体的にし、資料上の概念や事例を厚めに使う。",
    }[abstraction_mode]

    evidence_text = gather_outline_evidence_text(outline_item, ev_map)
    must_terms = outline_item.get("must_terms", [])
    forbidden_topics = outline_item.get("forbidden_topics", [])

    prompt = f"""
以下の条件で、レポート本文の1段落だけを書いてください。
本文1段落のみ出力してください。

課題文:
{theme}

レポート全体の中心主張:
{thesis}

この段落の役割:
{outline_item.get("role", "")}

この段落の中心主張:
{outline_item.get("main_claim", "")}

前段落との接続:
{outline_item.get("relation_to_prev", "")}

この段落で使う根拠:
{evidence_text}

必ず使う概念:
{', '.join(must_terms)}

既に説明済みの主張:
- {' / '.join(used_claims) if used_claims else 'なし'}

既に使用済みの事例・固有例:
- {' / '.join(used_examples) if used_examples else 'なし'}

この段落で再説明してはいけない内容:
- {' / '.join(forbidden_topics) if forbidden_topics else 'なし'}

目安字数:
約{paragraph_target}字

条件:
- 本文1段落のみ
- 見出し禁止
- 箇条書き禁止
- 同じ定義や事例の再説明禁止
- 既出主張の言い換えで字数を稼がない
- 不足分は因果、比較、意義の明確化で補う
- 資料にない新事例を足さない
- {style_instruction}
- {abstraction_instruction}
"""
    return clean_text(
        call_text(
            client,
            model,
            "各段落に固有の役割を持たせ、既出内容を繰り返さずに本文を作る。本文のみ返す。",
            prompt,
            temperature=0.25,
            max_output_tokens=1400,
        )
    )


def generate_single_pass_report(
    client: OpenAI,
    model: str,
    theme: str,
    target_length: int,
    argument_map: Dict[str, Any],
    evidences: List[Evidence],
    style_mode: str,
    abstraction_mode: str,
) -> str:
    thesis = str(argument_map.get("thesis", "")).strip()
    raw_paragraphs = argument_map.get("paragraphs", []) or []
    paragraphs = [normalize_outline_item(p, i) for i, p in enumerate(raw_paragraphs)]
    ev_map = evidence_map_by_id(evidences)

    if not paragraphs:
        raise ValueError("段落設計が空です。")

    # 結論・接続で少し削られるので、初稿はやや厚めに作る
    usable_target = max(1400, int(target_length * 1.03))
    paragraph_target = max(320, int(usable_target / max(len(paragraphs), 1)) - 20)

    used_claims: List[str] = []
    used_examples: List[str] = []
    built: List[str] = []

    for item in paragraphs:
        para = generate_paragraph(
            client=client,
            model=model,
            theme=theme,
            outline_item=item,
            ev_map=ev_map,
            thesis=thesis,
            used_claims=used_claims,
            used_examples=used_examples,
            style_mode=style_mode,
            abstraction_mode=abstraction_mode,
            paragraph_target=paragraph_target,
        )
        if para:
            built.append(para)

        main_claim = item.get("main_claim", "").strip()
        if main_claim:
            used_claims.append(main_claim)

        for eid in item.get("evidence_ids", []):
            ev = ev_map.get(eid)
            if not ev:
                continue
            if ev.example and ev.example != "なし":
                used_examples.append(ev.example)
            elif ev.topic:
                used_examples.append(ev.topic)

    report = "\n\n".join(p for p in built if p.strip())
    return clean_text(report)

# ============================================================
# 12. 仕上げ / 批評 / リライト
# ============================================================
def patch_missing_terms(
    client: OpenAI,
    model: str,
    theme: str,
    report: str,
    evidences: List[Evidence],
    min_terms: int = 3,
) -> str:
    terms = important_terms_from_evidences(evidences, top_k=12)
    present = [t for t in terms if t in report]
    if len(present) >= min_terms:
        return report

    missing = [t for t in terms if t not in report][: max(2, min_terms - len(present) + 2)]
    evidence_text = join_evidence_briefs(evidences, 5)
    prompt = f"""
以下の本文に、資料由来の概念がやや不足しています。
本文全体を自然に整えつつ、欠けている概念を無理なく組み込んでください。
本文のみ出力してください。

課題文:
{theme}

参考根拠:
{evidence_text}

できれば入れたい概念:
{', '.join(missing)}

現在の本文:
{report}

条件:
- 本文のみ
- 新しい外部事例を追加しない
- 既存の論旨を壊さない
- 概念を機械的に列挙しない
"""
    rewritten = clean_text(
        call_text(
            client,
            model,
            "資料由来の概念を自然に補う。本文のみ返す。",
            prompt,
            temperature=0.18,
            max_output_tokens=4200,
        )
    )
    return rewritten if rewritten else report


def dedupe_report_semantically(
    client: OpenAI,
    model: str,
    theme: str,
    report: str,
    evidences: List[Evidence],
) -> str:
    evidence_text = join_evidence_briefs(evidences, 6)
    prompt = f"""
以下の本文について、意味が重複している文・段落を統合し、
情報量を落とさずに自然な本文へ整えてください。
本文のみ出力してください。

課題文:
{theme}

参考根拠:
{evidence_text}

本文:
{report}

条件:
- 本文のみ
- 新情報の追加禁止
- 同じ主張の言い換え反復を削除する
- 同じ事例の再説明を削除する
- 削った分は因果関係・比較・意義の明確化に充てる
- 段落ごとの役割が重ならないようにする
- 資料固有語はなるべく残す
"""
    rewritten = clean_text(
        call_text(
            client,
            model,
            "意味重複を削除し、段落の役割分担を明確化する。本文のみ返す。",
            prompt,
            temperature=0.15,
            max_output_tokens=4200,
        )
    )
    return rewritten if rewritten else report


def compress_report_if_too_long(client: OpenAI, model: str, report: str, target_length: int) -> str:
    if length_band_status(report, target_length, strict=False) != "long":
        return report
    targets = build_length_targets(target_length, strict=False)
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
    rewritten = clean_text(
        call_text(
            client,
            model,
            "文字数と重複を同時に自然に圧縮する。本文のみ返す。",
            prompt,
            temperature=0.2,
            max_output_tokens=4200,
        )
    )
    return rewritten if rewritten else report


def append_report_if_too_short(
    client: OpenAI,
    model: str,
    theme: str,
    report: str,
    evidences: List[Evidence],
    target_length: int,
) -> str:
    status = length_band_status(report, target_length, strict=False)
    if status != "short":
        return report

    targets = build_length_targets(target_length, strict=False)
    shortage = max(0, targets["min"] - len(report))
    if shortage <= 0:
        return report

    evidence_text = join_evidence_briefs(evidences, 6)
    must_terms = important_terms_from_evidences(evidences, top_k=10)

    prompt = f"""
以下の本文は字数が不足しています。
ただし追加段落の継ぎ足しではなく、本文全体を再構成して
約{targets['ideal']}字に近づけてください。
本文のみ出力してください。

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

参考根拠:
{evidence_text}

できれば残したい概念:
{', '.join(must_terms)}

現在の本文:
{report}

条件:
- 本文のみ
- 既出論点の再説明禁止
- 同じ事例の再利用禁止
- 不足分は因果関係、比較、含意、段落間の接続の明確化で補う
- 新しい外部事例を追加しない
- 資料固有語をできるだけ残す
- 同じ主張の言い換えで字数を稼がない
"""
    rewritten = clean_text(
        call_text(
            client,
            model,
            "字数不足時は追記ではなく全文再構成で補う。本文のみ返す。",
            prompt,
            temperature=0.2,
            max_output_tokens=4200,
        )
    )
    return rewritten if rewritten else report


def enforce_length_strictly(
    client: OpenAI,
    model: str,
    theme: str,
    report: str,
    evidences: List[Evidence],
    target_length: int,
    max_rounds: int = 3,
) -> str:
    text = clean_text(report)

    for _ in range(max_rounds):
        status = length_band_status(text, target_length, strict=True)

        if status == "ok":
            return text

        if status == "short":
            text = append_report_if_too_short(
                client=client,
                model=model,
                theme=theme,
                report=text,
                evidences=evidences,
                target_length=target_length,
            )
            text = clean_text(text)
            continue

        if status == "long":
            text = compress_report_if_too_long(
                client=client,
                model=model,
                report=text,
                target_length=target_length,
            )
            text = clean_text(text)
            continue

    return text


def ensure_complete_text(
    client: OpenAI,
    model: str,
    theme: str,
    report: str,
    evidences: List[Evidence],
) -> str:
    text = clean_text(report)
    if not is_truncated_text(text):
        return text
    evidence_text = join_evidence_briefs(evidences, 5)
    for _ in range(2):
        prompt = f"""
以下のレポート本文は末尾が途中で切れている可能性があります。
最後の流れを壊さず、自然に完結させてください。
補完部分のみ出力してください。

課題文:
{theme}

参考根拠:
{evidence_text}

現在の本文末尾:
{text[-4500:]}

条件:
- 補完部分のみ
- 新しい論点を追加しない
- 既存の結論や最後の議論を自然に閉じる
- 150〜450字程度
"""
        addition = clean_text(
            call_text(
                client,
                model,
                "途中で切れたレポート末尾を自然に完結させる。補完部分のみ返す。",
                prompt,
                temperature=0.2,
                max_output_tokens=1000,
            )
        )
        if addition:
            text = clean_text(text + "\n\n" + addition)
        if not is_truncated_text(text):
            return text
    return force_close_text(text)


def critique_report(
    client: OpenAI,
    model: str,
    theme: str,
    report: str,
    evidences: List[Evidence],
) -> Dict[str, Any]:
    evidence_text = join_evidence_briefs(evidences, 6)
    prompt = f"""
以下のレポート本文を講義レポートとして批評し、JSONのみで返してください。

必須キー:
- overall_score: 0-100
- revision_needed: true/false
- strengths: 配列 2〜4個
- weaknesses: 配列 2〜5個
- missing_terms: 配列 0〜5個
- repetition_risk: 0-3
- external_example_risk: 0-3
- comment: 1〜3文

課題文:
{theme}

参考根拠:
{evidence_text}

本文:
{report}
"""
    try:
        return call_json(
            client,
            model,
            "講義レポートの品質を採点し、JSONのみ返す。",
            prompt,
            temperature=0.1,
            max_output_tokens=1600,
        )
    except Exception:
        return {
            "overall_score": 80,
            "revision_needed": False,
            "strengths": ["資料概念が一定数入っている"],
            "weaknesses": [],
            "missing_terms": [],
            "repetition_risk": 1,
            "external_example_risk": detect_external_example_risk(report, evidences),
            "comment": "自動批評に失敗したため簡易評価を返しました。",
        }


def rewrite_once(
    client: OpenAI,
    model: str,
    theme: str,
    report: str,
    critique: Dict[str, Any],
    evidences: List[Evidence],
    style_mode: str,
) -> str:
    style_instruction = {
        "標準": "自然で読みやすい文体にする。",
        "やや硬め": "少しフォーマルで簡潔な文体にする。",
        "やや柔らかめ": "少し柔らかいが幼くしない文体にする。",
    }[style_mode]
    evidence_text = join_evidence_briefs(evidences, 6)
    prompt = f"""
以下のレポート本文を、批評を踏まえて1回だけ改善してください。
本文のみ出力してください。

課題文:
{theme}

参考根拠:
{evidence_text}

現在の本文:
{report}

批評:
{json.dumps(critique, ensure_ascii=False, indent=2)}

条件:
- 本文のみ
- 新しい外部事例を追加しない
- 同じ主張の言い換え反復を減らす
- 資料固有概念は維持する
- {style_instruction}
"""
    rewritten = clean_text(
        call_text(
            client,
            model,
            "批評を踏まえて本文を一度だけ改善する。本文のみ返す。",
            prompt,
            temperature=0.22,
            max_output_tokens=4200,
        )
    )
    return rewritten if rewritten else report


def finalize_report(
    client: OpenAI,
    model: str,
    theme: str,
    report: str,
    evidences: List[Evidence],
    target_length: int,
    style_mode: str,
) -> Tuple[str, Dict[str, Any]]:
    report = patch_missing_terms(client, model, theme, report, evidences, min_terms=FAST_SETTINGS["min_source_terms"])
    report = dedupe_report_semantically(client, model, theme, report, evidences)
    report = append_report_if_too_short(client, model, theme, report, evidences, target_length)
    report = dedupe_report_semantically(client, model, theme, report, evidences)
    report = compress_report_if_too_long(client, model, report, target_length)
    report = ensure_complete_text(client, model, theme, report, evidences)

    critique = critique_report(client, model, theme, report, evidences)
    if critique.get("revision_needed", False) and int(critique.get("overall_score", 0)) < 84:
        report = rewrite_once(client, model, theme, report, critique, evidences, style_mode)
        report = dedupe_report_semantically(client, model, theme, report, evidences)
        report = ensure_complete_text(client, model, theme, report, evidences)
        critique = critique_report(client, model, theme, report, evidences)

    return report, critique

# ============================================================
# 13. 実行パイプライン
# ============================================================
def run_fast_pipeline(
    client: OpenAI,
    model: str,
    theme: str,
    uploaded_files,
    target_length: int,
    style_mode: str,
    abstraction_mode: str,
    progress: ProgressUI,
) -> Dict[str, Any]:
    cfg = FAST_SETTINGS

    progress.set_flow_step("PDF抽出", "PDFから意味チャンクを取り出しています。")
    chunks = extract_chunks(uploaded_files, cfg["chunk_char_min"], cfg["chunk_char_max"])
    if not chunks:
        raise ValueError("有効なテキストを抽出できませんでした。PDFが画像だけの可能性があります。")
    progress.metrics([("抽出チャンク", len(chunks)), ("目標文字数", target_length)])

    progress.set_flow_step("候補選定", "課題文との関連が高い候補を絞っています。")
    prefiltered = local_prefilter(chunks, theme, keep=cfg["local_keep"])
    progress.metrics([("抽出チャンク", len(chunks)), ("ローカル候補", len(prefiltered)), ("API精査上限", cfg["api_keep"])])

    progress.set_flow_step("根拠精査", "上位候補だけをAPIで精査しています。")
    evidences: List[Evidence] = []
    total = min(cfg["api_keep"], len(prefiltered))
    for i, chunk in enumerate(prefiltered[:total], start=1):
        evidences.append(build_evidence_one_call(client, model, theme, chunk))
        progress.subprogress(i, total, f"根拠精査 {i}/{total}")

    evidences.sort(key=lambda x: (x.final_score, x.precise_score, x.specificity_score), reverse=True)
    evidences = cluster_duplicates(evidences, threshold=0.42)
    selected = select_representatives(evidences, limit=cfg["final_keep"])
    progress.metrics([("精査件数", len(evidences)), ("採用根拠", len(selected)), ("高速最終候補", cfg["final_keep"])])

    progress.set_flow_step("論点設計", "重複しにくい段落設計を作っています。")
    argument_map = build_argument_map(client, model, theme, selected, points=4)

    progress.set_flow_step("本文生成", "段落ごとに本文を生成しています。")
    draft = generate_single_pass_report(
        client=client,
        model=model,
        theme=theme,
        target_length=target_length,
        argument_map=argument_map,
        evidences=selected,
        style_mode=style_mode,
        abstraction_mode=abstraction_mode,
    )

    progress.set_flow_step("字数補正", "字数と資料概念の不足を整えています。")
    report, critique = finalize_report(
        client=client,
        model=model,
        theme=theme,
        report=draft,
        evidences=selected,
        target_length=target_length,
        style_mode=style_mode,
    )

    strict_bounds = build_length_targets(target_length, strict=True)
    final_len = len(report)
    if final_len < strict_bounds["min"]:
        raise ValueError(
            f"指定字数を満たせませんでした。現在 {final_len} 字、必要最低 {strict_bounds['min']} 字です。"
        )

    progress.set_flow_step("重複整理", "意味重複を最終確認しています。")
    report = dedupe_report_semantically(client, model, theme, report, selected)
    report = ensure_complete_text(client, model, theme, report, selected)

    progress.finish("生成が完了しました。")
    return {
        "report": report,
        "draft": draft,
        "argument_map": argument_map,
        "selected_evidences": selected,
        "critique": critique,
        "char_count": len(report),
    }

# ============================================================
# 14. UI
# ============================================================
def get_user_id() -> str:
    if "rf_user_id" not in st.session_state:
        st.session_state["rf_user_id"] = "friend"
    return st.session_state["rf_user_id"]


st.sidebar.markdown("---")
st.sidebar.caption(f"1日あたり上限: {MAX_DAILY_RUNS} 回")
used_today = get_usage_count(get_user_id())
st.sidebar.caption(f"本日の使用回数: {used_today} / {MAX_DAILY_RUNS}")

with st.form("report_form"):
    theme = st.text_area(
        "課題文",
        height=160,
        placeholder="例：科学技術の発展がマーケティングチャネルの構造と機能にどのような変化をもたらしたか、講義資料に基づいて論じなさい。",
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        target_length = st.number_input(
            "目標文字数",
            min_value=400,
            max_value=MAX_TARGET_CHARS,
            value=min(2000, MAX_TARGET_CHARS),
            step=100,
        )
    with col2:
        style_mode = st.selectbox("文体", ["標準", "やや硬め", "やや柔らかめ"], index=0)
    with col3:
        abstraction_mode = st.selectbox("抽象度", ["標準", "抽象度高め", "抽象度低め"], index=0)

    uploaded_files = st.file_uploader(
        f"講義PDF（最大 {MAX_PDFS} 個 / 各 {MAX_FILE_MB}MB まで）",
        type=["pdf"],
        accept_multiple_files=True,
    )

    submitted = st.form_submit_button("生成する", use_container_width=True)

st.markdown('<div class="small-muted">資料固有の概念を優先しつつ、重複を抑えた本文を目指します。</div>', unsafe_allow_html=True)

if submitted:
    if not theme.strip():
        st.error("課題文を入力してください。")
        st.stop()

    if not uploaded_files:
        st.error("PDFを1つ以上アップロードしてください。")
        st.stop()

    if len(uploaded_files) > MAX_PDFS:
        st.error(f"PDFは最大 {MAX_PDFS} 個までです。")
        st.stop()

    for f in uploaded_files:
        size_mb = len(f.getvalue()) / (1024 * 1024)
        if size_mb > MAX_FILE_MB:
            st.error(f"{f.name} は {MAX_FILE_MB}MB を超えています。")
            st.stop()

    ok, new_count = check_and_increment_usage(get_user_id(), MAX_DAILY_RUNS)
    if not ok:
        st.error(f"本日の利用上限（{MAX_DAILY_RUNS}回）に達しました。")
        st.stop()

    progress = ProgressUI()
    try:
        client = get_client()
        result = run_fast_pipeline(
            client=client,
            model=MODEL_NAME,
            theme=theme.strip(),
            uploaded_files=uploaded_files,
            target_length=int(target_length),
            style_mode=style_mode,
            abstraction_mode=abstraction_mode,
            progress=progress,
        )

        st.markdown("---")
        st.subheader("生成結果")
        st.caption(f"文字数: {result['char_count']} 字")
        st.text_area("レポート本文", value=result["report"], height=520)

        with st.expander("論点設計JSON"):
            st.json(result["argument_map"])

        with st.expander("採用根拠"):
            for ev in result["selected_evidences"]:
                st.markdown(f"**{ev.file} p.{ev.page} / {ev.chunk_id}**")
                st.write(f"- topic: {ev.topic}")
                st.write(f"- proposition: {ev.proposition}")
                st.write(f"- evidence: {ev.evidence}")
                st.write(f"- example: {ev.example}")
                st.write(f"- terminology: {', '.join(ev.terminology)}")
                st.write(f"- final_score: {ev.final_score}")
                st.markdown("---")

        with st.expander("批評"):
            st.json(result["critique"])

        st.success(f"実行回数を記録しました（本日 {new_count}/{MAX_DAILY_RUNS} 回）。")

    except Exception as e:
        st.error(f"エラーが発生しました: {e}")
