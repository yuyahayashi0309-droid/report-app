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
            <div class=\"stage-box\">
                <div class=\"stage-title\">{html.escape(title)}</div>
                <div class=\"stage-sub\">{html.escape(subtitle)}</div>
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
    return response.output_text.strip()

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
def build_argument_map(client: OpenAI, model: str, theme: str, evidences: List[Evidence], points: int = 3) -> str:
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
    return call_text(client, model, "講義資料から論点設計を行う。プレーンテキストのみ返す。", prompt, temperature=0.2, max_output_tokens=1500)


def generate_single_pass_report(
    client: OpenAI,
    model: str,
    theme: str,
    target_length: int,
    argument_map: str,
    evidences: List[Evidence],
    style_mode: str,
    abstraction_mode: str,
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
    source = join_evidence_briefs(evidences, 8)
    must_terms = important_terms_from_evidences(evidences, top_k=8)
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
- 資料に明示されていない企業名・ブランド名・事例は原則出さない
- 最後は短く自分の考察で締める
- {style_instruction}
- {abstraction_instruction}
"""
    return clean_text(
        call_text(
            client,
            model,
            "資料密着型の大学レポート本文を書く。本文のみ返す。",
            prompt,
            temperature=0.42,
            max_output_tokens=5200,
        )
    )

# ============================================================
# 12. 仕上げ
# ============================================================
def patch_missing_terms(
    client: OpenAI,
    model: str,
    theme: str,
    text: str,
    evidences: List[Evidence],
    min_terms: int,
) -> str:
    important = important_terms_from_evidences(evidences, top_k=12)
    missing = [t for t in important if t not in text][:min_terms]
    if not missing:
        return text
    ref_blocks = []
    for ev in evidences[:6]:
        if any(t in ev.terminology or t in ev.topic or t in ev.proposition for t in missing):
            ref_blocks.append(render_evidence_brief(ev))
    ref_text = "\n\n".join(ref_blocks[:4])
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
- 資料外例は原則避ける
"""
    return clean_text(
        call_text(
            client,
            model,
            "最小修正で概念不足を補う。本文のみ返す。",
            prompt,
            temperature=0.25,
            max_output_tokens=3200,
        )
    )


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
    data = call_json(
        client,
        model,
        "資料固有性、重複、論点カバー、AIっぽさを厳しく評価する。JSONのみ返す。",
        prompt,
        temperature=0.0,
        max_output_tokens=900,
    )
    data["external_example_risk"] = detect_external_example_risk(report, evidences)
    data["abstract_term_pressure"] = count_abstract_term_hits(report)
    return data


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
        "標準": "自然で読みやすくする。",
        "やや硬め": "少しフォーマルで締まった文体にする。",
        "やや柔らかめ": "少し柔らかくするが、幼くしない。",
    }[style_mode]
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
- 資料外例は原則避ける
- {style_instruction}
"""
    return clean_text(
        call_text(
            client,
            model,
            "最小限の改稿で品質を上げる。本文のみ返す。",
            prompt,
            temperature=0.28,
            max_output_tokens=4200,
        )
    )


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
    return clean_text(
        call_text(
            client,
            model,
            "文字数と重複を同時に自然に圧縮する。本文のみ返す。",
            prompt,
            temperature=0.2,
            max_output_tokens=4200,
        )
    )


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
    target_addition = max(500, min(shortage + 200, 900))
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
- 既出の定義を繰り返さない
- 既出の事例を別表現でなぞらない
- 各段落で資料由来の概念を明示する
- 約{target_addition}字ぶんの中身を増やす
"""
    addition = clean_text(
        call_text(
            client,
            model,
            "字数不足を、未展開論点の補足だけで埋める追加段落を書く。本文のみ返す。",
            prompt,
            temperature=0.24,
            max_output_tokens=2000,
        )
    )
    if not addition:
        return report
    return clean_text(report + "\n\n" + addition)


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
    report = append_report_if_too_short(client, model, theme, report, evidences, target_length)
    report = compress_report_if_too_long(client, model, report, target_length)
    report = ensure_complete_text(client, model, theme, report, evidences)

    critique = critique_report(client, model, theme, report, evidences)
    if critique.get("revision_needed", False) and int(critique.get("overall_score", 0)) < 84:
        report = rewrite_once(client, model, theme, report, critique, evidences, style_mode)
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

    progress.set_flow_step("論点設計", "論点マップを作っています。")
    argument_map = build_argument_map(client, model, theme, selected, points=3)

    progress.set_flow_step("本文生成", "本文を組み立てています。")
    report = generate_single_pass_report(client, model, theme, target_length, argument_map, selected, style_mode, abstraction_mode)

    progress.set_flow_step("字数補正", "文字数と資料語の不足を整えています。")
    report, critique = finalize_report(client, model, theme, report, selected, target_length, style_mode)

    progress.finish("生成が完了しました。")
    return {
        "mode": "高速",
        "chunks": chunks,
        "selected_evidences": selected,
        "argument_map": argument_map,
        "report": report,
        "critique": critique,
    }

# ============================================================
# 14. UI入力 / 実行
# ============================================================
for key, default in {"result": None}.items():
    if key not in st.session_state:
        st.session_state[key] = default

left, right = st.columns([1.18, 0.82], gap="large")

with left:
    st.subheader("入力")
    user_id = st.text_input("識別名（ニックネームでOK）", placeholder="例：yuya / tanaka / a123")
    uploaded_files = st.file_uploader("PDFを複数アップロード", type=["pdf"], accept_multiple_files=True)
    theme = st.text_area(
        "課題文 / テーマ",
        placeholder="例：科学技術の発展がマーケティング・チャネル設計に与えた影響を、ディスインターメディエーションと電子市場を中心に論じなさい。",
        height=130,
    )
    focus_points = st.text_input("特に重視したい観点（任意）", placeholder="例：チャネル設計、ディスインターメディエーション、チャネル・コンフリクト")
    preferred_concepts = st.text_input("扱いたい概念（任意）", placeholder="例：電子市場、垂直型システム、EDI、リテール・リンク")
    excluded_topics = st.text_input("除外したい話題（任意）", placeholder="例：ブランド価値、感情分析、新製品開発")
    source_scope = st.text_input("使う範囲・章指定（任意）", placeholder="例：7章中心、7マーケティング・チャネル p.11〜38 を優先")
    target_length = st.number_input(
        "目標文字数",
        min_value=300,
        max_value=MAX_TARGET_CHARS,
        value=min(2000, MAX_TARGET_CHARS),
        step=100,
        help=f"このテスト版では {MAX_TARGET_CHARS} 字までです。",
    )

with right:
    st.subheader("設定")
    st.info("この版は友人テスト用のため、高速版のみです。")
    style_mode = st.selectbox("文体", ["標準", "やや硬め", "やや柔らかめ"], index=0)
    abstraction_mode = st.selectbox("抽象度", ["標準", "抽象度高め", "抽象度低め"], index=0)
    used_today = get_usage_count(user_id) if user_id.strip() else 0
    st.metric("本日の利用回数", f"{used_today}/{MAX_DAILY_RUNS}")
    st.caption(f"PDFは {MAX_PDFS} 個まで / 1ファイル {MAX_FILE_MB}MB まで")

max_pdfs = MAX_PDFS
max_file_mb = MAX_FILE_MB
if uploaded_files and len(uploaded_files) > max_pdfs:
    st.error(f"PDFは {max_pdfs} 個までです。")
    st.stop()

if uploaded_files:
    for f in uploaded_files:
        size_mb = f.size / (1024 * 1024)
        if size_mb > max_file_mb:
            st.error(f"{f.name} は {max_file_mb}MB を超えています。")
            st.stop()

confirm_generate = st.checkbox("この設定で本当に生成する", value=False)
generate_clicked = st.button("生成する", use_container_width=True, type="primary")


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


def validate_inputs(user_id_value: str, uploaded, theme_text: str, confirm_flag: bool) -> None:
    if not confirm_flag:
        st.warning("生成前にチェックを入れてください。")
        st.stop()
    if not user_id_value.strip():
        st.error("識別名を入力してください。")
        st.stop()
    if not uploaded:
        st.error("PDFを1つ以上アップロードしてください。")
        st.stop()
    if not theme_text.strip():
        st.error("課題文またはテーマを入力してください。")
        st.stop()


if generate_clicked:
    validate_inputs(user_id, uploaded_files, theme, confirm_generate)
    ok, used = check_and_increment_usage(user_id.strip(), daily_limit=MAX_DAILY_RUNS)
    if not ok:
        st.error("本日の利用上限に達しました。")
        st.stop()

    client = get_client()
    effective_theme = build_user_guidance(theme, focus_points, preferred_concepts, excluded_topics, source_scope)
    progress_ui = ProgressUI()

    try:
        result = run_fast_pipeline(
            client,
            MODEL_NAME,
            effective_theme,
            uploaded_files,
            int(target_length),
            style_mode,
            abstraction_mode,
            progress_ui,
        )
        result["effective_theme"] = effective_theme
        result["used_today"] = used
        st.session_state.result = result
        st.success("生成が完了しました。")
    except Exception as e:
        st.exception(e)
        st.stop()

# ============================================================
# 15. 出力表示
# ============================================================
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
        st.download_button(
            "本文をテキスト保存",
            data=report,
            file_name="reportflow_report.txt",
            mime="text/plain",
            use_container_width=True,
        )

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

    with tab4:
        st.subheader("内部評価")
        st.json(critique)
        with st.expander("重要概念候補"):
            st.write(", ".join(important_terms_from_evidences(selected, top_k=16)))
