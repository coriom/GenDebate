import json
import time
import random
import re
from pathlib import Path
from typing import List, Dict
from openai import OpenAI

# ==============================
# GLOBAL CONFIGURATION
# ==============================

OPENAI_API_KEY = ""             # ⚠️ Must be set before running
MODEL          = "gpt-4o-mini"  # Model used to generate the debate
TARGET_TURNS   = 20             # Number of utterances per debate session
SESSIONS       = 3              # Number of debate sessions to run
DEBATE_PATH    = "DEBATE"       # File where debates + analyses are appended
SLEEP_BETWEEN  = 1.0            # Pause (in seconds) between two sessions
SEED           = 123            # RNG seed for reproducibility (None => non-deterministic)
REACT_MIN_PCT  = 0.85           # Minimum fraction of lines that must contain a reply marker ↪P#
MAX_ATTEMPTS   = 3              # Max retries if generation is not reactive enough

# Analysis model parameters
ANALYSIS_MODEL   = "gpt-4o-mini"                 # Model used to analyze the generated debate
ANALYSIS_MAXTOK  = 2000                          # Max tokens for the analysis response
ANALYSIS_OUTFILE = Path("ANALYSES/analysis_last_session.md")  # File to store the latest analysis
APPEND_ANALYSIS_IN_DEBATE = True                 # If True, also append analysis to the DEBATE file

# ==============================
# PERSONAS
# ==============================
# Each persona represents a speaker with:
# - id: numeric identifier used to derive P#
# - speak_prob: base probability of speaking at each turn
# - style, traits, etc. used in prompts only (not enforced by code)

PERSONAS: List[Dict] = [
    {"id":1,"label":"China, 25","nationality":"CN","age":25,
    "traits":["proud"],"stance_trailer":"loves it","speak_prob":0.22,
    "style":"enthusiastic, proud, sprinkles light Chinese folklore refs"},
    {"id":2,"label":"Japan, 25","nationality":"JP","age":25,
    "traits":["proud","hostile"],"stance_trailer":"hostile","speak_prob":0.20,
     "style":"sharp, competitive, critiques staging and rhythm"},
    {"id":3,"label":"France, 35","nationality":"FR","age":35,
     "traits":["critic","animation-savvy"],"stance_trailer":"neutral (gets the gist)","speak_prob":0.18,
     "style":"analytical, technical animation vocabulary"},
    {"id":4,"label":"USA, 5","nationality":"US","age":5,
     "traits":["very naive"],"stance_trailer":"loves it","speak_prob":0.15,
     "style":"spontaneous, very short sentences, wonder-filled"},
    {"id":5,"label":"Russia, 12","nationality":"RU","age":12,
     "traits":["snobbish","prefers to play outside"],"stance_trailer":"not interested","speak_prob":0.10,
     "style":"blasé, tries to derail to something else"},
    {"id":6,"label":"Italy, 70","nationality":"IT","age":70,
     "traits":["ultimate snob","knows the Ne Zha myth"],"stance_trailer":"distant","speak_prob":0.15,
     "style":"erudite, biting, occasional absurd scatological aside (very brief, non-graphic)"},
]

# ==============================
# BASE PROMPT FOR DEBATE GENERATION
# ==============================
# This is the shared context + hard rules for the "multi-character conversation engine".
# You later inject:
# - T = number of lines
# - react_pct = minimum % of lines that must contain reply markers.

BASE_CONTEXT = """
Context
Six people just watched the trailer for the Chinese animated film “Ne Zha” (2019, director Jiaozi).
They have NOT watched the full movie; they do know its origin, the director, key myth beats from Fengshen Yanyi (Investiture of the Gods:
rebellious child, fate vs. choice), and what’s visible in the trailer (humor+action, CGI, vivid palette, epic score).

Core rules
- Discuss only what is inferable from the trailer + the myth (no precise movie spoilers).
- Lively, respectful tone, PG-13, no national stereotypes.
- Voices: P1 (CN, enthusiastic), P2 (JP, sharp), P3 (FR, analytical),
P4 (US, 5yo, very short), P5 (RU, blasé), P6 (IT, erudite + brief absurd aside, non-graphic).
- Themes: identity/rebellion, fate vs choice, cultural reading, animation/design, music, humor/action.

Conversation rules (mandatory)
- REAL interaction: at least {react_pct}% of lines must reply explicitly to someone using the marker ↪P#.
- When replying, put the marker right after the tag: `P#: ↪Pk ...` (e.g., `P3: ↪P2 ...`) and build on 1–2 recent ideas.
- Ask several questions across the debate (at least 3 total, varied).
- Use occasional micro-quotes “...” (1–3 words) from earlier lines to show active listening.
- No line repeats a previous one verbatim; every line advances the discussion.

Output format
Write exactly T lines, one utterance per line, strictly as:
P1: ...      (or)      P1: ↪P2 ...
P2: ...
P3: ...
P4: ...
P5: ...
P6: ...
(only these tags; no intro/outro outside the dialogue).
"""

# ==============================
# PROMPT FOR DEBATE ANALYSIS
# ==============================
# The analysis agent reads the transcript and returns:
# - A structured Markdown review
# - A JSON summary embedded at the end

ANALYSE_PROMPT = """
You are an expert conversation and media analyst. Evaluate the debate transcript about the “Ne Zha” (2019, dir. Jiaozi) trailer.

# Inputs
- Personas (reference JSON; use to judge voice fidelity):
{personas_json}

- Target constraints:
    - target_turns={target_turns}
    - min_reply_ratio={react_min_pct}  # fraction of lines that should contain a reply marker (↪P#)

- Transcript (one utterance per line; tags P1..P6):
<<<TRANSCRIPT
{transcript}
TRANSCRIPT>>>

# Tasks
1) Relevance & Understanding
    - Does each speaker mostly stay within what’s inferable from the trailer + Ne Zha myth (no full-movie spoilers)?
    - Is the trailer’s core set-up (rebellious child, fate vs. choice, humor+action, vivid CGI, epic score) correctly understood?

2) Form (Dialogue Quality)
    - Is there real back-and-forth? Estimate actual reply ratio = (lines with “↪P#”) / total.
    - Check flow and coherence (questions, micro-quotes, referencing previous points).
    - Adherence to format: exactly one utterance per line; correct P# tags; no headers/footers.
    - No immediate speaker repetition (P# twice in a row).

3) Content (Substance & Accuracy)
    - Are themes (identity/rebellion, fate vs choice, cultural reading, animation/design, music, humor/action) genuinely discussed?
    - Any factual misinterpretations about the Ne Zha myth or trailer elements? Call them out briefly.
    - Tone & safety: PG-13, no national stereotypes.

4) Persona Fidelity
    - Do voices match the persona briefs (P1 enthusiastic & proud, P2 sharp/critical, P3 analytical/technical, P4 very short/childlike, P5 blasé, P6 erudite with brief non-graphic absurd aside)?
    - Cite 1–2 short evidence snippets per persona.

5) Trailer Improvement Suggestions
   - Based on the discussion, give **concrete, trailer-specific** improvement ideas (editing rhythm, music cues, shot choices, captioning, title cards, cultural framing, etc.).
    - Prioritize 3–6 high-impact, actionably phrased recommendations.

# Scoring (0–5 each; integers)
- relevance_score
- form_score
- content_score
- persona_fidelity_score
- overall_score (not an average: your holistic judgment)

# Output Format (Markdown)
## Summary (5–8 sentences)
A concise overall verdict.

## Relevance & Understanding
- Findings:
- Issues (if any):

## Form (Dialogue Quality)
- Reply ratio (observed vs required): X/Y = Z% vs {react_min_pct} requirement
- Format adherence:
- Flow & coherence:

## Content (Substance & Accuracy)
- Strengths:
- Gaps / misinterpretations (quote and correct):

## Persona Fidelity
- P1:
- P2:
- P3:
- P4:
- P5:
- P6:

## Trailer Improvement Suggestions (Actionable)
1.
2.
3.
4.
5.
6.

## JSON Summary
```json
{{
    "relevance_score": 0,
    "form_score": 0,
    "content_score": 0,
    "persona_fidelity_score": 0,
    "overall_score": 0,
    "observed_reply_ratio": 0.0,
    "turns": {target_turns}
}}
"""
# (End of ANALYSE_PROMPT — Notes removed)


# ==============================
# I/O UTILITIES
# ==============================

def _append_lines(lines, path: str, meta: dict):
    """
    Append a debate session to a text file.
    Adds a header with timestamp + metadata, then each debate line.
    """
    from datetime import datetime
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    md = " ".join(f"{k}={v}" for k,v in meta.items())
    header = f"\n=== SESSION {ts} | {md} ===\n"
    with open(path, "a", encoding="utf-8", newline="\n") as f:
        f.write(header)
        for l in lines:
            f.write(l.rstrip() + "\n")

def _append_analysis_block(analysis_md: str, path: str):
    """
    Append an analysis block (Markdown string) to the same file as the debates.
    """
    from datetime import datetime
    ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    header = f"\n=== ANALYSIS {ts} ===\n"
    with open(path, "a", encoding="utf-8", newline="\n") as f:
        f.write(header)
        f.write(analysis_md.rstrip() + "\n")


# ==============================
# SPEAKER SAMPLING
# ==============================

def _weighted_no_repeat_choice(ids, weights, rng, last):
    """Weighted random choice over `ids` with weights, strictly avoiding `last` if possible."""
    if len(ids) == 1:
        return ids[0]
    # Filter out `last` to avoid immediate repetition
    mask = [(i, w) for i, w in zip(ids, weights) if i != last]
    mask_ids, mask_w = zip(*mask)
    total = sum(mask_w)
    # Normalize weights (fallback to uniform if all are 0)
    norm_w = [w/total for w in mask_w] if total > 0 else [1/len(mask_ids)]*len(mask_ids)
    return rng.choices(mask_ids, weights=norm_w, k=1)[0]

def _sample_speaker_plan(turns: int, rng: random.Random) -> List[str]:
    """
    Build a speaker plan (sequence of P# tags) of length `turns`:
    - uses persona.speak_prob as base sampling weights
    - avoids immediate repetition (no same P# twice in a row)
    """
    ids   = [p["id"] for p in PERSONAS]
    tags  = {p["id"]: f"P{p['id']}" for p in PERSONAS}
    probs = [max(0.0, float(p.get("speak_prob", 0.0))) for p in PERSONAS]
    total = sum(probs) or 1.0
    weights = [w/total for w in probs]

    order, last = [], None
    for _ in range(turns):
        choice = _weighted_no_repeat_choice(ids, weights, rng, last)
        order.append(tags[choice])
        last = choice
    return order


# ==============================
# BUILD USER PROMPT FOR GENERATION
# ==============================

def _build_user_prompt(turns: int, speaker_order: List[str]) -> str:
    """
    Build the user prompt sent to the generation model:
    - injects BASE_CONTEXT with the minimum reply ratio
    - embeds personas as JSON
    - adds a strict speaker order constraint
    """
    personas_json = json.dumps(PERSONAS, ensure_ascii=False, indent=2)
    order_str = ", ".join(speaker_order)
    return (
        BASE_CONTEXT.format(react_pct=int(REACT_MIN_PCT*100))
        + "\nPersonas (input)\n" + personas_json
        + "\n\nSTRICT ORDER CONSTRAINT\n"
        + f"- T={turns}\n- EXACT speaker order (no one speaks twice in a row): {order_str}\n"
        + "- Follow this order strictly: line 1 = first tag, line 2 = second tag, etc.\n"
        + "- Use `↪P#` right after the tag when replying (e.g., `P4: ↪P1 ...`).\n"
    )


# ==============================
# POST-PROCESSING MODEL OUTPUT
# ==============================

def _postprocess_to_order(text: str, expected_order: List[str]) -> List[str]:
    """
    Keep only the lines that follow the expected speaker order.
    For each expected tag in `expected_order`, we scan forward in the model output
    and take the first line that starts with 'P#:' for that tag.
    """
    raw = [l.strip() for l in text.splitlines() if l.strip()]
    out, j = [], 0
    for tag in expected_order:
        found = None
        # Find the next line starting with 'P#:'
        while j < len(raw):
            if raw[j].startswith(tag + ":"):
                found = raw[j]
                j += 1
                break
            j += 1
        if not found:
            break
        out.append(found)
    # Truncate to the exact length of the expected plan
    return out[:len(expected_order)]

def _is_reactive_enough(lines: List[str], threshold: float) -> bool:
    """
    Check if the proportion of lines that contain a reply marker (↪P#)
    reaches at least the given `threshold`.
    """
    if not lines:
        return False
    # We only inspect the beginning of the line (tag + optional reply marker)
    reactive = sum(1 for l in lines if "↪P" in l[:10])
    return (reactive / len(lines)) >= threshold


# ==============================
# GENERATE ONE DEBATE SESSION
# ==============================

def _generate_once(client: OpenAI, rng: random.Random) -> List[str]:
    """
    Generate a single complete debate session:
    - sample a non-repeating speaker order
    - call the chat model
    - post-process to enforce order and valid tags
    - retry a few times if the reactive ratio is too low
    """
    for _ in range(MAX_ATTEMPTS):
        # Speaker plan with no immediate repetition
        speaker_order = _sample_speaker_plan(TARGET_TURNS, rng)
        user_prompt = _build_user_prompt(TARGET_TURNS, speaker_order)
        messages = [
            {"role": "system", "content": "You are a multi-character conversation engine. Follow the ORDER and FORMAT STRICTLY."},
            {"role": "user", "content": user_prompt},
        ]
        # Call generation model
        out = client.chat.completions.create(model=MODEL, messages=messages, temperature=0.8, max_tokens=1500)
        text = (out.choices[0].message.content or "").strip()

        # Filter: keep only lines matching expected tags and order
        lines = _postprocess_to_order(text, speaker_order)
        # Safety: keep only lines with one of the allowed P# tags
        lines = [l for l in lines if ":" in l and l.split(":",1)[0] in ("P1","P2","P3","P4","P5","P6")]
        lines = lines[:TARGET_TURNS]

        # Check if debate is reactive enough
        if _is_reactive_enough(lines, REACT_MIN_PCT):
            return lines

    # If all attempts fail, return the last version (even if suboptimal)
    return lines if 'lines' in locals() else []


# ==============================
# READ LAST SESSION FOR ANALYSIS
# ==============================

def _read_last_session_from_debate(debate_path: Path) -> dict:
    """
    Extract the LAST session block from the DEBATE file.
    Returns a dict with:
    - 'header'    : full session header line
    - 'metadata'  : part after the '|', if present
    - 'transcript': content of the session (P# lines only)
    """
    p = Path(debate_path)
    text = p.read_text(encoding="utf-8") if p.exists() else ""
    if not text.strip():
        raise FileNotFoundError(f"No content found in '{debate_path}'. Generate a debate first.")

    # Match all session headers of the form "=== SESSION ... ==="
    header_pattern = re.compile(r"^=== SESSION .*?===\s*$", re.MULTILINE)
    headers = list(header_pattern.finditer(text))
    if not headers:
        raise ValueError("No '=== SESSION ... ===' header found in DEBATE.")

    # Last header => last session
    last = headers[-1]
    start = last.end()
    # end = start of previous header, or end of file if only one session
    end = headers[-2].start() if len(headers) >= 2 else None
    block = text[start:end].strip() if end else text[start:].strip()
    header_line = last.group(0).strip()
    metadata = header_line.split("|", 1)[1].strip() if "|" in header_line else ""
    return {"header": header_line, "metadata": metadata, "transcript": block}


# ==============================
# BUILD ANALYSIS PROMPT
# ==============================

def _build_analysis_prompt(personas: list, target_turns: int, react_min_pct: float, transcript: str) -> str:
    """
    Prepare the full prompt sent to the analysis model:
    - personas JSON
    - target number of turns
    - minimum reply ratio
    - raw transcript
    """
    return ANALYSE_PROMPT.format(
        personas_json=json.dumps(personas, ensure_ascii=False, indent=2),
        target_turns=target_turns,
        react_min_pct=react_min_pct,
        transcript=transcript
    )

def _analyze_last_debate(client: OpenAI, debate_path: Path) -> str:
    """
    Read the last session from the DEBATE file,
    run the analysis model on it,
    and save the result as Markdown.
    """
    data = _read_last_session_from_debate(debate_path)
    prompt = _build_analysis_prompt(PERSONAS, TARGET_TURNS, REACT_MIN_PCT, data["transcript"])
    completion = client.chat.completions.create(
        model=ANALYSIS_MODEL,
        messages=[
            {"role": "system", "content": "You are an expert conversation and media analyst."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
        max_tokens=ANALYSIS_MAXTOK,
    )
    analysis_md = (completion.choices[0].message.content or "").strip()
    ANALYSIS_OUTFILE.parent.mkdir(parents=True, exist_ok=True)
    ANALYSIS_OUTFILE.write_text(analysis_md, encoding="utf-8")
    return analysis_md


# ==============================
# PUBLIC HELPER: ANALYZE ARBITRARY TRANSCRIPT
# ==============================

def analyse_debate(transcript: str) -> str:
    """
    Optional helper to analyze an arbitrary transcript string
    (without going through the DEBATE file).
    """
    client = OpenAI(api_key=OPENAI_API_KEY)
    personas_json = json.dumps(PERSONAS, ensure_ascii=False, indent=2)
    prompt = ANALYSE_PROMPT.format(
        personas_json=personas_json,
        target_turns=TARGET_TURNS,
        react_min_pct=REACT_MIN_PCT,
        transcript=transcript
    )
    messages = [
        {"role": "system", "content": "You are an expert conversation and media analyst."},
        {"role": "user", "content": prompt},
    ]
    # ⚠️ Depending on your SDK version, this may need to be:
    # client.chat.completions.create(...) (as used elsewhere).
    out = client.chat_completions.create(model=ANALYSIS_MODEL, messages=messages, temperature=0.2, max_tokens=ANALYSIS_MAXTOK)  # type: ignore
    return (out.choices[0].message.content or "").strip()


# ==============================
# MAIN LOOP
# ==============================

def main():
    """
    Full pipeline:
    - Initialize RNG + OpenAI client
    - Run N sessions:
        - generate a debate
        - append it to DEBATE
        - analyze it and save analysis file
        - optionally append analysis to DEBATE as well
    """
    rng = random.Random(SEED) if SEED is not None else random.Random()
    client = OpenAI(api_key=OPENAI_API_KEY)

    for i in range(SESSIONS):
        lines = _generate_once(client, rng)
        if not lines:
            print(f"[Session {i+1}] Nothing to save.")
        else:
            _append_lines(lines, DEBATE_PATH, meta={"turns": len(lines), "session": i+1})
            print(f"[Session {i+1}] OK — {len(lines)} lines appended (probabilistic order, no immediate repeats).")

            analysis_md = _analyze_last_debate(client, Path(DEBATE_PATH))
            print(f"[Session {i+1}] Analysis saved to {ANALYSIS_OUTFILE}")
            if APPEND_ANALYSIS_IN_DEBATE and analysis_md:
                _append_analysis_block(analysis_md, DEBATE_PATH)
                print(f"[Session {i+1}] Analysis also appended to {DEBATE_PATH}")

        # Small pause between sessions if configured
        if i < SESSIONS - 1 and SLEEP_BETWEEN > 0:
            time.sleep(SLEEP_BETWEEN)

main()
