# Book ↔ Audiobook Alignment Project

**Study recap | 5 November 2026**  
**Scope:** What we discussed, implemented in notebook cells, and observed so far. This is a learning prototype, **not yet a finished full-book alignment pipeline**.

## 1. The goal, in plain language

We have a **book** (text extracted from a PDF) and a **recording of that book**. We want to build a training dataset for a speech-to-text (STT) model:

```text
ENTIRE BOOK                                  ENTIRE AUDIOBOOK
PDF → book.txt                               audiobook.mp3 / .wav
       │                                             │
       └──────────────────┬──────────────────────────┘
                          ▼
               Find speech regions (VAD)
                          ▼
          Cut temporary sections (~60 s, overlap)
                          ▼
            Transcribe temporary sections
                          ▼
            Match sections to book passages
                          ▼
           Verify/correct spoken words and boundaries
                          ▼
                  Forced alignment
               (word-level timestamps)
                          ▼
           Resolve overlaps + cut final clips
                          ▼
       3–30 s audio files + corresponding correct text
```

**Why not simply split the audiobook every 10 seconds?** A cut can happen in the middle of a word or sentence. Without word timing, we cannot reliably know which text belongs to each clip.

**Why not align the entire several-hour recording at once?** Long recordings are more expensive and error-prone to process. We work on manageable sections, but preserve their locations within the *original* recording.

**Key distinction:** The ~59.9-second clip was a **test section**, not the ultimate unit of work. The system must eventually cover **every section of the entire book and audiobook**.

---

## 2. The three kinds of text

| Text | What it means | Why we need it |
|---|---|---|
| **Book text** | Text extracted from the PDF, e.g. `book.txt` | The intended reference text, but may contain PDF-extraction errors |
| **STT transcription** | Words predicted from an audio section by a speech-to-text model | Helps locate a section in the book; can contain recognition errors |
| **Verified alignment text** | The words **actually spoken** in that section, checked/corrected as needed | Correct input to forced alignment |

Never assume either the PDF text or the STT transcription is automatically perfect. If the narrator says an introduction absent from the book, the audio and alignment text must still correspond. Either **retain the introduction with its verified transcription** or **trim that audio and remove the matching words from the text**. Do not remove only one side.

### Example from our project

- **Transcription begins:** `qadrli bolajonlar o'zbek xalq ertaklari ... qadim zamonda bir podsho o'tgan ...`
- **Book extract begins:** `ain ochilar y adim zamonda bir podsho o'tgan ekan ...`
- **Explanation:** The audio has an introduction not present in the book passage; the PDF extraction also damaged the phrase `qadim zamonda`.
- **Our shared anchor:** `bir podsho o'tgan`.

The anchor helps us locate the story, **but an anchor's character position is not an audio timestamp**.

---

## 3. Files and notebooks we used

Approximate project structure seen in your screenshots (some files were created or rearranged during our work):

```text
book_audio_alignment/
├── data/
│   └── test/
│       ├── book.txt                        # Reference book text
│       ├── test_audio.mp3                  # Audio used in duration check
│       ├── raw_api_responses/
│       │   └── section_000.response        # Raw transcription JSON, etc.
│       ├── all_section_transcriptions.json # Collected section texts
│       ├── book_matches_draft.json         # Book-matching results
│       ├── alignment_plan.json             # Proposed candidate/review statuses
│       ├── alignment_candidates.json       # Draft alignment manifest
│       └── alignment_drafts/
│           ├── section_000.txt
│           └── ...
├── src/
│   └── notebooks/
│       ├── normalization.ipynb
│       ├── vad.ipynb
│       ├── connecting_text.ipynb           # Early single-section experiment
│       ├── full_book_processing.ipynb      # Batch/sequential processing
│       ├── temporary_sections/
│       │   ├── section_000.wav
│       │   └── ...
│       └── temporary_sections.json         # Section timing metadata (inspect schema)
└── scripts/
    ├── transcribe.py
    ├── transcribe_sections.py
    ├── pdf_to_txt.py
    └── ...
```

Paths are **relative to the notebook's current working directory**. Check `Path.cwd()` if a notebook cannot find a file. File existence is not proof that its content has been manually verified.

---

## 4. Step-by-step: theory, logic, code

### Stage A. Extract and clean the book/transcription

**Idea:** PDFs may produce broken words, strange apostrophes, page headers, and extra spaces. STT outputs can have punctuation and casing different from the book. Normalize both to compare them fairly.

```python
import re


def normalize(text):
    text = text.lower()
    for mark in ["‘", "’", "ʻ", "ʼ", "`"]:
        text = text.replace(mark, "'")
    text = re.sub(r"[^\w\s']", " ", text)
    return " ".join(text.split())

book_words = normalize(book).split()
spoken_text = normalize(transcription)
```

**Code notes:**

- `lower()` makes capitalization consistent.
- `replace()` unifies several apostrophe shapes often found in Uzbek text.
- `re.sub(pattern, replacement, text)` replaces characters matching a regular expression. Here, we remove most punctuation while retaining word characters, spaces, and `'`.
- `split()` produces words; `" ".join(...)` puts them back with single spaces.
- **Important:** This cleaned text is useful for *searching*. Keep the original/verified text separately; aggressive normalization can remove useful punctuation and lose the exact original character offsets.

### Stage B. Detect speech with Silero VAD

**VAD = Voice Activity Detection.** Silero VAD finds approximate regions containing speech and helps avoid cutting at arbitrary points or long silences.

Conceptually, VAD returns intervals like:

```python
speech_regions = [
    {"start": 0.8, "end": 5.3},
    {"start": 5.9, "end": 12.4},
]
```

*These values are illustrative. Some VAD APIs use sample indices rather than seconds; check units before cutting.*

**What VAD does NOT do:** recognize words, tell us what page is being spoken, or provide reliable word-level timing. VAD boundaries are not the same as forced-alignment boundaries.

### Stage C. Create temporary audio sections

**Idea:** Group speech regions into manageable sections (in our experiment, roughly **60 seconds**) with a **small overlap** between neighboring sections. The overlap protects words at section edges.

```text
Original audio: [-------------------- continuous recording --------------------]
Section 000:    [----------]
Section 001:              [----------]
                                   ^ possible shared audio
Section 002:                       [----------]
```

For **every temporary section**, we must retain the exact time offset in the original audiobook, not just the WAV filename:

```json
{
  "section": "section_001",
  "start": 58.9,
  "end": 118.9
}
```

*Example schema only; the exact structure of your `temporary_sections.json` must be inspected.*

Why record this? If a word occurs **3.0 seconds into a section** that starts at **58.9 seconds** in the original, its full-audiobook start time is **61.9 seconds**.

### Stage D. Transcribe each temporary section

We used an STT API and stored raw responses in files such as `section_000.response`. A response contained `segments`, each with `start`, `end`, and `text` fields.

**Typical extraction logic:**

```python
import json
from pathlib import Path

response_path = Path("../../data/test/raw_api_responses/section_000.response")
data = json.loads(response_path.read_text(encoding="utf-8"))

transcription = " ".join(
    segment["text"].strip()
    for segment in data["segments"]
)
print(transcription)
```

**Code notes:**

- `Path(...)` represents a file path.
- `read_text(encoding="utf-8")` reads the contents as text.
- `json.loads(...)` **parses** a JSON string into Python dictionaries/lists.
- `segment["text"]` extracts each segment's recognized words.
- `.strip()` removes spaces at the edges; `" ".join(...)` combines segments into one transcript.

The model's segment times are **not guaranteed to be precise word timestamps**, and transcription is not forced alignment.

### Stage E. Find the matching passage in the book

We first tested **RapidFuzz** on a single transcript. RapidFuzz compares strings and produces an approximate similarity score from **0 to 100**.

```python
from rapidfuzz import fuzz

score = fuzz.ratio(spoken_text, candidate_book_passage)
```

**How the window search worked:**

1. Turn the cleaned book into `book_words`.
2. Count the words in the transcript to choose an approximate passage length (`window_size`).
3. Slide a candidate window across the book.
4. Compare each candidate with `fuzz.ratio`.
5. Keep the candidate with the largest score and its starting word index.

Illustrative simplified search:

```python
window_size = len(spoken_text.split())
step = max(1, window_size // 10)

best_score = -1
best_start = 0
best_passage = ""

for start in range(0, len(book_words), step):
    candidate = " ".join(book_words[start:start + window_size])
    score = fuzz.ratio(spoken_text, candidate)
    if score > best_score:
        best_score = score
        best_start = start
        best_passage = candidate
```

**Python concepts:** `range(...)` generates positions; list slicing `[a:b]` gets items from `a` up to but not including `b`; `max(1, ...)` stops the step from becoming zero; `best_score` remembers the strongest match.

**Crucial caveat:** A high RapidFuzz score indicates **text similarity**, not proof of correct audio/text boundaries and not calibrated confidence.

### Stage F. Locate introductions and compare mismatches

Our first match scored **75.51%**, with the first best passage beginning near the start of the book. Later full-book matching produced **82.35%** for `section_000`.

Both results pointed to the same issue: an unmatched spoken introduction plus PDF extraction errors. The score can change when the search window or matching algorithm changes.

We used a **shared phrase (anchor)** to examine how much text occurs before the main story:

```python
anchor = "bir podsho o'tgan"
audio_pos = normalize(transcription).find(anchor)
book_pos = normalize(book).find(anchor)

if audio_pos != -1 and book_pos != -1:
    intro_text = normalize(transcription)[:audio_pos]
    print(intro_text)
```

- `.find(phrase)` returns the phrase's starting **character index**, or `-1` if absent.
- `text[:position]` takes text *before* the position.
- This helps locate an introduction in **text**. It does not automatically tell us the introduction's length in **seconds**.

We also used `difflib.SequenceMatcher` for word-by-word differences:

```python
from difflib import SequenceMatcher

matcher = SequenceMatcher(
    None, book_story.split(), audio_story.split(), autojunk=False
)
for tag, i1, i2, j1, j2 in matcher.get_opcodes():
    if tag == "equal":
        continue
    print(tag, book_story.split()[i1:i2], audio_story.split()[j1:j2])
```

Its tags are `equal`, `replace`, `insert`, and `delete` (relative to the order of the two supplied sequences). These expose likely PDF or STT errors. **Review them rather than treating every difference as an instruction to copy one source over the other.**

### Stage G. Inspect a section's audio and possibly trim it

We discussed listening to local time ranges with `soundfile` and IPython audio playback:

```python
import soundfile as sf
from IPython.display import Audio, display

audio, sr = sf.read(audio_path)

def listen(start, end):
    clip = audio[int(start * sr):int(end * sr)]
    display(Audio(clip.T, rate=sr))
```

- `sr` is sample rate (samples per second).
- `start * sr` converts a time in seconds into a sample index.
- `.T` transposes the audio array for the notebook player, useful for multichannel audio.

Two legitimate ways to handle an introduction:

1. **Keep it:** verify the transcription, and align the whole section using the words actually spoken.
2. **Trim it:** find the real audio timestamp, cut the audio and its corresponding words consistently, and record the new time offset.

**Our latest approach is to preserve the introduction in `section_000`'s transcription draft**, not to discard it automatically. No exact introduction boundary has been verified in the screenshots.

### Stage H. Scale from one test section to all available sections

We created `full_book_processing.ipynb` so the single-section experiment wouldn't turn into ten thousand manual edits.

**Check which sections already have response files:**

```python
sections_dir = Path("temporary_sections")
responses_dir = Path("../../data/test/raw_api_responses")

audio_files = sorted(sections_dir.glob("*.wav"))
completed = {
    file.stem for file in responses_dir.glob("section_*.response")
}
pending = [f for f in audio_files if f.stem not in completed]

print("Audio sections:", len(audio_files))
print("Missing responses:", len(pending))
```

**Python concepts:**

- `glob("*.wav")` finds matching files.
- `sorted(...)` orders filenames; zero-padded names like `section_000` make this order meaningful.
- `file.stem` is the filename minus extension.
- `{... for ...}` constructs a **set**, useful for fast membership checks.
- `[...]` here constructs a **list** of pending files.

**Observed output:** `Total audio sections: 10`, `Already transcribed: 10`, `Still need transcription: 0`. This means **10 of 10 existing temporary files have matching response filenames**. It does **not** prove these 10 sections cover the *entire audiobook*, or that every response is valid.

We also aggregated transcription text into `all_section_transcriptions.json`, so each section has an ID and transcription together.

### Stage I. Match sections sequentially, not independently

Searching the **entire** book for **every** section is wasteful and can return repeated sentences from the wrong chapter.

Our sequential approach:

1. Search the whole book for the **first** section.
2. Save its best matching word range, e.g. `[start_word, end_word)`.
3. For the **next** section, search near the previous result while allowing overlap and some forward progress.
4. Try candidate lengths around **80%, 100%, and 120%** of the transcript's word count.
5. Refine the strongest candidate around its start position.
6. Save `section`, `start_word`, `end_word`, `similarity`, `transcription`, `book_text`, `needs_review` into `book_matches_draft.json`.

**Core logic from the notebook:**

```python
# For sections after section_000:
search_start = max(0, previous_start - 30)
search_end = min(len(book_words) - 1, previous_end + word_count * 3)

window_sizes = [
    max(1, int(word_count * factor))
    for factor in (0.8, 1.0, 1.2)
]
```

These offsets and multipliers are **heuristics**, not universal laws. Search errors can compound: if one section matches the wrong passage, the next section may search in the wrong area too. A production pipeline needs safeguards and occasional re-anchoring/global checks.

### Stage J. Validate the match sequence

We loaded `book_matches_draft.json` and checked:

- **Low similarity:** below 85% → review flag (a chosen threshold, not a scientific guarantee).
- **Invalid word range:** `end_word <= start_word`.
- **Backward match:** section starts before the prior section's start.
- **Large forward gap:** `current_start - previous_end > 40` words → possible missing passage.

```python
gap = current_start - previous_end
```

| Gap | Meaning |
|---|---|
| `gap < 0` | Book word ranges overlap; possibly normal with overlapping audio |
| `gap == 0` | Ranges meet exactly |
| `gap > 0` | Uncovered book words; could be expected or could show a mistake |

**Important:** The intended audio overlap matters; a positive or negative text gap by itself is not enough to label a section correct or wrong.

**Observed results from your screenshots:**

| Section | Match similarity | Matched book word range | Gap from previous | Flag |
|---|---:|---:|---:|---|
| `section_000` | **82.35%** | `0–51` | N/A | Low similarity |
| `section_001` | **98.65%** | `51–151` | `0` | None |
| `section_002` | **95.03%** | `150–237` | `-1` | None |
| `section_003` | **98.26%** | `240–330` | `3` | None |
| `section_004`–`009` | Not fully visible in screenshot | Not fully visible | Not fully visible | No additional review flags in summary |

**Total matched sections:** 10. **Sections flagged for review:** 1. A good score is a **candidate**, not a verified audio-text pair.

### Stage K. Create an alignment plan and text drafts

We proposed two stages of output:

- `alignment_plan.json`: the section, audio path, proposed book text, score, status (`candidate` vs `review`), and issues.
- `alignment_candidates.json` plus `alignment_drafts/section_XXX.txt`: an editable text file per audio section and a manifest tracking its path and whether it has been verified.

Initial draft selection:

```python
if match["similarity"] < 85:
    text = match["transcription"]  # e.g., section_000 with introduction
    source = "transcription"
else:
    text = match["book_text"]      # matching book passage
    source = "book"
```

Then create the draft files without overwriting corrections:

```python
draft_dir = Path("../../data/test/alignment_drafts")
draft_dir.mkdir(parents=True, exist_ok=True)

text_path = draft_dir / "section_000.txt"
if not text_path.exists():
    text_path.write_text(text.strip(), encoding="utf-8")
```

- `mkdir(parents=True, exist_ok=True)` creates directories if missing without error if already present.
- `if not text_path.exists()` avoids silently overwriting manual fixes.
- The manifest used `"verified": False` because no automated string score proves that every word was actually spoken.

**Potential limitation in our initial draft logic:** Choosing book text at a score of ≥85 can still introduce missing or extra words at the **start/end** of a section. Before alignment, we need boundary-aware correction rather than blindly trusting similarity.

---

## 5. Key Python libraries and utilities covered

| Library or tool | Purpose in this project |
|---|---|
| `pathlib.Path` | File and directory paths; `.glob()`, `.stem`, `.exists()`, `.read_text()`, `.write_text()`, `.mkdir()` |
| `json` | Parse/save JSON API responses and intermediate manifests (`loads`, `dumps`) |
| `re` | Normalize characters and whitespace using regular expressions |
| `rapidfuzz.fuzz` | Score approximate text similarity for locating passages |
| `difflib.SequenceMatcher` | Explain which words match, were inserted/deleted, or were replaced |
| `soundfile` | Load/save audio files and inspect sample rate |
| `IPython.display.Audio` | Listen to short audio slices in a notebook |
| `subprocess` + `ffprobe` | Inspect original audio duration without loading the whole file |
| **Silero VAD** | Locate speech/silence regions before temporary segmentation |
| **STT model/API** | Produce a rough transcription for each section (not final alignment) |

### `ffprobe` example: check full-audio duration

```python
import subprocess

result = subprocess.run(
    [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(original_audio),
    ],
    capture_output=True,
    text=True,
    check=True,
)
duration_seconds = float(result.stdout.strip())
```

- `subprocess.run` runs an external program.
- `capture_output=True` captures what it prints; `text=True` returns strings.
- `check=True` raises an error if the program fails.
- `float(...)` converts a printed duration into a number.

**Caveat:** `test_audio.mp3` may only be a sample. We did not see a confirmed result proving the 10 sections cover the complete audiobook.

---

## 6. Important conceptual differences

| Concept | Question it answers | Output |
|---|---|---|
| **VAD** | Where is someone speaking? | Speech time ranges |
| **Temporary segmentation** | How do we divide large audio into manageable overlapping parts? | ~60-second section WAVs + offsets |
| **STT transcription** | What words might have been spoken? | Predicted text |
| **Fuzzy text matching** | Where in the book does this speech belong? | Approximate word range + similarity |
| **Text review** | What words were *actually* spoken in this exact audio interval? | Corrected/verified text |
| **Forced alignment** | When was each known word spoken? | Per-word start and end times |
| **Final chunking** | How can we export usable training examples? | 3–30-second audio + matching text |

### Forced alignment is *not* transcription

A transcriber predicts the words from audio. A forced aligner takes **audio + known text** and estimates *when* the words occur. Correct text is therefore essential.

Example of the **future** output format (times are illustrative):

```json
[
  {"word": "qadim", "start": 0.12, "end": 0.54},
  {"word": "zamonda", "start": 0.55, "end": 1.10},
  {"word": "bir", "start": 1.15, "end": 1.35}
]
```

If those timestamps were relative to a section starting at 125.0 seconds in the full audiobook, we would add **125.0** to each value to obtain full-audiobook timestamps. Later we must also resolve duplicate words across overlapping sections.

For **Uzbek**, we must select/verify a compatible forced-alignment model; we cannot assume that a particular English-focused or WhisperX-based workflow supports the language accurately.

---

## 7. What is done, what is *not* done

### Demonstrated or visible in our screenshots

- One ~59.9-second section successfully transcribed with an Uzbek-language STT response.
- Text extracted from the book, with observable PDF errors.
- RapidFuzz matching tested; we saw 75.51% in the original single-section experiment.
- 10 temporary section files discovered and 10 response filenames found in the batch notebook.
- Sequential book matching run over 10 sections, with `section_000` scoring 82.35% and the remaining summary showing only **one** review flag.
- Mismatch/anchor inspection performed for `section_000`.

### Code proposed, but successful execution/verification not shown for every step

- Duration/metadata coverage check across the original audio.
- All alignment-plan and editable-draft files being created and reviewed.
- Actual audio trimming at an exact verified introduction timestamp.
- Consistent, word-accurate correction of each section's beginning and end.

### Still to build

1. **Verify whole-audiobook coverage.** Confirm original duration and that temporary section offsets cover the *entire* recording; generate missing sections if not.
2. **Complete boundary-aware text preparation** for every section, including the introduction, PDF errors, and overlaps.
3. **Choose and test an Uzbek-capable forced aligner** on a verified section.
4. **Run alignment on all sections**, retaining section offsets and reliable word timestamps.
5. **Deduplicate overlapping words/audio**, and handle any alignment failures.
6. **Cut final 3–30-second clips** on safe word/silence boundaries and save accurate matching text.
7. **Validate the export** (no lost speech, no missing text, no repeated overlap, correct durations) before training.

## 8. Best starting point when we resume

Continue in **`src/notebooks/full_book_processing.ipynb`**. Don't make another notebook yet. Start by checking `temporary_sections.json` and the **total duration of the original full audiobook** against the full set of section `[start, end]` ranges. This resolves the biggest uncertainty: **we have processed 10 existing sections, but have not established that they cover the whole book's audio**. Then finish review of each section's actual spoken text and select the forced alignment model.

**Bottom line:** We have built and tested the *matching/preparation* portion of an audiobook alignment pipeline. We have **not** performed forced alignment or exported the final STT chunks yet.
