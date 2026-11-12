"""Forced alignment with NVIDIA's Uzbek NeMo model.

"Forced" alignment means we already know WHAT is said (the text), and the
model only has to find WHEN each word is said in the audio.

The model is a CTC model: it marks each piece of a word ("▁de", "v", "dan")
with a short spike, one 80 ms frame long, and says "blank" everywhere else.
So a word's start and end are the times of its first and last spike. The
real sound lasts a little longer on both sides; chunks.py listens to the
audio to cut only in silence.
"""

import torch
import nemo.collections.asr as nemo_asr
from nemo.collections.asr.parts.utils.aligner_utils import (
    BLANK_TOKEN,
    get_batch_variables,
    viterbi_decoding,
    add_t_start_end_to_utt_obj,
    Segment,
    Word,
)

from uzbek_text import MODEL_CHARS

MODEL_NAME = "nvidia/stt_uz_fastconformer_hybrid_large_pc"


def load_model():
    """Load the Uzbek model: on the GPU if PyTorch can use it, otherwise on the CPU."""
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = nemo_asr.models.EncDecHybridRNNTCTCBPEModel.from_pretrained(
        model_name=MODEL_NAME,
        map_location=device,
    )
    model.change_decoding_strategy(decoder_type="ctc")
    model.eval()

    print("Model loaded on", device)
    return model


def quiet_edges(log_probs, frame_count):
    """Tell the aligner that the first and last frame of the file are silent.

    A section file starts and ends suddenly, often in the middle of a sound.
    In that first frame the model "hears" a piece of a word that is not
    there, and the aligner pulls the first word back to 0 s: "Oʻsha" started
    at 121.55 s instead of 122.19 s, and "Qadrli" covered 11 s of music.

    log_probs[0, frame, token] is the model's score for `token` in `frame`.
    The blank token is the last one. We make blank certain (score 0) and
    every other token impossible (a very low score) in those two frames.
    """
    blank = log_probs.shape[-1] - 1
    for frame in (0, frame_count - 1):
        log_probs[0, frame, :] = -10000.0
        log_probs[0, frame, blank] = 0.0


def make_word(word, log_probs, frame_duration):
    """Turn one aligned NeMo word into a dict with its time and score.

    The score is the average log-probability of the word's spikes:
    0 means certain, -1 is about 37 %, -3 is about 5 %.
    """
    frame_scores = []

    for token in word.tokens:
        if token.text == BLANK_TOKEN:
            continue

        first_frame = round(token.t_start / frame_duration)
        last_frame = round(token.t_end / frame_duration)
        for frame in range(first_frame, last_frame):
            frame_scores.append(log_probs[0, frame, token.token_id].item())

    return {
        "word": word.text,
        "start": round(word.t_start, 2),
        "end": round(word.t_end, 2),
        "score": round(sum(frame_scores) / len(frame_scores), 2),
    }


def align(model, audio_path, text):
    """Find when each word of `text` is spoken in `audio_path`.

    Returns a list like
    [{"word": "oʻzbek", "start": 12.72, "end": 13.12, "score": -0.4}, ...].
    Times are in seconds, counted from the start of this audio file.
    Prepare `text` with uzbek_text.to_model_text first.
    """
    unknown = sorted({char for char in text if char not in MODEL_CHARS and not char.isspace()})
    if unknown:
        print(f"Warning: {audio_path} has characters the model does not know: {' '.join(unknown)}")

    (
        log_probs,
        token_ids,
        audio_lengths,
        token_lengths,
        utterances,
        frame_duration,
    ) = get_batch_variables(
        audio=[str(audio_path)],
        model=model,
        gt_text_batch=[text],
        align_using_pred_text=False,
    )

    quiet_edges(log_probs, int(audio_lengths[0]))

    alignments = viterbi_decoding(
        log_probs,
        token_ids,
        audio_lengths,
        token_lengths,
        viterbi_device="cpu",
    )

    utterance = add_t_start_end_to_utt_obj(
        utterances[0],
        alignments[0],
        frame_duration,
    )

    words = []
    for segment in utterance.segments_and_tokens:
        if isinstance(segment, Segment):
            for word in segment.words_and_tokens:
                if isinstance(word, Word):
                    words.append(make_word(word, log_probs, frame_duration))

    if len(words) != len(text.split()):
        print(f"Warning: {audio_path} has {len(text.split())} words, but {len(words)} were aligned")

    return words