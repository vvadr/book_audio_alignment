"""Forced alignment with NVIDIA's Uzbek NeMo model.

"Forced" alignment means we already know WHAT is said (the text), and the
model only has to find WHEN each word is said in the audio.
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

MODEL_NAME = "nvidia/stt_uz_fastconformer_hybrid_large_pc"
MIN_FRAME_SCORE = -3.0


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


def make_word(word, log_probs, frame_duration):
    """Turn one aligned NeMo word into a dict with its time, score and trimmed seconds.

    The score is the average log-probability of the word's pieces over the
    frames they cover: 0 means certain, -1 is about 37 %, -3 is about 5 %.
    Frames the model was not sure about (below MIN_FRAME_SCORE) are cut off
    both ends of the word. That is how "Qadrli" loses its 11 s of music.
    """
    frame_numbers = []
    frame_scores = []

    for token in word.tokens:
        if token.text == BLANK_TOKEN:
            continue

        first_frame = round(token.t_start / frame_duration)
        last_frame = round(token.t_end / frame_duration)
        for frame in range(first_frame, last_frame):
            frame_numbers.append(frame)
            frame_scores.append(log_probs[0, frame, token.token_id].item())

    sure_frames = [
        frame for frame, frame_score in zip(frame_numbers, frame_scores)
        if frame_score >= MIN_FRAME_SCORE
    ]

    if sure_frames:
        start = sure_frames[0] * frame_duration
        end = (sure_frames[-1] + 1) * frame_duration
    else:
        start = word.t_start
        end = word.t_end

    trimmed = (word.t_end - word.t_start) - (end - start)

    return {
        "word": word.text,
        "start": round(start, 2),
        "end": round(end, 2),
        "score": round(sum(frame_scores) / len(frame_scores), 2),
        "trimmed": round(trimmed, 2),
    }


def align(model, audio_path, text):
    """Find when each word of `text` is spoken in `audio_path`.

    Returns a list like
    [{"word": "oʻzbek", "start": 12.72, "end": 13.12, "score": -0.4, "trimmed": 0.0}, ...].
    Times are in seconds, counted from the start of this audio file.
    """
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