"""Decoder trio for T-one: greedy, beam+kenlm, beam+kenlm+street hotwords.

The acoustic model runs ONCE per utterance; its log-probabilities (segmented
into phrases by the official T-one logprob splitter) are decoded by all three
decoders, so only the decoder configuration differs between variants.

Facts verified against the installed versions (tone 0.1.0, pyctcdecode 0.5.0):
- StreamingCTCModel.forward returns float16 log-probabilities, shape (B, L, 35).
  pyctcdecode detects non-probability input and applies log_softmax itself,
  so the same array feeds GreedyCTCDecoder.forward and
  BeamSearchDecoderCTC.decode unchanged.
- pyctcdecode 0.5.0 BeamSearchDecoderCTC.decode accepts hotwords and
  hotword_weight per call and builds the HotwordScorer per call: a single
  decoder instance is safely reused for both beam variants on identical
  logits.
- Official T-one beam settings (tone/decoder.py): alpha=0.4, beta=0.9,
  beam_width=200.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Iterable

import numpy as np
from pyctcdecode.decoder import BeamSearchDecoderCTC, build_ctcdecoder

from tone.decoder import LABELS, GreedyCTCDecoder
from tone.logprob_splitter import StreamingLogprobSplitter
from tone.onnx_wrapper import StreamingCTCModel

BEAM_WIDTH = 200  # official T-one default (tone/decoder.py, BeamSearchCTCDecoder.forward)
PIPELINE_PADDING = 2400  # 300ms left/right padding (StreamingCTCPipeline.PADDING)


@dataclass
class DecoderTrio:
    """Greedy + KenLM beam decoders shared by every config over one logprob matrix.

    ``decode_plan`` decodes an arbitrary list of configs (label, kind, weight)
    on the SAME logprobs: greedy and plain beam run once and are reused, each
    canonical weight gets its own pyctcdecode call with street hotwords.
    """

    greedy: GreedyCTCDecoder
    beam: BeamSearchDecoderCTC
    beam_width: int = BEAM_WIDTH

    def decode_plan(
        self,
        logprobs: np.ndarray,
        plan: list[tuple[str, str, float | None]],
        hotwords: Iterable[str],
    ) -> dict[str, str]:
        """Decode ONE logprob matrix according to ``plan``: (label, kind, weight).

        kinds: "greedy", "beam_no_hotwords", "canonical" (beam + street
        hotwords at ``weight``). The greedy and the plain beam runs are decoded
        once and cached, so an N-config plan costs at most two non-hotword
        decodes + one decode per canonical weight, all on identical logprobs.
        """
        greedy_text: str | None = None
        beam_text: str | None = None
        out: dict[str, str] = {}
        for label, kind, weight in plan:
            if kind == "greedy":
                if greedy_text is None:
                    greedy_text = self.greedy.forward(logprobs)
                out[label] = greedy_text
            elif kind == "beam_no_hotwords":
                if beam_text is None:
                    beam_text = self.beam.decode(logprobs, beam_width=self.beam_width)
                out[label] = beam_text
            elif kind == "canonical":
                out[label] = self.beam.decode(
                    logprobs,
                    beam_width=self.beam_width,
                    hotwords=list(hotwords) if hotwords else None,
                    hotword_weight=float(weight),
                )
            else:  # pragma: no cover - guarded by build_mic_config_plan
                raise ValueError(f"unknown config kind: {kind!r}")
        return out

    def decode_all(
        self,
        logprobs: np.ndarray,
        hotwords: Iterable[str] | None,
        hotword_weight: float,
    ) -> dict[str, str]:
        """Legacy trio (greedy / beam / beam+hotwords@w) via the generic plan."""
        plan = [
            ("greedy", "greedy", None),
            ("beam_kenlm", "beam_no_hotwords", None),
            ("beam_kenlm_hotwords", "canonical", float(hotword_weight)),
        ]
        return self.decode_plan(logprobs, plan, hotwords or ())


def load_acoustic_model(model_path: str | None = None, num_threads: int | None = None) -> StreamingCTCModel:
    """Official ONNX acoustic model; hf_hub_download caches the artifact.

    ``num_threads`` (benchmark only) sets ONNX Runtime intra/inter-op threads
    explicitly; ``None`` keeps the library default. The official wrapper is used
    unchanged — only the InferenceSession options differ.
    """
    if not model_path:
        model_path = StreamingCTCModel.download_from_hugging_face()
    if num_threads is None:
        return StreamingCTCModel.from_local(model_path)
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.intra_op_num_threads = int(num_threads)
    options.inter_op_num_threads = int(num_threads)
    return StreamingCTCModel(ort.InferenceSession(str(model_path), sess_options=options))


def resolve_model_path(model_path: str | None = None) -> str:
    """Local model.onnx path, downloading the official HF artifact if needed."""
    if model_path:
        return str(model_path)
    return StreamingCTCModel.download_from_hugging_face()


def load_beam_decoder(
    kenlm_path: str | None = None,
    alpha: float = 0.4,
    beta: float = 0.9,
) -> BeamSearchDecoderCTC:
    """Official T-one kenlm.bin (HF-hub cached) or an explicitly given file."""
    if not kenlm_path:
        from huggingface_hub import hf_hub_download

        kenlm_path = hf_hub_download("t-tech/T-one", "kenlm.bin")
    return build_ctcdecoder(
        labels=list(LABELS),
        kenlm_model_path=str(kenlm_path),
        alpha=alpha,  # official T-one values (tone/decoder.py from_local)
        beta=beta,
    )


def decode_with_scores(
    decoder: BeamSearchDecoderCTC,
    logprobs: np.ndarray,
    *,
    beam_width: int = BEAM_WIDTH,
    hotwords: Iterable[str] | None = None,
    hotword_weight: float = 0.0,
) -> tuple[str, float | None, float | None]:
    """Beam decode identical to ``decoder.decode`` but also returning scores.

    ``pyctcdecode``'s ``decode`` is exactly
    ``decode_beams(..., prune_history=True)[0][0]``; this helper performs the same
    call and additionally returns the winning beam's ``logit_score`` (acoustic)
    and ``combined_score`` (KenLM + hotword adjusted), so the benchmark can store
    a confidence without a second decode pass.
    """
    beams = decoder.decode_beams(
        logprobs,
        beam_width=beam_width,
        hotwords=list(hotwords) if hotwords else None,
        hotword_weight=float(hotword_weight),
        prune_history=True,  # same as BeamSearchDecoderCTC.decode
    )
    if not beams:
        return "", None, None
    text, _lm_state, _word_frames, logit_score, combined_score = beams[0]
    return text, float(logit_score), float(combined_score)


def collect_phrase_logprobs(
    model: StreamingCTCModel,
    audio: np.ndarray,
) -> tuple[list[np.ndarray], list[tuple[int, int]]]:
    """Run the acoustic model ONCE over full audio; return per-phrase logprobs.

    Mirrors StreamingCTCPipeline.forward_offline (left/right 300ms padding,
    2400-sample chunks, official logprob splitter) but returns the raw
    per-phrase log-probability matrices instead of decoding them.
    """
    if audio.dtype != np.int32:
        audio = audio.astype(np.int32)
    padded = np.pad(audio, (PIPELINE_PADDING, PIPELINE_PADDING))
    padded = np.pad(padded, (0, -len(padded) % model.AUDIO_CHUNK_SAMPLES))
    chunks = np.split(padded, len(padded) // model.AUDIO_CHUNK_SAMPLES)

    splitter = StreamingLogprobSplitter()
    model_state = None
    splitter_state = None
    phrase_logprobs: list[np.ndarray] = []
    phrase_frames: list[tuple[int, int]] = []
    for i, chunk in enumerate(chunks):
        logprobs, model_state_next = model.forward(chunk[None, :, None], model_state)
        phrases, splitter_state_next = splitter.forward(
            logprobs[0],
            splitter_state,
            is_last=i == len(chunks) - 1,
        )
        for p in phrases:
            phrase_logprobs.append(np.asarray(p.logprobs, dtype=np.float32))
            phrase_frames.append((int(p.start_frame), int(p.end_frame)))
        model_state, splitter_state = model_state_next, splitter_state_next
    return phrase_logprobs, phrase_frames

