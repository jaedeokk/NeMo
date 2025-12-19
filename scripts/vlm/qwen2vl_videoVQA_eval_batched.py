#!/usr/bin/env python3
"""
VideoVQA eval for NeMo Qwen2-VL with:
  (1) video_id -> video{video_id}.mp4 resolution
  (2) per-video caching of video tensors (pixel_values_videos, video_grid_thw)
  (3) per-video question batching for greedy decoding

Dataset JSON format:
[
  {"answer":"regard","category_id":10,"id":170859,"question":"what is the video doing?","video_id":7010},
  ...
]

Metric:
- Exact-match accuracy (since GT answer is a single word string)
"""

import argparse
import json
import os
import re
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import torch
from tqdm import tqdm
from transformers import AutoProcessor
from qwen_vl_utils import process_vision_info

import nemo.lightning as nl
from nemo.collections.vlm import Qwen2VLModel
from nemo.collections import llm, vlm
from nemo.collections.vlm.qwen2vl.data.multimodal_tokens import VIDEO_TOKEN_INDEX

HF_VIDEO_PLACEHOLDER_ID = 151656  # HF processor placeholder for <video>


# -------------------------
# Normalisation / metric
# -------------------------

_punct = re.compile(r"[^\w\s]")
_ws = re.compile(r"\s+")

def normalize_answer(s: str) -> str:
    s = s.strip().lower()
    s = _punct.sub(" ", s)
    s = _ws.sub(" ", s).strip()
    return s

def exact_match_accuracy(pred: str, gt: str) -> float:
    return 1.0 if normalize_answer(pred) == normalize_answer(gt) else 0.0


# -------------------------
# Model build / load
# -------------------------

def build_finetune_arch(
    model_size: str,
    tokenizer,
    dct: bool = False,
    max_sequence_length: int = 4096,
    projector_type: str = "mcore_mlp",
) -> Qwen2VLModel:
    size_info = {
        "2B": {"hf_model_name": "Qwen/Qwen2-VL-2B-Instruct", "llmconfig_class": llm.Qwen2Config1P5B},
        "7B": {"hf_model_name": "Qwen/Qwen2-VL-7B-Instruct", "llmconfig_class": llm.Qwen2Config7B},
    }
    if model_size not in size_info:
        raise ValueError(f"Unsupported model_size={model_size}. Choose from {list(size_info.keys())}")

    llm_config_class = size_info[model_size]["llmconfig_class"]
    language_transformer_config = llm_config_class(seq_length=max_sequence_length)

    vision_in_ch = 96 if dct else 3
    vision_transformer_config = vlm.Qwen2VLVisionConfig(in_channels=vision_in_ch)

    vision_projection_config = vlm.MultimodalProjectorConfig(
        projector_type=projector_type,
        input_size=vision_transformer_config.ffn_hidden_size,
        hidden_size=language_transformer_config.hidden_size,
        ffn_hidden_size=vision_transformer_config.ffn_hidden_size,
    )

    qwen2vl_config = vlm.Qwen2VLConfig(
        language_transformer_config=language_transformer_config,
        vision_transformer_config=vision_transformer_config,
        vision_projection_config=vision_projection_config,
        language_model_from_pretrained=None,
        freeze_language_model=False,
        freeze_vision_model=True,
    )
    return Qwen2VLModel(qwen2vl_config, tokenizer=tokenizer)


def load_model_and_processor(args):
    size_info = {
        "2B": {"hf_model_name": "Qwen/Qwen2-VL-2B-Instruct"},
        "7B": {"hf_model_name": "Qwen/Qwen2-VL-7B-Instruct"},
    }
    hf_model_name = size_info[args.model_size]["hf_model_name"]

    strategy = nl.MegatronStrategy(
        tensor_model_parallel_size=args.tp_size,
        pipeline_model_parallel_size=args.pp_size,
        ckpt_include_optimizer=False,
    )
    trainer = nl.Trainer(
        devices=args.tp_size * args.pp_size,
        max_steps=1,
        accelerator="gpu",
        strategy=strategy,
        plugins=nl.MegatronMixedPrecision(precision="bf16-mixed"),
        val_check_interval=1,
        limit_val_batches=1,
    )

    processor = AutoProcessor.from_pretrained(
        hf_model_name,
        min_pixels=args.min_pixels,
        max_pixels=args.max_pixels,
    )
    hf_tokenizer = processor.tokenizer

    fabric = trainer.to_fabric()
    if args.load_from_hf:
        model = fabric.import_model("hf://" + hf_model_name, Qwen2VLModel)
    else:
        if args.local_model_path is None:
            raise ValueError("--local_model_path is required when not using --load_from_hf")
        model = build_finetune_arch(
            model_size=args.model_size,
            tokenizer=hf_tokenizer,
            dct=False,
            max_sequence_length=args.max_seq_len,
            projector_type=args.projector_type,
        )
        model = fabric.load_model(args.local_model_path, model)

    model = model.module.cuda()
    model.eval()
    return model, processor, hf_tokenizer


# -------------------------
# Dataset helpers
# -------------------------

def load_json_list(path: str) -> List[Dict[str, Any]]:
    with open(path, "r") as f:
        obj = json.load(f)
    if not isinstance(obj, list):
        raise ValueError("Expected a JSON list.")
    return obj

def extract_id(item: Dict[str, Any], fallback_idx: int) -> str:
    if "id" in item and item["id"] is not None:
        return str(item["id"])
    if "question_id" in item and item["question_id"] is not None:
        return str(item["question_id"])
    return str(fallback_idx)

def extract_question(item: Dict[str, Any]) -> str:
    if "question" in item and item["question"]:
        return str(item["question"])
    raise KeyError("Expected key `question` in each JSON item.")

def extract_gt_answer(item: Dict[str, Any]) -> Optional[str]:
    if "answer" in item and item["answer"] is not None:
        return str(item["answer"])
    return None

def resolve_video_path_from_video_id(item: Dict[str, Any], video_root: Optional[str], pattern: str) -> str:
    if "video_id" not in item:
        raise KeyError("Expected key `video_id` in each JSON item.")
    vid = item["video_id"]
    vid_str = str(int(vid)) if isinstance(vid, (int, float)) or (isinstance(vid, str) and vid.isdigit()) else str(vid)
    filename = pattern.format(video_id=vid_str)

    if filename.startswith("http://") or filename.startswith("https://"):
        return filename
    if os.path.isabs(filename) and os.path.exists(filename):
        return filename
    if os.path.exists(filename):
        return filename
    if video_root is None:
        return filename
    return os.path.join(video_root, filename)


# -------------------------
# Video caching
# -------------------------

@torch.no_grad()
def get_video_tensors(processor, video_path: str) -> Tuple[torch.Tensor, torch.Tensor, int]:
    # 1) video load (once)
    messages = [{
        "role": "user",
        "content": [
            {"type": "video", "video": video_path},
            {"type": "text", "text": " "}  # dummy text (but MUST exist)
        ],
    }]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    _, video_inputs = process_vision_info(messages)

    # 2) IMPORTANT: call processor with BOTH text and videos
    out = processor(
        text=[text],
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )

    # 3) count expanded video placeholder tokens inside input_ids
    input_ids = out["input_ids"].clone()
    input_ids[input_ids == HF_VIDEO_PLACEHOLDER_ID] = VIDEO_TOKEN_INDEX
    n_video_tokens = int((input_ids == VIDEO_TOKEN_INDEX).sum().item())

    pixel_values_videos = out["pixel_values_videos"].clone().to("cuda")
    video_grid_thw = out["video_grid_thw"].clone().to("cuda")

    merge_factor = 8  # Qwen2-VL typical: temporal 2 x spatial 2 x spatial 2
    n_video_tokens = n_video_tokens // merge_factor

    return pixel_values_videos, video_grid_thw, n_video_tokens




# -------------------------
# Batched greedy decoding
# -------------------------

def _build_texts(processor, questions: List[str], one_word: bool = True) -> List[str]:
    texts = []
    for q in questions:
        if one_word:
            prompt = (
                "You are a video question answering assistant.\n"
                "Watch the video and answer the question with exactly ONE word.\n"
                "Do not use punctuation. Do not use multiple words.\n\n"
                f"Question: {q}\n"
                "One-word answer:"
            )
        else:
            prompt = f"Question: {q}\nAnswer:"
        messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
        texts.append(processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))
    return texts


@torch.no_grad()
def generate_batch_with_cached_video(
    model,
    processor,
    hf_tokenizer,
    questions: List[str],
    pixel_values_videos: torch.Tensor,
    video_grid_thw: torch.Tensor,
    n_video_tokens: int,
    osl: int,
    one_word: bool = True,
) -> List[str]:
    """
    Batched greedy decoding while reusing cached video tensors.

    IMPORTANT:
    NeMo Qwen2-VL expects the number of VIDEO_TOKEN_INDEX tokens in input_ids to match
    the number of visual features produced from (pixel_values_videos, video_grid_thw).
    Since we avoid calling the processor with videos here, we manually prefix the required
    number of VIDEO_TOKEN_INDEX tokens (= T*H*W from video_grid_thw).
    """
    # Build chat-templated texts (TEXT ONLY)
    texts = _build_texts(processor, questions, one_word=one_word)

    text_inputs = processor(
        text=texts,
        padding=True,
        return_tensors="pt",
    )

    text_input_ids = text_inputs["input_ids"].clone().to("cuda")         # (B, L)
    text_attention_mask = text_inputs["attention_mask"].clone().to("cuda")

    # Ensure (video_prefix + text_prompt) fits within the model's max sequence length.
    # For HF-imported NeMo Qwen2-VL this is typically 4096.
    max_total_len = getattr(processor, "model_max_length", None)
    if max_total_len is None or max_total_len > 100000:
        max_total_len = 4096
    # Reserve some room for generation (osl) if desired; keep it simple: fit the prompt only.
    max_prefix = max_total_len - int(text_input_ids.shape[1])
    if max_prefix < 0:
        # Text alone exceeds max; hard truncate text.
        text_input_ids = text_input_ids[:, :max_total_len]
        text_attention_mask = text_attention_mask[:, :max_total_len]
        max_prefix = 0
    if n_video_tokens > max_prefix:
        # Truncate video tokens to fit; this drops some visual tokens but prevents crashes.
        n_video_tokens = max_prefix

    # Manual prefix of video tokens
    B, L = text_input_ids.shape
    video_prefix = torch.full((B, n_video_tokens), VIDEO_TOKEN_INDEX, device="cuda", dtype=text_input_ids.dtype)
    input_ids = torch.cat([video_prefix, text_input_ids], dim=1)         # (B, n+L)

    prefix_mask = torch.ones((B, n_video_tokens), device="cuda", dtype=text_attention_mask.dtype)
    attention_mask = torch.cat([prefix_mask, text_attention_mask], dim=1)

    generated_ids = input_ids
    generated_mask = attention_mask

    finished = torch.zeros(B, dtype=torch.bool, device="cuda")
    eos_id = hf_tokenizer.eos_token_id

    for _ in range(osl):
        logits = model(
            input_ids=generated_ids,
            attention_mask=generated_mask,
            position_ids=None,
            pixel_values_videos=pixel_values_videos,
            video_grid_thw=video_grid_thw,
        )
        next_token = torch.argmax(logits[:, -1], dim=-1)  # (B,)

        next_token = torch.where(finished, torch.full_like(next_token, eos_id), next_token)

        generated_ids = torch.cat([generated_ids, next_token.unsqueeze(1)], dim=1)
        generated_mask = torch.cat(
            [generated_mask, torch.ones((B, 1), device="cuda", dtype=generated_mask.dtype)],
            dim=1,
        )

        finished = finished | (next_token == eos_id)
        if finished.all():
            break

    generated_ids[generated_ids < 0] = 0

    # Decode only generated part (exclude prefix + prompt)
    prompt_len = input_ids.shape[1]
    out = generated_ids[:, prompt_len:]
    answers = processor.batch_decode(out, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    return [a.strip() for a in answers]


def chunked(seq, n: int):
    for i in range(0, len(seq), n):
        yield seq[i:i+n]


# -------------------------
# Main eval loop
# -------------------------

def run_videovqa_eval(args):
    model, processor, hf_tokenizer = load_model_and_processor(args)
    data = load_json_list(args.json_path)

    preds: List[Dict[str, Any]] = []
    accs: List[float] = []
    missing = 0

    grouped = defaultdict(list)  # video_id -> list[(global_idx, item)]
    for idx, item in enumerate(data):
        grouped[item["video_id"]].append((idx, item))

    video_cache: Dict[Any, Tuple[torch.Tensor, torch.Tensor, int]] = {}

    for video_id, items in tqdm(grouped.items(), desc="VideoVQA (grouped)"):
        _, first_item = items[0]
        video_path = resolve_video_path_from_video_id(first_item, args.video_root, args.video_pattern)

        if not (video_path.startswith("http://") or video_path.startswith("https://")) and not os.path.exists(video_path):
            for idx, item in items:
                qid = extract_id(item, idx)
                preds.append({"id": qid, "answer": "", "error": f"missing video: {video_path}"})
                missing += 1
            continue

        if video_id not in video_cache:
            video_cache[video_id] = get_video_tensors(processor, video_path)

        pixel_values_videos, video_grid_thw, n_video_tokens = video_cache[video_id]

        bs = max(1, args.text_batch_size)
        for sub in chunked(items, bs):
            idxs = [i for i, _ in sub]
            batch_items = [it for _, it in sub]
            qids = [extract_id(it, i) for i, it in zip(idxs, batch_items)]
            questions = [extract_question(it) for it in batch_items]

            answers = generate_batch_with_cached_video(
                model=model,
                processor=processor,
                hf_tokenizer=hf_tokenizer,
                questions=questions,
                pixel_values_videos=pixel_values_videos,
                video_grid_thw=video_grid_thw,
                n_video_tokens=n_video_tokens,
                osl=args.osl,
                one_word=not args.free_form,
            )

            for i, it, qid, q, ans in zip(idxs, batch_items, qids, questions, answers):
                out = {"id": qid, "answer": ans}
                if args.save_question:
                    out["question"] = q
                if args.save_video:
                    out["video"] = video_path
                preds.append(out)

                gt = extract_gt_answer(it)
                if gt is not None:
                    accs.append(exact_match_accuracy(ans, gt))

    with open(args.output_json, "w") as f:
        json.dump(preds, f, ensure_ascii=False, indent=2)

    print(f"Saved predictions to {args.output_json}")
    if missing:
        print(f"[WARN] missing videos: {missing}")

    if accs:
        mean_acc = sum(accs) / len(accs)
        print(f"Exact-match accuracy (mean over {len(accs)} samples): {mean_acc:.4f}")
        if args.output_metrics_json:
            with open(args.output_metrics_json, "w") as f:
                json.dump({"mean_exact_match": mean_acc, "num_scored": len(accs), "num_total": len(data)}, f, indent=2)
            print(f"Saved metrics to {args.output_metrics_json}")
    else:
        print("No ground-truth `answer` field found; accuracy not computed.")


def parse_args():
    p = argparse.ArgumentParser(description="VideoVQA eval (video_id caching + text batching) for NeMo Qwen2-VL")

    p.add_argument("--load_from_hf", action="store_true", help="Load Qwen2-VL from Hugging Face.")
    p.add_argument("--local_model_path", type=str, default=None, help="NeMo .nemo / checkpoint path (if not HF).")

    p.add_argument("--json_path", type=str, required=True, help="Path to annotations JSON (a list).")
    p.add_argument("--video_root", type=str, default=None, help="Root folder that contains mp4 files.")
    p.add_argument("--video_pattern", type=str, default="video{video_id}.mp4",
                   help='Filename pattern resolved from video_id. Use "{video_id}" placeholder.')

    p.add_argument("--output_json", type=str, default="pred_videovqa.json", help="Where to save predictions.")
    p.add_argument("--output_metrics_json", type=str, default=None, help="Optional: save metrics JSON.")

    p.add_argument("--model_size", type=str, default="2B", choices=["2B", "7B"], help="Qwen2-VL size.")
    p.add_argument("--tp_size", type=int, default=1, help="Tensor parallel size.")
    p.add_argument("--pp_size", type=int, default=1, help="Pipeline parallel size.")

    p.add_argument("--osl", type=int, default=30, help="Max generated tokens.")
    p.add_argument("--text_batch_size", type=int, default=8, help="Questions per batch for a given video_id.")
    p.add_argument("--free_form", action="store_true", help="Do not enforce one-word prompt style.")

    p.add_argument("--min_pixels", type=int, default=64 * 28 * 28, help="AutoProcessor min_pixels.")
    p.add_argument("--max_pixels", type=int, default=256 * 28 * 28, help="AutoProcessor max_pixels.")
    p.add_argument("--max_seq_len", type=int, default=4096, help="LLM seq_len when building local NeMo arch.")
    p.add_argument("--projector_type", type=str, default="mcore_mlp", help="NeMo projector type for local arch.")

    p.add_argument("--save_question", action="store_true", help="Include question in output json.")
    p.add_argument("--save_video", action="store_true", help="Include resolved video path in output json.")

    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_videovqa_eval(args)
