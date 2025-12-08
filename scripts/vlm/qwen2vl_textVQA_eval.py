# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Example:
  pip install qwen_vl_utils && python scripts/vlm/qwen2vl_generate.py --load_from_hf
"""

import argparse

import requests
import torch
from PIL import Image
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor

import nemo.lightning as nl
from nemo.collections.vlm import Qwen2VLConfig2B, Qwen2VLModel
from nemo.utils import logging

from nemo.collections import llm, vlm
from nemo.collections.vlm import MultimodalProjectorConfig, Qwen2VLVisionConfig, Qwen2VLConfig

from io import BytesIO
import os
from tqdm import tqdm

def build_messages(image_path_or_url: str, question: str):
    """
    Construct the Qwen2VL message format for a single VQA sample.
    
    Parameters:
        image_path_or_url (str): Local image path or image URL.
        question (str): TextVQA question string.
    
    Returns:
        list: A 'messages' list compatible with Qwen2VL processor.
    """
    return [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image_path_or_url,
                },
                {
                    "type": "text",
                    "text": question,
                },
            ],
        }
    ]


def generate_one_answer(model, processor, hf_tokenizer, inputs, osl: int):
    """지금 네 greedy loop를 함수로 분리한 버전."""
    import torch

    with torch.no_grad():
        input_ids = inputs['input_ids'].clone().to("cuda")
        # special image token → NeMo용 ID
        input_ids[input_ids == 151655] = -200

        image_grid_thw = inputs['image_grid_thw'].clone().to("cuda")
        pixel_values = inputs['pixel_values'].clone().to("cuda")

        generated_ids = input_ids
        for _ in range(osl):
            output = model(
                pixel_values=pixel_values,
                input_ids=input_ids,
                position_ids=None,
                attention_mask=None,
                image_grid_thw=image_grid_thw,
            )

            next_token_ids = torch.argmax(output[:, -1], dim=-1, keepdim=True)
            generated_ids = torch.cat([generated_ids, next_token_ids], dim=-1)
            input_ids = generated_ids

            if next_token_ids.item() == hf_tokenizer.eos_token_id:
                break

        generated_ids[generated_ids < 0] = 0
        trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
        texts = processor.batch_decode(
            trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        return texts[0].strip()

def run_textvqa_eval(args, processor, hf_tokenizer, model):
    import json
    from qwen_vl_utils import process_vision_info

    with open(args.json_path, "r") as f:
        ann = json.load(f)

    data = ann["data"]
    results = []

    for item in tqdm(data, desc="TextVQA val"):
        qid = item["question_id"]
        question = item["question"]
        image_id = item["image_id"]

        image_path = os.path.join(args.image_folder, f"{image_id}.jpg")
        if not os.path.exists(image_path):
            print(f"[WARN] missing image: {image_path}")
            continue

        messages = build_messages(image_path, question)

        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)

        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )

        answer = generate_one_answer(model, processor, hf_tokenizer, inputs, args.osl)

        results.append({
            "question_id": qid,
            "answer": answer,
        })

    with open(args.output_json, "w") as f:
        json.dump(results, f)
    print(f"Saved predictions to {args.output_json}")


def build_finetune_arch(tokenizer, dct=False, max_sequence_length=4096, projector_type="mcore_mlp"):
    SIZE_INFO_MAP = {
        "2B": {"hf_model_name": "Qwen/Qwen2-VL-2B-Instruct", "llmconfig_class": llm.Qwen2Config1P5B},
        "7B": {"hf_model_name": "Qwen/Qwen2-VL-7B-Instruct", "llmconfig_class": llm.Qwen2Config7B},
    }
    model_size = "2B"
    _, llm_config_class = (
        SIZE_INFO_MAP[model_size]["hf_model_name"],
        SIZE_INFO_MAP[model_size]["llmconfig_class"],
    )

    language_transformer_config = llm_config_class(
        seq_length=max_sequence_length,
    )

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
        language_model_from_pretrained=None,  #
        freeze_language_model=False,
        freeze_vision_model=True,
    )

    return Qwen2VLModel(qwen2vl_config, tokenizer=tokenizer)



def load_image(image_url: str) -> Image.Image:
    # pylint: disable=C0115,C0116
    try:
        response = requests.get(image_url, stream=True)
        response.raise_for_status()
        image = Image.open(response.raw)
        return image
    except requests.exceptions.RequestException as e:
        print(f"Error loading image from {image_url}: {e}")
        return None


def main(args) -> None:
    # pylint: disable=C0115,C0116
    strategy = nl.MegatronStrategy(
        tensor_model_parallel_size=args.tp_size,
        pipeline_model_parallel_size=args.pp_size,
        ckpt_include_optimizer=False,
    )
    trainer = nl.Trainer(
        devices=args.tp_size * args.pp_size,
        max_steps=1000,
        accelerator="gpu",
        strategy=strategy,
        plugins=nl.MegatronMixedPrecision(precision="bf16-mixed"),
        val_check_interval=1000,
        limit_val_batches=50,
    )

    # Tokenize the input texts
    # The default range for the number of visual tokens per image in the model is 4-16384. You can set min_pixels
    # and max_pixels according to your needs, such as a token count range of 256-1280, to balance speed and memory
    # usage.
    min_pixels = 16 * 28 * 28
    max_pixels = 64 * 28 * 28
    processor = AutoProcessor.from_pretrained(
        "Qwen/Qwen2-VL-2B-Instruct", min_pixels=min_pixels, max_pixels=max_pixels
    )
    hf_tokenizer = processor.tokenizer

    fabric = trainer.to_fabric()
    # Decide whether to import or load the model based on the input arguments
    if args.load_from_hf:
        model = fabric.import_model("hf://Qwen/Qwen2-VL-2B-Instruct", Qwen2VLModel)
    else:
        # model = Qwen2VLModel(Qwen2VLConfig2B(), tokenizer=hf_tokenizer)
        model = build_finetune_arch(
        tokenizer=hf_tokenizer,
        dct=False,                  #
        max_sequence_length=min_pixels,   
        projector_type="mcore_mlp", # 
        )
        model = fabric.load_model(args.local_model_path, model)
    model = model.module.cuda()
    model.eval()
    if args.json_path is not None and args.image_folder is not None:
        run_textvqa_eval(args, processor, hf_tokenizer, model)
        return

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": args.image_url,
                },
                {"type": "text", "text": "Describe this image."},
            ],
        }
    ]

    # Preparation for inference
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)

    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )

    with torch.no_grad():
        input_ids = inputs['input_ids'].clone().to("cuda")
        # convert special tokens to nemo image ID
        input_ids[input_ids == 151655] = -200
        image_grid_thw = inputs['image_grid_thw'].clone().to("cuda")
        pixel_values = inputs['pixel_values'].clone().to("cuda")

        # Greedy generation loop
        generated_ids = input_ids
        for _ in range(args.osl):
            output = model(
                pixel_values=pixel_values,
                input_ids=input_ids,
                position_ids=None,
                attention_mask=None,
                image_grid_thw=image_grid_thw,
            )

            next_token_ids = torch.argmax(output[:, -1], dim=-1, keepdim=True)
            generated_ids = torch.cat([generated_ids, next_token_ids], dim=-1)

            input_ids = generated_ids
            # If the generated token is the end of sequence token, stop generating
            if next_token_ids.item() == hf_tokenizer.eos_token_id:
                break

        generated_ids[generated_ids < 0] = 0
        generated_ids_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
        generated_texts = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

        logging.info("======== GENERATED TEXT OUTPUT ========")
        logging.info(f"{args.image_url}, \t\t{generated_texts}")
        logging.info("=======================================")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Qwen2VL Multimodal Inference")
    parser.add_argument(
        "--load_from_hf",
        action="store_true",
        help="Flag to indicate whether to load the model from Hugging Face hub.",
    )
    parser.add_argument(
        "--local_model_path",
        type=str,
        default=None,
        help="Local path to the model if not loading from Hugging Face.",
    )
    parser.add_argument(
        "--json_path",
        type=str,
        default="/datasets/TextVQA/val/TextVQA_0.5.1_val.json",
        help="path of the JSON file to use.",
    )
    parser.add_argument(
        "--image_folder",
        type=str,
        default="/datasets/TextVQA/val/train_images/",
        help="path of the local image files.",
    )
    parser.add_argument(
    "--output_json",
    type=str,
    default="pred_textvqa_val.json",
    help="Where to save predictions (question_id, answer).",
    )
    parser.add_argument('--osl', type=int, default=30, help='output seq length')
    parser.add_argument('--tp_size', type=int, default=1, help='tensor parallel size')
    parser.add_argument('--pp_size', type=int, default=1, help='pipeline parallel size')
    args = parser.parse_args()

    main(args)