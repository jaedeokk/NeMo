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

from nemo.collections.vlm.qwen2vl.data.multimodal_tokens import VIDEO_TOKEN_INDEX

HF_VIDEO_PLACEHOLDER_ID = 151656  


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
    # min_pixels = 16 * 28 * 28
    # max_pixels = 64 * 28 * 28
    min_pixels = 64 * 28 * 28
    max_pixels = 256 * 28 * 28
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

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": args.image_url,
                    # "max_pixels": 360 * 420,
                    # "fps": 1.0,
                },
                {"type": "text", "text": "Describe the video"},
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
        input_ids[input_ids == HF_VIDEO_PLACEHOLDER_ID] = VIDEO_TOKEN_INDEX
        video_grid_thw = inputs['video_grid_thw'].clone().to("cuda")
        pixel_values_videos = inputs['pixel_values_videos'].clone().to("cuda")
        attention_mask = inputs["attention_mask"].clone().to("cuda")  
        generated_ids = input_ids
        generated_mask = attention_mask
        for _ in range(args.osl):
            output = model(
                input_ids=generated_ids,
                attention_mask=generated_mask,   
                position_ids=None,
                pixel_values_videos=pixel_values_videos,
                video_grid_thw=video_grid_thw,
            )
            next_token_ids = torch.argmax(output[:, -1], dim=-1, keepdim=True)
            generated_ids = torch.cat([generated_ids, next_token_ids], dim=-1)

            generated_mask = torch.cat(
                [generated_mask, torch.ones((generated_mask.size(0), 1), device=generated_mask.device, dtype=generated_mask.dtype)],
                dim=1
            )

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
        "--image_url",
        type=str,
        default="https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg",
        help="URL of the image to use for inference.",
    )
    parser.add_argument('--osl', type=int, default=30, help='output seq length')
    parser.add_argument('--tp_size', type=int, default=1, help='tensor parallel size')
    parser.add_argument('--pp_size', type=int, default=1, help='pipeline parallel size')
    args = parser.parse_args()

    main(args)