import argparse
from pathlib import Path

from nemo.collections.llm import import_ckpt
from nemo.collections import vlm

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-p", "--path-or-model-id", type=Path,
        default='Qwen/Qwen2-VL-2B-Instruct',
        help="Path to the Hugging Face model directory or HF model ID."
    )
    parser.add_argument(
        "-o", "--output-path", type=Path,
        default=None,
        help="Path to save the converted NeMo checkpoint."
    )
    args = parser.parse_args()

    path_or_model_id = args.path_or_model_id
    if not path_or_model_id.exists():  # it's a model id
        path_or_model_id = f"hf://{path_or_model_id}"

    # Import the model and convert to NeMo 2.0 format
    import_ckpt(
        model=vlm.Qwen2VLModel(vlm.Qwen2VLConfig2B()),  # Model configuration
        source=path_or_model_id,
        output_path=args.output_path,
    )


if __name__ == '__main__':
    main()
