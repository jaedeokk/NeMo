# DCT Tokenizer Experiment

This experiment implements fine-tuning of the Qwen2-VL-2B-Instruct model using the Cambrian737k dataset with DCT (Discrete Cosine Transform) tokenization for video processing. The setup is optimized for Qwen2-VL models and includes custom modifications to the NeMo framework for compatibility.

## Setup Instructions

### 1. Environment Setup

Start the Docker environment:

```bash
bash ./make-and-run-docker.sh
```

Inside the Docker container, install the required Transformers version:

```bash
# Pin Transformers to v4.51.3 since NeMo framework does not support Qwen2-VL under the latest version
pip install "transformers==4.51.3"
```


For processing discrete cosine transform (dct), install the torch-dct:
```bash
# Tensor dct module
pip install torch-dct
```


```bash
cd experiments/dct_tokenizer
```

### 2. Model Preparation

The `prepare_nemo_checkpoint.py` script downloads the pretrained model from HuggingFace and converts it to the NeMo2 format.
```bash
python prepare_nemo_checkpoint.py -o /models/Qwen2-VL-2B-Instruct-nemo
```

### 3. Dataset Preparation

This experiment uses the [Cambrian737k dataset](https://huggingface.co/datasets/LanguageBind/Cambrian737k), a large-scale multimodal dataset containing image-text conversations.

#### Download the Dataset

```bash
cd /datasets/
git clone https://huggingface.co/datasets/LanguageBind/Cambrian737k
cd -
```

**Expected dataset structure:**
```
/datasets/
└── Cambrian737k/
    ├── Cambrian737k/
    │   ├── cambrian737k.json         # Metadata file
    │   ├── ai2d.tar
    │   ├── chartqa.tar
    │   └── ... (many tar files)
    ├── lmmseval/
    └── mmvp_cache/
```

#### Convert to WebDataset Format

The raw dataset needs to be converted to WebDataset format for efficient training. The data preparation script
- Filters the dataset by checking if image files exist
- Converts images to JPEG format and stores as binary data
- Creates WebDataset shards with conversation data
- Generates metadata for efficient data loading

```bash
# Convert dataset to WebDataset format
python data_preparation.py --data-dir /datasets/Cambrian737k/Cambrian737k --output-dir /datasets/Cambrian737k-wds
### Expected log messges below.
# 0 conversations will be saced
# Filtering done and saved to /datasets/Cambrian737k/Cambrian737k/Cambrian737k_filtered.json.
# # writing /datasets/Cambrian737k-wds/pretrain-0.tar 0 0.0 GB 0
# Processing images: 100%|█████████████████████████| 696248/696248 [08:42<00:00, 1331.46it/s]
# Dataset successfully converted to the webdataset format.
```

We now prepare dataset for Energon (NeMo's data loading system).
```bash
energon prepare /datasets/Cambrian737k-wds
### During the preparation process, you'll need to make several selections. Here's how we configured ours. TL;DR: 8,1,1 / y / 11
# Found 70 tar files in total. The first and last ones are:
# - pretrain-0.tar
# - pretrain-9.tar
# If you want to exclude some of them, cancel with ctrl+c and specify an exclude filter in the command line.
# Please enter a desired train/val/test split like "0.5, 0.2, 0.3" or "8,1,1": 8,1,1
# Indexing shards  [####################################]  70/70
# Sample 0, keys:
#  - jpg.jpg
#  - jpg.json
# Sample 1, keys:
#  - jpg.jpg
#  - jpg.json
# Found the following part types in the dataset: png.json, jpg.json, jpg.jpg, png.jpg
# Do you want to create a dataset.yaml interactively? [Y/n]: y
# The following sample types are available:
# 0. CaptioningSample
# 1. ImageClassificationSample
# 2. ImageSample
# 3. InterleavedSample
# 4. MultiChoiceVQASample
# 5. OCRSample
# 6. Sample
# 7. SimilarityInterleavedSample
# 8. TextSample
# 9. VQASample
# 10. VidQASample
# 11. Crude sample (plain dict for cooking)
# Please enter a number to choose a class: 11
# CrudeWebdataset does not need a field map. You will need to provide a `Cooker` for your dataset samples in your `TaskEncoder`.
# Furthermore, you might want to add `subflavors` in your meta dataset specification.
# Done
```

### 4. Launch Training

The training is configured through `qwen_launch.sh` with the following key parameters.
Launch fine-tuning process:
```bash
bash qwen_launch.sh
```
# Bug Report
First, to reproduce this bug, the path of the image to be tested (if not specified, removing the flag will make it work with the default URL) and the path of the model saved as .distcp are required. This saving was done by training 4.training above with PP and TP greater than or equal to 2 (in my case, 2 each).

---
## Case 1: Huggingface loading
If the following code is executed first, the model will be directly retrieved from Hugging Face and loaded. However, in this case as well, after the model is loaded, it does not work if TP/PP are set to values greater than 2.
```bash
bash eval_qwen_default.sh
```
In this case, the error message comes out as follows.
### Error Message

```bash
[rank3]
 File "/opt/megatron-lm/megatron/core/transformer/transformer_block.py", line 544, in forward
hidden_states = make_viewless_tensor(inp=hidden_states, requires_grad=True, keep_graph=True)
 File "/opt/megatron-lm/megatron/core/utils.py", line 595, in make_viewless_tensor  
 if inp._base is None:                                                                                             
AttributeError: 'NoneType' object has no attribute '_base'     

[rank0]
File "/workspace/scripts/vlm/qwen2vl_generate.py", line 127, in main 
 generated_ids = torch.cat([generated_ids, next_token_ids], dim=-1)
RuntimeError: Sizes of tensors must match except in dimension 1. Expected size 1 but got size 4096 for tensor number 1 in the list. 
```
## Case 2: Local checkpoint loading
`CKPT_DIR` should be filled first.

```bash
bash eval_qwen.sh
```
In this case, the error message appears as follows regardless of tensor parallelisms.
```bash
File "/workspace/scripts/vlm/qwen2vl_generate.py", line 79, in main 
    model = fabric.load_model(args.local_model_path, model)  
File "/workspace/nemo/lightning/fabric/fabric.py", line 86, in load_model  
     self.load(path, {"state_dict": dist_model})
     ..
     ..
File "/opt/megatron-lm/megatron/core/dist_checkpointing/strategies/torch.py", line 558, in _validate_global_shapes
     raise KeyError
     KeyError: "module.vision_projection.0.weight from model not in state dict: ['module.language_model.decoder.final_layernorm._extra_state/shard_0_1', 'module.language_model.decoder.final_layernorm.weight', 'module.language_model.decoder.layers.mlp.linear_fc1._extra_state/shard_0_28', 'module.language_model.decoder.layers.mlp.linear_fc1._extra_state/shar..
..
```


## Training Output

TBA

## Customization

### Modifying Training Parameters

Edit `qwen_launch.sh` to adjust:
- `NUM_PROC_PER_NODE`: Number of processes per node
- `DEVICES`: Number of GPUs to use
- `MBS`/`GBS`: Micro and global batch sizes
- `MINPIXELS`/`MAXPIXELS`: Image resolution range
- `EXPERIMENT_NAME`: Name for this training run


## Troubleshooting

### Common Issues

1. **Transformers Version Conflict**: Ensure you're using `transformers==4.51.3`
2. **CUDA Out of Memory**: Reduce batch size (`MBS`/`GBS`) or use fewer devices
3. **Dataset Loading Issues**: Verify the WebDataset conversion completed successfully
4. **Model Loading Errors**: Check that the model path is correct and files are downloaded

## References

- [Qwen2-VL Model](https://huggingface.co/Qwen/Qwen2-VL-2B-Instruct)
- [Cambrian737k Dataset](https://huggingface.co/datasets/LanguageBind/Cambrian737k)
- [NeMo Framework](https://github.com/NVIDIA/NeMo)
- [WebDataset Documentation](https://webdataset.github.io/webdataset/)
