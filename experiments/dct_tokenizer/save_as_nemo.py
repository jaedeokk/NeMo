from nemo.collections import vlm

model = vlm.Qwen2VLModel.load_from_checkpoint(
    checkpoint_path="./experiments_finetune/Qwen2VL_finetune_2B2_CP_MBS1_GBS4_seqpack--reduced_train_loss=3.2087-epoch=0-consumed_samples=8000.0/weights",  
    strict=False
)
model.save_to("qwen2vl_ft.nemo")
print("Saved to qwen2vl_ft.nemo")