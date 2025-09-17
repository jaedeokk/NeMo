import torch
import torch.nn as nn

class MyFront(nn.Module):
    def __init__(self, in_dim: int,out_dim: int):
        super().__init__()
        # 예시: 패치 벡터 차원 보존형 선형층(여기만 학습)
        self.proj = nn.Conv2d(in_dim,out_dim,stride=1,padding=0,kernel_size=1)

    def forward(self, pixel_values: torch.Tensor, image_grid_thw: torch.Tensor):
        """
        pixel_values: (num_patches_total, dim)  # Qwen2VL 프로세서 출력
        image_grid_thw: (B, 3)  # 각 이미지별 (T,H,W)
        """
        # 여기에 DCT/필터 등 커스텀 연산을 넣어도 됨 (torch 연산으로!)
        x = self.proj(pixel_values)
        return x
