import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
from transformers import AutoImageProcessor, AutoModel


class DINORewardCalculator:
    def __init__(self, device="cuda" if torch.cuda.is_available() else "cpu", model_name="/root/autodl-tmp/checkpoints/dinov3/dinov3-vitl16-pretrain-lvd1689m"):
        """
        初始化 DINOv3 奖励计算器

        依赖:
          - transformers >= 4.56.0
          - 模型在 HuggingFace 上是 gated 访问,需要先在 model card 申请权限并登录

        :param model_name: HuggingFace 上的 DINOv3 模型权重地址
        """
        self.device = device
        print(f"Loading DINO model ({model_name}) on {self.device}...")

        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

    @staticmethod
    def _to_pil(image):
        if isinstance(image, np.ndarray):
            return Image.fromarray(image)
        return image

    @torch.no_grad()  # 冻结梯度，节省显存
    def get_embedding(self, image):
        """
        获取图片的 DINOv3 语义特征 (L2 归一化后的 [CLS] token)
        :param image: 单张图片 (PIL.Image / numpy.ndarray, RGB) 或图片列表
        :return: shape (N, D) 的归一化特征 tensor
        """
        # 统一成 list,processor 内部会自动 batch
        if isinstance(image, (list, tuple)):
            images = [self._to_pil(img) for img in image]
        else:
            images = [self._to_pil(image)]

        # DINOv3 processor 自动进行 Resize, Crop 和 Normalize
        inputs = self.processor(images=images, return_tensors="pt").to(self.device)

        # DINOv3 的 HF 实现没有 pooler_output,从 last_hidden_state 取 [CLS] token
        # token 顺序: [CLS, register_tokens..., patch_tokens...],CLS 始终在 index 0
        outputs = self.model(**inputs)
        embedding = outputs.last_hidden_state[:, 0]

        # L2 归一化 (单位超球面,便于距离计算)
        embedding = F.normalize(embedding, p=2, dim=1)
        return embedding

    def compute_handoff_reward(self, current_img, expert_ref_img, temperature=1.0):
        """
        计算当前状态图片与专家参考图片的 Reward
        公式: R = exp( - ||f_cur - f_exp||^2 / temperature )

        说明:
          两端都做了 L2 归一化,所以 ||f_cur - f_exp||^2 ∈ [0, 4],
          因此 reward ∈ [exp(-4/temperature), 1]。
          temperature 是 RBF kernel 的 bandwidth(不是 softmax temperature),
          值越大,reward 对距离越不敏感。
        """
        # 把两张图打成一个 batch,只过一次 forward
        embeddings = self.get_embedding([current_img, expert_ref_img])
        l2_dist_sq = ((embeddings[0] - embeddings[1]) ** 2).sum().item()
        reward = float(np.exp(-l2_dist_sq / temperature))
        return reward, l2_dist_sq


if __name__ == "__main__":
    # --- 本地测试脚本 ---
    print("Testing DINO Reward Calculator...")
    calculator = DINORewardCalculator(device="cpu")  # 测试时用CPU即可

    # 用纯色图做测试,语义比随机噪声更稳定
    img1 = Image.new("RGB", (224, 224), color=(135, 206, 235))   # 天蓝
    img2 = Image.new("RGB", (224, 224), color=(135, 206, 235))   # 完全相同
    img3 = Image.new("RGB", (224, 224), color=(255, 0, 0))        # 纯红

    r_same, d_same = calculator.compute_handoff_reward(img1, img2)
    r_diff, d_diff = calculator.compute_handoff_reward(img1, img3)
    print(f"Same pair  -> L2^2: {d_same:.6f}, Reward: {r_same:.4f}")
    print(f"Diff pair  -> L2^2: {d_diff:.6f}, Reward: {r_diff:.4f}")

    assert d_same < 1e-5, f"Identical images should have near-zero distance, got {d_same}"
    assert r_same > r_diff, "Same pair should score higher than different pair"
    print("Sanity check passed.")
