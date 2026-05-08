"""Unit test for DINoV3RewardModel.

Usage:
    cd RLinf/tests/unit_tests
    python test_dinov3_reward_model.py

Prerequisites:
    - transformers >= 4.56.0
    - huggingface-cli login (for DINOv3 gated access)
    - An expert reference image at the path set below
"""

import sys
from pathlib import Path

# Add RLinf to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
import numpy as np
from omegaconf import OmegaConf
from PIL import Image


def create_mock_expert_image(path: str, size: int = 224):
    """Create a dummy expert reference image for testing."""
    img = Image.new("RGB", (size, size), color=(100, 150, 200))
    img.save(path)
    return path


def test_dinov3_reward_model():
    """Test DINoV3RewardModel instantiation and forward pass."""
    from rlinf.models.embodiment.reward import get_reward_model_class, reward_model_registry

    # 1. Verify registration
    assert "dinov3" in reward_model_registry, "dinov3 not registered"
    print("[1/6] Model registration: OK")

    # 2. Create mock expert image
    expert_path = "/tmp/test_expert_ref.png"
    create_mock_expert_image(expert_path)

    # 3. Build config (same structure as training config)
    cfg = OmegaConf.create({
        "model_name": "/root/autodl-tmp/checkpoints/dinov3/dinov3-vitl16-pretrain-lvd1689m",
        "temperature": 1.0,
        "expert_ref_image_path": expert_path,
        "image_size": [3, 224, 224],
        "normalize": False,
        "precision": "fp32",
    })

    # 4. Instantiate model
    model_cls = get_reward_model_class("dinov3")
    model = model_cls(cfg)
    model.eval()
    print("[2/6] Model instantiation: OK")

    # 5. Test forward() with NHWC uint8 (env worker format)
    batch_size = 4
    dummy_images_nhwc = torch.randint(0, 256, (batch_size, 224, 224, 3), dtype=torch.uint8)

    with torch.no_grad():
        outputs = model(dummy_images_nhwc)

    assert "loss" in outputs, "Missing 'loss' in outputs"
    assert "accuracy" in outputs, "Missing 'accuracy' in outputs"
    assert "logits" in outputs, "Missing 'logits' in outputs"
    assert "probabilities" in outputs, "Missing 'probabilities' in outputs"
    assert outputs["probabilities"].shape == (batch_size,), \
        f"Expected probabilities shape ({batch_size},), got {outputs['probabilities'].shape}"
    assert (outputs["probabilities"] >= 0).all() and (outputs["probabilities"] <= 1).all(), \
        "Probabilities out of [0, 1] range"
    print(f"[3/6] forward() NHWC uint8: OK (rewards={outputs['probabilities'].tolist()})")

    # 6. Test forward() with NCHW float (after worker dtype conversion)
    dummy_images_nchw = torch.randn(batch_size, 3, 224, 224)  # float, random noise
    with torch.no_grad():
        outputs2 = model(dummy_images_nchw)
    assert outputs2["probabilities"].shape == (batch_size,)
    print(f"[4/6] forward() NCHW float: OK (rewards={outputs2['probabilities'].tolist()})")

    # 7. Test compute_reward()
    obs_dict = {"images": dummy_images_nhwc}
    rewards = model.compute_reward(obs_dict)
    assert rewards.shape == (batch_size,)
    assert (rewards >= 0).all() and (rewards <= 1).all()
    print(f"[5/6] compute_reward(): OK (rewards={rewards.tolist()})")

    # 8. Test identical images give reward ≈ 1.0
    # Create an image matching the expert reference color (100, 150, 200)
    same_img = torch.zeros((1, 224, 224, 3), dtype=torch.uint8)
    same_img[0, :, :, 0] = 100
    same_img[0, :, :, 1] = 150
    same_img[0, :, :, 2] = 200
    same_batch = same_img.repeat(2, 1, 1, 1)  # two identical images
    with torch.no_grad():
        outputs_same = model(same_batch)
    r0, r1 = outputs_same["probabilities"][0].item(), outputs_same["probabilities"][1].item()
    assert abs(r0 - r1) < 1e-5, f"Identical images should give identical rewards: {r0} vs {r1}"
    assert r0 > 0.99, f"Identical image to expert ref should give high reward, got {r0}"
    print(f"[6/6] Identical images sanity check: OK (reward={r0:.6f})")

    print("\n=== All tests passed! ===")


def test_without_expert_ref():
    """Test that missing expert_ref raises appropriate error."""
    from rlinf.models.embodiment.reward import get_reward_model_class

    cfg = OmegaConf.create({
        "model_name": "/root/autodl-tmp/checkpoints/dinov3/dinov3-vitl16-pretrain-lvd1689m",
        "temperature": 1.0,
        "expert_ref_image_path": None,  # Missing!
        "image_size": [3, 224, 224],
        "normalize": False,
        "precision": "fp32",
    })

    model_cls = get_reward_model_class("dinov3")
    model = model_cls(cfg)
    model.eval()

    dummy = torch.randint(0, 256, (1, 224, 224, 3), dtype=torch.uint8)
    try:
        with torch.no_grad():
            model(dummy)
        assert False, "Should have raised RuntimeError"
    except RuntimeError as e:
        assert "expert_ref_image_path" in str(e)
        print("[Bonus] Missing expert_ref error handling: OK")


if __name__ == "__main__":
    test_dinov3_reward_model()
    test_without_expert_ref()
