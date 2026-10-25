"""G1 OmniPicker 电力工具任务的数据变换。

训练数据键(LeRobot): observation.state / observation.images.cam_head /
observation.images.cam_wrist_r / action
推理客户端发送:      {state, images: {cam_head, cam_wrist_r}, prompt}
这里统一成模型需要的: {state, image, image_mask, actions, prompt}
"""
import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class G1OmnipickerInputs(transforms.DataTransformFn):
    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        base_image = _parse_image(data["images"]["cam_head"])
        wrist_image = _parse_image(data["images"]["cam_wrist_r"])

        inputs = {
            "state": data["state"],
            "image": {
                "base_0_rgb": base_image,
                "left_wrist_0_rgb": wrist_image,
                "right_wrist_0_rgb": np.zeros_like(base_image),  # 第三路补零占位
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                "right_wrist_0_rgb": np.True_ if self.model_type == _model.ModelType.PI0_FAST else np.False_,
            },
        }
        if "actions" in data:
            inputs["actions"] = data["actions"]
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]
        return inputs


@dataclasses.dataclass(frozen=True)
class G1OmnipickerOutputs(transforms.DataTransformFn):
    action_dim: int = 18

    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"][..., :self.action_dim])}