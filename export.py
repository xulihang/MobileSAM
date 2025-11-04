import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from mobile_sam import sam_model_registry, SamPredictor
from mobile_sam.utils.onnx import SamOnnxModel

import onnxruntime
from onnxruntime.quantization import QuantType
from onnxruntime.quantization.quantize import quantize_dynamic

def show_mask(mask, ax):
    color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    
def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   
    
def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))  


checkpoint = "weights/mobile_sam.pt"
model_type = "vit_t"

sam = sam_model_registry[model_type](checkpoint=checkpoint)

onnx_model_path = None  # Set to use an already exported model, then skip to the next section.

import warnings

onnx_model_path = "sam_onnx_example.onnx"

onnx_model = SamOnnxModel(sam, return_single_mask=True)

dynamic_axes = {
    "point_coords": {1: "num_points"},
    "point_labels": {1: "num_points"},
}

embed_dim = sam.prompt_encoder.embed_dim
embed_size = sam.prompt_encoder.image_embedding_size
mask_input_size = [4 * x for x in embed_size]
dummy_inputs = {
    "image_embeddings": torch.randn(1, embed_dim, *embed_size, dtype=torch.float),
    "point_coords": torch.randint(low=0, high=1024, size=(1, 5, 2), dtype=torch.float),
    "point_labels": torch.randint(low=0, high=4, size=(1, 5), dtype=torch.float),
    "mask_input": torch.randn(1, 1, *mask_input_size, dtype=torch.float),
    "has_mask_input": torch.tensor([1], dtype=torch.float),
    "orig_im_size": torch.tensor([1500, 2250], dtype=torch.float),
}
output_names = ["masks", "iou_predictions", "low_res_masks"]

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=torch.jit.TracerWarning)
    warnings.filterwarnings("ignore", category=UserWarning)
    with open(onnx_model_path, "wb") as f:
        torch.onnx.export(
            onnx_model,
            tuple(dummy_inputs.values()),
            f,
            export_params=True,
            verbose=False,
            opset_version=16,
            do_constant_folding=True,
            input_names=list(dummy_inputs.keys()),
            output_names=output_names,
            dynamic_axes=dynamic_axes,
        )    




def export_image_encoder(full_model, out_path="sam_image_encoder.onnx", input_size=1024, opset=13):
    """
    从已加载的完整模型 full_model 中查找 image encoder 子模块并导出为 ONNX。
    尝试常见属性名: image_encoder, vision_encoder, encoder 等。
    """
    # 尝试常见属性名
    candidates = ["image_encoder", "vision_encoder", "encoder", "vision", "img_encoder"]
    encoder = None
    for name in candidates:
        if hasattr(full_model, name):
            encoder = getattr(full_model, name)
            break

    # 如果仍未找到，遍历属性查找包含 'encoder' 关键字的模块
    if encoder is None:
        for k, v in vars(full_model).items():
            if "encoder" in k and hasattr(v, "__call__"):
                encoder = v
                break

    if encoder is None:
        raise RuntimeError("image encoder 子模块未找到，请检查模型对象的属性名（例如 image_encoder / vision_encoder 等）。")

    encoder = encoder.eval().cpu()

    # 构造 dummy 输入（1,3,H,W）
    dummy = torch.randn(1, 3, input_size, input_size, dtype=torch.float32)

    input_names = ["image"]
    output_names = ["image_embeddings"]
    dynamic_axes = {
        "image": {0: "batch", 2: "height", 3: "width"},
        "image_embeddings": {0: "batch"}
    }

    with torch.no_grad():
        torch.onnx.export(
            encoder,
            dummy,
            out_path,
            opset_version=opset,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            do_constant_folding=True,
        )

    print(f"Exported image encoder to {out_path}")

if __name__ == "__main__":
    # ...existing code: 加载/构建完整 SAM 模型，通常会得到一个变量例如 `sam` 或 `model` ...
    # 假设现有代码会赋值到 `sam`，在加载完 checkpoint / model 后调用：
    try:
        # ...existing code that creates 'sam'...
        export_image_encoder(sam, out_path="sam_image_encoder.onnx", input_size=1024)
    except NameError:
        # 如果主脚本中模型变量名不同，请替换 `sam` 为实际变量名或手动调用 export_image_encoder
        print("模型变量未命名为 'sam'，请在脚本中确认模型变量名并调用 export_image_encoder(model, ...)")