import os
import cv2
import numpy as np
import onnxruntime
import matplotlib.pyplot as plt

# 完全独立于 mobile_sam 的预处理 + 使用 onnxruntime 生成 image_embeddings（如果需要）

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

def preprocess_for_encoder(image, target_size=1024, mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]):
    """
    Resize (keep aspect), pad to square (target_size x target_size), normalize to 0-1 and apply ImageNet mean/std.
    Returns:
      - tensor: np.float32 shape (1,3,target_size,target_size)
      - scale: float (scaling factor applied to original)
      - padded_size: (h, w) shape after padding (target_size, target_size)
    Assumes input image is RGB uint8.
    """
    h0, w0 = image.shape[:2]
    scale = target_size / max(h0, w0)
    new_w, new_h = int(w0 * scale), int(h0 * scale)
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    # pad bottom and right
    canvas = np.zeros((target_size, target_size, 3), dtype=np.uint8)
    canvas[:new_h, :new_w, :] = resized
    # to float 0..1
    img = canvas.astype(np.float32) / 255.0
    img = (img - mean) / std
    # HWC -> CHW
    img = img.transpose(2, 0, 1)[None, :].astype(np.float32)
    return img, scale, (new_h, new_w)

def transform_coords(coords, orig_size, scale):
    """
    coords: (N,2) x,y in original image coordinates
    orig_size: (h, w)
    scale: scaling factor used in preprocessing (new = old * scale)
    Returns coords transformed into padded/resized image space.
    """
    coords = np.array(coords, dtype=np.float32)
    transformed = coords * scale
    return transformed

# paths
onnx_model_path = "sam_onnx_example.onnx"
# optional image encoder onnx (if main ONNX already takes embeddings this will be used to compute them)
encoder_onnx_path = "sam_image_encoder.onnx"  # place encoder ONNX here if needed
if not os.path.exists(onnx_model_path):
    raise FileNotFoundError(f"ONNX model not found: {onnx_model_path}")

ort_session = onnxruntime.InferenceSession(onnx_model_path)

# read image
image_path = 'notebooks/images/picture2.jpg'
image = cv2.imread(image_path)
if image is None:
    raise FileNotFoundError(f"Image not found: {image_path}")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Decide how to supply embeddings:
# If the ONNX model expects "image_embeddings" as an input, we must provide them.
# Try to compute embeddings by running encoder_onnx_path with the same preprocessing.
input_names = [inp.name for inp in ort_session.get_inputs()]
need_embeddings = "image_embeddings" in input_names

image_embeddings = None
# Default preprocessing target size; if encoder ONNX exists, try to infer expected input size from its inputs.
if need_embeddings:
    if not os.path.exists(encoder_onnx_path):
        raise FileNotFoundError(
            "ONNX model expects 'image_embeddings' but no encoder ONNX found. "
            "Export the image encoder to sam_image_encoder.onnx or modify onnx_model to accept raw images."
        )
    enc_sess = onnxruntime.InferenceSession(encoder_onnx_path)
    # Attempt to infer target size from encoder input shape
    enc_inp = enc_sess.get_inputs()[0]
    shape = enc_inp.shape  # e.g. [1, 3, 1024, 1024] or [None, 3, 1024, 1024]
    # pick last two dims if available and valid
    try:
        target_size = int(shape[-1])
    except Exception:
        target_size = 1024
    tensor, scale, (res_h, res_w) = preprocess_for_encoder(image, target_size=target_size)
    # run encoder
    enc_input_name = enc_sess.get_inputs()[0].name
    enc_outs = enc_sess.run(None, {enc_input_name: tensor})
    # Assume first output is embeddings and already in correct shape for the main ONNX
    image_embeddings = enc_outs[0].astype(np.float32)
else:
    # If main ONNX accepts raw image input, preprocess directly for that input.
    # Find candidate input name for image (common names: "image", "images", "pixel_values")
    raw_input_name = None
    candidates = ["image", "images", "pixel_values", "input_image"]
    for c in candidates:
        if c in input_names:
            raw_input_name = c
            break
    # If not found, take the first input and assume it's the image input
    if raw_input_name is None:
        raw_input_name = ort_session.get_inputs()[0].name
    # choose a target size for preprocessing; try to read the shape
    inp = ort_session.get_inputs()[0]
    shp = inp.shape
    try:
        target_size = int(shp[-1])
    except Exception:
        target_size = 1024
    tensor, scale, (res_h, res_w) = preprocess_for_encoder(image, target_size=target_size)
    # For raw image input we will feed `tensor` under the detected raw_input_name later.

# prepare prompt (points)
input_point = np.array([[250, 375]], dtype=np.float32)  # x,y in original image coords
input_label = np.array([1], dtype=np.float32)
# add dummy second point as original code did (to satisfy model expecting >=2 pts)
onnx_coord = np.concatenate([input_point, np.array([[0.0, 0.0]], dtype=np.float32)], axis=0)[None, :, :]
onnx_label = np.concatenate([input_label, np.array([-1], dtype=np.float32)], axis=0)[None, :]

# transform coords to encoder's resized/padded space
# use scale produced by preprocessing; if not computed above, compute here with default target_size
try:
    scale  # noqa: F821
except NameError:
    # compute scale as target_size / max(orig)
    target_size = target_size if 'target_size' in locals() else 1024
    scale = target_size / max(image.shape[:2])

onnx_coord_transformed = transform_coords(onnx_coord[0], image.shape[:2], scale)[None, :, :].astype(np.float32)
# the model in original example also concatenated a second zero point; keep shape (1,2,2)

onnx_mask_input = np.zeros((1, 1, 256, 256), dtype=np.float32)
onnx_has_mask_input = np.zeros((1,), dtype=np.float32)

# Build inputs dict according to what the main ONNX expects
ort_inputs = {}
for inp in ort_session.get_inputs():
    name = inp.name
    if name == "image_embeddings":
        ort_inputs[name] = image_embeddings
    elif name in ("point_coords", "point_coordinates", "point_coords_coord"):
        ort_inputs[name] = onnx_coord_transformed
    elif name in ("point_labels", "point_label"):
        ort_inputs[name] = onnx_label
    elif name == "mask_input":
        ort_inputs[name] = onnx_mask_input
    elif name == "has_mask_input":
        ort_inputs[name] = onnx_has_mask_input
    elif name in ("orig_im_size", "original_size", "orig_size"):
        ort_inputs[name] = np.array(image.shape[:2], dtype=np.float32)
    else:
        # If ONNX expects raw image (we preprocessed into `tensor`) feed it
        if 'tensor' in locals() and name == raw_input_name:
            ort_inputs[name] = tensor
        # otherwise skip or fill zeros for optional inputs
        else:
            # Try to create a dummy array matching expected shape
            shp = []
            for s in inp.shape:
                if isinstance(s, str) or s is None:
                    shp.append(1)
                else:
                    shp.append(int(s))
            ort_inputs[name] = np.zeros(tuple(shp), dtype=np.float32)

# Run inference
outs = ort_session.run(None, ort_inputs)
# assume first output is masks, third is low_res_logits like original example
masks = outs[0]
# apply threshold; choose a default threshold (user can adjust)
mask_threshold = 0.0
masks = masks > mask_threshold
# masks may be in model's output resolution; resize to original image for visualization
# pick first mask in batch and first channel
mask = masks[0, 0].astype(np.uint8)  # may be float bool
mask_resized = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)

plt.figure(figsize=(10,10))
plt.imshow(image)
show_mask(mask_resized, plt.gca())
show_points(input_point, input_label, plt.gca())
plt.axis('off')
plt.show()