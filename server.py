#!/usr/bin/env python3

import os
import time
import datetime
from bottle import route, run, template, request, static_file
import json
import sys
import cv2
import numpy as np
import onnxruntime

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

@route('/getmask', method='POST')
def getmask():
    upload = request.files.get('upload')
    boxes = request.forms.get('boxes')
    boxes = json.loads(boxes)
    name, ext = os.path.splitext(upload.filename)
    print(ext.lower())
    if ext.lower() not in ('.png','.jpg','.jpeg'):
        return "File extension not allowed."
        
    timestamp=str(int(time.time()*1000))
    savedName=timestamp+ext
    save_path = "./uploaded/"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    file_path = "{path}/{file}".format(path=save_path, file=savedName)
    mask_path = "{path}/{file}".format(path=save_path, file=savedName+"-mask.png")
    if os.path.exists(file_path)==True:
        os.remove(file_path)
    upload.save(file_path)
    image = cv2.imread(file_path)
    mask = gen_mask(image, boxes, [])
    mask = mask * 255
    cv2.imwrite(mask_path, mask)
    return static_file(savedName+"-mask.png", root='uploaded')  
    

@route('/<filepath:path>')
def server_static(filepath):
    return static_file(filepath, root='www')

def gen_mask(image, boxes, points=None):
    ort_session = onnxruntime.InferenceSession(onnx_model_path)
    input_names = [inp.name for inp in ort_session.get_inputs()]
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

    # Ensure boxes is a list of boxes. Support single-box flat list like [x0,y0,x1,y1]
    if boxes is None:
        boxes = []
    elif len(boxes) > 0 and (not hasattr(boxes[0], '__iter__') or isinstance(boxes[0], (int, float))):
        boxes = [boxes]

    # Prepare merged mask (0/1) same spatial dims as original image
    mask_merged = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)

    for idx, box in enumerate(boxes):
        # allow per-box point specification via points parameter
        point = None
        if points is not None:
            try:
                point = points[idx]
            except Exception:
                # if points is a single point for all boxes
                if not isinstance(points, (list, tuple)) or len(points) == 2:
                    point = points
                else:
                    point = None

        # prepare prompt: use a bounding box plus an optional point
        # box format: [x0, y0, x1, y1] in original image coordinates
        input_box = np.array(box, dtype=np.float32)

        if point is None:
            input_point_arr = np.array([], dtype=np.float32)
        else:
            input_point_arr = np.array([point], dtype=np.float32)

        input_label = np.array([0], dtype=np.float32)

        # Build box corner coords and labels (labels 2,3 for box corners as requested)
        onnx_box_coords = input_box.reshape(2, 2).astype(np.float32)  # [[x0,y0],[x1,y1]]
        onnx_box_labels = np.array([2, 3], dtype=np.float32)

        # If the user provided no points, use only the box corners as prompts.
        # Otherwise concatenate user points + box corners as before.
        if input_point_arr is None or input_point_arr.size == 0:
            onnx_coord = onnx_box_coords[None, :, :]
            onnx_label = onnx_box_labels[None, :].astype(np.float32)
        else:
            onnx_coord = np.concatenate([input_point_arr.astype(np.float32), onnx_box_coords], axis=0)[None, :, :]
            onnx_label = np.concatenate([input_label.astype(np.float32), onnx_box_labels], axis=0)[None, :].astype(np.float32)

        # transform coords to encoder's resized/padded space
        try:
            scale  # noqa: F821
        except NameError:
            target_size = target_size if 'target_size' in locals() else 1024
            scale = target_size / max(image.shape[:2])

        onnx_coord = transform_coords(onnx_coord[0], image.shape[:2], scale)[None, :, :].astype(np.float32)

        onnx_mask_input = np.zeros((1, 1, 256, 256), dtype=np.float32)
        onnx_has_mask_input = np.zeros((1,), dtype=np.float32)

        # Build inputs dict according to what the main ONNX expects
        ort_inputs = {}
        for inp in ort_session.get_inputs():
            name = inp.name
            if name == "image_embeddings":
                ort_inputs[name] = image_embeddings
            elif name in ("point_coords", "point_coordinates", "point_coords_coord"):
                ort_inputs[name] = onnx_coord
            elif name in ("point_labels", "point_label"):
                ort_inputs[name] = onnx_label
            elif name == "mask_input":
                ort_inputs[name] = onnx_mask_input
            elif name == "has_mask_input":
                ort_inputs[name] = onnx_has_mask_input
            elif name in ("orig_im_size", "original_size", "orig_size"):
                ort_inputs[name] = np.array(image.shape[:2], dtype=np.float32)
            else:
                # If ONNX expects raw image (we preprocessed into `tensor`) feed it if we can
                if 'tensor' in locals() and 'raw_input_name' in locals() and name == raw_input_name:
                    ort_inputs[name] = tensor
                else:
                    # Try to create a dummy array matching expected shape
                    shp = []
                    for s in inp.shape:
                        if isinstance(s, str) or s is None:
                            shp.append(1)
                        else:
                            shp.append(int(s))
                    ort_inputs[name] = np.zeros(tuple(shp), dtype=np.float32)

        # Run inference. Try to match the pattern masks, _, _ = ort_session.run(...). Fall back if the model returns fewer outputs.
        try:
            masks, _, _ = ort_session.run(None, ort_inputs)
        except Exception:
            outs = ort_session.run(None, ort_inputs)
            masks = outs[0]

        # apply threshold; choose a default threshold (user can adjust)
        mask_threshold = 0.0
        masks = masks > mask_threshold
        # pick first mask in batch and first channel
        mask = masks[0, 0].astype(np.uint8)
        mask_resized = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)

        # accumulate into merged mask (logical OR)
        mask_merged = np.logical_or(mask_merged, mask_resized).astype(np.uint8)

    return mask_merged

# paths
onnx_model_path = "decoder.onnx"
encoder_onnx_path = "encoder.onnx"


if __name__ == '__main__':
    if len(sys.argv)==2:
        service_port=sys.argv[1]
    else:
        service_port=8289
    run(host='127.0.0.1', port=service_port)