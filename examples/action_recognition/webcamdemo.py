# SET YOUR ENV.INI
#CLOUD_ZOO_URL = "degirum/yolov8_pose"
#AISERVER_HOSTNAME_OR_IP = "https://cs.degirum.com/"
import degirum as dg, dgtools
from collections import deque
import numpy as np
import torch
import openvino as ov
import sys

from action_utils import (average_clip, extract_yolo_results, reformat_posesamples, 
                           UniformSampleFrames, PoseDecode, PoseCompact, 
                           Resize, CenterCrop, GeneratePoseTarget, FormatShape, 
                           PackActionInputs)


def preprocess(results, imgsz):
    left_kp = [1, 3, 5, 7, 9, 11, 13, 15]
    right_kp = [2, 4, 6, 8, 10, 12, 14, 16]

    pose_data = reformat_posesamples(results, imgsz) #(h,w)
    frame_sampler = UniformSampleFrames(clip_len=48, num_clips=1,test_mode=True)
    pose_data = frame_sampler.transform(pose_data)
    pose_decoder = PoseDecode()
    pose_data = pose_decoder.transform(pose_data)
    pose_compacter = PoseCompact(hw_ratio=1.0, allow_imgpad=True)
    pose_data = pose_compacter.transform(pose_data)
    resizer = Resize(scale=(-1, 64))
    pose_data = resizer.transform(pose_data)
    cropper = CenterCrop(crop_size=64)
    pose_data = cropper.transform(pose_data)
    # The original uses two, we can also hardcode this to two as well
    targetter = GeneratePoseTarget(sigma=0.6, use_score=True, with_kp=True, with_limb=False, double=True, left_kp=left_kp, right_kp=right_kp)
    pose_data = targetter.transform(pose_data)
    shaper = FormatShape(input_format='NCTHW_Heatmap')
    pose_data = shaper.transform(pose_data)
    packer = PackActionInputs()
    pose_data = packer.transform(pose_data)

    return pose_data

def extract_bbox_kps(results):
    bboxes = np.zeros((2,4))
    bbox_scores = np.zeros(2)

    kps = np.zeros((2,17,2))
    kp_scores = np.zeros((2,17))

    for i in range(len(results)):
        # Max two people.
        if i < 2:
            bbox_scores[i] = results[i]['score']
            bboxes[i] = results[i]['bbox']

            for j in range(len(results[i]['landmarks'])):
                kps[i][j] = results[i]['landmarks'][j]['landmark']
                kp_scores[i][j] = results[i]['landmarks'][j]['score']
    
    result_dict = { 'bboxes' : bboxes, 
                    'bbox_scores' : bbox_scores, 
                    'keypoints' : kps, 
                    'keypoints_visible': kps, 
                    'keypoint_scores' : kp_scores }
    
    return result_dict

target = dg.CLOUD
camera_id = 0

zoo = dg.connect(target, dgtools.get_cloud_zoo_url(), dgtools.get_token())


model = zoo.load_model("yolo_v8n_pose--640x640_float_openvino_cpu_1")
#model.overlay_show_probabilities = True # show probabilities on overlay image

core = ov.Core()
#REFACTOR HARDCODED
model_onnx = core.read_model(model='posec3d_ntu60_bone_2x.onnx')
# Load model on device
compiled_onnx = core.compile_model(model=model_onnx, device_name='CPU')

#hardcoded
label_map = [x.strip() for x in open('label_map_ntu60.txt').readlines()]

WINDOW_SIZE = 72
STEP_SIZE = 6

camera_h = 0
camera_w = 0

frame_window = deque()

# AI prediction loop
# Press 'x' or 'q' to stop
with dgtools.Display("AI Camera") as display:    
    for res in dgtools.predict_stream(model, camera_id):
        camera_h, camera_w, _ = res.image.shape

        pose_results = extract_bbox_kps(res.results)
        
        frame_window.append(pose_results)

        
        if len(frame_window) >= WINDOW_SIZE + STEP_SIZE:
            for i in range(STEP_SIZE):
                frame_window.popleft()
        
        # #PERFORM INFERENCE
        if len(frame_window) == WINDOW_SIZE:
            pose_data = preprocess(frame_window, (camera_h, camera_w))
            input = torch.clone(pose_data['inputs'])
            res_onnx = compiled_onnx([input.detach().cpu().numpy()])[0]

            num_segs = res_onnx.shape[0] # I think I can hardcode this because batch = 1 // len(pose_data['data'])
            res_onnx = average_clip(res_onnx, num_segs=num_segs)
            index = res_onnx.argmax().item()
            action_label = label_map[index]
            sys.stdout.write("\033[K")
            print(action_label)

        display.show(res)


