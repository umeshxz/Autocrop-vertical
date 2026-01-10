import argparse
import os
import re
import shutil
import subprocess
import time

import cv2
import numpy as np
from scenedetect import VideoManager, SceneManager
from scenedetect.detectors import ContentDetector
from tqdm import tqdm
from ultralytics import YOLO

# --- Constants ---
# ASPECT_RATIO = 10 / 16

# Load the YOLO model once
# Load the YOLO model once
model = YOLO('yolov8n.pt')

# Load the Haar Cascade for face detection once
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


def analyze_scene_content(video_path, scene_start_time, scene_end_time):
    """
    Analyzes the middle frame of a scene to detect people and faces.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return []

    fps = cap.get(cv2.CAP_PROP_FPS)

    start_frame = scene_start_time.get_frames()
    end_frame = scene_end_time.get_frames()
    middle_frame_number = int(start_frame + (end_frame - start_frame) / 2)

    cap.set(cv2.CAP_PROP_POS_FRAMES, middle_frame_number)

    ret, frame = cap.read()
    if not ret:
        cap.release()
        return []

    results = model([frame], verbose=False)

    detected_objects = []

    for result in results:
        if len(detected_objects) >= 1: return detected_objects
        boxes = result.boxes
        for box in boxes:
            if len(detected_objects) >= 1: return detected_objects
            if box.cls[0] == 0:
                x1, y1, x2, y2 = [int(i) for i in box.xyxy[0]]
                person_box = [x1, y1, x2, y2]

                person_roi_gray = cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(person_roi_gray, scaleFactor=1.1, minNeighbors=5,
                                                      minSize=(30, 30))

                face_box = None
                if len(faces) > 0:
                    fx, fy, fw, fh = faces[0]
                    face_box = [x1 + fx, y1 + fy, x1 + fx + fw, y1 + fy + fh]

                detected_objects.append({'person_box': person_box, 'face_box': face_box})

    cap.release()
    return detected_objects


def detect_scenes(video_path):
    video_manager = VideoManager([video_path])
    scene_manager = SceneManager()
    scene_manager.add_detector(ContentDetector())
    video_manager.set_downscale_factor()
    video_manager.start()
    scene_manager.detect_scenes(frame_source=video_manager)
    scene_list = scene_manager.get_scene_list()
    fps = video_manager.get_framerate()
    video_manager.release()
    return scene_list, fps


def get_enclosing_box(boxes):
    if not boxes:
        return None
    min_x = min(box[0] for box in boxes)
    min_y = min(box[1] for box in boxes)
    max_x = max(box[2] for box in boxes)
    max_y = max(box[3] for box in boxes)
    return [min_x, min_y, max_x, max_y]


def decide_cropping_strategy(scene_analysis, frame_height, aspect_ratio):
    num_people = len(scene_analysis)
    if num_people == 0:
        return 'LETTERBOX', None
    if num_people == 1:
        target_box = scene_analysis[0]['face_box'] or scene_analysis[0]['person_box']
        return 'TRACK', target_box
    person_boxes = [obj['person_box'] for obj in scene_analysis]
    group_box = get_enclosing_box(person_boxes)
    group_width = group_box[2] - group_box[0]
    max_width_for_crop = frame_height * aspect_ratio
    if group_width < max_width_for_crop:
        return 'TRACK', group_box
    else:
        return 'LETTERBOX', None


def calculate_crop_box(target_box, frame_width, frame_height, aspect_ratio):
    target_center_x = (target_box[0] + target_box[2]) / 2
    crop_height = frame_height
    crop_width = int(crop_height * aspect_ratio)
    x1 = int(target_center_x - crop_width / 2)
    y1 = 0
    x2 = int(target_center_x + crop_width / 2)
    y2 = frame_height
    if x1 < 0:
        x1 = 0
        x2 = crop_width
    if x2 > frame_width:
        x2 = frame_width
        x1 = frame_width - crop_width
    return x1, y1, x2, y2


def get_video_resolution(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Could not open video file {video_path}")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return width, height


def run_conversion(input_path, output_path, aspect_ratio):
    script_start_time = time.time()

    input_video = input_path
    final_output_video = output_path

    # Define temporary file paths based on the output name
    base_name = os.path.splitext(final_output_video)[0]
    temp_video_output = f"{base_name}_temp_video.mp4"
    temp_audio_output = f"{base_name}_temp_audio.aac"

    # Clean up previous temp files if they exist
    if os.path.exists(temp_video_output): os.remove(temp_video_output)
    if os.path.exists(temp_audio_output): os.remove(temp_audio_output)
    if os.path.exists(final_output_video): os.remove(final_output_video)

    print("🎬 Step 1: Detecting scenes...")
    step_start_time = time.time()
    scenes, fps = detect_scenes(input_video)
    step_end_time = time.time()

    if not scenes:
        print("❌ No scenes were detected. Aborting.")
        exit()

    print(f"✅ Found {len(scenes)} scenes in {step_end_time - step_start_time:.2f}s. Here is the breakdown:")
    # for i, (start, end) in enumerate(scenes):
    #     print(f"  - Scene {i+1}: {start.get_timecode()} -> {end.get_timecode()}")

    print("\n🧠 Step 2: Analyzing scene content and determining strategy...")
    step_start_time = time.time()
    original_width, original_height = get_video_resolution(input_video)

    OUTPUT_HEIGHT = original_height
    OUTPUT_WIDTH = int(OUTPUT_HEIGHT * aspect_ratio)
    if OUTPUT_WIDTH % 2 != 0:
        OUTPUT_WIDTH += 1

    scenes_analysis = []
    for i, (start_time, end_time) in enumerate(tqdm(scenes, desc="Analyzing Scenes", disable=True)):
        analysis = analyze_scene_content(input_video, start_time, end_time)
        strategy, target_box = decide_cropping_strategy(analysis, original_height, aspect_ratio)
        scenes_analysis.append({
            'start_frame': start_time.get_frames(),
            'end_frame': end_time.get_frames(),
            'analysis': analysis,
            'strategy': strategy,
            'target_box': target_box
        })
    step_end_time = time.time()
    print(f"✅ Scene analysis complete in {step_end_time - step_start_time:.2f}s.")

    print("\n📋 Step 3: Generated Processing Plan")
    
    # Group scenes into chunks with same strategy
    chunks = []
    if not scenes_analysis:
        return

    current_chunk = {
        'strategy': scenes_analysis[0]['strategy'],
        'scenes': [scenes_analysis[0]]
    }
    
    for i in range(1, len(scenes_analysis)):
        scene = scenes_analysis[i]
        if scene['strategy'] == current_chunk['strategy']:
            current_chunk['scenes'].append(scene)
        else:
            chunks.append(current_chunk)
            current_chunk = {
                'strategy': scene['strategy'],
                'scenes': [scene]
            }
    chunks.append(current_chunk)
    
    print(f"✅ Grouped {len(scenes_analysis)} scenes into {len(chunks)} processing chunks.")

    print("\n✂️ Step 4: Processing video segments (FFmpeg Native)...")
    step_start_time = time.time()
    
    chunk_files = []
    temp_dir = f"{base_name}_segments"
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)
        
    for i, chunk in enumerate(tqdm(chunks, desc="Rendering Chunks")):
        chunk_strategy = chunk['strategy']
        chunk_output = os.path.join(temp_dir, f"chunk_{i:04d}.mp4")
        chunk_files.append(chunk_output)
        
        # Calculate start and end for the whole chunk
        # Note: We need accurate frame trimming.
        # But for 'crop' expression, we need relative frames from 0.
        
        # Construct specific filter command
        # 1. Trim the input to the chunk duration
        # 2. Apply filter (Dynamic Crop or Letterbox)
        
        chunk_start_frame = chunk['scenes'][0]['start_frame']
        chunk_end_frame = chunk['scenes'][-1]['end_frame']
        
        # FFmpeg 'select' or 'trim' uses seconds usually for -ss/to, or frames for filter
        # Ideally using -ss before input is fast, but precise frame matching can be tricky with GOP.
        # We will use -ss (fast seek) and re-encode.
        
        # Calculate time (approximation from frames? No, use the stored timecodes if possible, 
        # but here we have frames. Let's use start_frame/fps)
        start_sec = chunk_start_frame / fps
        end_sec = chunk_end_frame / fps
        duration_sec = end_sec - start_sec
        
        vf_filters = []
        
        if chunk_strategy == 'LETTERBOX':
            # Scale to fit width, then pad height
            # scale=w=OUTPUT_WIDTH:h=-1:flags=lanczos,pad=w=OUTPUT_WIDTH:h=OUTPUT_HEIGHT:x=0:y=(oh-ih)/2
            vf_filters.append(f"scale={OUTPUT_WIDTH}:-2") # -2 ensures even dim
            vf_filters.append(f"pad={OUTPUT_WIDTH}:{OUTPUT_HEIGHT}:(ow-iw)/2:(oh-ih)/2:color=black")
            
        elif chunk_strategy == 'TRACK':
            # Dynamic Crop
            # x = if(between(n, 0, len1), x1, if(between(n, len1, len2), x2, ...))
            # Note: 'n' in the filter starts at 0 for the trimmed segment.
            
            # We need to build the IF chain.
            expr_parts = []
            
            # Calculate crop boxes and relative durations
            rel_start = 0
            
            # Use 'default' x as the last one to be safe
            default_x = 0
            
            for scene in chunk['scenes']:
                s_len = scene['end_frame'] - scene['start_frame']
                target_box = scene['target_box']
                # Recalculate crop box for this specific scene target
                # (function logic copied/inlined to ensure accessing variables)
                
                # Logic from calculate_crop_box:
                target_center_x = (target_box[0] + target_box[2]) / 2
                crop_w = int(original_height * aspect_ratio)
                x1 = int(target_center_x - crop_w / 2)
                # Bounds check
                if x1 < 0: x1 = 0
                if x1 + crop_w > original_width: x1 = original_width - crop_w
                
                # For this frame range [rel_start, rel_start + s_len]
                # We want x to be x1
                # expr: between(n, start, end)
                # Note: 'between' is inclusive.
                expr_parts.append(f"between(n,{rel_start},{rel_start + s_len})*{x1}")
                
                rel_start += s_len
                default_x = x1
            
            # Refined expression: Summing them works if ranges don't overlap.
            # "between(n,A,B)*VAL + between(n,B,C)*VAL2..."
            # Since 'n' increases, only one term is 1 (true) at a time.
            x_expr = "+".join(expr_parts)
            if not x_expr: x_expr = "0"
            
            vf_filters.append(f"crop=w={OUTPUT_WIDTH}:h={OUTPUT_HEIGHT}:x='{x_expr}':y=0")
            
        vf_string = ",".join(vf_filters)
        
        # Execute FFmpeg for this chunk
        cmd = [
            'ffmpeg', '-y', '-loglevel', 'error',
            '-ss', f"{start_sec:.3f}",
            '-t', f"{duration_sec:.3f}", # -t is duration
            '-i', input_video,
            '-vf', vf_string,
            '-c:v', 'h264_videotoolbox', '-b:v', '5000k', # Hardware encoder on Mac
            '-an', # No audio in segments, we merge original audio later
            chunk_output
        ]
        
        try:
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        except subprocess.CalledProcessError as e:
            print(f"❌ Chunk rendering failed: {e.stderr.decode()}")
            # Critical failure
            exit()

    # Concat chunks
    print("🔗 Concatenating segments...")
    concat_list_path = f"{base_name}_concat_list.txt"
    with open(concat_list_path, 'w') as f:
        for p in chunk_files:
            abs_p = os.path.abspath(p)
            f.write(f"file '{abs_p}'\n")
            
    concat_cmd = [
        'ffmpeg', '-y', '-loglevel', 'error',
        '-f', 'concat', '-safe', '0',
        '-i', concat_list_path,
        '-c', 'copy',
        temp_video_output
    ]
    subprocess.run(concat_cmd, check=True)
    
    # Cleanup Segments
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
    if os.path.exists(concat_list_path):
        os.remove(concat_list_path)
    step_end_time = time.time()
    print(f"✅ Video processing complete in {step_end_time - step_start_time:.2f}s.")

    print("\n🔊 Step 5: Extracting original audio...")
    step_start_time = time.time()
    audio_extract_command = [
        'ffmpeg', '-y', '-i', input_video, '-vn', '-acodec', 'copy', temp_audio_output
    ]
    try:
        subprocess.run(audio_extract_command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        step_end_time = time.time()
        print(f"✅ Audio extracted in {step_end_time - step_start_time:.2f}s.")
    except subprocess.CalledProcessError as e:
        print("\n❌ Audio extraction failed.")
        print("Stderr:", e.stderr.decode())
        exit()

    print("\n✨ Step 6: Merging video and audio...")
    step_start_time = time.time()
    merge_command = [
        'ffmpeg', '-y', '-i', temp_video_output, '-i', temp_audio_output,
        '-c:v', 'copy', '-c:a', 'copy', final_output_video
    ]
    try:
        subprocess.run(merge_command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        step_end_time = time.time()
        print(f"✅ Final video merged in {step_end_time - step_start_time:.2f}s.")
    except subprocess.CalledProcessError as e:
        print("\n❌ Final merge failed.")
        print("Stderr:", e.stderr.decode())
        exit()

    # Clean up temp files
    os.remove(temp_video_output)
    os.remove(temp_audio_output)

    script_end_time = time.time()
    print(f"\n🎉 All done! Final video saved to {final_output_video}")
    print(f"⏱️  Total execution time: {script_end_time - script_start_time:.2f} seconds.")

# if __name__ == '__main__':
    # parser = argparse.ArgumentParser(description="Smartly crops a horizontal video into a vertical one.")
    # parser.add_argument('-i', '--input', type=str, required=True, help="Path to the input video file.")
    # parser.add_argument('-o', '--output', type=str, required=True, help="Path to the output video file.")
    # args = parser.parse_args()
    #
    # input_v = args.input
    # final_output = args.output
    run_conversion('/Users/umeshyadav/Downloads/test.mp4', '/Users/umeshyadav/Downloads/test_1.mp4')