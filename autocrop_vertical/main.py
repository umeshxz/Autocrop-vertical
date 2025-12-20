import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from scenedetect import VideoManager, SceneManager
from scenedetect.detectors import ContentDetector
from tqdm import tqdm
from ultralytics import YOLO

# --- Constants ---
ASPECT_RATIO = 10/16

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
                faces = face_cascade.detectMultiScale(person_roi_gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
                
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

def decide_cropping_strategy(scene_analysis, frame_height, aspect_ratio=ASPECT_RATIO):
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

def calculate_crop_box(target_box, frame_width, frame_height, aspect_ratio=ASPECT_RATIO):
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

def calculate_panning_crop_box(target_box, pan_state, last_crop_box, frame_width, frame_height, aspect_ratio=ASPECT_RATIO):
    crop_height = frame_height
    crop_width = int(crop_height * aspect_ratio)

    if pan_state == 'center' or not last_crop_box:
        return calculate_crop_box(target_box, frame_width, frame_height)

    last_x1, _, last_x2, _ = last_crop_box

    if pan_state == 'right':
        x1 = last_x2 + 1
        x2 = x1 + crop_width
        if x2 > frame_width:
            x2 = frame_width
            x1 = x2 - crop_width
    elif pan_state == 'left':
        x1 = last_x1 - crop_width - 1
        if x1 < 0:
            x1 = 0
        x2 = x1 + crop_width

    return x1, 0, x2, crop_height

def get_video_resolution(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Could not open video file {video_path}")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return width, height

def run_conversion(input_path, output_path, aspect_ratio=ASPECT_RATIO):
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
    for i, (start, end) in enumerate(scenes):
        print(f"  - Scene {i+1}: {start.get_timecode()} -> {end.get_timecode()}")


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
        strategy, target_box = decide_cropping_strategy(analysis, original_height)

        scene_duration = (end_time.get_frames() - start_time.get_frames()) / fps
        needs_panning = strategy == 'TRACK' and scene_duration > 3

        scenes_analysis.append({
            'start_frame': start_time.get_frames(),
            'end_frame': end_time.get_frames(),
            'analysis': analysis,
            'strategy': strategy,
            'target_box': target_box,
            'needs_panning': needs_panning
        })
    step_end_time = time.time()
    print(f"✅ Scene analysis complete in {step_end_time - step_start_time:.2f}s.")

    print("\n📋 Step 3: Generated Processing Plan")
    for i, scene_data in enumerate(scenes_analysis):
        num_people = len(scene_data['analysis'])
        strategy = scene_data['strategy']
        start_time = scenes[i][0].get_timecode()
        end_time = scenes[i][1].get_timecode()
        # print(f"  - Scene {i+1} ({start_time} -> {end_time}): Found {num_people} person(s). Strategy: {strategy}")

    print("\n✂️ Step 4: Processing video frames...")
    step_start_time = time.time()
    
    command = [
        'ffmpeg', '-loglevel', 'info', '-y', '-f', 'rawvideo', '-vcodec', 'rawvideo',
        '-s', f'{OUTPUT_WIDTH}x{OUTPUT_HEIGHT}', '-pix_fmt', 'bgr24',
        '-r', str(fps), '-i', '-', '-c:v', 'libx264',
        '-preset', 'fast', '-crf', '23', '-an', temp_video_output
    ]

    ffmpeg_process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)

    cap = cv2.VideoCapture(input_video)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    frame_number = 0
    current_scene_index = 0
    
    pan_state = 'center'
    last_pan_time = 0
    last_crop_box = None

    with tqdm(total=total_frames, desc="Applying Plan", disable=True) as pbar:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if current_scene_index < len(scenes_analysis) - 1 and \
               frame_number >= scenes_analysis[current_scene_index + 1]['start_frame']:
                current_scene_index += 1
                pan_state = 'center'
                last_pan_time = 0
                last_crop_box = None

            scene_data = scenes_analysis[current_scene_index]
            strategy = scene_data['strategy']
            target_box = scene_data['target_box']

            if strategy == 'TRACK':
                if scene_data['needs_panning']:
                    scene_time = (frame_number - scene_data['start_frame']) / fps
                    if scene_time > last_pan_time + 3:
                        if pan_state == 'center':
                            pan_state = 'right'
                        elif pan_state == 'right':
                            pan_state = 'left'
                        else: # left
                            pan_state = 'right'
                        last_pan_time = scene_time

                crop_box = calculate_panning_crop_box(target_box, pan_state, last_crop_box, original_width, original_height)
                last_crop_box = crop_box
                processed_frame = frame[crop_box[1]:crop_box[3], crop_box[0]:crop_box[2]]
                output_frame = cv2.resize(processed_frame, (OUTPUT_WIDTH, OUTPUT_HEIGHT))
            else: # LETTERBOX
                scale_factor = OUTPUT_WIDTH / original_width
                scaled_height = int(original_height * scale_factor)
                scaled_frame = cv2.resize(frame, (OUTPUT_WIDTH, scaled_height))
                
                output_frame = np.zeros((OUTPUT_HEIGHT, OUTPUT_WIDTH, 3), dtype=np.uint8)
                y_offset = (OUTPUT_HEIGHT - scaled_height) // 2
                output_frame[y_offset:y_offset + scaled_height, :] = scaled_frame
            
            ffmpeg_process.stdin.write(output_frame.tobytes())
            frame_number += 1
            pbar.update(1)
    
    ffmpeg_process.stdin.close()
    stderr_output = ffmpeg_process.stderr.read().decode()
    ffmpeg_process.wait()
    cap.release()

    if ffmpeg_process.returncode != 0:
        print("\n❌ FFmpeg frame processing failed.")
        print("Stderr:", stderr_output)
        exit()
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

    merge_audio_keep_original(video_path=final_output_video , audio_path='/Users/umeshyadav/Documents/test_audio.mp3',
                           output_path='/Users/umeshyadav/Documents/test_a1.mp4')

def _has_audio_track(video_path):
    """Return True if the video contains an audio stream (uses ffprobe)."""
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "a",
        "-show_entries", "stream=index",
        "-of", "csv=p=0",
        video_path
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    return bool(proc.stdout.strip())

def merge_audio_keep_original(video_path, audio_path, output_path):
    """
    Merge `audio_path` into `video_path` while keeping the video's original audio level unchanged,
    and keeping the new audio at 10% volume.

    Rules:
      - If audio shorter than video -> loop it.
      - If audio longer than video  -> trimmed to video duration.
      - Original video audio unchanged; new audio reduced to 10% before mixing.
    """
    video_has_audio = _has_audio_track(video_path)

    # Quote paths for shell safety
    vq = shlex.quote(video_path)
    aq = shlex.quote(audio_path)
    outq = shlex.quote(output_path)

    if video_has_audio:
        # Use video as input 0 and loop external audio as input 1.
        # Filter chain:
        #  [1:a] -> volume=0.1, aresample -> [a1]
        #  [0:a] -> aresample -> [a0]
        #  [a0][a1] -> amix (inputs=2, duration=first so it follows original audio/video length) -> [mixed]
        # Map 0:v and [mixed]
        cmd = (
            "ffmpeg -y "
            f"-i {vq} "
            f"-stream_loop -1 -i {aq} "
            "-filter_complex "
            "\"[1:a]volume=0.1,aresample=48000[a1];"
            "[0:a]aresample=48000[a0];"
            "[a0][a1]amix=inputs=2:duration=first:dropout_transition=0[mix]\" "
            "-map 0:v -map \"[mix]\" "
            "-c:v copy -c:a aac -b:a 192k "
            "-shortest "
            f"{outq}"
        )
    else:
        # No original audio -> just use external audio (looped, volume=0.1)
        cmd = (
            "ffmpeg -y "
            f"-i {vq} "
            f"-stream_loop -1 -i {aq} "
            "-filter_complex "
            "\"[1:a]volume=0.1,aresample=48000[a1]\" "
            "-map 0:v -map \"[a1]\" "
            "-c:v copy -c:a aac -b:a 192k "
            "-shortest "
            f"{outq}"
        )

    print("Running ffmpeg command:")
    print(cmd)
    subprocess.run(cmd, shell=True, check=True)

def remove_audio(video_path, output_path):
    """
    Removes all audio streams from the input video.
    Video is copied without re-encoding.
    """
    cmd = (
        "ffmpeg -y "
        f"-i {shlex.quote(video_path)} "
        "-c:v copy -an "
        f"{shlex.quote(output_path)}"
    )

    print("Running:", cmd)
    subprocess.run(cmd, shell=True, check=True)

def ffprobe_json(path):
    """Return ffprobe JSON for the file."""
    cmd = [
        "ffprobe",
        "-v", "error",
        "-print_format", "json",
        "-show_format",
        "-show_streams",
        str(path)
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        print(f"ffprobe failed for {path}:\n{res.stderr}", file=sys.stderr)
        sys.exit(res.returncode)
    return json.loads(res.stdout)

def get_video_stream_info(path):
    """Return (width, height, duration_seconds) for the first video stream."""
    info = ffprobe_json(path)
    # find first video stream
    streams = info.get("streams", [])
    vstream = None
    for s in streams:
        if s.get("codec_type") == "video":
            vstream = s
            break
    if not vstream:
        print(f"No video stream found in {path}", file=sys.stderr)
        sys.exit(1)

    width = int(vstream.get("width"))
    height = int(vstream.get("height"))

    # duration: prefer format.duration then stream.duration
    duration = None
    fmt = info.get("format", {})
    if fmt.get("duration"):
        duration = float(fmt["duration"])
    else:
        # fallback to stream duration
        duration = float(vstream.get("duration", 0.0))

    if duration <= 0:
        print(f"Could not determine duration for {path} (got {duration})", file=sys.stderr)
        sys.exit(1)

    return width, height, duration

def build_ffmpeg_command(fg_path, bg_path, out_path, shrink_percent, fg_width, fg_height, duration, crf, preset):
    """
    Strategy:
    - Use -stream_loop -1 for background input so it loops indefinitely (ffmpeg option must appear before -i).
    - Limit output duration with -t <duration> so the looped background is trimmed to the foreground length.
    - Filter graph:
        [0:v] -> background (scale to fg resolution)
        [1:v] -> foreground (scale width to shrink% of foreground width; height auto with -2)
        overlay them with foreground centered
    - Map video from filter output, map foreground audio (input index 1) if present.
    """

    shrink_factor = shrink_percent / 100.0

    # Use numeric canvas size (fg_width x fg_height) for precise scaling of background
    filter_complex = (
        # note: input order in the command will be: bg (index 0, looped), fg (index 1)
        # scale background to foreground canvas, ensure valid SAR
        f"[0:v]scale={fg_width}:{fg_height},setsar=1[bgscaled];"
        # scale foreground width to shrink% of its original width; -2 makes height even
        f"[1:v]scale=iw*{shrink_factor}:-2[fgscaled];"
        # overlay centered
        "[bgscaled][fgscaled]overlay=(main_w-overlay_w)/2:(main_h-overlay_h)/2:format=auto[outv]"
    )

    cmd = [
        "ffmpeg",
        "-y",
        # loop background infinitely (will be trimmed by -t)
        "-stream_loop", "-1",
        "-i", str(bg_path),
        "-i", str(fg_path),
        "-filter_complex", filter_complex,
        "-map", "[outv]",
        # map foreground audio (input 1)
        "-map", "1:a?",
        "-c:v", "libx264",
        "-crf", str(crf),
        "-preset", preset,
        "-c:a", "copy",
        # limit output duration to foreground duration (avoid overrun if bg longer/looped)
        "-t", f"{duration:.3f}",
        "-movflags", "+faststart",
        str(out_path)
    ]

    return cmd

def run(cmd):
    print("Running:\n", " ".join(shlex.quote(c) for c in cmd))
    subprocess.run(cmd, check=True)

def main():
    parser = argparse.ArgumentParser(description="Overlay foreground (shrunken) on a looped/truncated background video.")
    parser.add_argument("--foreground", "-f", required=True, help="Foreground video path (canvas source).")
    parser.add_argument("--background", "-b", required=True, help="Background video path (will be looped/trimmed to match foreground).")
    parser.add_argument("--output", "-o", required=True, help="Output path.")
    parser.add_argument("--shrink", type=float, default=50.0, help="Shrink percent for foreground width (e.g., 50 => 50%% width). Default: 50")
    parser.add_argument("--crf", type=int, default=18, help="CRF for libx264 (lower => higher quality). Default: 18")
    parser.add_argument("--preset", type=str, default="medium", help="x264 preset (ultrafast, fast, medium, slow). Default: medium")
    args = parser.parse_args()

    fg_path = Path(args.foreground)
    bg_path = Path(args.background)
    out_path = Path(args.output)

    # basic checks
    if not fg_path.exists():
        print(f"Foreground not found: {fg_path}", file=sys.stderr)
        sys.exit(1)
    if not bg_path.exists():
        print(f"Background not found: {bg_path}", file=sys.stderr)
        sys.exit(1)

    # check_tool("ffmpeg")
    # check_tool("ffprobe")

    # gather foreground info (canvas dims + duration)
    fg_w, fg_h, fg_duration = get_video_stream_info(fg_path)
    print(f"Foreground resolution: {fg_w}x{fg_h}, duration: {fg_duration:.3f} s")

    # Build and run ffmpeg command
    cmd = build_ffmpeg_command(
        fg_path=fg_path,
        bg_path=bg_path,
        out_path=out_path,
        shrink_percent=args.shrink,
        fg_width=fg_w,
        fg_height=fg_h,
        duration=fg_duration,
        crf=args.crf,
        preset=args.preset
    )

    run(cmd)
    print(f"Done — output written to: {out_path}")

def test_main(input_path, output_path):
    if not Path(input_path).exists():
        print(f"Input file does not exist: {input_path}", file=sys.stderr)
        sys.exit(1)

    fg_w, fg_h, fg_duration = get_video_stream_info(input_path)
    print(f"Foreground resolution: {fg_w}x{fg_h}, duration: {fg_duration:.3f} s")

    # Build and run ffmpeg command
    cmd = build_ffmpeg_command(
        fg_path=input_path,
        bg_path='/Users/umeshyadav/Downloads/bg.mp4',
        out_path=output_path,
        shrink_percent=50,
        fg_width=fg_w,
        fg_height=fg_h,
        duration=fg_duration,
        crf=18,
        preset='fast'
    )

    run(cmd)
    print(f"Finished. Output written to: {output_path}")

def test2():
    cmd = build_ffmpeg_command(
        fg_path=fg_path,
        bg_path=bg_path,
        out_path=out_path,
        shrink_percent=args.shrink,
        fg_width=fg_w,
        fg_height=fg_h,
        duration=fg_duration,
        crf=args.crf,
        preset=args.preset
    )

    run(cmd)

if __name__ == '__main__':
    # parser = argparse.ArgumentParser(description="Smartly crops a horizontal video into a vertical one.")
    # parser.add_argument('-i', '--input', type=str, required=True, help="Path to the input video file.")
    # parser.add_argument('-o', '--output', type=str, required=True, help="Path to the output video file.")
    # args = parser.parse_args()
    #
    # input_v = args.input
    # final_output = args.output
    # run_conversion('/Users/umeshyadav/Documents/test.mp4', '/Users/umeshyadav/Documents/test_1.mp4')
    merge_audio_keep_original(video_path='/Users/umeshyadav/Documents/test.mp4', audio_path='/Users/umeshyadav/Documents/test_audio.mp3',
                           output_path='/Users/umeshyadav/Documents/test_a1.mp4')
    test_main('/Users/umeshyadav/Documents/test_a1.mp4', '/Users/umeshyadav/Documents/test_a2_3h.mp4')
    # remove_audio('/Users/umeshyadav/Documents/test_a1.mp4', '/Users/umeshyadav/Documents/test_a2.mp4')
    # run_conversion('/Users/umeshyadav/Documents/test.mp4', '/Users/umeshyadav/Documents/test_1.mp4')