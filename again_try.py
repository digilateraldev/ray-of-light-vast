# import cv2
# import numpy as np
import rembg
# from rembg import new_session
from moviepy import VideoFileClip
import cv2
# import numpy as np
import gc
# from rembg import new_session
import os
import re
from PIL import Image
# import moviepy as vfx 
import argparse
import shutil
# from faster_whisper import WhisperModel
from moviepy import VideoFileClip, AudioFileClip, CompositeVideoClip, TextClip
from df.enhance import enhance, init_df, load_audio, save_audio
import numpy as np
import ffmpeg
from pydub import AudioSegment
import librosa
import uuid
import mediapipe as mp
import soundfile as sf
import subprocess
import time
from datetime import timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing
import json
# from moviepy.config import change_settings
# change_settings({"IMAGEMAGICK_BINARY": 'imagemagick'})


os.environ["IMAGEMAGICK_BINARY"] = '/usr/bin/convert'
session = rembg.new_session("u2net_human_seg",providers=["CUDAExecutionProvider", "CPUExecutionProvider"])




# Fix for the ANTIALIAS deprecation in Pillow
if not hasattr(Image, 'ANTIALIAS'):
    Image.ANTIALIAS = Image.LANCZOS

# def ensure_output_directory(base_directory, sub_directory):
#     # Create the full path for the subdirectory
#     full_path = os.path.join(base_directory, sub_directory)
    
#     if not os.path.exists(full_path):
#         os.makedirs(full_path)
#         print(f"Created directory: {full_path}")
#     return full_path

uuid_dir = str(uuid.uuid1())
temp_folder = f"audio_processing_{uuid_dir}"
os.makedirs(temp_folder, exist_ok=True)
# frames_dir = ensure_output_directory(temp_folder, "frames_dir")
# processed_dir = ensure_output_directory(temp_folder, "processed_dir")
normalvideo = os.path.join(temp_folder,"normal_video.mp4")
output_video_with_audio =  os.path.join(temp_folder, 'final_video_with_audio.mp4')  # Final video path




output_frames_dir = 'extracted_frames'  # Directory to save extracted frames
output_rembg_dir = 'rembg_frames'  # Directory to save frames after background removal
final_output_dir = 'final_output' 
# Overlay images
template_image_path = 'digi.png'  # Foreground overlay (on top)
background_image_path = 'bg1.jpg'  # Background overlay (behind)

# Define bounding box for rembg image placement
box_x1, box_y1 = 109, 173   # Top-left corner of the box
box_x2, box_y2 = 969, 1820  # Bottom-right corner of the box

# Create directories if they don't exist
# output_frames_dir  = ensure_output_directory(temp_folder, output_frames_dir)
# output_rembg_dir = ensure_output_directory(temp_folder, output_rembg_dir)
# final_output_dir = ensure_output_directory(temp_folder, final_output_dir)



# Load Haar Cascade for face detection
# face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Load MediaPipe Selfie Segmentation
mp_selfie_segmentation = mp.solutions.selfie_segmentation
selfie_segmentation = mp_selfie_segmentation.SelfieSegmentation(model_selection=1)



def add_text_overlay(video_clip, text1, text2, text3):
    # Add text overlays to the video
    text_clip_1 = TextClip(text=text1, font_size=60, color='blue', font="./Poppins-Bold.ttf", bg_color='#00000000',size=(1080,151))
    text_clip_1 = text_clip_1.with_position([0, 1580]).with_duration(video_clip.duration).rotated(4)

    text_clip_2 = TextClip(text=text2, font_size=40, color='black', font="./Poppins-Regular.ttf", bg_color='#00000000',size=(1080,140))
    text_clip_2 = text_clip_2.with_position([0, 1650]).with_duration(video_clip.duration).rotated(4)

    text_clip_3 = TextClip(text=text3, font_size=40, color='black', font="./Poppins-Regular.ttf", bg_color='#00000000',size=(1080,140))
    text_clip_3 = text_clip_3.with_position([0, 1700]).with_duration(video_clip.duration).rotated(4)

    return CompositeVideoClip([video_clip, text_clip_1, text_clip_2, text_clip_3])






# Function to apply MediaPipe Selfie Segmentation and sharpening
def apply_selfie_segmentation_and_sharpening(frame):
    # Convert the frame to RGB
    
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Process the frame
    results = selfie_segmentation.process(rgb_frame)
    
    # Create a mask from the segmentation result
    mask = results.segmentation_mask > 0.5

    # Apply the mask to the frame
    mask = mask.astype(np.uint8) * 255
    mask = cv2.GaussianBlur(mask, (101, 101), 0)
    segmented_frame = cv2.bitwise_and(frame, frame, mask=mask)

    # Apply sharpening
    sharpening_kernel = np.array([[-1, -1, -1],
                                  [-1,  9, -1],
                                  [-1, -1, -1]])
    sharpened_image = cv2.filter2D(segmented_frame, -1, sharpening_kernel)
    return sharpened_image

# Function to process frames with MediaPipe Selfie Segmentation, sharpening, and rembg
def process_frame_with_segmentation_and_rembg(frame):
    # Apply MediaPipe Selfie Segmentation and sharpening
    img = apply_selfie_segmentation_and_sharpening(frame)
    if img is None:
        print(f"Failed to remove background from frame.")
        return None
    return img

# Function to convert Kelvin temperature to RGB color
def kelvin_to_rgb(kelvin):
    kelvin = max(1000, min(40000, kelvin))  # Clamp to valid range
    if kelvin < 6600:
        red = 255
        green = max(0, min(255, 99.4708025861 * np.log(kelvin / 100) - 161.1195681661))
        blue = max(0, min(255, 138.5177312231 * np.log(kelvin / 100) - 305.0447927307))
    else:
        red = max(0, min(255, 329.698727446 * ((kelvin / 100 - 60) ** -0.1332047592)))
        green = max(0, min(255, 288.1221695283 * ((kelvin / 100 - 60) ** -0.0755148492)))
        blue = 255
    return np.array([blue, green, red], dtype=np.float32) / 255.0

# Function to apply the ray of light effect from two light sources (left and right)
def apply_ray_of_light_to_frame_vec(frame, light_left_x, light_left_y, light_right_x, light_right_y, kelvin, light_intensity):
    # frame: HxWx3 uint8 (RGB or BGR — keep consistent)
    h, w = frame.shape[:2]

    # Create coordinate grids
    y_coords, x_coords = np.indices((h, w))

    # Distances (vectorized)
    dist_left = np.sqrt((x_coords - light_left_x) ** 2 + (y_coords - light_left_y) ** 2)
    dist_right = np.sqrt((x_coords - light_right_x) ** 2 + (y_coords - light_right_y) ** 2)

    max_distance = np.sqrt((w // 2) ** 2 + (h // 2) ** 2)
    mask_left = np.clip(1 - (dist_left / max_distance), 0, 1)
    mask_right = np.clip(1 - (dist_right / max_distance), 0, 1)

    # Smooth masks using Gaussian blur via cv2 (faster than scipy for many systems)
    # Convert masks to float32 images and blur
    mask_left = cv2.GaussianBlur((mask_left * 255).astype(np.uint8), (51, 51), 0).astype(np.float32) / 255.0
    mask_right = cv2.GaussianBlur((mask_right * 255).astype(np.uint8), (51, 51), 0).astype(np.float32) / 255.0

    # Scale by intensity
    mask_left = np.clip(mask_left * light_intensity, 0, 1)[:, :, np.newaxis]
    mask_right = np.clip(mask_right * light_intensity, 0, 1)[:, :, np.newaxis]

    # Convert frame to float [0,1]
    frame_float = frame.astype(np.float32) / 255.0

    # Kelvins -> RGB color (function already in your file)
    light_color = kelvin_to_rgb(kelvin)  # returns [B,G,R] normalized
    light_color = light_color[np.newaxis, np.newaxis, :]  # shape (1,1,3)

    # Apply effects
    effect = mask_left * light_color + mask_right * light_color
    result = frame_float + effect
    result = np.clip(result, 0, 1)
    result_uint8 = (result * 255).astype(np.uint8)
    return result_uint8

def process_single_frame_in_memory(frame_bgr, intensity):
    """ Accepts BGR uint8 frame (from cv2), returns final RGBA PIL Image or BGR uint8 depending on your choice.
        We'll return an RGBA NumPy array (HxWx4 uint8) to keep alpha available for template compositing.
    """
    # Convert to RGB because rembg/remove often works on RGB arrays or PIL
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

    # remove background (uses your global session)
    try:
        rembg_result = rembg.remove(frame_rgb, session=session)
    except Exception as e:
        # fallback: if rembg fails, use original
        rembg_result = frame_rgb
        print("rembg failure:", e)

    # Ensure numpy uint8 and 3/4 channels
    if isinstance(rembg_result, Image.Image):
        rembg_np = np.array(rembg_result)
    else:
        rembg_np = np.array(rembg_result)

    # If rembg returned RGBA or RGB, normalize to RGBA
    if rembg_np.ndim == 2:
        rembg_np = np.stack([rembg_np]*3, axis=-1)
    if rembg_np.shape[-1] == 3:
        alpha_ch = np.full((rembg_np.shape[0], rembg_np.shape[1], 1), 255, dtype=np.uint8)
        rembg_np = np.concatenate([rembg_np, alpha_ch], axis=-1)

    rembg_rgba = rembg_np  # HxWx4 RGBA

    # Compose rembg into background template (resizing kept)
    # Reuse your resize_inside_box but operate with PIL once per frame
    rembg_pil = Image.fromarray(rembg_rgba)
    rembg_resized = resize_inside_box(rembg_pil, (box_x1, box_y1, box_x2, box_y2))
    background = Image.open(background_image_path).convert('RGBA')
    background_resized = background.resize(Image.open(template_image_path).size, Image.ANTIALIAS)
    combined = background_resized.copy()
    combined.paste(rembg_resized, (box_x1, box_y1), rembg_resized)  # uses alpha channel

    # Convert combined to NumPy RGB for lighting processing
    combined_rgb = np.array(combined.convert("RGB"))  # HxWx3 (RGB)
    # Apply vectorized ray-of-light (returns RGB uint8)
    # pick intensity numeric: convert intensity argument to numeric if necessary
    kelvin = 3500
    try:
        processed_rgb = apply_ray_of_light_to_frame_vec(combined_rgb, combined_rgb.shape[1]//4, combined_rgb.shape[0]//2,
                                                       3*combined_rgb.shape[1]//4, combined_rgb.shape[0]//2,
                                                       kelvin, intensity)
    except Exception as e:
        print("Light processing failed:", e)
        processed_rgb = combined_rgb

    # Convert processed to RGBA (add alpha fully opaque)
    alpha_channel = np.full((processed_rgb.shape[0], processed_rgb.shape[1], 1), 255, dtype=np.uint8)
    processed_rgba = np.concatenate([processed_rgb, alpha_channel], axis=-1)

    # Overlay the top template (template3.png) as you did before
    template = Image.open(template_image_path).convert('RGBA')
    processed_pil = Image.fromarray(processed_rgba)
    # Ensure same size before alpha_composite
    if processed_pil.size != template.size:
        processed_pil = processed_pil.resize(template.size, Image.ANTIALIAS)
    final_image = Image.alpha_composite(processed_pil, template)

    # Return final as BGR uint8 (ready for cv2.VideoWriter) to avoid more conversions later
    final_rgb = np.array(final_image.convert("RGB"))
    final_bgr = cv2.cvtColor(final_rgb, cv2.COLOR_RGB2BGR)
    return final_bgr

# Fast parallel in-memory frame processing + streaming writer
def process_video_frames_in_memory(input_video_path, output_video_path, intensity_numeric):
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        print("Cannot open video:", input_video_path)
        return False

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    # Determine output size from template
    template = Image.open(template_image_path).convert('RGBA')
    out_width, out_height = template.size
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_video_path, fourcc, fps, (out_width, out_height))

    max_workers = min(max(1, multiprocessing.cpu_count() - 2), 8)
    CHUNK = 10

    frame_idx = 0
    print(f"Starting in-memory processing: fps={fps}, frames={frame_count}, workers={max_workers}, chunk={CHUNK}")

    executor = ThreadPoolExecutor(max_workers=max_workers)

    while True:
        frames_buffer = []
        for _ in range(CHUNK):
            ret, frame = cap.read()
            if not ret:
                break
            frame_resized = cv2.resize(frame, (out_width, out_height), interpolation=cv2.INTER_AREA)
            frames_buffer.append((frame_idx, frame_resized))
            frame_idx += 1

        if not frames_buffer:
            break

        futures = {executor.submit(process_single_frame_in_memory, fr, intensity_numeric): idx
                   for idx, fr in frames_buffer}

        results = {}
        for fut in as_completed(futures):
            idx = futures[fut]
            try:
                results[idx] = fut.result()
            except Exception as e:
                print(f"Frame {idx} failed:", e)
                results[idx] = np.zeros((out_height, out_width, 3), dtype=np.uint8)

        # Write frames in correct order
        for idx, _ in frames_buffer:
            writer.write(results[idx])

        if frame_idx % 50 == 0:
            print(f"Processed {frame_idx}/{frame_count} frames...")

    cap.release()
    writer.release()
    executor.shutdown(wait=True)
    print(f"Video processing complete. Saved to {output_video_path}")
    return True

    # Fallback simple processing if chunk parallel loop aborted above:
    # If the chunk approach aborted above to keep the snippet robust, process frames sequentially as a fallback:
    # We'll re-open and process sequentially but still use the faster vectorized light function:
    # cap.release()
    # print("Falling back to sequential streaming processing (re-open video).")
    # cap = cv2.VideoCapture(input_video_path)
    # # fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    # writer = cv2.VideoWriter(output_video_path, fourcc, fps, (out_width, out_height))
    # idx = 0
    # while True:
    #     ret, frame = cap.read()
    #     if not ret:
    #         break
    #     frame_resized = cv2.resize(frame, (out_width, out_height), interpolation=cv2.INTER_AREA)
    #     processed = process_single_frame_in_memory(frame_resized, intensity_numeric)
    #     writer.write(processed)
    #     idx += 1
    #     if idx % 50 == 0:
    #         print(f"Processed {idx} frames...")

    # cap.release()
    # writer.release()
    # print("Video processing (sequential fallback) done.")
    # return True

# Function to process frames with ray of light effect

# Create a temporary folder for storing intermediate files
# Function to clean up the temp folder after processing
def clean_up_temp_folder(temp_folder):
    try:
        if os.path.exists(temp_folder):
            shutil.rmtree(temp_folder)
            print(f"Deleted temporary folder: {temp_folder}")
    except Exception as e:
        print(f"Error while deleting temporary folder: {e}")

# Main function that takes the input video file as a command-line argument
def main(input_video_file, temp_folder):
    normalvideo = os.path.join(temp_folder,"normal_video.mp4")
    final_output_path = os.path.join(temp_folder, 'final_output.wav')  # Final Audio Path before final_video
    output_video_with_audio =  os.path.join(temp_folder, 'final_video_with_audio.mp4')  # Final video path
    # Final Audio substile path
    # Step 1: Extract audio from video
    extracted_audio_path = extract_audio_from_video(input_video_file, os.path.join(temp_folder, "extracted_audio.wav"))
    if not extracted_audio_path:
        return
    
    # Step 3: Convert the extracted audio to wav format (if necessary)
    # wav_audio_path = convert_to_wav(extracted_audio_path, output_file=os.path.join(temp_folder, "temp.wav"))
    # if not wav_audio_path:
    #     return

    # Step 4: Initialize DeepFilterNet model
    model, df_state, _ = init_df()

    # Load the audio file with the appropriate sample rate for DeepFilterNet
    audio, _ = load_audio(extracted_audio_path, sr=df_state.sr())

    # Denoise the audio using DeepFilterNet
    enhanced = enhance(model, df_state, audio)

    # Convert enhanced audio tensor back to NumPy array and remove the batch dimension
    enhanced = enhanced.squeeze().cpu().numpy()

    # Save the denoised audio to a wav file in the temp folder
    result_wav_path = os.path.join(temp_folder, "result3.wav")
    save_audio(result_wav_path, enhanced, df_state.sr())

    # Set file paths for the processing steps
    input_audio_path = result_wav_path  # Output from DeepFilterNet
    pitch_corrected_path = os.path.join(temp_folder, 'corrected_pitch.wav')
    aligned_audio_path = os.path.join(temp_folder, 'aligned_audio.wav')
    # Step 5: Pitch correction
    def correct_pitch(input_path, output_path):
        y, sr = librosa.load(input_path, sr=None)
        n_steps = 0  # Adjust this value to correct pitch (positive to shift up, negative to shift down)
        y_corrected = librosa.effects.pitch_shift(y, n_steps=n_steps, sr=sr)
        sf.write(output_path, y_corrected, sr)
        print(f"Pitch correction applied and saved to {output_path}")

    correct_pitch(input_audio_path, pitch_corrected_path)

    # Step 6: Time alignment
    def time_alignment(input_path, output_path, stretch_rate=1.0):
        y, sr = librosa.load(input_path, sr=None)
        y_stretched = librosa.effects.time_stretch(y, rate=stretch_rate)
        sf.write(output_path, y_stretched, sr)
        print(f"Time alignment applied and saved to {output_path}")

    time_alignment(pitch_corrected_path, aligned_audio_path)

    # Step 7: Final normalization using SoX
    def normalize_audio(input_path, output_path):
        sox_path = "sox"  # Change as needed
        command = [sox_path, input_path, output_path, 'gain', '-1']
        try:
            subprocess.run(command, check=True)
            print(f"Final normalization applied and saved to {output_path}")
        except subprocess.CalledProcessError as e:
            print(f"Error during normalization: {e}")

    normalize_audio(aligned_audio_path, final_output_path)

    # Step 8: Apply gain boost using pydub
    boost_db = 5  # Boost gain (you can adjust this)
    audio = AudioSegment.from_wav(final_output_path)
    # Apply gain boost
    boosted_audio = audio.apply_gain(boost_db)
    boosted_audio.export(final_output_path, format="wav")
    y_boosted, sr_boosted = librosa.load(final_output_path, sr=None)
    # Apply dynamic range compression to prevent clipping
    compressed_audio = boosted_audio.compress_dynamic_range(threshold=-40.0, ratio=2.0)
    compressed_audio.export(final_output_path, format="wav")
    print(f"Gain boosted by {boost_db} dB and saved to {final_output_path}")

    # Step 9: Merge the final audio back into the video
    video_with_audio_obj=merge_audio_video( normalvideo , final_output_path, output_video_with_audio)
    return video_with_audio_obj
    # print("Processing complete. Final video with processed audio saved at", output_video_with_audio)

# Subtitile
def boostaudio(audio, temp_folder):
    boost_sub = os.path.join(temp_folder, 'boost_sub.mp3')  # Final Audio substile path
    
    # Load the audio file
    audio = AudioSegment.from_file(audio)  # Change the filename and format as needed
    # Increase the volume (in dB)
    increase_db = 10.0  # Change this value to increase or decrease the volume
    louder_audio = audio + increase_db
    # Export the modified audio
    louder_audio.export(boost_sub, format="mp3")  # Change the filename and format as needed

# def whisperfun(audio, temp_folder):
#     audio_subtitiles = os.path.join(temp_folder, 'sub.srt')
#     # Load the model (choose from tiny, base, small, medium, large)
#     model = WhisperModel.load_model("medium")
#     # Load your audio file
#     audio_path = audio  # Replace with your audio file path
#     # Transcribe the audio into English
#     result = model.transcribe(audio_path, language='en')
#     # Get the segments with timestamps and text
#     segments = result['segments']
    
    # Function to convert seconds to the SRT timestamp format
    # def format_timestamp(seconds):
    #     hours = int(seconds // 3600)
    #     minutes = int((seconds % 3600) // 60)
    #     seconds = seconds % 60
    #     milliseconds = int((seconds - int(seconds)) * 1000)
    #     return f"{hours:02}:{minutes:02}:{int(seconds):02},{milliseconds:03}"

    # # Save the transcription in SRT format
    # with open(audio_subtitiles, "w", encoding="utf-8") as srt_file:
    #     for i, segment in enumerate(segments):
    #         start = format_timestamp(segment['start'])
    #         end = format_timestamp(segment['end'])
    #         text = segment['text'].strip()

    #         srt_file.write(f"{i + 1}\n")
    #         srt_file.write(f"{start} --> {end}\n")
    #         srt_file.write(f"{text}\n\n")

    # print("Transcription saved in SRT format.")

# Function to convert any audio format to wav using ffmpeg-python
# def convert_to_wav(input_file, output_file="temp.wav"):
#     try:
#         # Use ffmpeg to convert input audio to wav
#         ffmpeg.input(input_file).output(output_file, format="wav").run(overwrite_output=True)
#         print(f"Converted {input_file} to {output_file}")
#         return output_file
#     except ffmpeg.Error as e:
#         print(f"Failed to convert {input_file}. Error: {e}")
#         return None

# Function to parse the SRT file and extract subtitles
# def parse_srt(srt_file):
#     with open(srt_file, 'r') as file:
#         content = file.read()
        
#     # Regular expression to match the SRT format
#     pattern = re.compile(r'(\d+)\n([\d:,]+) --> ([\d:,]+)\n(.*?)\n', re.DOTALL)
#     matches = pattern.findall(content)

#     subtitles = []
#     for match in matches:
#         start_time_str = match[1].replace(',', '.')
#         end_time_str = match[2].replace(',', '.')
#         text = match[3].strip()
        
#         # Convert time strings to seconds
#         start_time = convert_time_to_seconds(start_time_str)
#         end_time = convert_time_to_seconds(end_time_str)

#         subtitles.append((start_time, end_time, text))
    
#     return subtitles

# # Helper function to convert HH:MM:SS.milliseconds to seconds
# def convert_time_to_seconds(time_str):
#     h, m, s = time_str.split(':')
#     return float(h) * 3600 + float(m) * 60 + float(s)



# Audio File
# Function to extract audio from video
def extract_audio_from_video(input_video, output_audio="extracted_audio.wav"):
    try:
        ffmpeg.input(input_video).output(output_audio, format="wav").run(overwrite_output=True)
        print(f"Extracted audio from {input_video} to {output_audio}")
        return output_audio
    except ffmpeg.Error as e:
        print(f"Failed to extract audio from {input_video}. Error: {e}")
        return None

# Function to merge audio and video using moviepy
def merge_audio_video(video_no_audio_path, audio_path, output_video_path):
    try:
        # Load video (without audio)
        video_clip = VideoFileClip(video_no_audio_path)
        
        # Load audio
        audio_clip = AudioFileClip(audio_path)

        # Set the audio to the video
        final_video = video_clip.with_audio(audio_clip)

        # Write the result to the output video file
        return final_video

        # print(f"Merged {audio_path} with {video_no_audio_path} into {output_video_path}")
    
    except Exception as e:
        print(f"Failed to merge audio and video using moviepy. Error: {e}")

# Function to overlay rembg image onto bg.png and template.png


def resize_inside_box(image, box):
    """
    Resizes an image while maintaining aspect ratio, and places it inside a given bounding box.
    """
    x1, y1, x2, y2 = box
    target_width = x2 - x1
    target_height = y2 - y1
    img_width, img_height = image.size

    # Calculate scaling factor while keeping aspect ratio
    scale_factor = min(target_width / img_width, target_height / img_height)
    new_width = int(img_width * scale_factor)
    new_height = int(img_height * scale_factor)

    # Resize the image while maintaining aspect ratio
    resized_image = image.resize((new_width, new_height), Image.ANTIALIAS)

    # Create a transparent background with the bounding box size
    box_image = Image.new("RGBA", (target_width, target_height), (0, 0, 0, 0))

    # Center the resized image inside the bounding box
    paste_x = (target_width - new_width) // 2
    paste_y = (target_height - new_height) // 2
    box_image.paste(resized_image, (paste_x, paste_y), resized_image)
    return box_image



def startpoint(input_video, output_video, text1, text2, text3, intensity):
    try:
        if intensity == "hard":
            light_intensity = 0.25
        elif intensity == "soft":
            light_intensity = 0.15
        else:
            light_intensity = 0.0
        print("intensity value: ", light_intensity)

        # Process frames in-memory and create a temp `normalvideo`
        normalvideo = os.path.join(temp_folder, "normal_video.mp4")
        success = process_video_frames_in_memory(input_video, normalvideo, light_intensity)
        if not success:
            print("Frame processing failed.")
            return

        # Continue with audio processing and final merge (main)
        clip2=main(input_video, temp_folder)

        # At this point main() will produce output_video_with_audio in your temp_folder
        # clip2 = VideoFileClip(output_video_with_audio)
        # Export the final video with text overlay
        text1 = f"Dr. {text1}"
        clip2 = add_text_overlay(clip2, text1, text2, text3)
        fps = clip2.fps if hasattr(clip2, 'fps') else 30
        clip2.write_videofile(output_video, codec="libx264", audio_codec="aac", fps=fps)
        print("Process completed!")
    finally:
        clean_up_temp_folder(temp_folder)
if __name__ == "__main__":
    # Step 1: Create the argument parser
    parser = argparse.ArgumentParser(description="Process a video file with background removal and audio enhancement.")
    
    # Step 2: Define the arguments
    parser.add_argument("video_path", type=str, help="Path to the input video file.")
    parser.add_argument("output_video", type=str, default="result.mp4", help="Path to save the final output video.")
    parser.add_argument("text1", type=str, default="ROhit Chawuhan", help="Text overlay 1.")
    parser.add_argument("text2", type=str, default="Animator", help="Text overlay 2.")
    parser.add_argument("text3", type=str, default="Andheri West, MUmbai 400058", help="Text overlay 3.")
    parser.add_argument("--intensity", type=str, choices=["soft", "hard"], default="default", help="Set light intensity (soft=0.15 / hard=0.25 / default=0)")
    

    # Step 3: Parse the arguments
    args = parser.parse_args()
    try:
        starttime=time.time()
        # vid_path=rotate_video(args.video_path,f'{args.video_path}_rotated.mp4')
        vid_path=os.path.join(temp_folder,'rotated_video.mp4')
        cmd = [
        "ffmpeg",
        "-i", args.video_path,
        "-c:a", "copy", vid_path
        ]
        subprocess.run(cmd)
        startpoint(vid_path, args.output_video, args.text1, args.text2, args.text3, args.intensity)
    finally:
        endtime=time.time()
        elapsed_time=endtime-starttime
        print(timedelta(seconds=elapsed_time))

