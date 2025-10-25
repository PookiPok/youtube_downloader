"""
YouTube Video Face Emotion Analyzer
Extracts frames from a YouTube video, detects faces, analyzes emotions,
and saves cropped faces organized by emotion category.

Required packages:
pip install opencv-python deepface yt-dlp pillow numpy tensorflow tf-keras
"""
import sys
import cv2
import os
from deepface import DeepFace
import yt_dlp
from pathlib import Path
import numpy as np
from datetime import datetime

class YouTubeFaceEmotionAnalyzer:
    def __init__(self, output_dir="emotion_faces"):
        """
        Initialize the analyzer with output directory.
        
        Args:
            output_dir: Base directory to save cropped face images
        """
        self.output_dir = output_dir
        self.emotions = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
        self.setup_directories()
        
    def setup_directories(self):
        """Create directories for each emotion category."""
        for emotion in self.emotions:
            emotion_path = os.path.join(self.output_dir, emotion)
            Path(emotion_path).mkdir(parents=True, exist_ok=True)
        print(f"✓ Created emotion directories in '{self.output_dir}'")
    
    def download_video(self, youtube_url):
        """
        Download YouTube video to temporary location.
        
        Args:
            youtube_url: URL of the YouTube video
            
        Returns:
            Path to downloaded video file
        """
        print(f"Downloading video from: {youtube_url}")
        
        ydl_opts = {
            'format': 'best[ext=mp4]',
            'outtmpl': 'temp_video.%(ext)s',
            'quiet': False,
        }
        
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([youtube_url])
            print("✓ Video downloaded successfully")
            return "temp_video.mp4"
        except Exception as e:
            print(f"✗ Error downloading video: {e}")
            return None
    
    def detect_faces(self, frame):
        """
        Detect faces in a frame using OpenCV's Haar Cascade.
        
        Args:
            frame: Image frame from video
            
        Returns:
            List of face bounding boxes (x, y, w, h)
        """
        # Load face cascade classifier
        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        
        # Convert to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )
        

        return faces
    
    def analyze_emotion(self, face_img):
        """
        Analyze emotion of a face using DeepFace.
        
        Args:
            face_img: Cropped face image
            
        Returns:
            Dominant emotion string or None if analysis fails
        """
        try:
            # DeepFace expects RGB format
            face_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
            
            # Analyze emotion
            result = DeepFace.analyze(
                face_rgb,
                actions=['emotion'],
                enforce_detection=False,
                silent=True
            )
            
            # Extract dominant emotion
            if isinstance(result, list):
                result = result[0]
            
            dominant_emotion = result['dominant_emotion']
            return dominant_emotion
            
        except Exception as e:
            print(f"  Warning: Could not analyze emotion: {e}")
            return None
    
    def save_face(self, face_img, emotion, frame_num, face_num):
        """
        Save cropped face image to appropriate emotion folder.
        
        Args:
            face_img: Cropped face image
            emotion: Detected emotion
            frame_num: Frame number
            face_num: Face number in frame
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"frame{frame_num:06d}_face{face_num}_{timestamp}.jpg"
        filepath = os.path.join(self.output_dir, emotion, filename)
        
        cv2.imwrite(filepath, face_img)
        return filepath
    
    def process_video(self, youtubes_url, frame_skip=30):
        """
        Main processing function: download video, extract frames, detect faces,
        analyze emotions, and save cropped faces.
        
        Args:
            youtube_url: URL of YouTube video
            frame_skip: Process every Nth frame (default: 30, ~1 per second for 30fps video)
        """
        result = []
        for i, youtube_url in enumerate(youtubes_url, 1):
            # Download video
            print(youtube_url)
            video_path = self.download_video(youtube_url)
            if not video_path:
                return
            
            # Open video
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                print("✗ Error: Could not open video file")
                return
            
            # Get video properties
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            
            print(f"\nVideo Info:")
            print(f"  Total frames: {total_frames}")
            print(f"  FPS: {fps}")
            print(f"  Processing every {frame_skip} frames\n")
            
            frame_count = 0
            processed_count = 0
            face_count = 0
            emotion_stats = {emotion: 0 for emotion in self.emotions}
            
            print("Processing video frames...")
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Process only every Nth frame
                if frame_count % frame_skip == 0:
                    processed_count += 1
                    
                    # Detect faces in frame
                    faces = self.detect_faces(frame)
                    
                    if len(faces) > 0:
                        print(f"Frame {frame_count}: Found {len(faces)} face(s)")
                    
                    # Process each detected face
                    for i, (x, y, w, h) in enumerate(faces):
                        # Add padding to face crop
                        padding = int(0.2 * w)
                        x1 = max(0, x - padding)
                        y1 = max(0, y - padding)
                        x2 = min(frame.shape[1], x + w + padding)
                        y2 = min(frame.shape[0], y + h + padding)
                        
                        # Crop face
                        face_img = frame[y1:y2, x1:x2]
                        
                        # Analyze emotion
                        emotion = self.analyze_emotion(face_img)
                        
                        if emotion:
                            # Save cropped face
                            filepath = self.save_face(face_img, emotion, frame_count, i)
                            face_count += 1
                            emotion_stats[emotion] += 1
                            print(f"  Face {i+1}: {emotion} → {os.path.basename(filepath)}")
                
                frame_count += 1
                
                # Progress update
                if frame_count % 300 == 0:
                    progress = (frame_count / total_frames) * 100
                    print(f"\nProgress: {progress:.1f}% ({frame_count}/{total_frames} frames)\n")
            
            # Cleanup
            cap.release()
            
            # Remove temporary video file
            try:
                os.remove(video_path)
                print("\n✓ Cleaned up temporary video file")
            except:
                pass
            
            # Print summary
            print("\n" + "="*60)
            print("PROCESSING COMPLETE")
            print("="*60)
            print(f"Total frames processed: {processed_count}")
            print(f"Total faces detected: {face_count}")
            print(f"\nEmotion breakdown:")
            for emotion, count in sorted(emotion_stats.items(), key=lambda x: x[1], reverse=True):
                if count > 0:
                    percentage = (count / face_count * 100) if face_count > 0 else 0
                    print(f"  {emotion.capitalize():10s}: {count:4d} ({percentage:5.1f}%)")
            print("="*60)
            result.append({
                'total_frames': processed_count,
                'total_faces': 'face_count',
            })
        return result
    
    def process_single_video(self, youtube_url, frame_skip=30):
        """
        Main processing function: download video, extract frames, detect faces,
        analyze emotions, and save cropped faces.
        
        Args:
            youtube_url: URL of YouTube video
            frame_skip: Process every Nth frame (default: 30, ~1 per second for 30fps video)
        """
        result = []
        # Download video
        video_path = self.download_video(youtube_url)
        if not video_path:
            return
        
        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print("✗ Error: Could not open video file")
            return
        
        # Get video properties
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        print(f"\nVideo Info:")
        print(f"  Total frames: {total_frames}")
        print(f"  FPS: {fps}")
        print(f"  Processing every {frame_skip} frames\n")
        
        frame_count = 0
        processed_count = 0
        face_count = 0
        emotion_stats = {emotion: 0 for emotion in self.emotions}
        
        print("Processing video frames...")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process only every Nth frame
            if frame_count % frame_skip == 0:
                processed_count += 1
                
                # Detect faces in frame
                faces = self.detect_faces(frame)
                
                if len(faces) > 0:
                    print(f"Frame {frame_count}: Found {len(faces)} face(s)")
                
                # Process each detected face
                for i, (x, y, w, h) in enumerate(faces):
                    # Add padding to face crop
                    padding = int(0.2 * w)
                    x1 = max(0, x - padding)
                    y1 = max(0, y - padding)
                    x2 = min(frame.shape[1], x + w + padding)
                    y2 = min(frame.shape[0], y + h + padding)
                    
                    # Crop face
                    face_img = frame[y1:y2, x1:x2]
                    
                    # Analyze emotion
                    emotion = self.analyze_emotion(face_img)
                    
                    if emotion:
                        # Save cropped face
                        filepath = self.save_face(face_img, emotion, frame_count, i)
                        face_count += 1
                        emotion_stats[emotion] += 1
                        print(f"  Face {i+1}: {emotion} → {os.path.basename(filepath)}")
            
            frame_count += 1
            
            # Progress update
            if frame_count % 300 == 0:
                progress = (frame_count / total_frames) * 100
                print(f"\nProgress: {progress:.1f}% ({frame_count}/{total_frames} frames)\n")
        
        # Cleanup
        cap.release()
        
        # Remove temporary video file
        try:
            os.remove(video_path)
            print("\n✓ Cleaned up temporary video file")
        except:
            pass
        
        # Print summary
        print("\n" + "="*60)
        print("PROCESSING COMPLETE")
        print("="*60)
        print(f"Total frames processed: {processed_count}")
        print(f"Total faces detected: {face_count}")
        print(f"\nEmotion breakdown:")
        for emotion, count in sorted(emotion_stats.items(), key=lambda x: x[1], reverse=True):
            if count > 0:
                percentage = (count / face_count * 100) if face_count > 0 else 0
                print(f"  {emotion.capitalize():10s}: {count:4d} ({percentage:5.1f}%)")
        print("="*60)
        result.append({
            'total_frames': processed_count,
            'total_faces': 'face_count',
        })
        return result

def load_youtube_links(file_path):
    """Load YouTube links from a text file."""
    links = []
    
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            for line_num, line in enumerate(file, 1):
                line = line.strip()
                
                # Skip empty lines
                if not line:
                    continue
                
                # Check if line contains multiple links separated by delimiter
                if ',' in line or '\t' in line or ' ' in line:
                    # Try splitting by common delimiters
                    parts = re.split(r'[,\t\s]+', line)
                    for part in parts:
                        part = part.strip()
                        if part and ('youtube.com' in part or 'youtu.be' in part):
                            links.append(part)
                            print(f"Line {line_num}: {part}")
                else:
                    # Single link per line
                    if 'youtube.com' in line or 'youtu.be' in line:
                        links.append(line)
  
                        print(f"Line {line_num}: {line}")
    
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        sys.exit(1)
    except Exception as e:
        print(f"Error reading file: {e}")
        sys.exit(1)
    
    return links


def main():

    frame_skip = 30
    youtube_links = []
    if len(sys.argv) < 2:
        print("Usage: python script.py <path_to_txt_file>")
        """Main execution function."""
        print("="*60)
        print("YouTube Video Face Emotion Analyzer")
        print("="*60)
        
        # Get YouTube URL from user
        youtube_links = input("\nEnter YouTube video URL: ").strip()
        
        if not youtube_links:
            print("✗ Error: No URL provided")
            return
        
        # Optional: Set frame skip (process every Nth frame)
        try:
            frame_skip = int(input("Process every N frames (default 30): ").strip() or "30")
        except ValueError:
            frame_skip = 30
    else:
        file_path = sys.argv[1]
        print(f"Loading YouTube links from: {file_path}\n")
        youtube_links = load_youtube_links(file_path)
        
        print(f"\n{'='*50}")
        print(f"Total links found: {len(youtube_links)}")
        print(f"{'='*50}\n")
        
    # Create analyzer and process video
    analyzer = YouTubeFaceEmotionAnalyzer(output_dir="emotion_faces")
    analyzer.process_video(youtube_links, frame_skip=frame_skip)
    
    print(f"\n✓ All cropped faces saved to 'emotion_faces' folder")


if __name__ == "__main__":
    main()