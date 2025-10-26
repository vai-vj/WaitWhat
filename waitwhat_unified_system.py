"""
WaitWhat - Fixed Sync & Video Display Issues
Complete working version with proper UI updates
"""

import gradio as gr
import cv2
import numpy as np
import threading
import time
from datetime import datetime
import json
import os
import tempfile
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification
import yt_dlp
import whisper

# Your LLM system
from waitwhat_question_answer_engine import WaitWhatQuestionAnswerEngine

# =====================
# EMOTION DETECTOR WITH UI CALLBACKS
# =====================
class VITEmotionDetector:
    def __init__(self):
        self.processor = AutoImageProcessor.from_pretrained("trpakov/vit-face-expression")
        self.model = AutoModelForImageClassification.from_pretrained("trpakov/vit-face-expression")
        
        haar_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self.face_cascade = cv2.CascadeClassifier(haar_path)
        
        self.is_monitoring = False
        self.cap = None
        self.current_emotion = "neutral"
        self.sadness_start = None
        self.confusion_events = []
        
        # UI callback for live updates
        self.ui_callback = None
        
    def start_monitoring(self, confusion_callback=None, ui_callback=None):
        self.confusion_callback = confusion_callback
        self.ui_callback = ui_callback
        
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            return False, "Could not access webcam"
        
        self.is_monitoring = True
        monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        monitor_thread.start()
        return True, "Emotion monitoring started"
    
    def stop_monitoring(self):
        self.is_monitoring = False
        if self.cap:
            self.cap.release()
    
    def _monitor_loop(self):
        last_processed = 0.0
        
        while self.is_monitoring:
            if not self.cap:
                break
                
            ret, frame = self.cap.read()
            if not ret:
                continue
            
            now = time.time()
            if now - last_processed < 2.0:
                time.sleep(0.1)
                continue
                
            try:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80))
                
                if len(faces) > 0:
                    (x, y, w, h) = sorted(faces, key=lambda r: r[2]*r[3], reverse=True)[0]
                    
                    pad = int(0.25 * w)
                    x1, y1 = max(0, x - pad), max(0, y - pad)
                    x2, y2 = min(frame.shape[1], x + w + pad), min(frame.shape[0], y + h + pad)
                    face_crop = frame[y1:y2, x1:x2]
                    
                    if face_crop.size > 0:
                        image = Image.fromarray(cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB))
                        inputs = self.processor(images=image, return_tensors="pt")
                        
                        with torch.no_grad():
                            outputs = self.model(**inputs)
                            logits = outputs.logits
                            predicted_class_idx = logits.argmax(-1).item()
                            label = self.model.config.id2label[predicted_class_idx]
                        
                        self.current_emotion = label
                        
                        # Notify UI of emotion change
                        if self.ui_callback:
                            try:
                                self.ui_callback('emotion_update', {'emotion': label})
                            except:
                                pass  # Ignore callback errors
                        
                        # Sadness detection
                        if label.lower() == "sad":
                            if self.sadness_start is None:
                                self.sadness_start = datetime.now()
                                print(f"😔 Confusion started!")
                        else:
                            if self.sadness_start is not None:
                                duration = (datetime.now() - self.sadness_start).total_seconds()
                                
                                if duration > 1.5:
                                    print(f"🎯 CONFUSION DETECTED! Duration: {duration:.1f}s")
                                    
                                    confusion_event = {
                                        'duration': duration,
                                        'timestamp': datetime.now().strftime('%H:%M:%S')
                                    }
                                    
                                    self.confusion_events.append(confusion_event)
                                    
                                    # Notify confusion callback
                                    if self.confusion_callback:
                                        self.confusion_callback(confusion_event)
                                    
                                    # Notify UI
                                    if self.ui_callback:
                                        try:
                                            self.ui_callback('confusion_detected', confusion_event)
                                        except:
                                            pass
                                
                                self.sadness_start = None
                
                last_processed = now
                
            except Exception as e:
                print(f"Emotion error: {e}")
                self.current_emotion = "error"
            
            time.sleep(0.3)

# =====================
# FIXED YOUTUBE DOWNLOADER WITH PROPER FILE HANDLING
# =====================
def download_youtube_fixed(url):
    """Fixed YouTube downloader that works with Gradio"""
    try:
        import tempfile
        import random
        
        # Create unique temp file
        temp_dir = tempfile.gettempdir()
        random_id = random.randint(1000, 9999)
        output_file = os.path.join(temp_dir, f"waitwhat_{random_id}.mp4")
        
        ydl_opts = {
            'outtmpl': output_file,
            'format': 'best[height<=720][ext=mp4]/best[ext=mp4]/best',
            'quiet': False,
            'no_warnings': False,
        }
        
        print(f"📥 Downloading to: {output_file}")
        
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            # Get info first
            info = ydl.extract_info(url, download=False)
            title = info.get('title', 'YouTube Video')
            description = info.get('description', 'Downloaded educational video')
            
            # Download
            ydl.download([url])
            
            # Verify file exists
            if os.path.exists(output_file):
                print(f"✅ Video downloaded successfully: {output_file}")
                return output_file, title, description
            else:
                print(f"❌ File not found after download: {output_file}")
                return None, title, "Download failed - file not found"
                
    except Exception as e:
        print(f"YouTube download error: {e}")
        return None, "Failed Video", f"Error: {str(e)}"

# =====================
# MAIN SYSTEM WITH PROPER UI SYNC
# =====================
class WaitWhatSynced:
    def __init__(self):
        self.emotion_detector = VITEmotionDetector()
        self.llm_engine = WaitWhatQuestionAnswerEngine()
        
        self.current_video_path = None
        self.video_metadata = {}
        self.is_system_running = False
        self.current_questions = []
        self.latest_confusion_event = None
        
        # UI state for live updates
        self.ui_state = {
            'status': "System ready",
            'emotion': "Not monitoring",
            'events': "No events",
            'questions': []
        }
        
    def start_system(self, video_source, youtube_url=None, uploaded_file=None):
        """Start system with proper video handling"""
        print(f"🚀 Starting system with: {video_source}")
        
        video_path = None
        
        if video_source == "YouTube URL" and youtube_url:
            print(f"📥 Processing YouTube URL: {youtube_url}")
            video_path, title, description = download_youtube_fixed(youtube_url)
            
            if video_path and os.path.exists(video_path):
                self.current_video_path = video_path
                self.video_metadata = {'title': title, 'description': description}
                print(f"✅ YouTube video ready")
            else:
                return "❌ Failed to download YouTube video", "Download failed", None
                
        elif video_source == "Upload Video" and uploaded_file:
            print(f"📁 Processing uploaded file: {uploaded_file}")
            if os.path.exists(uploaded_file):
                self.current_video_path = uploaded_file
                self.video_metadata = {
                    'title': 'Uploaded Educational Video',
                    'description': 'User uploaded learning content'
                }
                video_path = uploaded_file
                print(f"✅ Upload video ready")
            else:
                return "❌ Uploaded file not found", "File error", None
        else:
            return "❌ Please provide video source", "No video provided", None
        
        # Start emotion monitoring with UI callback
        success, msg = self.emotion_detector.start_monitoring(
            confusion_callback=self.handle_confusion,
            ui_callback=self.ui_update_callback
        )
        
        if success:
            self.is_system_running = True
            self.ui_state['status'] = f"🔍 Monitoring: {self.video_metadata['title']}"
            
            return (
                f"✅ WaitWhat started for: {self.video_metadata['title']}", 
                f"🎬 Video loaded + 👀 Monitoring emotions for: {self.video_metadata['title']}",
                video_path  # Return video path for player
            )
        else:
            return f"❌ {msg}", "Monitoring failed", None
    
    def ui_update_callback(self, event_type, data):
        """Handle UI updates from background threads"""
        if event_type == 'emotion_update':
            self.ui_state['emotion'] = f"Current: {data['emotion']}"
        elif event_type == 'confusion_detected':
            self.ui_state['events'] = f"⚠️ Confusion at {data['timestamp']} ({data['duration']:.1f}s)"
    
    def stop_system(self):
        self.emotion_detector.stop_monitoring()
        self.is_system_running = False
        self.ui_state = {
            'status': "⏹️ System stopped",
            'emotion': "Not monitoring", 
            'events': "No events",
            'questions': []
        }
        return "⏹️ WaitWhat system stopped", "Ready to start new session"
    
    def handle_confusion(self, confusion_event):
        """Handle confusion with question generation"""
        print(f"🤔 Processing confusion event...")
        
        self.latest_confusion_event = confusion_event
        
        # Generate mock transcript
        transcript = self.get_mock_transcript(self.video_metadata.get('title', ''))
        
        # Generate questions with LLM
        try:
            context = {
                'video_title': self.video_metadata.get('title', 'Educational Video'),
                'video_description': self.video_metadata.get('description', 'Learning content')
            }
            
            clip_content = {
                'clip_transcript': transcript,
                'start_time': '0:00',
                'end_time': '0:20'
            }
            
            questions_result = self.llm_engine.generate_confusion_questions(context, clip_content)
            
            if questions_result['success']:
                self.current_questions = questions_result['questions']
                self.ui_state['questions'] = self.current_questions.copy()
                print(f"💡 Generated {len(self.current_questions)} questions!")
                for i, q in enumerate(self.current_questions, 1):
                    print(f"   {i}. {q}")
            else:
                print("❌ Question generation failed, using defaults")
                self.current_questions = [
                    "What is the main concept being explained?",
                    "How does this process work?",
                    "Why is this important to understand?"
                ]
                self.ui_state['questions'] = self.current_questions.copy()
                
        except Exception as e:
            print(f"❌ LLM error: {e}")
            self.current_questions = [
                "Can you explain this concept more simply?",
                "What are the key points I should remember?",
                "How does this relate to what I already know?"
            ]
            self.ui_state['questions'] = self.current_questions.copy()
    
    def get_mock_transcript(self, video_title):
        """Generate contextual mock transcript"""
        if any(word in video_title.lower() for word in ['llm', 'language', 'ai', 'neural', 'transformer']):
            return "Large Language Models use transformer architecture with attention mechanisms to understand and generate human language through complex neural networks."
        elif any(word in video_title.lower() for word in ['python', 'programming', 'code']):
            return "Python programming involves writing clean, readable code using proper syntax and following best practices for software development."
        else:
            return "This educational content explains important concepts that require careful attention and understanding to master effectively."
    
    def answer_question(self, selected_question):
        """Generate answer using LLM"""
        if not selected_question:
            return "❌ Please select a question first"
        
        try:
            context = {
                'video_title': self.video_metadata.get('title', 'Educational Video'),
                'video_description': self.video_metadata.get('description', 'Learning content')
            }
            
            clip_content = {
                'clip_transcript': self.get_mock_transcript(self.video_metadata.get('title', '')),
                'start_time': '0:00',
                'end_time': '0:20'
            }
            
            answer_result = self.llm_engine.answer_selected_question(context, clip_content, selected_question)
            
            if answer_result['success']:
                return f"""## 🎯 WaitWhat Explanation

**Video:** {self.video_metadata.get('title', 'Video')}
**Confusion Time:** {self.latest_confusion_event['timestamp'] if self.latest_confusion_event else 'Recent'}
**Your Question:** {selected_question}

**AI Answer:**
{answer_result['answer']}

---
**Continue watching - WaitWhat is still monitoring for confusion!** 🤖"""
            else:
                return f"❌ Failed to generate answer: {answer_result.get('error', 'Unknown error')}"
                
        except Exception as e:
            return f"❌ Error: {str(e)}"
    
    def get_live_status(self):
        """Get current status for UI refresh"""
        if not self.is_system_running:
            return (
                "⏹️ **System Stopped**",
                "Not monitoring",
                "No events",
                []
            )
        
        return (
            self.ui_state['status'],
            f"**Emotion:** {self.emotion_detector.current_emotion}",
            self.ui_state['events'],
            self.ui_state['questions']
        )

# =====================
# GRADIO INTERFACE WITH PROPER SYNC
# =====================
waitwhat = WaitWhatSynced()

with gr.Blocks(theme=gr.themes.Soft(), title="WaitWhat!?") as demo:
    
    gr.HTML("""
        <div style="text-align: center; padding: 20px; background: linear-gradient(45deg, #667eea, #764ba2); color: white; border-radius: 10px; margin-bottom: 20px;">
            <h1 style="margin: 0; font-size: 2.2em;">🎯 WaitWhat!?</h1>
            <p style="margin: 10px 0;">Watch Video + AI Detects Confusion + Get Help</p>
        </div>
    """)
    
    with gr.Row():
        # LEFT - Video & Controls  
        with gr.Column(scale=2):
            gr.Markdown("### 🎬 Video Setup")
            
            video_source = gr.Radio(
                choices=["YouTube URL", "Upload Video"],
                value="YouTube URL",
                label="Video Source"
            )
            
            youtube_url = gr.Textbox(
                label="YouTube URL",
                placeholder="https://www.youtube.com/watch?v=...",
                value="https://www.youtube.com/watch?v=5sLYAQS9sWQ"
            )
            
            uploaded_video = gr.Video(
                label="Upload Video File",
                visible=False
            )
            
            # VIDEO PLAYER - Fixed to show video properly
            video_player = gr.Video(
                label="📺 Your Educational Video",
                height=350,
                autoplay=False,
                show_download_button=True
            )
            
            with gr.Row():
                start_btn = gr.Button("🚀 Start WaitWhat", variant="primary", size="lg")
                stop_btn = gr.Button("⏹️ Stop", variant="secondary")
                refresh_btn = gr.Button("🔄 Refresh Status", variant="secondary")
            
            system_status = gr.Markdown("**Ready:** Select video source and start system")
            
        # RIGHT - AI Assistant with Live Updates
        with gr.Column(scale=2):
            gr.Markdown("### 🤖 Live AI Assistant")
            
            monitoring_status = gr.Markdown("System not started")
            emotion_status = gr.Markdown("Emotion detection inactive")
            events_status = gr.Markdown("No events yet")
            
            gr.Markdown("### 💡 Confusion Questions")
            
            questions_dropdown = gr.Dropdown(
                choices=[],
                label="Available Questions",
                value=None,
                interactive=True
            )
            
            answer_btn = gr.Button("🆘 Get AI Answer", variant="primary")
            
            ai_response = gr.Markdown("""### 🎯 Live Demo Instructions:

**Test Flow:**
1. **Video**: Use pre-filled YouTube URL or upload your video
2. **Start**: Click "🚀 Start WaitWhat" 
3. **Watch**: Video should appear in player above
4. **Express**: Make sad face for 2+ seconds when confused
5. **Refresh**: Click "🔄 Refresh Status" to see updates
6. **Questions**: Select generated question from dropdown
7. **Answer**: Get AI explanation

**System Status:** Ready to start!""")
    
    # Dynamic input visibility
    def update_inputs(source):
        return (
            gr.update(visible=(source == "YouTube URL")),
            gr.update(visible=(source == "Upload Video"))
        )
    
    video_source.change(
        update_inputs,
        inputs=[video_source],
        outputs=[youtube_url, uploaded_video]
    )
    
    # Main system functions
    def start_system_handler(source, youtube_url, uploaded_file):
        """Handle system start with proper video return"""
        status, monitoring, video_path = waitwhat.start_system(source, youtube_url, uploaded_file)
        print(f"🎬 Returning video path to Gradio: {video_path}")
        return status, monitoring, gr.update(value=video_path)
        
    
    
    def refresh_status_handler():
        """Refresh all status displays"""
        status, emotion, events, questions = waitwhat.get_live_status()
        
        # Update dropdown with new questions
        choices = questions if questions else []
        dropdown_update = gr.update(
            choices=choices,
            value=choices[0] if choices else None
        )
        
        return status, emotion, events, dropdown_update
    
    # Event connections
    start_btn.click(
        start_system_handler,
        inputs=[video_source, youtube_url, uploaded_video],
        outputs=[system_status, monitoring_status, video_player]
    )
    
    stop_btn.click(
        waitwhat.stop_system,
        outputs=[system_status, monitoring_status]
    )
    
    refresh_btn.click(
        refresh_status_handler,
        outputs=[monitoring_status, emotion_status, events_status, questions_dropdown]
    )
    
    answer_btn.click(
        waitwhat.answer_question,
        inputs=[questions_dropdown],
        outputs=[ai_response]
    )

if __name__ == "__main__":
    print("🎯 WaitWhat?")
    print("🔧 Fixes Applied:")
    print("   ✅ Video display in Gradio player")
    print("   ✅ UI synchronization with backend")
    print("   ✅ Live status updates")
    print("   ✅ Question dropdown updates")
    print()
    print("📱 Access at: http://localhost:7860")
    print("🔄 Remember to click 'Refresh Status' to see updates!")
    
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True
    )