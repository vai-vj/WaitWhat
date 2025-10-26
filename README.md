# WaitWhat?
*Catch confusion before it catches you.*  
AI-driven clarity for every moment of confusion.  

---

## 👥 Team
*Team Name:* weWon()  
*Project:* WaitWhat?  

---

## 🧠 Overview
*WaitWhat?* enhances learning and engagement by detecting moments of confusion while watching videos — using *facial emotion recognition* — and then providing *AI-powered clarifications* automatically.

Whether you’re watching an educational lecture, tutorial, or documentary, *WaitWhat?* ensures you never stay confused for long.  

---

## Key Features

### Emotion Detection  
- Uses *VIT-face-expression* transformer and *OpenCV* to monitor facial cues.  
- Detects confusion or sadness in real time using your webcam.  

### Timestamp Tracking  
- Logs the exact video timestamp where confusion occurs.  
- Automatically triggers contextual question generation.

### Speech Transcription 
- Used Whisper model from audio to text transcription
- If video is more than 10 mins , the text transcription is summarized using Bart transformer 

### AI Chatbot Integration  
- Generates clarifying “What?”, “How?”, “Why?” questions related to the confusing moment.  
- Uses *OpenAI-powered LLM* (through WaitWhatQuestionAnswerEngine) for intelligent, human-like explanations.  

###  Multi-Source Video Support  
- Works with *YouTube videos* via official API and *uploaded local files*.  
- Displays the video inline using a synchronized Gradio player.  

---

##  Demo Flow

###  Step-by-Step
1. *Choose video source:*  
   - Paste a YouTube URL, or  
   - Upload a local .mp4 file.  

2. *Start monitoring:*  
   Click *“ Start WaitWhat”* to begin facial emotion detection.  

3. *Watch the video:*  
   The system monitors your expressions in real-time.  

4. *Show confusion:*  
   Make a confused or puzzled face for ~2 seconds — the system detects it as confusion!  

5. *Refresh status:*  
   Click *“ Refresh Status”* to see detected emotions and confusion timestamps.  

6. *Get AI help:*  
   Choose a generated clarifying question and click *“ Get AI Answer”*.  

7. *Continue watching:*  
   The system keeps monitoring while providing real-time explanations.  

---

##  Tech Stack

| Component | Technology Used |
|------------|----------------|
| *Frontend/UI* | Gradio |
| *Emotion Detection* | OpenCV, vit-face-expression |
| *AI/NLP Models* | OpenAI API, Bart Transformer |
| *Speech/Transcription* | Whisper |
| *ML Frameworks* | PyTorch, TensorFlow |
| *Video Handling* | yt_dlp, moviepy, YouTube API |
| *Backend Integration* | Custom LLM Engine (WaitWhatQuestionAnswerEngine) |

---

## YouTube Video

You can test the application using the following sample YouTube video:

[🔗 Watch on YouTube](https://www.youtube.com/watch?v=1p2ZrPKap90)
