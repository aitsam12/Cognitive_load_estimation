













import os
import csv
import time
import random
import requests
import pyaudio
import io
import wave
import numpy as np
from psychopy import visual, core, monitors
from datetime import datetime
import warnings

# Suppress FP16 warning
warnings.filterwarnings("ignore", message="FP16 is not supported on CPU; using FP32 instead")

# Your OpenAI API key
OPENAI_API_KEY = "api_key"

# File paths
audio_folder = "StroopTaskCode\\stroop_recordings"
results_csv_path = "StroopTaskCode\\stroopresultsALL.csv"

# Create the necessary folders
if not os.path.exists(audio_folder):
    os.makedirs(audio_folder)

# Function to transcribe audio using Whisper API
def transcribe_audio(audio_file_path):
    retries = 3
    backoff = 1
    for attempt in range(retries):
        try:
            with open(audio_file_path, "rb") as audio_file:
                headers = {"Authorization": f"Bearer {OPENAI_API_KEY}"}
                files = {"file": ("audio.wav", audio_file, "audio/wav")}
                data = {"model": "whisper-1"}

                response = requests.post(
                    "https://api.openai.com/v1/audio/transcriptions",
                    headers=headers,
                    files=files,
                    data=data,
                    timeout=15  # Increase timeout to 60 seconds
                )

                if response.status_code == 200:
                    transcript = response.json().get("text", "").strip().lower()
                    for word in ['red', 'blue', 'green', 'yellow']:
                        if word in transcript:
                            return word, transcript  # Return both the matched word and the full transcript
                    return "unrecognized", transcript  # Return unrecognized along with transcript
                else:
                    raise RuntimeError(f"Whisper API request failed: {response.status_code}")
        except Exception as e:
            print(f"Error in transcription attempt {attempt + 1}: {e}")
            time.sleep(backoff)
            backoff *= 2  # Exponential backoff
    return "error", "N/A"

# Function to record speech during a Stroop trial
def record_speech_segment(duration=3.0):
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=1024)
    frames = []
    
    start_time = time.time()
    print("Recording...")
    try:
        while time.time() - start_time < duration:
            data = stream.read(1024, exception_on_overflow=False)  # Handle buffer overflows
            frames.append(data)
    except Exception as e:
        print(f"Error during recording: {e}")
    finally:
        stream.stop_stream()
        stream.close()
        p.terminate()

    audio_data = b''.join(frames)
    return audio_data

# Function to save audio to file
def save_audio(audio_data, participant_name, segment_number):
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    audio_file_name = f"{participant_name}_{timestamp}_segment_{segment_number}.wav"
    audio_file_path = os.path.join(audio_folder, audio_file_name)

    with wave.open(audio_file_path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframes(audio_data)

    return audio_file_path

# Function to detect the start of speech using energy thresholding
def detect_speech_onset(audio_data, threshold=2000, rate=16000):
    audio_samples = np.frombuffer(audio_data, dtype=np.int16)
    energy = np.abs(audio_samples)

    for i, sample in enumerate(energy):
        if sample > threshold:
            onset_time = i / rate
            return onset_time
    return None

# Function to display a message for a given duration
def show_message(text, duration):
    message = visual.TextStim(win, text=text, color='black', height=0.25)
    message.draw()
    win.flip()
    core.wait(duration)

# Create congruent and incongruent trials without consecutive repeated words
def create_blocked_trials(num_trials):
    trials = []
    words = ['RED', 'GREEN', 'BLUE', 'YELLOW']
    colors = {'RED': 'red', 'GREEN': 'green', 'BLUE': 'blue', 'YELLOW': 'yellow'}
    num_congruent = int(num_trials * 0.2)
    num_incongruent = num_trials - num_congruent

    last_word = None  # Track the last word used

    for _ in range(num_congruent):
        word = random.choice([w for w in words if w != last_word])  # Avoid repetition of last word
        trials.append({'word': word, 'color': colors[word], 'type': 'congruent'})
        last_word = word  # Update last word

    for _ in range(num_incongruent):
        word = random.choice([w for w in words if w != last_word])  # Avoid repetition of last word
        possible_colors = [c for c in colors.values() if c != colors[word]]
        trials.append({'word': word, 'color': random.choice(possible_colors), 'type': 'incongruent'})
        last_word = word  # Update last word

    random.shuffle(trials)
    return trials

# Function to run a single Stroop trial without immediate transcription
def run_trial(trial, segment_number, participant_name):
    # Display the word with the specified color
    word_stim = visual.TextStim(win, text=trial['word'], color=trial['color'], height=0.35)
    word_stim.draw()
    win.flip()

    # Record speech for 3 seconds
    start_time = time.time()
    audio_data = record_speech_segment(duration=3.0)
    end_time = time.time()

    # Save audio data
    audio_file_path = save_audio(audio_data, participant_name, segment_number)

    # Detect speech onset
    speech_onset = detect_speech_onset(audio_data)
    if speech_onset is not None:
        reaction_time = speech_onset
    else:
        reaction_time = end_time - start_time  # Default to full duration if speech isn't detected

    # Collect trial information without transcription
    displayed_word = trial['word']
    condition = trial['type']
    correct_answer = trial['color']

    # Return all collected data including the path to the audio file
    return {
        'unix_time': time.time(),
        'current_time': datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S'),
        'displayed_word': displayed_word,
        'condition': condition,
        'correct_answer': correct_answer,
        'reaction_time': reaction_time,
        'audio_file_path': audio_file_path
    }

# Create a custom monitor
mon = monitors.Monitor('stroopmoni')
mon.setWidth(53)
mon.setDistance(60)
mon.setSizePix((800, 600))

# Initialize PsychoPy window
win = visual.Window(size=(800, 600), color="white", fullscr=True, screen=1)

def clean_memory():
    """Cleans up PsychoPy memory"""
    win.flip()  # Flip the window to refresh
    core.wait(0.5)  # Small delay to ensure resources are freed

def main():
    # Get participant's name
    participant_name = input("Please enter the participant's name: ")

    # Create results CSV if it doesn't exist
    if not os.path.isfile(results_csv_path):
        with open(results_csv_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Unix Timestamp', 'Current Timestamp', 'Displayed Word', 'Condition',
                             'Correct Answer', 'User Answer', 'Reaction Time', 'Audio File', 'Full Transcription'])

    # Instructions and practice test (3 trials)
    show_message("Welcome to the Stroop Task!", 3)
    show_message("Instructions: Speak the color of the word, not the word itself.", 5)

    practice_trials = create_blocked_trials(6)
    segment_number = 0
    all_trial_data = []  # Store trial data

    # Run practice trials
    for trial in practice_trials:
        trial_data = run_trial(trial, segment_number, participant_name)
        all_trial_data.append(trial_data)
        segment_number += 1

    # After practice, clean memory
    clean_memory()

    # Main experiment in smaller blocks
    show_message("Main experiment will start.", 3)
    main_trials = create_blocked_trials(30)
    block_size = 10
    for i in range(0, len(main_trials), block_size):
        block = main_trials[i:i + block_size]
        for trial in block:
            trial_data = run_trial(trial, segment_number, participant_name)
            all_trial_data.append(trial_data)
            segment_number += 1

        # Clean memory after every 10 trials
        clean_memory()

    # Display thank you message when the experiment finishes
    show_message("Thank you for participating!", 3)

    # After all trials, perform transcription and update CSV
    with open(results_csv_path, mode='a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        for trial_data in all_trial_data:
            audio_file_path = trial_data['audio_file_path']
            
            # Show current file being transcribed
            print(f"Transcribing file: {audio_file_path}")
            
            user_answer, full_transcription = transcribe_audio(audio_file_path)
            writer.writerow([
                trial_data['unix_time'], trial_data['current_time'], trial_data['displayed_word'],
                trial_data['condition'], trial_data['correct_answer'], user_answer,
                trial_data['reaction_time'], audio_file_path, full_transcription
            ])

    # Close PsychoPy window
    win.close()

if __name__ == "__main__":
    main()
