import os
import time
import threading
import cv2
import numpy as np
import face_recognition
import speech_recognition as sr
import openai
import pyttsx3
from gpiozero import Robot, DigitalInputDevice
from picamera2 import Picamera2


OPENAI_API_KEY = "your-api-key-here"  # Replace with your OpenAI API key
KNOWN_FACES_DIR = "known_faces"
NAVIGATION_MODEL = "gpt-4"  # or "gpt-3.5-turbo"
VOICE_MODEL = "gpt-3.5-turbo"  # Faster for voice responses


robot = Robot(left=(12, 13), right=(18, 19))  # GPIO pins for motors
picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration(main={"format": 'XRGB8888', "size": (640, 480)}))
picam2.start()


recognizer = sr.Recognizer()
microphone = sr.Microphone()
engine = pyttsx3.init()


known_face_encodings = []
known_face_names = []

for filename in os.listdir(KNOWN_FACES_DIR):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        image = face_recognition.load_image_file(os.path.join(KNOWN_FACES_DIR, filename))
        encoding = face_recognition.face_encodings(image)[0]
        known_face_encodings.append(encoding)
        known_face_names.append(os.path.splitext(filename)[0])


openai.api_key = OPENAI_API_KEY

class NavigationSystem:
    def __init__(self):
        self.current_location = "unknown"
        self.obstacles = []
        self.last_detected_faces = []
    
    def analyze_scene(self, frame):

        rgb_frame = frame[:, :, ::-1]
        
     
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
        
        self.last_detected_faces = []
        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"
            
            if True in matches:
                first_match_index = matches.index(True)
                name = known_face_names[first_match_index]
            
            self.last_detected_faces.append(name)
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 100, 200)
        self.obstacles = np.sum(edges > 0) / edges.size > 0.1  # Threshold for obstacle detection
        
        return frame
    
    def get_navigation_decision(self):
        scene_description = f"Current location: {self.current_location}. "
        
        if self.obstacles:
            scene_description += "There are obstacles in front. "
        
        if self.last_detected_faces:
            scene_description += f"Detected faces: {', '.join(self.last_detected_faces)}. "
        else:
            scene_description += "No faces detected. "
        
        
 '''commands -
         FORWARD
         BACKWARD
         LEFT
         RIGHT
         STOP
        
      '''  
        
        try:
            response = openai.ChatCompletion.create(
                model=NAVIGATION_MODEL,
                messages=[{"role": "system", "content": prompt}],
                temperature=0.1,
                max_tokens=10
            )
            return response.choices[0].message.content.strip().upper()
        except Exception as e:
            print(f"Navigation API error: {e}")
            return "STOP"

class VoiceSystem:
    def __init__(self):
        self.is_listening = False
    
    def listen(self):
        with microphone as source:
            print("Listening...")
            audio = recognizer.listen(source, phrase_time_limit=5)
        
        try:
            text = recognizer.recognize_google(audio)
            print(f"You said: {text}")
            return text
        except sr.UnknownValueError:
            print("Could not understand audio")
            return None
        except sr.RequestError as e:
            print(f"Speech recognition error: {e}")
            return None
    
    def speak(self, text):
        engine.say(text)
        engine.runAndWait()
    
    def ask_openai(self, prompt):
        try:
            response = openai.ChatCompletion.create(
                model=VOICE_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=150
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"OpenAI API error: {e}")
            return "I'm sorry, I encountered an error."

def main():
    nav_system = NavigationSystem()
    voice_system = VoiceSystem()
    
   
    def navigation_worker():
        while True:
            frame = picam2.capture_array()
            analyzed_frame = nav_system.analyze_scene(frame)
            
            cv2.imshow('Robot View', analyzed_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
            command = nav_system.get_navigation_decision()
            print(f"Navigation command: {command}")
            
            if command == "FORWARD":
                robot.forward(speed=0.5)
            elif command == "BACKWARD":
                robot.backward(speed=0.5)
            elif command == "LEFT":
                robot.left(speed=0.5)
            elif command == "RIGHT":
                robot.right(speed=0.5)
            else:
                robot.stop()
            
            time.sleep(0.1)
    
    nav_thread = threading.Thread(target=navigation_worker, daemon=True)
    nav_thread.start()

    while True:
        voice_system.speak("I'm ready for your command.")
        command = voice_system.listen()
        
        if command:
            if "stop" in command.lower() or "halt" in command.lower():
                voice_system.speak("Stopping all operations.")
                robot.stop()
                break
            elif "move" in command.lower() or "go" in command.lower():
                # Let navigation system handle movement
                voice_system.speak("I'll navigate based on what I see.")
            else:
                response = voice_system.ask_openai(command)
                voice_system.speak(response)
        
        time.sleep(1)
    
    cv2.destroyAllWindows()
    picam2.stop()

if __name__ == "__main__":
    main()
