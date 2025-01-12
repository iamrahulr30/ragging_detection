import os ,time
import face_recognition
import pickle
from collections import defaultdict

VIOLENCE_FRAMES_DIR = "ragging_frames"
PROCESSED_FRAMES_DIR = "proc_ragging_frames"
IDENTIFIED_OUTPUT_FILE = "detected_individuals.txt"

os.makedirs(PROCESSED_FRAMES_DIR, exist_ok=True)

def group_frames_by_time(folder_path):
    grouped_frames = defaultdict(list)
    
    for filename in os.listdir(folder_path):
        if filename.endswith(".png"):
            time_key = filename.split('_')[1].split('-')[1]  # Extract time (e.g., 142447 from frame_20241223-142447_10.png)
            grouped_frames[time_key].append(os.path.join(folder_path, filename))
    
    return grouped_frames

def recognize_faces_in_grouped_frames(grouped_frames):
    # Load saved encodings
    with open("faces/processed_faces/encodings.pkl", "rb") as f:
        data = pickle.load(f)
    
    known_encodings = data["encodings"]
    known_names = data["names"]

    for time_key, frames in grouped_frames.items():
        identified_individuals = set()  # To store unique individuals identified
        
        for frame_path in frames:
            image = face_recognition.load_image_file(frame_path)
            face_locations = face_recognition.face_locations(image)
            face_encodings = face_recognition.face_encodings(image, face_locations)
            
            for face_encoding in face_encodings:
                matches = face_recognition.compare_faces(known_encodings, face_encoding, tolerance=0.6)
                if True in matches:
                    match_index = matches.index(True)
                    identified_individuals.add(known_names[match_index])
                else:
                    identified_individuals.add("Unknown")
            
            # Move the processed frame to the processed folder
            os.rename(frame_path, os.path.join(PROCESSED_FRAMES_DIR, os.path.basename(frame_path)))
        
        # Write the result to the file once for the entire time group
        with open(IDENTIFIED_OUTPUT_FILE, "a") as file:
            file.write(f"Time: {time_key}\n")
            file.write(f"Individuals Identified: {', '.join(identified_individuals)}\n")
            file.write("------\n")
        print(f"Processed frames for time: {time_key}")

def main():
    grouped_frames = group_frames_by_time(VIOLENCE_FRAMES_DIR)
    recognize_faces_in_grouped_frames(grouped_frames)

if __name__ == "__main__":
    while True:
        main()
        print("time out")
        time.sleep(2)
