import cv2
import numpy as np
import tensorflow as tf
import time
import os

# Constants
SEQ_LEN = 20  # Number of frames in each sequence
IMG_HEIGHT = 64
IMG_WIDTH = 64
SAVE_DIR = "ragging_frames"  # Directory for saving detected frames
VIDEO_PATHS = [ "voilent_videos/V_10.mp4" , "voilent_videos/V_1000.mp4"  , "voilent_videos/V_119.mp4" ]  # Replace with your video file path
os.makedirs(SAVE_DIR, exist_ok=True)

# Load the pre-trained model
model = tf.keras.models.load_model(
    'models/convlstm_model___Date_Time_2024_09_17__12_50_39___Loss_0.32116425037384033___Accuracy_0.8600000143051147.h5'
)

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def preprocess_frames(frames):
    """Resize and normalize frames."""
    processed_frames = [cv2.resize(frame, (IMG_HEIGHT, IMG_WIDTH)) / 255.0 for frame in frames]
    return np.array(processed_frames)

def predict_ragging(frames):
    """Predict whether the sequence indicates ragging."""
    frames = np.expand_dims(frames, axis=0)
    predictions = model.predict(frames, verbose=0)
    return predictions

def run_ragging_detection(VIDEO_PATH):
    """Main loop for detecting ragging in a video file."""
    cap = cv2.VideoCapture(VIDEO_PATH)  # Open video file
    frames_buffer = []  # Buffer for storing frames in a sequence

    if not cap.isOpened():
        print(f"Error: Could not open video {VIDEO_PATH}")
        return

    cv2.namedWindow('Video Feed', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Video Feed', 400, 300) 

    while True:
        ret, frame = cap.read()

        if not ret:  # End of video
            break
        

        cv2.imshow('Video Feed', frame)  # Display video
        frames_buffer.append(frame)

        # Process once buffer is full
        if len(frames_buffer) == SEQ_LEN:
            processed_frames = preprocess_frames(frames_buffer)
            prediction = predict_ragging(processed_frames)

            confidence = prediction[0][0]
            result = "Ragging" if confidence > 0.7 else "Non-Ragging"
            print(f"Prediction: {result} (Confidence: {confidence})")

            if confidence > 0.5:  # Save frames if detected
                print("Ragging detected! Saving frames...")
                timestamp = time.strftime("%Y%m%d-%H%M%S")
                for idx, frame in enumerate(frames_buffer):
                    file_name = os.path.join(SAVE_DIR, f"frame_{timestamp}_{idx}.png")
                    cv2.imwrite(file_name, frame)

            frames_buffer = []  # Reset buffer

        # Exit condition
        if cv2.waitKey(25) & 0xFF == ord('q'):  # 25ms delay for video playback
            break

    cap.release()
    cv2.destroyAllWindows()

# Run the detection loop
if __name__ == "__main__":
    for vids in VIDEO_PATHS :
        run_ragging_detection(vids)
        time.sleep(2)
