import cv2
import numpy as np
import tensorflow as tf
import asyncio
import time
import os

# Constants
SEQ_LEN = 20  # Number of frames in each sequence
IMG_HEIGHT = 64
IMG_WIDTH = 64
SAVE_DIR = "ragging_frames"  # Directory for saving detected frames
os.makedirs(SAVE_DIR, exist_ok=True)

# Load the pre-trained model
model = tf.keras.models.load_model(
    'models/convlstm_model___Date_Time_2024_09_17__12_50_39___Loss_0.32116425037384033___Accuracy_0.8600000143051147.h5' ,  
    # 'models/convlstm_model___Date_Time_2024_09_17__14_44_44___Loss_0.3518313765525818___Accuracy_0.871999979019165.h5',
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
    predictions = model.predict(frames)
    return predictions

async def run_ragging_detection():
    """Main loop for detecting ragging in live video."""
    cap = cv2.VideoCapture(0)  # Open webcam
    frames_buffer = []  # Buffer for storing frames in a sequence

    print("Press 'q' to exit.")

    while True:
        ret, frame = cap.read()

        if not ret:
            print("Failed to capture frame")
            break

        cv2.imshow('Live Video Feed', frame)  # Display live feed
        frames_buffer.append(frame)

        # Process once buffer is full
        if len(frames_buffer) == SEQ_LEN:
            processed_frames = preprocess_frames(frames_buffer)
            prediction = predict_ragging(processed_frames)
            
            confidence = prediction[0][0]
            result = "Ragging" if confidence > 0.7 else "Non-Ragging"
            print(f"Prediction: {result} (Confidence: {confidence})")

            if confidence > 0.5:  # Adjust threshold if needed
                print("Ragging detected! Saving frames...")
                timestamp = time.strftime("%Y%m%d-%H%M%S")
                for idx, frame in enumerate(frames_buffer):
                    file_name = os.path.join(SAVE_DIR, f"frame_{timestamp}_{idx}.png")
                    cv2.imwrite(file_name, frame)

            frames_buffer = []  # Reset the buffer

        # Exit condition
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Run the detection loop
if __name__ == "__main__":
    asyncio.run(run_ragging_detection())
