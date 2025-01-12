import os
import face_recognition as face_reg
import pickle

def encode_faces_from_folder(folder_path):
    encodings = []
    names = []
    
    for filename in os.listdir(folder_path):
        if filename.endswith((".jpg", ".png")):
            image_path = os.path.join(folder_path, filename)
            name = filename.split('.')[0]  
            print(name)
            
            image = face_reg.load_image_file(image_path)
            encodings_found = face_reg.face_encodings(image)
            
            if encodings_found:
                encodings.append(encodings_found[0])  
                names.append(name)

    with open("faces/processed_faces/encodings.pkl", "wb") as f:
        pickle.dump({"encodings": encodings, "names": names}, f)
    
    print(f"Encodings saved for {len(names)} individuals. , {names}")


encode_faces_from_folder("faces/base/")
    