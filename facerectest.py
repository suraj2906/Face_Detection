import face_recognition
import os
import cv2

knowndir = "C:\\Users\\SHAH\\Documents\\Face_Recognition\\facerec\\known"
unknowndir = "C:\\Users\\SHAH\\Documents\\Face_Recognition\\facerec\\unknown"
tolerance = 0.5
frame_thickness = 3
font_thickness = 2
model = "hog"

known_faces = []
known_names = []

print("Processing known faces")

for filename in os.listdir(knowndir):
    image = face_recognition.load_image_file(f"{knowndir}/{filename}")
    encoding = face_recognition.face_encodings(image)[0]
    known_faces.append(encoding)
    known_names.append(filename)

print("Processing unkown faces")

for filename in os.listdir(unknowndir):
    print(filename)
    image = face_recognition.load_image_file(f"{unknowndir}/{filename}")
    locations = face_recognition.face_locations(image, model=model)
    encodings = face_recognition.face_encodings(image, locations)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    for face_encoding, face_location in zip(encodings, locations):
        results = face_recognition.compare_faces(known_faces, face_encoding, tolerance)
        match = None
        if True in results:
            match = known_names[results.index(True)]
            print(f"Match found: {match}")
            top_left = (face_location[3], face_location[0])
            bottom_right = (face_location[1], face_location[2])
            color = [255, 0, 0]
            cv2.rectangle(image, top_left, bottom_right, color, frame_thickness)
            top_left = (face_location[3], face_location[2])
            bottom_right = (face_location[1], face_location[2] + 22)
            cv2.rectangle(image, top_left, bottom_right, color, cv2.FILLED)
            cv2.putText(image, match, (face_location[3] + 10, face_location[2] + 15), cv2.FONT_HERSHEY_SIMPLEX, 
            0.5, [200,200,200], font_thickness)
    
    cv2.namedWindow('Faces found', cv2.WINDOW_NORMAL)
    cv2.imshow('Faces found', image)
    cv2.waitKey(0)
    cv2.destroyWindow('Faces found')

