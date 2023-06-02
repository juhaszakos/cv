import os
import cv2
import re
import numpy as np
import time
import pickle

script_path = os.path.abspath(__file__)
script_dir = os.path.dirname(script_path)

subjects_dir = "subjects"
subject_path = os.path.join(script_dir, subjects_dir)

haar_model_file_name = "haarcascade_frontalface_default.xml"
haar_model_path = os.path.join(script_dir, haar_model_file_name)

subjects_subdirs_path_regex_raw = r"^(.*\\{}\\)(.*)$"
subjects_subdirs_path_regex = subjects_subdirs_path_regex_raw.format(subjects_dir)

lbp_model_name = "face_recognizer.yaml"
lbp_model_path = os.path.join(script_dir, lbp_model_name)

labels_file_name = "labels.pkl"
labels_file_path = os.path.join(script_dir, labels_file_name)

def train_model():
    response = input("Would you like to add new subject? (Y/N):")
    if response.lower() == "y":
        add_new_subject()
    elif response.lower() != "n":
        raise Exception(response + " is not a valid answer!")

    images, labels = label_data_for_training()
    model = cv2.face.LBPHFaceRecognizer_create()
    start_time = time.time()
    print("Model training started!")
    model.train(images, labels)
    end_time = time.time()
    print("Model training finished! Took: " + str(end_time - start_time))
    model.save(lbp_model_path)

def label_data_for_training():
    images = []
    labels = []
    for root, dirs, files in os.walk(subject_path):
        if len(dirs) > 0:
            continue
        label = re.search(subjects_subdirs_path_regex, root).group(2)
        for file in files:
            image_path = os.path.join(root, file)
            images.append(cv2.imread(image_path, cv2.IMREAD_GRAYSCALE))
            labels.append(label)

    label_dict = {label: idx for idx, label in enumerate(set(labels))}

    # save labels for the main application
    with open(labels_file_path, "wb") as file:
        pickle.dump(label_dict, file)

    numeric_labels = [label_dict[label] for label in labels]
    return (images, np.array(numeric_labels))

def add_new_subject():
    name_of_subject = input("Enter name of subject:")
    path = create_subject_directory(name_of_subject)
    capture = cv2.VideoCapture(0)
    if not capture.isOpened():
        raise IOError("Cannot open webcam")
    classifier = cv2.CascadeClassifier(haar_model_path)

    counter = 0
    adding_subject = True
    while adding_subject:
        _, frame = capture.read()
        frame_monochrome = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_monochrome_normalized = cv2.equalizeHist(frame_monochrome)
        faces = classifier.detectMultiScale(
             frame_monochrome_normalized ,
            # scaleFactor=1.05,
             minNeighbors=5
            # minSize=(100, 100)
        )

        if (len(faces) > 0):
            x, y, w, h = faces[0]  # only one face is expected during new subject auditing
            cropped_face = frame[y:y + h, x: x + w]
            cropped_face = cv2.resize(cropped_face, (250, 250))
            image_name = name_of_subject + str(counter) + ".jpg"
            image_path = os.path.join(path, image_name)
            cv2.imwrite(image_path, cropped_face)
            counter += 1
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

        cv2.imshow('Adding new subjet.', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            adding_subject = False


def create_subject_directory(subject_name):
    path = os.path.join(subject_path, subject_name)
    os.mkdir(path)
    return path

if __name__ == '__main__':
    train_model()