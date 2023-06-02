import os
import cv2
import pickle

script_path = os.path.abspath(__file__)
script_dir = os.path.dirname(script_path)

lbp_model_name = "face_recognizer.yaml"
lbp_model_path = os.path.join(script_dir, lbp_model_name)

labels_file_name = "labels.pkl"
labels_file_path = os.path.join(script_dir, labels_file_name)


def recognize():
    capture = cv2.VideoCapture(0)
    if not capture.isOpened():
        raise IOError("Cannot open webcam")
    classifier = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(lbp_model_path)

    # getting the labels for the images
    label_dict = {}
    with open(labels_file_path, "rb") as file:
        label_dict = pickle.load(file)
    reversed_label_dict = {v: k for k, v in label_dict.items()}

    test = True
    while test:
        _, frame = capture.read()
        frame_monochrome = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_monochrome_normalized = cv2.equalizeHist(frame_monochrome)
        faces = classifier.detectMultiScale(
            frame_monochrome_normalized,
            # scaleFactor=1.05,
            minNeighbors=5
            # minSize=(100, 100)
        )
        for (x, y, w, h) in faces:
            cropped_face = frame_monochrome[y:y + h, x:x + w]
            cropped_face = cv2.resize(cropped_face, (250, 250))
            cv2.imshow('found face', cropped_face)
            id, confidence = recognizer.predict(cropped_face)
            print("confidence: " + str(confidence) + ", predicted id: " + str(id))
            print("labels: " + str(reversed_label_dict))
            if confidence <= 45:
                label = reversed_label_dict[id]
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(
                    img=frame,
                    text=label,
                    org=(x + w, y),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=1,
                    color=(0, 0, 0),
                    thickness=3
                )
            else:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.waitKey(1)

        cv2.imshow('capture', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            test = False

    cv2.destroyAllWindows()


if __name__ == '__main__':
    recognize()
