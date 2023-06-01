import cv2
import numpy as np


def test():
    capture = cv2.VideoCapture(0)
    if not capture.isOpened():
        raise IOError("Cannot open webcam")
    classifier = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    while True:
        _, frame = capture.read()
        frame_monochrome = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_monochrome_normalized = cv2.equalizeHist(frame_monochrome)
        faces = classifier.detectMultiScale(
            frame_monochrome_normalized,
            scaleFactor=1.05,
            minNeighbors=6,
            minSize=(100, 100)
        )
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cropped_face = frame[y:y + h, x:x + w]
            cv2.imshow('found face', cropped_face)
            cv2.waitKey(1)
        cv2.imshow('capture', frame)
        cv2.waitKey(1)

    cv2.destroyAllWindows()


if __name__ == '__main__':
    test()
