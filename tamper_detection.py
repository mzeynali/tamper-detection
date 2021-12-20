import cv2
import numpy as np
from time import sleep
from termcolor import colored
from skimage.metrics import structural_similarity as ssim


class TamperDetection:
    def __init__(self, input_size=500, show_result=False,
                 similarity_thresh=0.6, history=25,
                 multichannel=True) -> None:
        self.input_size = input_size
        self.similarity_thresh = similarity_thresh
        self.multichannel = multichannel
        self.flag = False
        self.show_result = show_result
        self.history = history
        self.fgbg = cv2.createBackgroundSubtractorMOG2(history=history)
        self.kernel = np.ones((5, 5), np.uint8)
        self.thresh_size = int(input_size * 0.001)
        self.base_frame = None
        self.mat = 0
        self.tamper_as_sim = False
        self.tampers_history = []
        self.frame_history = []

    def run(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (self.input_size, self.input_size))

        if len(self.frame_history) > 100:
            self.frame_history.pop(0)

        if not self.tamper_as_sim:
            self.frame_history.append(frame)

        area = 0
        tampered = 0
        bounding_rect = []
        fgmask = self.fgbg.apply(frame)
        fgmask = cv2.erode(fgmask, self.kernel, iterations=5)
        fgmask = cv2.dilate(fgmask, self.kernel, iterations=5)

        contours, _ = cv2.findContours(
            fgmask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
        )

        for i in range(len(contours)):
            # x,y,w,h
            bounding_rect.append(cv2.boundingRect(contours[i]))
        for i in range(len(contours)):
            if (
                bounding_rect[i][2] >= self.thresh_size
                or bounding_rect[i][3] >= self.thresh_size
            ):
                area += area + (bounding_rect[i][2] * bounding_rect[i][3])

        if area >= int(frame.shape[0]) * int(frame.shape[1]) / 3:
            tampered = 1

        if len(self.tampers_history) >= self.history:
            self.mat = np.convolve(self.tampers_history,
                                   np.ones(self.history), mode='valid')[0]
            self.tampers_history.pop(0)

        if self.mat or self.tamper_as_sim:  # trigger tamper
            self.base_frame = self.frame_history[0]
            s = ssim(self.base_frame, frame, multichannel=self.multichannel)

            if s < self.similarity_thresh:
                tampered = 1
                self.tamper_as_sim = True
            else:
                self.tamper_as_sim = False
        else:
            self.tamper_as_sim = False

        self.tampers_history.append(tampered)

        if self.show_result:
            if self.mat:
                cv2.putText(frame, "TAMPERING DETECTED", (5, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            else:
                cv2.putText(frame, "TAMPERING NOT DETECTED", (5, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            cv2.imshow('Tamper Detection', frame)
            cv2.waitKey(1)
            sleep(0.01)

        if self.mat:
            print("[INFO] TAMPERING", colored("DETECTED",'red') )


if __name__ == "__main__":
    td = TamperDetection(show_result=False)
    cap = cv2.VideoCapture(
        "./videos/1.mp4")

    while cap.isOpened():
        frame = cap.read()[1]
        td.run(frame)
