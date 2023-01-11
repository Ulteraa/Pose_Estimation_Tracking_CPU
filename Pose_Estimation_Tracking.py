import cv2
import mediapipe as mp
import numpy as np

def pose_estimation_open_cv(path):
    mp_pose = mp.solutions.pose
    mp_draw = mp.solutions.drawing_utils
    pose = mp_pose.Pose()

    cap = cv2.VideoCapture(path)

    while True:
        ret, img = cap.read()
        img = cv2.resize(img, (600, 400))

        # do Pose detection
        results = pose.process(img)
        #-----------------------------------------------------------------------------
        mp_draw.draw_landmarks(img, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                               mp_draw.DrawingSpec((255, 0, 0), 2, 2),
                               mp_draw.DrawingSpec((255, 0, 255), 2, 2)
                               )
        # ------------------------------------------------------------------
        cv2.imshow("Pose Estimation", img)

        cv2.waitKey(2)
if __name__=='__main__':
    path = 'v3.mp4'
    pose_estimation_open_cv(path)