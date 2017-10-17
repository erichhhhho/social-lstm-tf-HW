
import cv2
import numpy as np

cap = cv2.VideoCapture('seq_eth.avi')
#cap = cv2.VideoCapture('C:\Users\khan1\Desktop\pythonproject\test_1.avi')
print(cap)
while (cap.isOpened()):
    _, frame = cap.read()
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    lower_green = np.array([0, 60, 0])
    upper_green = np.array([200, 255, 200])
    mask = cv2.inRange(rgb, lower_green, upper_green)
    res = cv2.bitwise_and(frame, frame, mask=mask)

    cv2.imshow('frame', frame)
    cv2.imshow('mask', mask)
    cv2.imshow('res', res)

    cv2.waitKey(0)


cv2.destroyAllWindows()
cap.release()