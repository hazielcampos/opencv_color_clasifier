import numpy as np
import cv2
from clasifier import Clasifier


webcam = cv2.VideoCapture(0)
clas = Clasifier(((0, 53), (320, 240)))

clas.start()
try:
    while True:
        ret, frame = webcam.read()
        if not ret:
            break
        frame = cv2.resize(frame, (320, 240))
        clas.set_frame(frame)
        
        display_frame = clas.get_display_frame()

        cv2.imshow("Color Detection", display_frame)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            webcam.release()
            cv2.destroyAllWindows()
            break
except KeyboardInterrupt:
    print("User finished")
finally:
    clas.terminate()