import numpy as np
import cv2
import sys
import dlib
import fps

detector = dlib.get_frontal_face_detector()
win = dlib.image_window()

fpsWithTick = fps.fpsWithTick()
cap = cv2.VideoCapture(0)

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    #print("Processing file: {}".format(f))
    #img = dlib.load_rgb_image(frame)
    # The 1 in the second argument indicates that we should upsample the image
    # 1 time.  This will make everything bigger and allow us to detect more
    # faces.

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    dets = detector(frame, 1)
    print("Number of faces detected: {}".format(len(dets)))
    for i, d in enumerate(dets):
        print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(
            i, d.left(), d.top(), d.right(), d.bottom()))



    fps_output = str(fpsWithTick.get())
    cv2.putText(frame, "fps = " + fps_output, (0, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255))
    win.clear_overlay()
    win.set_image(frame)
    win.add_overlay(dets)


    #cv2.imshow('frame',frame)
    #if cv2.waitKey(1) & 0xFF == ord('q'):
    #    break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()