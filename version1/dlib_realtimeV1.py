import numpy as np
import cv2
import sys
import dlib
import fps
import caffe

caffe.set_mode_cpu()

age_net_pretrained='./dex_chalearn_iccv2015.caffemodel'
age_net_model_file='./age.prototxt'

gender_net_pretrained='./gender.caffemodel'
gender_net_model_file='./gender.prototxt'

age_list=['0','1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20','21','22','23','24','25','26','27','28','29','30','31','32','33','34','35','36','37','38','39','40','41','42','43','44','45','46','47','48','49','50','51','52','53','54','55','56','57','58','59','60','61','62','63','64','65','66','67','68','69','70','71','72','73','74','75','76','77','78','79','80','81','82','83','84','85','86','87','88','89','90','91','92','93','94','95','96','97','98','99','100']
#age_list=['(0, 2)','(4, 6)','(8, 12)','(15, 20)','(25, 32)','(38, 43)','(48, 53)','(60, 100)']
gender_list=['Female','Male']

age_net = cv2.dnn.readNetFromCaffe(age_net_model_file, age_net_pretrained)
gender_net = cv2.dnn.readNetFromCaffe(gender_net_model_file, gender_net_pretrained)


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
        cropped_img = frame[d.top():d.bottom(), d.left():d.right()]
        #cv2.imwrite('deneme.png', cropped_img)
        cropped_img = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB)

        blob = cv2.dnn.blobFromImage(cropped_img, scalefactor=1.0, size=(224, 224), swapRB=False)

        age_net.setInput(blob)
        detections = age_net.forward()
        age = np.dot(detections, np.arange(0, 101))
        print 'predicted age:{}'.format(i), round(age)


        gender_net.setInput(blob)
        predict = gender_net.forward()

        print 'predicted gender:{}'.format(i), gender_list[predict[0].argmax()]
        # cv2.putText(img, age, (x-20, y-20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, lineType=cv2.LINE_AA)

    fps_output = str(fpsWithTick.get())
    cv2.putText(frame, "fps = " + fps_output, (0, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255))
    win.clear_overlay()
    win.set_image(frame)
    win.add_overlay(dets)


    cv2.imshow('frame',frame)
    #if cv2.waitKey(1) & 0xFF == ord('q'):
    #    break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()