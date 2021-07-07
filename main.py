import cv2
import cvzone

from cvzone.SelfiSegmentationModule import SelfiSegmentation
import os

cap=cv2.VideoCapture(0)
cap.set(3,640)
cap.set(3,480)

segmentor = SelfiSegmentation()
background_image = cv2.imread('images/1.jpg')
background_image=cv2.GaussianBlur(background_image,(13,13),0)
background_image = cv2.cvtColor(background_image, cv2.COLOR_BGR2RGB)
crop_background = background_image[0:480, 0:640]

while True:
    status,img=cap.read()
    imgout=segmentor.removeBG(img,crop_background,threshold=0.8)
    imagestack=cvzone.stackImages([img,imgout],2,1)
    cv2.imshow("image",imagestack)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cv2.destroyAllWindows()

