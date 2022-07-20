import cv2
import numpy as np
import glob

path="C:/Users/Hp/OneDrive/Desktop/Miniproject/trainingset/giloma/*.*"

imageno=1
for file in glob.glob(path):
    print(file)
    img=cv2.imread(file,0)
    dimension=img.shape
    print(dimension[0],dimension[1])
    imgcropped=img[50:dimension[0]-50,50:dimension[1]-50]
   
    converted_img=cv2.cvtColor(imgcropped,cv2.COLOR_GRAY2BGR)
    nl=cv2.fastNlMeansDenoisingColored(converted_img,None,10,10,7,21)
    #cv2.imshow("nlm",nl)
    #step3--->>>applying histogram equalization
    converted_img1=cv2.cvtColor(nl,cv2.COLOR_BGR2GRAY)
    equ=cv2.equalizeHist(converted_img1)
    cv2.imshow("res",equ)
   
    #cv2.imshow("hist",res)
    cv2.waitKey(1)
    #path1="C:\Users\Hp\OneDrive\Desktop\Miniproject\preprocessedtrainingset\preprocessedgiloma"
    cv2.imwrite(r"C:\Users\Hp\OneDrive\Desktop\Miniproject\preprocessedtrainingset\preprocessedgiloma\giloma"+str(imageno)+'.jpg',equ)
    imageno+=1