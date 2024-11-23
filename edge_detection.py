#import the stuff
import cv2 as cv

#load the image 
img = cv.imread("images/woman.jpeg")
cv.imshow("Person", img)

#color_to_BGR_and_show
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow("Person", gray)

#bring_the_haar
haar_cascade = cv.CascadeClassifier("face.xml")
faces_rect = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
print(f"Number of faces that found in picture is {len(faces_rect)}")


#draw_a_rectangle_aroud_the_face
for (x,y,w,h) in faces_rect:
    cv.rectangle(img, (x,y), (x+w,y+h), (0,255,0), thickness=5)

cv.imshow("Detected faces!", img)

#wait
cv.waitKey(0)