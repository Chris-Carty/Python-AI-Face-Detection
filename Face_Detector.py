import cv2

# Load pre-trained data on face frontals from opencv (haar cascade algorithm)
trained_face_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# example image to detect faces in
img = cv2.imread('group-test.jpg')

# make image grayscale (due to limitations of trained data)
grayscaled_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# detect faces
face_coordinates = trained_face_data.detectMultiScale(grayscaled_img)
#test: print(face_coordinates)

# Draw Rectangle for faces
for (x, y, w, h) in face_coordinates:
    cv2.rectangle(img, (x,y), (x+w, y+h), (0,255,0), 3)

# example image
cv2.imshow('ai-face-detection', img)
# keeps program running until key is pressed
cv2.waitKey()

print("Code Completed") 