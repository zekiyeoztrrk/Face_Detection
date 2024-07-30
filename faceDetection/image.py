import cv2
import numpy as np
import matplotlib.pyplot as plt

image_path = 'image.jpg'
image = cv2.imread(image_path)

# Grayscale 
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Gaussian Blurring
blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)

# Edge detection 
edges = cv2.Canny(blurred_image, 50, 150)

# Finding Contours
contours, _ = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contour_image = image.copy()
cv2.drawContours(contour_image, contours, -1, (0, 255, 0), 2)

# Face Detection
haar_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
face = haar_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

face_image = image.copy()
for (x, y, w, h) in face: 
    cv2.rectangle(face_image, (x, y), (x+w, y+h), (255, 0, 0), 2)
    
# Results
titles = ['Original Image', 'Gray Image', 'Blurred Image', 'Edge Image', 'Contour Image', 'Face Detection']
images = [image, gray_image, blurred_image, edges, contour_image, face_image]

for i in range(len(images)):
    plt.figure()  
    plt.imshow(images[i], cmap='gray')  
    plt.title(titles[i])  
    plt.axis('off') 
    plt.show()  

# The number of faces
plt.imshow(cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB))
plt.title(f'Face Detection\nYüz Sayısı: {len(face)}')
plt.xticks([]), plt.yticks([])

plt.tight_layout()
plt.show()

