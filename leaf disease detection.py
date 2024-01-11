import cv2
import numpy as np

# Load the leaf image
image = cv2.imread("C:\\Users\\PC\Desktop\\affect123.jpg")


# Convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


# Apply Gaussian blur to reduce noise
blurred = cv2.GaussianBlur(gray, (5, 5), 0)


# Define a lower and upper threshold for detecting disease-related color
lower_green = np.array([35, 30, 30])  # Adjust these values based on your leaf disease color
upper_green = np.array([80, 255, 255])

# Create a mask to segment the disease-affected regions
mask = cv2.inRange(image, lower_green, upper_green)

# Apply the mask to the original image
result = cv2.bitwise_and(image, image, mask=mask)

# Find contours in the masked image
contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# Define the minimum contour area threshold
min_contour_area = 500  # Adjust this value based on your specific application

# Iterate through the contours and draw bounding boxes around diseases
for contour in contours:
     x, y, w, h = cv2.boundingRect(contour)
     if cv2.contourArea(contour) > min_contour_area:
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 500), 1)  # Draw a red rectangle



# Display the result
cv2.imshow('Leaf Disease Detection', image)


# Display the grayscale image
cv2.imshow('Grayscale Image', gray)

# Display the blurred image
cv2.imshow('Blurred Image', blurred)
cv2.imshow('Binary Image',mask)
cv2.imshow('Threshold Image',result)

cv2.waitKey(0)
cv2.destroyAllWindows()