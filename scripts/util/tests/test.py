import os
import cv2
import numpy as np
from scipy.spatial import distance as dist
from detect import get_contour, draw_contour
script_dir = os.path.dirname(__file__)
    

    # Dictionary to hold the loaded template images
   
image = cv2.imread(os.path.join(script_dir, "pokemon.png"))

# # Assuming 'image' is your input image
# gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
# contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# centroids = []
# for cnt in contours:
#     M = cv2.moments(cnt)
#     if M["m00"] != 0:
#         cX = int(M["m10"] / M["m00"])
#         cY = int(M["m01"] / M["m00"])
#         centroids.append((cX, cY))
#         # Draw centroid for visualization
#         cv2.circle(image, (cX, cY), 5, (255, 0, 0), -1)
# # Calculate distances between each pair of centroids
# for i in range(len(centroids)):
#     for j in range(i + 1, len(centroids)):
#         d = dist.euclidean(centroids[i], centroids[j])
#         print(f"Distance between object {i} and object {j}: {d}")

#         # Optionally, draw a line between centroids and display the distance
#         cv2.line(image, centroids[i], centroids[j], (0, 255, 0), 2)
#         midpoint = ((centroids[i][0] + centroids[j][0]) // 2, (centroids[i][1] + centroids[j][1]) // 2)
#         cv2.putText(image, f"{d:.2f}", midpoint, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
# cv2.imshow("Distances", image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

contours = get_contour(image)
h, w = image.shape[:2]
center_point = (w // 2, h // 2)

# Optionally, draw the central point for visualization
cv2.circle(image, center_point, 5, (255, 0, 0), -1)
from scipy.spatial import distance as dist

for i, cnt in enumerate(contours):
    # Convert contour to a simple list of points
    contour_points = cnt.squeeze()
    
    # Find the closest contour point to the center
    closest_point = min(contour_points, key=lambda x: dist.euclidean(x, center_point))
    closest_point = tuple(closest_point)
    
    # Calculate the distance from the central point to this closest contour point
    distance_to_contour = dist.euclidean(center_point, closest_point)
    
    print(f"Distance from center to edge of contour {i}: {distance_to_contour:.2f}")

    # Draw the closest point and a line to it for visualization
    cv2.circle(image, closest_point, 5, (0, 255, 0), -1)
    cv2.line(image, center_point, closest_point, (255, 0, 0), 2)
cv2.imshow("Distances to Edges", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
