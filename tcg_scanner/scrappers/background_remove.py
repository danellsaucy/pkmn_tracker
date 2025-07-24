from rembg import remove
from PIL import Image
import cv2
import numpy as np

# Load and process the Pokemon card
input_image = Image.open(r"C:\Users\daforbes\Downloads\s-l1200.jpg")
output_image = remove(input_image)

# Convert to numpy array (compatible with numpy 1.26.4)
result_array = np.array(output_image)
print(f"Image shape: {result_array.shape}")  # Should be (height, width, 4) for RGBA

# Convert for OpenCV display (RGBA to BGRA)
result_cv = cv2.cvtColor(result_array, cv2.COLOR_RGBA2BGRA)

# Display the result
cv2.imshow('Pokemon Card - Background Removed', result_cv)
cv2.waitKey(0)
cv2.destroyAllWindows()