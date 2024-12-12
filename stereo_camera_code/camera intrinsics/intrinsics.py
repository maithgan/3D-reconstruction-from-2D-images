import cv2
import numpy as np
import glob

# Termination criteria for corner sub-pixel accuracy
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Define the checkerboard dimensions
CHECKERBOARD = (9, 7)  # Adjust rows and columns based on your checkerboard

# Define the real-world square size in millimeters
square_size = 20  # Example: 25 mm per square

# Prepare object points with real-world dimensions
objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
objp[:, :2] *= square_size  # Scale points to match real-world box size

# Arrays to store object points and image points from all the images
objpoints = []  # 3D points in real world space
imgpoints = []  # 2D points in image plane

# Load all checkerboard images
image_paths = glob.glob(r'C:\Users\sudeshi\Documents\Kaushek\shivam\cam_shivam\*.jpg')  # Update path to your images
print(f"Found {len(image_paths)} images for calibration.")

for idx, fname in enumerate(image_paths):
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find the checkerboard corners
    ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)

    # If found, add object points and image points
    if ret:
        objpoints.append(objp)

        # Refine corner locations
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners2)

        # Draw and display the corners
        img = cv2.drawChessboardCorners(img, CHECKERBOARD, corners2, ret)
        cv2.imshow(f'Checkerboard {idx + 1}', img)
        cv2.waitKey(500)
    else:
        print(f"Checkerboard not detected in image {fname}")

cv2.destroyAllWindows()

# Perform camera calibration
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

# Print the camera matrix and distortion coefficients
if ret:
    print("Camera calibration was successful!")
    print("\nCamera intrinsic matrix:\n", mtx)
    print("\nDistortion coefficients:\n", dist)
else:
    print("Camera calibration failed. Check your images and setup.")

# Optional: Save calibration results
np.savez("camera_calibration.npz", mtx=mtx, dist=dist, rvecs=rvecs, tvecs=tvecs)

# Undistort an example image
example_img = cv2.imread(image_paths[0])  # Use the first image for testing
h, w = example_img.shape[:2]
new_camera_mtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))

# Undistort
undistorted_img = cv2.undistort(example_img, mtx, dist, None, new_camera_mtx)

# Display undistorted image
cv2.imshow('Original Image', example_img)
cv2.imshow('Undistorted Image', undistorted_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
