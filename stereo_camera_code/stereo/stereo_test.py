import cv2
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt

def create_point_cloud_file(vertices, colors, filename):
    """
    Creates a PLY file with the given vertices (3D points) and colors.
    
    Args:
    - vertices: The 3D points, numpy array with shape (N, 3).
    - colors: The colors for each point, numpy array with shape (N, 3), where each row is [R, G, B] in [0, 255].
    - filename: The name of the PLY file to write.
    """
    # Ensure colors are in the expected format (0-255 range, 3 channels)
    colors = colors.reshape(-1, 3)
    vertices = np.hstack([vertices.reshape(-1, 3), colors])  # Combine points and colors
    
    # PLY header
    ply_header = '''ply
    format ascii 1.0
    element vertex %(vert_num)d
    property float x
    property float y
    property float z
    property uchar red
    property uchar green
    property uchar blue
    end_header
    '''
    
    # Write the data into the PLY file
    with open(filename, 'w') as f:
        f.write(ply_header % dict(vert_num=len(vertices)))  # Write the header with the number of vertices
        np.savetxt(f, vertices, '%f %f %f %d %d %d')  # Save vertices with RGB values

    print(f"Point cloud saved as {filename}")

# Parse calibration file (same function as before)
def parse_calibration_file(filepath):
    calib_data = {}

    with open(filepath, 'r') as file:
        for line in file:
            if '=' not in line:
                continue
            try:
                key, value = line.strip().split('=')
                key = key.strip()
                value = value.strip()

                if key.startswith("cam"):
                    calib_data[key] = np.array([list(map(float, row.split())) for row in value.strip("[]").split(';')])
                elif key in {"doffs", "baseline", "width", "height", "ndisp", "isint", "vmin", "vmax"}:
                    calib_data[key] = float(value) if '.' in value else int(value)
            except ValueError as e:
                print(f"Error parsing line: {line.strip()}")
                print(f"Error: {e}")
                continue

    return calib_data

# Compute Q matrix (same function as before)
def compute_q_matrix(calib_data):
    focal_length = calib_data["cam0"][0, 0]
    cx = calib_data["cam0"][0, 2]
    cy = calib_data["cam0"][1, 2]
    doffs = calib_data["doffs"]
    baseline = calib_data["baseline"]

    Q = np.array([
        [1, 0, 0, -cx],
        [0, 1, 0, -cy],
        [0, 0, 0, focal_length],
        [0, 0, -1 / baseline, doffs / baseline]
    ])
    return Q

# Main function for point cloud generation
def main():
    # Load rectified stereo images
    left_img = cv2.imread(r"C:\Users\sudeshi\Documents\Kaushek\shivam\im0.png", cv2.IMREAD_GRAYSCALE)
    right_img = cv2.imread(r"C:\Users\sudeshi\Documents\Kaushek\shivam\im1.png", cv2.IMREAD_GRAYSCALE)

    # Stereo matching (Block matching)
    stereo = cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities=16 * 10,
        blockSize=9,
        P1=8 * 3 * 11**2,
        P2=32 * 3 * 11**2,
        disp12MaxDiff=1,
        uniquenessRatio=10,
        speckleWindowSize=100,
        speckleRange=16
    )
    disparity = stereo.compute(left_img, right_img).astype(np.float32) / 16.0

    # Load calibration data and compute Q matrix
    calib_filepath = "calib.txt"  # Path to calibration file
    calib_data = parse_calibration_file(calib_filepath)
    Q = compute_q_matrix(calib_data)

    # Reproject to 3D
    points_3D = cv2.reprojectImageTo3D(disparity, Q)

    # Filter out non-finite (inf, -inf, NaN) points
    mask = np.isfinite(points_3D).all(axis=2)  # Check if all values are finite (no inf or NaN)
    points_3D = points_3D[mask]  # Apply mask to filter out invalid points
    colors = cv2.imread(r"C:\Users\sudeshi\Documents\Kaushek\shivam\im0.png")[mask]  # Use left image for coloring

    cv2.imshow('Limg',left_img)
    cv2.imshow('Rimg',right_img)
    cv2.waitKey(0)

    # Close all windows
    cv2.destroyAllWindows()
    # Display disparity map
    plt.imshow(disparity, cmap='gray')
    plt.colorbar()
    plt.title("Disparity Map")
    plt.show()
    
    print("Q Matrix:")
    print(Q)

    print("Point Cloud Sample:")
    print(points_3D[:10])

    # Save point cloud using the create_point_cloud_file function
    output_file = "output.txt"
    create_point_cloud_file(points_3D, colors, output_file)

    # Visualize point cloud using Open3D
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(points_3D)
    pc.colors = o3d.utility.Vector3dVector(colors / 255.0)

    # Save and visualize with Open3D
    o3d.io.write_point_cloud("Myoutput_open3d.ply", pc)
    o3d.visualization.draw_geometries([pc])

if __name__ == "__main__":
    main()
