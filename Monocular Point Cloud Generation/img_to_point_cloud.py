import cv2
import numpy as np
import open3d as o3d
from transformers import pipeline
from PIL import Image
import argparse
from matplotlib import pyplot as plt

# Initialize Depth-Estimation Model
pipe = pipeline(task="depth-estimation", model="LiheYoung/depth-anything-small-hf")

# Estimate depth of an image
def estimate_depth(image, plot_depth=False):
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    im_pil = Image.fromarray(img)
    depth = pipe(im_pil)["depth"]
    depth_map = np.array(depth, dtype=np.float32)

    # Normalize and filter the depth map
    depth_map = cv2.GaussianBlur(depth_map, (5, 5), 0)
    depth_map = depth_map / np.percentile(depth_map, 99)  # Normalize to [0, 1]

    if plot_depth:
        plt.figure(figsize=(8, 6))
        plt.imshow(depth_map, cmap="plasma")
        plt.colorbar(label="Depth")
        plt.title("Predicted Depth Map")
        plt.axis("off")
        plt.show()

    return depth_map

# Generate camera intrinsic matrix
def get_camera_intrinsics(H, W, fov=60.0):
    f = -0.5 * W / (np.tan(0.5 * fov * (np.pi / 180)))
    cx = 0.5 * W
    cy = 0.5 * H
    return np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]])

# Generate 3D point cloud from depth map
def generate_point_cloud(image, depth_map, camera_matrix):
    h, w = depth_map.shape
    points_3d = []
    colors = []

    for v in range(h):
        for u in range(w):
            Z = depth_map[v, u]
            if Z <= 0:
                continue
            X = (u - camera_matrix[0, 2]) * Z / camera_matrix[0, 0]
            Y = (v - camera_matrix[1, 2]) * Z / camera_matrix[1, 1]
            points_3d.append([X, Y, Z])
            # BGR to RGB
            colors.append(image[v, u][::-1])  

    return np.array(points_3d), np.array(colors) / 255.0

# Convert numpy arrays to Open3D PointCloud
def to_open3d_point_cloud(points, colors):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    if colors is not None:
        pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd

# Visualize the point cloud
def visualize_point_cloud(pcd):
    if len(pcd.points) == 0:
        print("No points to visualize.")
        return
    
    # Visualize the point cloud
    o3d.visualization.draw_geometries([pcd], window_name="Point Cloud")

# Visualize the mesh
def create_mesh_from_pointcloud(pcd):
    print("Processing 3D Mesh...")
    # Estimate normals for the point cloud
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
    )
    pcd.orient_normals_consistent_tangent_plane(k=10)
    
    print("Processing 3D Mesh...")
    # Ball-pivoting mesh reconstruction
    distances = pcd.compute_nearest_neighbor_distance()
    avg_distance = np.mean(distances)
    radius = 3 * avg_distance
    
    print("Processing 3D Mesh...")
    bpa_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
        pcd, o3d.utility.DoubleVector([radius, radius * 2])
    )

    # Visualize the mesh
    o3d.visualization.draw_geometries([bpa_mesh], window_name="3D Mesh")

# Main pipeline
def main():
    parser = argparse.ArgumentParser(description="Generate a unified point cloud from video.")
    parser.add_argument("--input", type=str, default="input\images\scene1.jpg", help="Path to the input Image.")
    parser.add_argument("--output", type=str, default="output\images\scene1.ply", help="Path to the output point cloud file.")
    parser.add_argument("--view_depth", type=int, default=0, help="1 to view the depth map.")
    parser.add_argument("--view_pc", type=int, default=0, help="1 to save the point cloud.")
    parser.add_argument("--save_pc", type=int, default=0, help="1 to save the point cloud.")
    args = parser.parse_args()

    print(f"Processing Image...")

    img = cv2.imread(args.input)
    if img is None:
        print("Error: Image not found or could not be loaded.")
    else:
        # Estimate depth
        depth_map = estimate_depth(img, plot_depth=(args.view_depth == 1))
        # Convert depth map to 16-bit unsigned integer format
        depth_16bit = (depth_map * 1000).astype(np.uint16)  # Scale and convert to 16-bit
        # Create Open3D Image from 16-bit depth map
        depth_image = o3d.geometry.Image(depth_16bit)

        # Create RGBD Image
        rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d.geometry.Image(img), depth_image, depth_scale=1000.0, depth_trunc=3.0
        )
        # Generate point cloud for current frame
        img_pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
            rgbd_image,
            o3d.camera.PinholeCameraIntrinsic(o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault)
        )
        flip_matrix = np.array([[-1, 0, 0, 0],
                                [0, -1, 0, 0],
                                [0, 0, -1, 0],
                                [0, 0, 0, 1]])
        
        # Flipping the point cloud as it will generate upside down
        img_pcd.transform(flip_matrix)

        if args.view_pc == 1:
            # Visualize the final merged point cloud
            print("Visualizing merged point cloud...")
            visualize_point_cloud(img_pcd)

        # Save the point cloud
        if args.save_pc == 1:
            o3d.io.write_point_cloud(args.output, img_pcd)
            print(f"Point cloud saved to {args.output}")
        
        create_mesh_from_pointcloud(img_pcd)


if __name__ == "__main__":
    main()