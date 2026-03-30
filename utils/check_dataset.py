import json
import numpy as np
import cv2
from scipy.spatial.transform import Rotation
import os
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch
import glob
import matplotlib.cm as cm

def quaternion_to_rotation_matrix(q):
    """Convert quaternion to rotation matrix using scipy"""
    # scipy expects quaternion in [x, y, z, w] format
    # but our data is in [w, x, y, z] format
    w, x, y, z = q
    quat_scipy = [x, y, z, w]  # Convert to scipy format
    
    # Create Rotation object from quaternion
    rotation = Rotation.from_quat(quat_scipy)
    return rotation.as_matrix()

def create_3d_box(center, wlh, yaw):
    """Create 3D bounding box with 8 vertices using yaw angle"""
    w, l, h = wlh
    
    # Create 8 vertices of bounding box (in object coordinate system)
    # nuScenes convention: x=right, y=forward, z=up
    corners = np.array([
        [-w/2, -l/2, -h/2],  # 0: bottom-left-back
        [w/2, -l/2, -h/2],   # 1: bottom-right-back
        [w/2, l/2, -h/2],    # 2: bottom-right-front
        [-w/2, l/2, -h/2],   # 3: bottom-left-front
        [-w/2, -l/2, h/2],   # 4: top-left-back
        [w/2, -l/2, h/2],    # 5: top-right-back
        [w/2, l/2, h/2],     # 6: top-right-front
        [-w/2, l/2, h/2]     # 7: top-left-front
    ])
    
    # Create rotation matrix from yaw angle (rotation around Y axis)
    # In camera coordinates: Y is up, rotation around Y axis
    cos_yaw = np.cos(yaw)
    sin_yaw = np.sin(yaw)
    
    rot_matrix = np.array([
        [cos_yaw, 0, sin_yaw],
        [0, 1, 0],
        [-sin_yaw, 0, cos_yaw]
    ])
    
    # Rotate corners and translate to world position
    rotated_corners = np.dot(corners, rot_matrix.T)
    rotated_corners += np.array(center)
    
    return rotated_corners

def read_lidar_data(lidar_path):
    """Read LiDAR point cloud data from .pcd.bin file"""
    try:
        # Read binary file
        points = np.fromfile(lidar_path, dtype=np.float32)
        # Reshape to Nx4 (x, y, z, intensity, ring index)
        points = points.reshape(-1, 5)
        print(f"Loaded {len(points)} LiDAR points")
        return points
    except Exception as e:
        print(f"Error reading LiDAR data: {e}")
        return None

def transform_lidar_to_camera(points_lidar, cam2lidar_rotation, cam2lidar_translation):
    """Transform LiDAR points to camera coordinate system"""
    # Ensure inputs are numpy arrays
    cam2lidar_R = np.array(cam2lidar_rotation)
    cam2lidar_t = np.array(cam2lidar_translation)
    
    # Transform: lidar -> camera
    # P_cam = R_lidar2cam * P_lidar + t_lidar2cam
    # where R_lidar2cam = R_cam2lidar^T, t_lidar2cam = -R_cam2lidar^T * t_cam2lidar
    
    R_lidar2cam = cam2lidar_R.T
    t_lidar2cam = -np.dot(R_lidar2cam, cam2lidar_t)
    
    # Apply transformation
    points_cam = np.dot(points_lidar[:, :3], R_lidar2cam.T) + t_lidar2cam
    
    # Keep intensity as 4th column
    points_cam_with_intensity = np.column_stack([points_cam, points_lidar[:, 3]])
    
    return points_cam_with_intensity

def project_3d_to_2d_camera_coords(points_3d, cam_intrinsic):
    """Project 3D points to 2D image plane (points already in camera coordinates)"""
    # print(f"\n=== Projection Debug Info ===")
    # print(f"Sample 3D point: {points_3d[0]}")
    # print(f"Input 3D points shape: {points_3d.shape}")
    
    # Ensure inputs are numpy arrays
    cam_intrinsic = np.array(cam_intrinsic)
    points_3d = np.array(points_3d)
    
    # Check if points are in front of camera (z > 0)
    z_positive = np.sum(points_3d[:, 2] > 0)
    # print(f"Points in front of camera (z > 0): {z_positive}/{len(points_3d)}")
    
    # Project to image plane
    x = points_3d[:, 0] / points_3d[:, 2]
    y = points_3d[:, 1] / points_3d[:, 2]
    
    # print(f"Normalized image coordinates - x range: [{np.min(x):.2f}, {np.max(x):.2f}]")
    # print(f"Normalized image coordinates - y range: [{np.min(y):.2f}, {np.max(y):.2f}]")
    
    # Apply camera intrinsic parameters
    fx = cam_intrinsic[0, 0]
    fy = cam_intrinsic[1, 1]
    cx = cam_intrinsic[0, 2]
    cy = cam_intrinsic[1, 2]
    
    # print(f"Camera intrinsic - fx: {fx}, fy: {fy}, cx: {cx}, cy: {cy}")
    
    u = fx * x + cx
    v = fy * y + cy
    
    result = np.column_stack([u, v])
    # print(f"Final 2D coordinates - u range: [{np.min(u):.2f}, {np.max(u):.2f}]")
    # print(f"Final 2D coordinates - v range: [{np.min(v):.2f}, {np.max(v):.2f}]")
    
    # Check if points are within image bounds
    valid_mask = (u >= 0) & (u < 1600) & (v >= 0) & (v < 900) & (points_3d[:, 2] > 0)
    
    return result, valid_mask

def draw_3d_box_on_image(image, box_corners_2d, color=(0, 255, 0), thickness=2):
    """Draw 3D bounding box on image using OpenCV"""
    if len(box_corners_2d) != 8:
        return
    
    # Define edges of the 3D box (12 edges)
    edges = [
        # Bottom face
        (0, 1), (1, 2), (2, 3), (3, 0),
        # Top face
        (4, 5), (5, 6), (6, 7), (7, 4),
        # Vertical edges
        (0, 4), (1, 5), (2, 6), (3, 7)
    ]
    
    # Draw each edge
    for start_idx, end_idx in edges:
        start_point = tuple(map(int, box_corners_2d[start_idx]))
        end_point = tuple(map(int, box_corners_2d[end_idx]))
        
        # Check if both points are valid (within image bounds)
        if (0 <= start_point[0] < image.shape[1] and 0 <= start_point[1] < image.shape[0] and
            0 <= end_point[0] < image.shape[1] and 0 <= end_point[1] < image.shape[0]):
            cv2.line(image, start_point, end_point, color, thickness)

def visualize_3d_boxes_on_image(data, data_root):
    """Visualize all 3D bounding boxes on the image"""
    # Read image
    image_path = os.path.join(data_root, data['image_path'])
    if not os.path.exists(image_path):
        print(f"Image not found: {image_path}")
        return None
    
    image = cv2.imread(image_path)
    if image is None:
        print(f"Failed to read image: {image_path}")
        return None
    
    print(f"Image loaded: {image.shape}")
    
    # Create a copy for drawing
    image_with_boxes = image.copy()
    
    # Process each 3D bounding box
    for i, bbox in enumerate(data['cam_bbox_3d']):
        # print(f"\nProcessing bbox {i+1}: {bbox['category']}")
        
        # Get bbox parameters
        center = np.array(bbox['center'])
        wlh = np.array(bbox['wlh'])
        
        # Use yaw field directly
        if 'yaw' in bbox:
            yaw = bbox['yaw']
        else:
            print(f"  Warning: No yaw field found for {bbox['category']}, using 0")
            yaw = 0
        
        # print(f"  Center: {center}")
        # print(f"  WLH: {wlh}")
        # print(f"  Yaw: {np.degrees(yaw):.1f}°")
        
        # Create 3D box corners using yaw
        box_corners_3d = create_3d_box(center, wlh, yaw)
        
        # Project to 2D (points already in camera coordinates)
        box_corners_2d, valid_mask = project_3d_to_2d_camera_coords(
            box_corners_3d, 
            data['cam_intrinsic']
        )
        
        # Check projection validity
        valid_corners = np.sum(valid_mask)
        print(f"  Valid corners: {valid_corners}/8")
        
        if valid_corners >= 4:  # Need at least 4 corners to draw a meaningful box
            # Choose color based on category
            if bbox['category'] == 'car':
                color = (0, 255, 0)  # Green
            elif bbox['category'] == 'pedestrian':
                color = (255, 0, 0)  # Blue
            elif bbox['category'] == 'truck':
                color = (0, 0, 255)  # Red
            else:
                color = (255, 255, 0)  # Cyan
            
            # Draw the 3D box
            draw_3d_box_on_image(image_with_boxes, box_corners_2d, color, thickness=2)
            
            # Draw box center point
            center_2d, _ = project_3d_to_2d_camera_coords(
                np.array([center]),  # Pass as list of single point
                data['cam_intrinsic']
            )
            
            if len(center_2d) > 0:
                center_point = tuple(map(int, center_2d[0]))
                if (0 <= center_point[0] < image.shape[1] and 
                    0 <= center_point[1] < image.shape[0]):
                    cv2.circle(image_with_boxes, center_point, 5, color, -1)
                    
                    # Add label
                    label = f"{bbox['category']}"
                    cv2.putText(image_with_boxes, label, 
                              (center_point[0] + 10, center_point[1] - 10),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        else:
            print(f"  Warning: Too few valid corners for {bbox['category']}")
    
    return image_with_boxes

def create_lidar_image_overlay(data, data_root):
    """Create image with LiDAR points and 3D boxes overlaid"""
    print("\n=== Creating LiDAR-Image Overlay ===")
    
    # Read image
    image_path = os.path.join(data_root, data['image_path'])
    if not os.path.exists(image_path):
        print(f"Image not found: {image_path}")
        return None
    
    image = cv2.imread(image_path)
    if image is None:
        print(f"Failed to read image: {image_path}")
        return None
    
    # Read LiDAR data
    lidar_path = os.path.join(data_root, data['lidar_path'])
    if not os.path.exists(lidar_path):
        print(f"LiDAR file not found: {lidar_path}")
        return None
    
    points_lidar = read_lidar_data(lidar_path)
    if points_lidar is None:
        return None
    
    # Transform LiDAR points to camera coordinates
    points_cam = transform_lidar_to_camera(
        points_lidar, 
        data['cam2lidar_rotation'], 
        data['cam2lidar_translation']
    )
    
    # Project LiDAR points to image
    points_2d, valid_mask = project_3d_to_2d_camera_coords(
        points_cam[:, :3],  # Only xyz coordinates
        data['cam_intrinsic']
    )
    
    # Create overlay image
    overlay_image = np.zeros_like(image, dtype=np.uint8)
    
    # Draw LiDAR points
    valid_points = points_2d[valid_mask]
    valid_points_3d = points_cam[valid_mask, :3]  # Get 3D coordinates for distance calculation
    
    print(f"Drawing {len(valid_points)} valid LiDAR points")
    
    # Calculate distances from camera (z coordinate in camera frame)
    distances = valid_points_3d[:, 2]  # Z coordinate is distance from camera
    
    # Create colormap for distances
    # Use a reversed colormap so that closer points are warmer colors
    colormap = cm.jet_r  # jet_r: red (close) -> blue (far)
    
    # Normalize distances to [0, 1] for colormap
    min_dist = np.min(distances)
    max_dist = np.max(distances)
    if max_dist > min_dist:
        normalized_distances = (distances - min_dist) / (max_dist - min_dist)
    else:
        normalized_distances = np.zeros_like(distances)
    
    # Get colors from colormap
    colors = colormap(normalized_distances)
    
    # Draw points with distance-based colors
    for i, (point_2d, color_rgba) in enumerate(zip(valid_points, colors)):
        # Convert to integer coordinates
        u, v = int(point_2d[0]), int(point_2d[1])
        
        # Convert RGBA to BGR for OpenCV (and scale to 0-255)
        color_bgr = (int(color_rgba[2] * 255), int(color_rgba[1] * 255), int(color_rgba[0] * 255))
        
        # Draw small circle for each point
        cv2.circle(overlay_image, (u, v), 2, color_bgr, 2)
    
    # # Draw 3D bounding boxes on top
    # for i, bbox in enumerate(data['cam_bbox_3d']):
    #     center = np.array(bbox['center'])
    #     wlh = np.array(bbox['wlh'])
    #     orientation = np.array(bbox['orientation'])
        
    #     # Create 3D box corners
    #     box_corners_3d = create_3d_box(center, wlh, orientation)
        
    #     # Project to 2D
    #     box_corners_2d, valid_mask = project_3d_to_2d_camera_coords(
    #         box_corners_3d, 
    #         data['cam_intrinsic']
    #     )
        
    #     # Check projection validity
    #     valid_corners = np.sum(valid_mask)
        
    #     if valid_corners >= 4:
    #         # Choose color based on category
    #         if bbox['category'] == 'car':
    #             color = (0, 255, 0)  # Green
    #         elif bbox['category'] == 'pedestrian':
    #             color = (255, 0, 0)  # Blue
    #         elif bbox['category'] == 'truck':
    #             color = (0, 0, 255)  # Red
    #         else:
    #             color = (255, 255, 0)  # Cyan
            
    #         # Draw the 3D box
    #         # draw_3d_box_on_image(overlay_image, box_corners_2d, color, thickness=2)
            
    #         # Draw box center point
    #         center_2d, _ = project_3d_to_2d_camera_coords(
    #             np.array([center]),
    #             data['cam_intrinsic']
    #         )
            
    #         if len(center_2d) > 0:
    #             center_point = tuple(map(int, center_2d[0]))
    #             if (0 <= center_point[0] < image.shape[1] and 
    #                 0 <= center_point[1] < image.shape[0]):
    #                 cv2.circle(overlay_image, center_point, 5, color, -1)
                    
    #                     # Add label
    #                     label = f"{bbox['category']}"
    #                     cv2.putText(overlay_image, label, 
    #                               (center_point[0] + 10, center_point[1] - 10),
    #                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    
    return overlay_image

def create_bev_visualization(data):
    """Create Bird's Eye View visualization of 3D boxes in XZ plane (camera coordinates)"""
    print("\n=== Creating BEV Visualization (XZ plane) ===")
    
    # Create figure with larger size for better visibility
    fig, ax = plt.subplots(figsize=(16, 12))
    
    # Set up coordinate system for camera coordinates
    # x: right, z: forward (camera view direction)
    # Range: 30m forward, ±15m left/right
    x_range = [-15, 15]  # meters (left-right)
    z_range = [0, 30]    # meters (forward)
    
    ax.set_xlim(x_range)
    ax.set_ylim(z_range)
    ax.set_xlabel('X (left-right) [m]', fontsize=14)
    ax.set_ylabel('Z (forward) [m]', fontsize=14)
    ax.set_title('Bird\'s Eye View - 3D Bounding Boxes (XZ plane, 30m range)', fontsize=16)
    ax.grid(True, alpha=0.3)
    
    # Increase tick label sizes
    ax.tick_params(axis='both', which='major', labelsize=12)
    
    # Draw ego vehicle (camera position) - make it larger
    ego_pos = [0, 0]
    ax.plot(ego_pos[0], ego_pos[1], 'ko', markersize=15, label='Ego Vehicle')
    
    # Process each 3D bounding box
    for i, bbox in enumerate(data['cam_bbox_3d']):
        center = np.array(bbox['center'])
        wlh = np.array(bbox['wlh'])
        
        # Use yaw field directly
        if 'yaw' in bbox:
            yaw = bbox['yaw']
        else:
            print(f"  Warning: No yaw field found for {bbox['category']}, using 0")
            yaw = 0
        
        # Get 2D center for BEV (x, z coordinates)
        center_2d = [center[0], center[2]]  # x and z coordinates
        
        # CORRECTED: In nuScenes convention:
        # w (width) = x direction (left-right) in 3D
        # l (length) = y direction (forward-backward) in 3D
        # h (height) = z direction (up-down) in 3D
        # But in BEV XZ plane: 
        # width should be in Z direction (forward-backward)
        # length should be in X direction (left-right)
        width = wlh[0]   # w: should be in Z direction for BEV
        length = wlh[1]  # l: should be in X direction for BEV
        
        # Create rectangle vertices around center (before rotation)
        # Rectangle vertices in local coordinate system
        half_w = width / 2
        half_l = length / 2
        
        # Define 4 corners relative to center 
        # In BEV XZ plane: X=right, Z=forward
        # When yaw=0: length should point in +X direction (right)
        # width is in Z direction (forward-backward)
        corners_local = np.array([
            [-half_l, -half_w],  # bottom-left (back-left): length in X, width in Z
            [half_l, -half_w],   # bottom-right (back-right): length in X, width in Z
            [half_l, half_w],    # top-right (front-right): length in X, width in Z
            [-half_l, half_w]    # top-left (front-left): length in X, width in Z
        ])
        
        # Apply rotation around center - CORRECTED for camera coordinate system
        # In camera coordinates: yaw is rotation around Y axis (pointing down)
        # In BEV XZ plane: we need to rotate the box so that length aligns with arrow
        cos_yaw = np.cos(yaw)
        sin_yaw = np.sin(yaw)
        
        # CORRECTED: For camera coordinate system in BEV XZ plane
        # When yaw=0: box length should point in +X direction (right)
        # When yaw=π/2: box length should point in +Z direction (forward)
        rotation_matrix = np.array([
            [cos_yaw, -sin_yaw],    # X component rotation
            [sin_yaw, cos_yaw]      # Z component rotation
        ])
        
        # Rotate corners
        corners_rotated = np.dot(corners_local, rotation_matrix.T)
        
        # Translate to world position
        corners_world = corners_rotated + np.array(center_2d)
        
        # Create polygon patch for rotated rectangle
        from matplotlib.patches import Polygon
        rect_polygon = Polygon(
            corners_world,
            alpha=0.8,
            linewidth=3,
            edgecolor='black'
        )
        
        # Choose color based on category
        if bbox['category'] == 'car':
            color = 'green'
        elif bbox['category'] == 'pedestrian':
            color = 'red'
        elif bbox['category'] == 'truck':
            color = 'blue'
        else:
            color = 'orange'
        
        rect_polygon.set_facecolor(color)
        ax.add_patch(rect_polygon)
        
        # Draw heading arrow to show vehicle orientation
        # In camera coordinates: X positive is right, Z positive is forward
        # Yaw is rotation around Y axis: X positive direction rotating towards Z positive direction is positive
        arrow_length = max(width, length) * 0.8
        
        # For camera coordinates, yaw should be applied correctly
        # The arrow should point in the direction the vehicle is facing
        # CORRECTED: Use the actual center_2d coordinates for arrow start and end
        arrow_end_x = center_2d[0] + arrow_length * np.cos(yaw)
        arrow_end_z = center_2d[1] + arrow_length * np.sin(yaw)
        
        # Create arrow patch with larger size
        arrow = FancyArrowPatch(
            (center_2d[0], center_2d[1]),  # Start from center
            (arrow_end_x, arrow_end_z),    # End at calculated position
            arrowstyle='->',
            mutation_scale=30,
            color='red',  # Changed to red for better visibility
            linewidth=3
        )
        ax.add_patch(arrow)
        
        # Add label with yaw information and dimensions - larger font
        label = f"{bbox['category'][:3]}\n{np.degrees(yaw):.0f}°\n{wlh[0]:.1f}x{wlh[1]:.1f}"
        ax.text(center_2d[0], center_2d[1], label, 
                ha='center', va='center', fontsize=10, weight='bold')
    
    # Make legend larger
    ax.legend(fontsize=14)
    
    # Set aspect ratio to equal for proper scaling
    ax.set_aspect('equal')
    
    # Add some padding around the plot
    plt.tight_layout()
    
    return fig

def create_visualization_folder(base_name):
    """Create folder structure for visualization results"""
    # Create main visualization folder
    viz_folder = "visualization_results"
    if not os.path.exists(viz_folder):
        os.makedirs(viz_folder)
    
    # Create subfolder for this frame
    frame_folder = os.path.join(viz_folder, base_name)
    if not os.path.exists(frame_folder):
        os.makedirs(frame_folder)
    
    return frame_folder

def filter_boxes_by_distance_fov(data, data_root, distance_threshold=30):
    """Filter out boxes that are too far away from the camera and outside the camera FOV"""
    # Create a copy of the data to avoid modifying the original
    filtered_data = data.copy()
    
    # Get camera intrinsic parameters
    cam_intrinsic = np.array(data['cam_intrinsic'])
    fx = cam_intrinsic[0, 0]
    fy = cam_intrinsic[1, 1]
    cx = cam_intrinsic[0, 2]
    cy = cam_intrinsic[1, 2]
    
    # Image dimensions (assuming 1600x900 based on your code)
    image_path = os.path.join(data_root, data['image_path'])
    img_height, img_width = cv2.imread(image_path).shape[:2]
    
    # FOV angles (in radians) - typical camera FOV
    fov_horizontal = 2 * np.arctan(cx / fx)  # Horizontal FOV
    fov_vertical = 2 * np.arctan(cy / fy)    # Vertical FOV
    
    print(f"Camera FOV - Horizontal: {np.degrees(fov_horizontal):.1f}°, Vertical: {np.degrees(fov_vertical):.1f}°")
    
    # Filter 3D bounding boxes
    valid_boxes = []
    total_boxes = len(data['cam_bbox_3d'])
    
    for i, bbox in enumerate(data['cam_bbox_3d']):
        center = np.array(bbox['center'])
        
        # Check distance from camera (z coordinate in camera frame)
        distance = center[2]
        
        # Check if box is too far away
        if distance > distance_threshold:
            print(f"  Box {i+1} ({bbox['category']}) filtered: distance {distance:.1f}m > {distance_threshold}m")
            continue
        
        # Check if box center is in front of camera
        if center[2] <= 0:
            print(f"  Box {i+1} ({bbox['category']}) filtered: behind camera (z={center[2]:.1f}m)")
            continue
        
        # Project box center to 2D image coordinates
        # Normalize coordinates
        x = center[0] / center[2]
        y = center[1] / center[2]
        
        # Apply camera intrinsic parameters
        u = fx * x + cx
        v = fy * y + cy
        
        # Check if center is within image bounds with some margin
        margin = 50  # pixels margin from image edges
        if (u < -margin or u > img_width + margin or 
            v < -margin or v > img_height + margin):
            print(f"  Box {i+1} ({bbox['category']}) filtered: center outside image bounds (u={u:.1f}, v={v:.1f})")
            continue
        
        # Check if box is within reasonable FOV limits
        # Calculate horizontal and vertical angles
        horizontal_angle = np.arctan2(center[0], center[2])
        vertical_angle = np.arctan2(center[1], center[2])
        
        # Add some margin to FOV (e.g., 10 degrees)
        fov_margin = np.radians(10)
        
        if (abs(horizontal_angle) > fov_horizontal/2 + fov_margin or 
            abs(vertical_angle) > fov_vertical/2 + fov_margin):
            print(f"  Box {i+1} ({bbox['category']}) filtered: outside FOV (h_angle={np.degrees(horizontal_angle):.1f}°, v_angle={np.degrees(vertical_angle):.1f}°)")
            continue
        
        # Box passed all filters
        valid_boxes.append(bbox)
        print(f"  Box {i+1} ({bbox['category']}) kept: distance={distance:.1f}m, center=({u:.1f}, {v:.1f})")
    
    # Update the filtered data
    filtered_data['cam_bbox_3d'] = valid_boxes
    
    print(f"\nFiltering results: {len(valid_boxes)}/{total_boxes} boxes kept")
    
    return filtered_data

def process_json_file(json_path, data_root):
    """Process a single JSON file"""
    print(f"\n{'='*60}")
    print(f"Processing: {os.path.basename(json_path)}")
    print(f"{'='*60}")
    
    # Read JSON data
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    data = filter_boxes_by_distance_fov(data, data_root, 30)
    
    # Create visualization folder
    base_name = os.path.splitext(os.path.basename(json_path))[0]
    frame_folder = create_visualization_folder(base_name)
    
    # Visualize 3D boxes on image
    image_with_boxes = visualize_3d_boxes_on_image(data, data_root)
    
    if image_with_boxes is not None:
        # Save the result
        output_path = os.path.join(frame_folder, 'projection_result.jpg')
        cv2.imwrite(output_path, image_with_boxes)
        print(f"\nResult saved to: {output_path}")
        
        # Create BEV visualization
        fig = create_bev_visualization(data)
        bev_path = os.path.join(frame_folder, 'bev_result.png')
        fig.savefig(bev_path, dpi=150, bbox_inches='tight')
        print(f"BEV visualization saved to: {bev_path}")
        plt.close(fig)
        
        # Create LiDAR-image overlay
        overlay_image = create_lidar_image_overlay(data, data_root)
        if overlay_image is not None:
            overlay_path = os.path.join(frame_folder, 'lidar_overlay.jpg')
            cv2.imwrite(overlay_path, overlay_image)
            print(f"LiDAR overlay saved to: {overlay_path}")
    
    return data

def main():
    # Path to mini_labels directory
    labels_dir = "/perception/users/cfx/WorkSpace_2025/nova_fastbev/fastbev/mini_labels"
    data_root = "/perception/users/cfx/WorkSpace_2025/nova_fastbev/fastbev"
    
    if not os.path.exists(labels_dir):
        print(f"Labels directory not found: {labels_dir}")
        return
    
    # Get all JSON files
    json_files = glob.glob(os.path.join(labels_dir, "*.json"))
    
    if not json_files:
        print(f"No JSON files found in {labels_dir}")
        return
    
    print(f"Found {len(json_files)} JSON files")
    
    # Process each JSON file
    for json_file in json_files:
        try:
            process_json_file(json_file, data_root)
        except Exception as e:
            print(f"Error processing {json_file}: {e}")
            continue

if __name__ == "__main__":
    main()
