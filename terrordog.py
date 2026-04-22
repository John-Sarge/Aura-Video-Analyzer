import cv2
import numpy as np
import pyrealsense2 as rs
import mediapipe as mp
import trimesh
import pyrender
import math
import time

class AuraAnalyzer:
    def __init__(self, obj_path):
        # 1. Initialize RealSense D435i
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
        config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
        self.pipeline.start(config)
        
        # Align Depth to Color to ensure coordinates match perfectly
        self.align = rs.align(rs.stream.color)

        # Pre-load the AVA background image for when detection is lost
        self.bg_img = cv2.imread('AVA_BACKGROUND.png', cv2.IMREAD_UNCHANGED)
        if self.bg_img is not None:
            # Resize it strictly to the 720p output window size
            self.bg_img = cv2.resize(self.bg_img, (1280, 720))

        # 2. Initialize MediaPipe Face Mesh for Head Tracking
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1, 
            refine_landmarks=True, 
            min_detection_confidence=0.5, 
            min_tracking_confidence=0.5
        )

        # 3. Initialize 3D Renderer (Pyrender)
        self.scene = pyrender.Scene(bg_color=[0, 0, 0, 255]) # Black background
        
        # Load the Terror Dog model
        sm = trimesh.load(obj_path)
        
        # FIX: Center and Auto-Scale the 3D model so it's guaranteed to be visible
        # If the OBJ was exported in very large or small units (e.g., mm vs meters), we won't see it without this.
        center = sm.bounds.mean(axis=0)
        sm.apply_translation(-center)
        max_extent = np.max(sm.extents)
        if max_extent > 0:
            sm.apply_scale(4.0 / max_extent) # Scale to a visible size of ~4 units

        # FIX: The user reported seeing the bottom of the model with the chin pointing left.
        # This resolves the 3D model orientation by mapping coordinates.
        # 1. Rotate 90 degrees around X-axis (tips the model up from its belly)
        rot_x = trimesh.transformations.rotation_matrix(math.radians(-90), [1, 0, 0])
        sm.apply_transform(rot_x)
        # 2. Rotate 90 degrees around Y-axis (turns the model horizontally so chin faces forward instead of left)
        rot_y = trimesh.transformations.rotation_matrix(math.radians(-90), [0, 1, 0])
        sm.apply_transform(rot_y)
        # 3. Rotate 180 degrees around Z-axis (flips the model upright since it was upside down)
        rot_z = trimesh.transformations.rotation_matrix(math.radians(180), [0, 0, 1])
        sm.apply_transform(rot_z)
        
        # FIX: The "Aura/Heat Signature" Visual Overhaul
        # We compute the surface direction (normals) of every polygon on the Terror Dog.
        # By mapping these 3D directions into Red, Green, and Blue colors, we create 
        # that flat, blotchy, retro "Matcap" / thermal look seen in the Ghostbusters film.
        normals = np.abs(sm.vertex_normals)
        colors_float = np.zeros((len(normals), 4), dtype=np.float32)
        
        # NEW COLOR MAPPING: Ghost1.png is mostly deep black with distinct patches of blue and green around the edges.
        # To get the black front face and colored outlines, we map the colors to the "grazing angles" (surfaces curving away from the camera).
        
        # Red is virtually non-existent, just a tiny bit on top/bottom ridges
        colors_float[:, 0] = np.power(normals[:, 1], 6) * 40.0   
        
        # Green forms the "outline". (1.0 - normal_Z) means faces pointing AT the camera are 0.0 (Black).
        # Faces curving away from the camera light up brightly! Raising to power 4 sharpens this outline band.
        colors_float[:, 1] = np.power(1.0 - normals[:, 2], 4) * 220.0  
        
        # Blue prominently highlights the sides and the horns (high X normal)
        colors_float[:, 2] = np.power(normals[:, 0], 3) * 255.0  
        colors_float[:, 3] = 255.0                               # Alpha (Fully opaque)
        
        # Add a little low-fi analog noise
        noise = np.random.uniform(-15, 20, size=colors_float[:, :3].shape)
        colors_float[:, :3] += noise
        
        # Subtract a heavier baseline to violently crush the entire front face and noise pixels down into pure black
        colors_float[:, :3] -= 35.0
        
        # Clip to valid 0-255 and apply to mesh
        sm.visual.vertex_colors = np.clip(colors_float, 0, 255).astype(np.uint8)

        # Calling from_trimesh(sm) dynamically generates the correct material WITH our vertex colors included.
        mesh = pyrender.Mesh.from_trimesh(sm, smooth=False)
        self.dog_node = self.scene.add(mesh)

        # Set ambient light back down so it doesn't artificially brighten the dark areas
        self.scene.ambient_light = [0.8, 0.8, 0.8]

        # Scaled Intrinsics for 720p (1280x720) instead of 480p width/height
        self.camera = pyrender.IntrinsicsCamera(fx=1000, fy=1000, cx=640, cy=360)
        self.cam_node = self.scene.add(self.camera, pose=np.eye(4))
        self.renderer = pyrender.OffscreenRenderer(1280, 720)

    def _translate_pose(self, x, y, z):
        pose = np.eye(4)
        pose[0,3] = x; pose[1,3] = y; pose[2,3] = z
        return pose

    def detect_colander_depth(self, depth_frame, face_landmarks, frame_shape):
        """
        Uses REAL depth distances to detect the colander hat instead of unreliable color heuristics.
        Checks if there is a physical object covering the region above the forehead.
        """
        img_h, img_w = frame_shape[:2]
        
        # Get face coordinates to dynamically scale our search region (solves "have to be pretty close" issue)
        forehead_x = int(np.clip(face_landmarks.landmark[10].x * img_w, 0, img_w - 1))
        forehead_y = int(np.clip(face_landmarks.landmark[10].y * img_h, 0, img_h - 1))
        
        chin_y = int(np.clip(face_landmarks.landmark[152].y * img_h, 0, img_h - 1))
        face_height = max(20, chin_y - forehead_y)  # face height in pixels
        
        left_cheek_x = int(np.clip(face_landmarks.landmark[234].x * img_w, 0, img_w - 1))
        right_cheek_x = int(np.clip(face_landmarks.landmark[454].x * img_w, 0, img_w - 1))
        face_width = max(20, right_cheek_x - left_cheek_x)  # face width in pixels

        # 1. Measure the exact physical distance to the user's face (in meters)
        # Using the nose instead of forehead for a much more stable depth reference without hair/hat brim interference
        nose_x = int(np.clip(face_landmarks.landmark[1].x * img_w, 0, img_w - 1))
        nose_y = int(np.clip(face_landmarks.landmark[1].y * img_h, 0, img_h - 1))
        face_depth = depth_frame.get_distance(nose_x, nose_y)
        
        if face_depth <= 0.1: # if invalid or too close, abort
            return False

        # 2. Define the Region of Interest (ROI) dynamically scaled to the user's apparent size
        # The colander sits above the forehead, roughly 10% to 60% of the face height up
        roi_top = max(0, forehead_y - int(face_height * 0.6))
        roi_bottom = max(0, forehead_y - int(face_height * 0.1)) 
        roi_left = max(0, forehead_x - int(face_width * 0.5))
        roi_right = min(img_w, forehead_x + int(face_width * 0.5))
        
        if roi_bottom <= roi_top or roi_right <= roi_left:
            return False
            
        # 3. Gather depth samples across the region where the hat should be
        # Stepping dynamically based on face size so we don't over/undersample at different distances
        step = max(2, int(face_width * 0.1))
        depth_samples = []
        invalid_samples = 0
        total_samples = 0
        
        for y in range(roi_top, roi_bottom, step):
            for x in range(roi_left, roi_right, step):
                total_samples += 1
                d = depth_frame.get_distance(x, y)
                # D435i struggles with shiny metal due to IR light scattering, frequently returning 0 (invalid depth).
                if d > 0:
                    depth_samples.append(d)
                else:
                    invalid_samples += 1
                    
        # Feature: If the area above the head is mostly returning "invalid" (0m) depth, 
        # it is almost definitively the shiny curved metal of the colander scattering the RealSense projector!
        if total_samples > 0 and (invalid_samples / total_samples) > 0.5:
            return True
            
        if not depth_samples:
            return False
            
        # Median depth of the objects sitting in the hat area
        hat_area_depth = np.median(depth_samples)
        
        # If the median object above the head is within 30cm of the nose depth, it's the colander!
        # Increased tolerance to 0.30m (30cm) to reduce flickering and account for the hat's shape
        depth_difference = abs(face_depth - hat_area_depth)
        
        return depth_difference < 0.30

    def get_head_pose(self, face_landmarks):
        """
        Convert Mediapipe landmarks to a 4x4 transformation matrix for Pyrender.
        """
        # Simplified orientation using nose (1), and sides of face (234, 454)
        nose = face_landmarks.landmark[1]
        left = face_landmarks.landmark[234]
        right = face_landmarks.landmark[454]
        
        # Calculate Mock Rotation (Yaw, Pitch, Roll)
        yaw = math.atan2((right.z - left.z), (right.x - left.x))
        # FIX: Invert the pitch so looking up/down matches the model's movement correctly
        pitch = -math.atan2(nose.y - left.y, nose.z - left.z) 
        
        pose = np.eye(4)
        
        # Translate to center of face but pushed back in Z space
        pose[0, 3] = (nose.x - 0.5) * 2.0 
        pose[1, 3] = -(nose.y - 0.5) * 2.0
        pose[2, 3] = -3.0 # Distance from camera
        
        # Applying rotation (Very basic Euler to Matrix)
        cy = math.cos(yaw); sy = math.sin(yaw)
        cp = math.cos(pitch); sp = math.sin(pitch)
        
        pose[0,0] = cy; pose[0,2] = sy
        pose[1,1] = cp; pose[1,2] = -sp
        pose[2,0] = -sy; pose[2,2] = cy * cp
        return pose

    def run(self):
        # Allow the OpenCV window to be resizable by the user
        cv2.namedWindow('Aura Video-Analyzer version 3.1', cv2.WINDOW_NORMAL)
        
        try:
            # Temporal smoothing for colander detection to prevent 1-frame flickers
            colander_streak = 0
            
            # Tracking for lost connection/interference static
            last_valid_pose = None
            last_seen_time = 0
            
            while True:
                frames = self.pipeline.wait_for_frames()
                
                # Align depth frames exactly with color frames
                aligned_frames = self.align.process(frames)
                color_frame = aligned_frames.get_color_frame()
                depth_frame = aligned_frames.get_depth_frame()
                
                if not color_frame or not depth_frame:
                    continue

                frame = np.asanyarray(color_frame.get_data())
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Default output is black (1280x720)
                output_image = np.zeros((720, 1280, 3), dtype=np.uint8)

                # Process face mesh
                results = self.face_mesh.process(rgb_frame)
                
                # Track whether a colander was found in this specific frame anywhere
                found_in_frame = False

                if results.multi_face_landmarks:
                    for face_landmarks in results.multi_face_landmarks:
                        # 1. Check for Colander Hat using DEPTH instead of color matching
                        has_colander = self.detect_colander_depth(depth_frame, face_landmarks, frame.shape)
                        
                        if has_colander:
                            found_in_frame = True
                            # 2. Get Head Pose
                            pose = self.get_head_pose(face_landmarks)
                            last_valid_pose = pose
                            break # Only process one face

                # Update the debounce streak
                if found_in_frame:
                    colander_streak = min(colander_streak + 2, 8) # builds confidence quickly
                    last_seen_time = time.time()
                else:
                    colander_streak = max(colander_streak - 1, 0) # drops confidence slowly
                    
                # RENDER LOGIC
                if colander_streak > 3 and last_valid_pose is not None:
                    # NORMAL TRACKING
                    self.scene.set_pose(self.dog_node, last_valid_pose)
                    color, depth = self.renderer.render(self.scene)
                    
                    output_image = cv2.cvtColor(color, cv2.COLOR_RGB2BGR)
                    output_image = cv2.GaussianBlur(output_image, (15, 15), 0)
                    
                    # Normal Light CRT Static
                    h, w, _ = output_image.shape
                    grain = np.random.randint(-15, 15, size=(h, w, 3), dtype=np.int16)
                    output_image = np.clip(output_image.astype(np.int16) + grain, 0, 255).astype(np.uint8)
                    output_image[::2, :, :] = (output_image[::2, :, :] * 0.5).astype(np.uint8)

                elif last_valid_pose is not None and (time.time() - last_seen_time) <= 5.0:
                    # INTERFERENCE / LOST TRACKING MODE (Less than 5 seconds since last seen)
                    
                    # Randomly jitter the last known pose to simulate "searching" or connection drops
                    jitter_pose = last_valid_pose.copy()
                    jitter_pose[0:3, 3] += np.random.uniform(-0.1, 0.1, 3) # Jitter XYZ translation
                    self.scene.set_pose(self.dog_node, jitter_pose)
                    
                    color, depth = self.renderer.render(self.scene)
                    output_image = cv2.cvtColor(color, cv2.COLOR_RGB2BGR)
                    
                    # Heavy Blur
                    output_image = cv2.GaussianBlur(output_image, (31, 31), 0)
                    
                    h, w, _ = output_image.shape
                    
                    # EXTREME Static Noise (like a lost TV channel)
                    grain = np.random.randint(-150, 150, size=(h, w, 3), dtype=np.int16)
                    
                    # Horizontal tracking tear/glitch effect (randomly shift pixels left or right)
                    if np.random.random() > 0.5:
                        shift = np.random.randint(-150, 150)
                        output_image = np.roll(output_image, shift, axis=1)
                        
                    output_image = np.clip(output_image.astype(np.int16) + grain, 0, 255).astype(np.uint8)
                    output_image[::2, :, :] = (output_image[::2, :, :] * 0.5).astype(np.uint8)
                else:
                    # Totally lost (over 5 seconds). Reset pose and go back to pure black output
                    last_valid_pose = None
                    
                    if self.bg_img is not None:
                        # Extract alpha channel to blend the transparent background image
                        alpha = self.bg_img[:, :, 3] / 255.0
                        
                        # Loop through B, G, R channels and overlay
                        for c in range(3):
                            output_image[:, :, c] = (1.0 - alpha) * output_image[:, :, c] + (alpha * self.bg_img[:, :, c])

                # Overlay a small Picture-in-Picture (PiP) of the raw color camera feed in the bottom right corner
                pip_w, pip_h = 320, 180
                
                # Reverse (flip horizontally) the raw camera feed so it mirrors the user naturally
                mirrored_frame = cv2.flip(frame, 1)
                small_frame = cv2.resize(mirrored_frame, (pip_w, pip_h))
                
                # Add a thin green border around the PiP to match the Ghostbusters aesthetic
                cv2.rectangle(small_frame, (0, 0), (pip_w - 1, pip_h - 1), (0, 255, 0), 2)
                
                # Place it in the bottom right corner of the 720p viewport (before the footer is added)
                output_image[720 - pip_h:720, 1280 - pip_w:1280] = small_frame

                # Add a footer for copyright and info
                footer_height = 25
                footer = np.zeros((footer_height, 1280, 3), dtype=np.uint8)
                cv2.putText(footer, "Copyright (C) 1984 Spengler. Modified by Seargeant, Charm City Ghostbusters, 2026. All rights reserved. | Aura Video-Analyzer v3.1", 
                            (10, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1, cv2.LINE_AA)
                
                # Concatenate the main image and the footer
                output_image = np.vstack((output_image, footer))

                # Show output
                cv2.imshow('Aura Video-Analyzer version 3.1', output_image)

                if cv2.waitKey(1) & 0xFF == 27: # ESC to quit
                    break
        finally:
            self.pipeline.stop()
            cv2.destroyAllWindows()

if __name__ == "__main__":
    analyzer = AuraAnalyzer('terrordog.obj')
    analyzer.run()