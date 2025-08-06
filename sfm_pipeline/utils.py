import numpy as np
import cv2

def serialize_keypoints(keypoints):
    """Serialize keypoints for pickle storage."""
    return [(kp.pt, kp.size, kp.angle, kp.response, kp.octave, kp.class_id) for kp in keypoints]

def deserialize_keypoints(serialized_kp):
    """Deserialize keypoints from pickle storage."""
    return [cv2.KeyPoint(pt[0], pt[1], size, angle, response, octave, class_id) 
            for pt, size, angle, response, octave, class_id in serialized_kp]

def serialize_matches(matches):
    """Serialize matches for pickle storage."""
    return [(m.queryIdx, m.trainIdx, m.imgIdx, m.distance) for m in matches]

def deserialize_matches(serialized_matches):
    """Deserialize matches from pickle storage."""
    return [cv2.DMatch(queryIdx, trainIdx, imgIdx, distance) 
            for queryIdx, trainIdx, imgIdx, distance in serialized_matches]

def get_aligned_matches(kp1, kp2, matches):
    """Get aligned point coordinates from keypoints and matches."""
    img1_indices = np.array([m.queryIdx for m in matches])
    img2_indices = np.array([m.trainIdx for m in matches])
    
    pts1 = np.array([kp1[i].pt for i in img1_indices])
    pts2 = np.array([kp2[i].pt for i in img2_indices])
    
    return pts1, pts2

def points_to_ply(points, colors, filename):
    """Save 3D points with colors to PLY format."""
    with open(filename, 'w') as f:
        f.write('ply\n')
        f.write('format ascii 1.0\n')
        f.write(f'element vertex {len(points)}\n')
        f.write('property float x\n')
        f.write('property float y\n')
        f.write('property float z\n')
        f.write('property uchar red\n')
        f.write('property uchar green\n')
        f.write('property uchar blue\n')
        f.write('end_header\n')
        
        for pt, color in zip(points, colors):
            f.write(f'{pt[0]} {pt[1]} {pt[2]} {int(color[0])} {int(color[1])} {int(color[2])}\n')