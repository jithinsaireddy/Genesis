import numpy as np

class AICamera:
    """
    AI-driven camera controller for cinematic shots.
    Handles 3D camera positioning and movement in virtual space.
    """
    def __init__(self):
        # Initialize camera at origin (0,0,0)
        self.position = np.array([0, 0, 0])
        self.rotation = np.array([0, 0, 0])  # Euler angles (pitch, yaw, roll)
        
    def move_camera(self, direction):
        """
        Move the camera in the specified direction.
        
        Args:
            direction (list): 3D vector representing movement in [x, y, z] directions
            
        Returns:
            np.array: New camera position after movement
        """
        self.position += np.array(direction)
        return self.position
        
    def rotate_camera(self, rotation):
        """
        Rotate the camera based on Euler angles.
        
        Args:
            rotation (list): Rotation angles in [pitch, yaw, roll] format
            
        Returns:
            np.array: New camera rotation angles
        """
        self.rotation += np.array(rotation)
        # Normalize angles to keep them in reasonable ranges
        self.rotation = np.mod(self.rotation + 180, 360) - 180
        return self.rotation

if __name__ == "__main__":
    # Test camera movements
    camera = AICamera()
    print("Initial Position:", camera.position)
    
    # Simulate a tracking shot
    print("Camera Position after tracking shot:", 
          camera.move_camera([1, 2, -3]))  # Move right(1), up(2), backward(-3)
    
    # Simulate a pan and tilt
    print("Camera Rotation after pan/tilt:", 
          camera.rotate_camera([15, 45, 0]))  # Tilt up 15°, pan right 45°
