import os
import cv2
import numpy as np

def abstract_frame_with_templates(frame):
    
    # if not hasattr(abstract_frame_with_templates, "counter"):
    #     abstract_frame_with_templates.counter = 0  # Initialize it for the first call
    
    
    # Dictionary to hold the loaded template images and their corresponding colors
    script_dir = os.path.dirname(__file__)
    

    # Dictionary to hold the loaded template images
   
    templates = {
    'player_back': (cv2.imread(os.path.join(script_dir, "player_back_dark.png"), 0), (0, 0, 255)),  # Red color for player_back
    'player_left': (cv2.imread(os.path.join(script_dir, "player_left_dark.png"), 0), (0, 0, 255)),   # Green color for player_left
    'player_right': (cv2.imread(os.path.join(script_dir, "player_right_dark.png"), 0), (0, 0, 255)),  # Blue color for player_right
    'player_front': (cv2.imread(os.path.join(script_dir, "player_front_dark.png"), 0), (0, 0, 255)),
    'player_back_white': (cv2.imread(os.path.join(script_dir, "player_back_white.png"), 0), (0, 0, 255)),  # Red color for player_back
    'player_left_white': (cv2.imread(os.path.join(script_dir, "player_left_white.png"), 0), (0, 0, 255)),   # Green color for player_left
    'player_right_white': (cv2.imread(os.path.join(script_dir, "player_right_white.png"), 0), (0, 0, 255)),  # Blue color for player_right
    'player_front_white': (cv2.imread(os.path.join(script_dir, "player_front_white.png"), 0), (0, 0, 255)),# Cyan color for player_front
    'doc': (cv2.imread(os.path.join(script_dir, "doc.png"), 0), (114, 255, 0)),  # Cyan color for player_front
    'grass': (cv2.imread(os.path.join(script_dir, "grass.png"), 0), (0, 255, 255)),  # Yellow color for grass
    # 'house1': (cv2.imread(os.path.join(script_dir, "house1.png"), 0), (255, 0, 255)),  # Magenta color for house1
    # 'house2': (cv2.imread(os.path.join(script_dir, "house2.png"), 0), (255, 255, 255)),  # White color for house2
     'fence': (cv2.imread(os.path.join(script_dir, "fence.png"), 0), (128, 128, 128)),  # Gray color for fence
     'door1': (cv2.imread(os.path.join(script_dir, "door1.png"), 0), (0, 128, 128)),  # Teal color for door1
     'door2': (cv2.imread(os.path.join(script_dir, "door2.png"), 0), (128, 0, 128)),  # Purple color for door2
    # 'door3': (cv2.imread(os.path.join(script_dir, "door3.png"), 0), (128, 128, 0))  # Olive color for door3
}


    # Convert frame to grayscale
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # show converted frame
    # cv2.imshow('Converted Frame', frame_gray)  
    # cv2.waitKey(1)
    # Initialize a blank canvas for drawing abstract representations
    abstract_frame = np.zeros_like(frame)

    for name, (template, color) in templates.items():
       
        # Perform template matching
        res = cv2.matchTemplate(frame_gray, template, cv2.TM_CCOEFF_NORMED)
        threshold = 0.8
        loc = np.where(res >= threshold)

        # Draw rectangles on the abstract_frame for each detected template with its assigned color
        for pt in zip(*loc[::-1]):  # loc[::-1] to switch from (row, col) to (x, y)
            cv2.rectangle(abstract_frame, pt, (pt[0] + template.shape[1], pt[1] + template.shape[0]), color, -1)


    # cv2.imshow('Overlay Frame', abstract_frame)  # Display the overlay frame
    # cv2.waitKey(1)
    
    return abstract_frame
    
    # # Increment the counter
    # abstract_frame_with_templates.counter += 1
    
    # # Check if it's time to save the frame
    # if abstract_frame_with_templates.counter % 10 == 0:
    #     save_dir = os.path.join(script_dir, "saved_frames")  # Define the directory to save frames
    #     if not os.path.exists(save_dir):
    #         os.makedirs(save_dir)  # Create the directory if it doesn't exist
    #     save_path = os.path.join(save_dir, f"frame_{abstract_frame_with_templates.counter}.png")
    #     cv2.imwrite(save_path, frame_gray)  # Save the current grayscale frame
    #     print(f"Saved frame to {save_path}")  # Optional: Print confirmation

    


# Example usage
if __name__ == "__main__":
    script_dir = os.path.dirname(__file__)
    frame = cv2.imread(os.path.join(script_dir,'pokemon2.png'))  # Load a frame
    abstract_frame = abstract_frame_with_templates(frame)
    
    
    # Display the result
    cv2.imshow('Abstract Frame', abstract_frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
