import cv2

# Constants for maximum display size, font scale, and text thickness
MAX_DISPLAY_SIZE = 1500
FONT_SCALE = 1.0
TEXT_THICKNESS = 2

# Define marker colors
MARKER_COLORS = {
    cv2.aruco.DICT_4X4_250: (0, 0, 255),
    cv2.aruco.DICT_5X5_250: (0, 255, 0),
    cv2.aruco.DICT_6X6_250: (255, 0, 0),
    cv2.aruco.DICT_7X7_250: (255, 255, 0)
}

def detect_and_mark_markers(image_path):
    # Load the image
    image = cv2.imread(image_path)

    # Define a list of ArUco dictionaries for different marker sizes
    aruco_dicts = [
        cv2.aruco.DICT_4X4_250,
        cv2.aruco.DICT_5X5_250,
        cv2.aruco.DICT_6X6_250,
        cv2.aruco.DICT_7X7_250
    ]

    # Create the detector parameters object
    parameters = cv2.aruco.DetectorParameters()

    # Loop through the dictionaries and detect markers
    for aruco_dict in aruco_dicts:
        detect_and_draw(image, aruco_dict, parameters)

    # Resize the image and display the result
    display_image(image)

def detect_and_draw(image, aruco_dict, parameters):
    # Get the dictionary object based on the constant
    dictionary = cv2.aruco.getPredefinedDictionary(aruco_dict)
    corners, ids, _ = cv2.aruco.detectMarkers(image, dictionary, parameters=parameters)
    draw_markers(image, corners, ids, MARKER_COLORS[aruco_dict])

def draw_markers(image, corners, ids, color):
    if ids is not None:
        for i in range(len(ids)):
            if len(corners) > i:
                current_corners = [corners[i]]
                cv2.aruco.drawDetectedMarkers(image, current_corners, ids[i], borderColor=color)
                org = (int(corners[i][0][0][0]), int(corners[i][0][0][1]))
                cv2.putText(image, str(ids[i][0]), org, cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, (0, 0, 0), TEXT_THICKNESS)

def display_image(image):
    # Resize the image to fit the screen or make it larger if needed
    height, width = image.shape[:2]
    scale_factor = MAX_DISPLAY_SIZE / max(height, width)
    if height > MAX_DISPLAY_SIZE or width > MAX_DISPLAY_SIZE:
        image = cv2.resize(image, None, fx=scale_factor, fy=scale_factor)
    else:
        image = cv2.resize(image, None, fx=scale_factor, fy=scale_factor)

    # Display the resulting image with detected markers and IDs
    cv2.imshow("ArUco Marker Detection Result", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # List of sample image paths
    sample_image_paths = ['images/img1.png', 'images/img2.ppm', 'images/img3.jpg','images/img4.png','images/img5.png']
    
    # Process and display each sample image
    for image_path in sample_image_paths:
        detect_and_mark_markers(image_path)