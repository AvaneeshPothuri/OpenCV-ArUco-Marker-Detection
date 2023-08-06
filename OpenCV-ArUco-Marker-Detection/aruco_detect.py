import cv2

def detect_and_mark_markers(image_path):
    # Load the image
    image = cv2.imread(image_path)

    # Define the ArUco dictionaries for 4X4, 5X5, 6X6, and 7X7 markers
    aruco_dict_4x4 = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)
    aruco_dict_5x5 = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_250)
    aruco_dict_6x6 = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
    aruco_dict_7x7 = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_7X7_250)

    # Create the detector parameters object
    parameters = cv2.aruco.DetectorParameters()

    # Detect markers for each dictionary size
    corners, ids, _ = cv2.aruco.detectMarkers(image, aruco_dict_4x4, parameters=parameters)
    draw_markers(image, corners, ids, color=(0, 0, 255))

    corners, ids, _ = cv2.aruco.detectMarkers(image, aruco_dict_5x5, parameters=parameters)
    draw_markers(image, corners, ids, color=(0, 255, 0))

    corners, ids, _ = cv2.aruco.detectMarkers(image, aruco_dict_6x6, parameters=parameters)
    draw_markers(image, corners, ids, color=(255, 0, 0))

    corners, ids, _ = cv2.aruco.detectMarkers(image, aruco_dict_7x7, parameters=parameters)
    draw_markers(image, corners, ids, color=(255, 255, 0))

    # Step 5: Resize the image to fit the screen or make it larger if it's smaller
    max_display_size = 800
    height, width = image.shape[:2]
    scale_factor = max_display_size / max(height, width)
    if height > max_display_size or width > max_display_size:
        image = cv2.resize(image, (int(width * scale_factor), int(height * scale_factor)))
    else:
        image = cv2.resize(image, (int(width * scale_factor), int(height * scale_factor)))

    # Step 6: Display the resulting image with detected markers and IDs
    cv2.imshow("ArUco Marker Detection Result", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def draw_markers(image, corners, ids, color):
    if ids is not None:
        for i in range(len(ids)):
            if len(corners) > i:
                # Reshape corners to match the format expected by drawDetectedMarkers
                current_corners = [corners[i]]
                cv2.aruco.drawDetectedMarkers(image, current_corners, ids[i], borderColor=color)
                org = (int(corners[i][0][0][0]), int(corners[i][0][0][1]))
                cv2.putText(image, str(ids[i][0]), org, cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)


if __name__ == "__main__":
    sample_image_path = 'images/img2.ppm'  
    detect_and_mark_markers(sample_image_path)
