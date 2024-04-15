import cv2
import numpy as np

def convert_to_ycbcr(frame):
    # Convert the frame to YCbCr color space
    ycbcr_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
    return ycbcr_frame

def gaussian_skin_color_classification(ycbcr_frame):
    # Define skin color range in YCbCr space
    lower_skin = np.array([0, 133, 77], dtype=np.uint8)
    upper_skin = np.array([255, 173, 127], dtype=np.uint8)

    # Create a binary mask for skin color
    skin_mask = cv2.inRange(ycbcr_frame, lower_skin, upper_skin)
    return skin_mask

def convert_binary_to_blocky(binary_frame, block_size):
    # Resize the binary frame to have a lower resolution
    binary_frame_resized = cv2.resize(binary_frame, (0, 0), fx=1/block_size, fy=1/block_size)

    # Upsample the binary frame to the original size
    blocky_frame = cv2.resize(binary_frame_resized, (binary_frame.shape[1], binary_frame.shape[0]))

    return blocky_frame

def block_to_candidate_regions(block_image, min_region_size):
    # Perform connected component analysis
    _, labels, stats, _ = cv2.connectedComponentsWithStats(block_image)

    # Filter candidate regions based on size
    candidate_regions = []

    for i in range(1, stats.shape[0]):  # Skip the background (label 0)
        region_size = stats[i, cv2.CC_STAT_AREA]

        if region_size >= min_region_size:
            candidate_regions.append(stats[i])

    return candidate_regions

def select_best_face_region(candidate_regions):
    # Sort candidate regions by area in descending order
    sorted_regions = sorted(candidate_regions, key=lambda x: x[4], reverse=True)

    # Return the region with the largest area (best face region)
    return sorted_regions[0] if sorted_regions else None

def draw_face_region(frame, face_region):
    if face_region is not None:
        x, y, w, h = face_region[0], face_region[1], face_region[2], face_region[3]
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Draw a green rectangle

def main():
    # Open a connection to the camera (you can change the parameter to 0 for the default camera)
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Couldn't open the camera.")
        return

    # Set the block size for blocky image
    block_size = 20
    min_region_size = 20

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        if not ret:
            print("Error: Couldn't read a frame.")
            break

        # Step 1: Convert to YCbCr color space
        ycbcr_frame = convert_to_ycbcr(frame)
        cv2.imshow('YCbCr Frame', ycbcr_frame)

        # Step 2: Gaussian skin color classification
        binary_frame = gaussian_skin_color_classification(ycbcr_frame)
        cv2.imshow('Binary Frame', binary_frame)

        # Step 3: Convert binary image to blocky image
        blocky_frame = convert_binary_to_blocky(binary_frame, block_size)
        cv2.imshow('Blocky Frame', blocky_frame)

        # Step 4: Block to candidate regions
        candidate_regions = block_to_candidate_regions(blocky_frame, min_region_size)

        # Step 5: Select the best face region
        best_face_region = select_best_face_region(candidate_regions)

        # Step 6: Draw the face region on the original frame
        draw_face_region(frame, best_face_region)

        # Display the original frame with the face region
        cv2.imshow('Original Frame with Face Region', frame)

        # Break the loop if 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the camera and close all windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
