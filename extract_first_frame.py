import cv2

# Path to the video file
video_path = '/home/robotics/cogvideox-factory/cogx-dataset/videos/19.mp4'

# Path to save the extracted image
image_path = 'img19.jpg'

# Open the video file
cap = cv2.VideoCapture(video_path)

# Read the first frame
ret, frame = cap.read()

# Check if the frame was successfully read
if ret:
    # Save the frame as an image
    cv2.imwrite(image_path, frame)
    print(f"First frame extracted and saved as {image_path}")
else:
    print("Failed to read the video file or extract the first frame.")

# Release the video capture object
cap.release()