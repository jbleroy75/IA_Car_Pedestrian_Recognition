import cv2

# Our Images
video = cv2.VideoCapture('video.mp4')

# Our Pre-trained car and pedestrian  classifier
car_tracker_file = 'car_detector.xml'
pedestrian_tracker_file = 'pedestrian_detector.xml'


# Create car classifier
car_tracker = cv2.CascadeClassifier(car_tracker_file)
pedestrian_tracker = cv2.CascadeClassifier(pedestrian_tracker_file)


# Run forever until car stops or something or crashes.
while True:

    # Read the current frame
    (read_successful, frame) = video.read()

    # Safe coding
    if read_successful:
        # Must Convert to grayscale
        grayscaled_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        break

    # Detect cars and pedestrian
    cars = car_tracker.detectMultiScale(grayscaled_frame)
    pedestrians = pedestrian_tracker.detectMultiScale(grayscaled_frame)
    print(cars, pedestrians)  # COORDONNES VOITURE

    # Draw rectangles around the cars
    for (x, y, w, h) in cars:
        cv2.rectangle(frame, (x+1, y+2), (x+w, y+h), (255, 0, 0), 2)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)

    # Draw rectangles around the pedestrians
    for (x, y, w, h) in pedestrians:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (235, 255, 255), 2)

    # Display the image with the faces spottedcle
    cv2.imshow('JB py', frame)

    # Don't autoclose (Wait here and the code and listen for a key press)
    key = cv2.waitKey(1)

    # Stop if Q key is pressed
    if key == 81 or key == 113:
        break

# Release the VideoCapture object
video.release

print("Code Completed")
