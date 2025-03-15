import cv2
import mediapipe as mp
import numpy as np
import math
import multiprocessing as mutp

class HandTrackingDynamic:
    """
    A class to perform hand tracking and dynamic gesture recognition using MediaPipe.
    """
    def __init__(self, mode=False, maxHands=2, detectionCon=0.5, trackCon=0.5, draw=True):
        """
        Initializes the hand tracking model with the given parameters.

        Parameters:
        mode (bool): Whether to treat the input images as a batch of static images.
        maxHands (int): Maximum number of hands to detect.
        detectionCon (float): Minimum confidence value for hand detection.
        trackCon (float): Minimum confidence value for hand tracking.
        """
        self.__mode__ = mode
        self.__maxHands__ = maxHands
        self.__detectionCon__ = detectionCon
        self.__trackCon__ = trackCon
        self.handsMp = mp.solutions.hands
        self.hands = self.handsMp.Hands(static_image_mode=self.__mode__,
                                        max_num_hands=self.__maxHands__,
                                        min_detection_confidence=self.__detectionCon__,
                                        min_tracking_confidence=self.__trackCon__)
        self.mpDraw = mp.solutions.drawing_utils
        self.tipIds = [4, 8, 12, 16, 20]
        self.draw = draw

    def findFingers(self, frame):
        """
        Processes the frame to find hand landmarks and optionally draws them.

        Parameters:
        frame (numpy.ndarray): The input image frame.
        draw (bool): Whether to draw the hand landmarks on the frame.

        Returns:
        numpy.ndarray: The frame with hand landmarks drawn (if draw=True).
        """
        imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert the image to RGB
        self.results = self.hands.process(imgRGB)  # Process the image to find hand landmarks
        #if self.results.multi_hand_landmarks:
            #for handLms in self.results.multi_hand_landmarks:
                #if self.draw:
                #    self.mpDraw.draw_landmarks(frame, handLms, self.handsMp.HAND_CONNECTIONS)  # Draw landmarks
        return frame

    def findHand(self, frame):
        """
        Finds the position of hand landmarks and returns them along with the bounding box.

        Parameters:
        frame (numpy.ndarray): The input image frame.
        handNo (int): The index of the hand to find.
        draw (bool): Whether to draw the landmarks and bounding box on the frame.

        Returns:
        list: A list of hand landmarks.
        tuple: The bounding box coordinates (xmin, ymin, xmax, ymax).
        """
        result = []
        if self.results.multi_hand_landmarks:
            for hid, myHand in enumerate(self.results.multi_hand_landmarks):
                xList = []
                yList = []
                lmsList = []
                bbox = None
                handedness = self.results.multi_handedness[hid].classification[0].label  # Get handedness (left/right hand)
                for id, lm in enumerate(myHand.landmark):
                    height, width, c = frame.shape
                    cx, cy = int(lm.x * width), int(lm.y * height)  # Convert normalized coordinates to pixel values
                    xList.append(cx)
                    yList.append(cy)
                    lmsList.append([id, cx, cy])
                    if self.draw:
                        cv2.circle(frame, (cx, cy), 2, (255, 0, 255), cv2.FILLED)  # Draw circles on landmarks
                        cv2.putText(frame, str(id), (cx+10, cy-10), cv2.FONT_HERSHEY_PLAIN, 0.5, (0, 0, 255))  # Label landmarks

                xmin, xmax, ymin, ymax = min(xList), max(xList), min(yList), max(yList)  # Get bounding box coordinates
                bbox = xmin, ymin, xmax, ymax
                if self.draw:
                    cv2.rectangle(frame, (xmin - 20, ymin - 20), (xmax + 20, ymax + 20), (0, 255, 0), 2)  # Draw bounding box
                result.append([lmsList, bbox, handedness])
        return result

    def touchFingers(self, lmsList, bbox, handedness, frame):
        """
        Draws the dynamic gesture based on the distance between thumb and index finger.

        Parameters:
        lmsList (list): List of hand landmarks.
        bbox (tuple): Bounding box coordinates (xmin, ymin, xmax, ymax).
        frame (numpy.ndarray): The input image frame.

        Returns:
        bool: True if the thumb and index finger are touching, False otherwise.
        """
        # Create a dynamic threshold based on the area ratio
        threshold = ((((bbox[2] - bbox[0]) * (bbox[3] - bbox[1]))*0.0001)**1.472)+12  # Adjust the multiplication as needed
        
        # Get the coordinates of the thumb (point 4) and index finger (point 8)
        thumb_x, thumb_y = lmsList[4][1], lmsList[4][2]
        index_x, index_y = lmsList[8][1], lmsList[8][2]
        
        # Calculate the Euclidean distance between the thumb and index finger points
        distance = math.sqrt((index_x - thumb_x)**2 + (index_y - thumb_y)**2)

        #print('Distance: {}'.format(distance))
        
        # Check if the distance is less than the dynamic threshold
        if distance < threshold:
            # If the distance is less than the threshold, draw the line or perform some action
            print('touchFingers finded in {} hand'.format(handedness))
            return True
        else:
            # If the distance is greater than the threshold, do nothing
            return False
            
class HandTrackingDraw:
    """
    A class to perform hand tracking and dynamic gesture recognition using MediaPipe.
    """
    def __init__(self):
        """
        Initializes the hand tracking model with the given parameters.
        """
        self.drawpoints = [[],[]]
        self.posEraser = [50,50,100,100]
        self.drawNum = [0,0]
        self.movingEraser = False
        self.drawHand = None

    def draw(self, lmsList, contactFingers):
        """
        Draws the dynamic gesture based on the distance between thumb and index finger.

        Parameters:
        lmsList (list): List of hand landmarks.
        contactFingers (bool): Whether the thumb and index finger are touching.
        """
        if len(lmsList) != 0 and not self.movingEraser:
            pos_x = int((lmsList[4][2] + lmsList[8][2]) / 2)
            pos_y = int((lmsList[4][3] + lmsList[8][3]) / 2)
            if lmsList[0][0] == "Left":
                if self.drawpoints[0] == []:
                    self.drawpoints[0].append([])
                if contactFingers:
                    self.drawpoints[0][self.drawNum[0]].append([pos_x, pos_y])
                elif self.drawpoints[0][self.drawNum[0]] != []:
                    self.drawNum[0] += 1
                    self.drawpoints[0].append([])
            elif lmsList[0][0] == "Right":
                if self.drawpoints[1] == []:
                    self.drawpoints[1].append([])
                if contactFingers:
                    self.drawpoints[1][self.drawNum[1]].append([pos_x, pos_y])
                elif self.drawpoints[1][self.drawNum[1]] != []:
                    self.drawNum[1] += 1
                    self.drawpoints[1].append([])
            
    def showDrawPoints(self, frame):
        """
        Draws the points collected for the dynamic gesture on the frame.

        Parameters:
        frame (numpy.ndarray): The input image frame.
        """
        if self.drawpoints != []:
            for hand in self.drawpoints:
                for points in hand:
                    points = np.array(points, np.int32)
                    points = points.reshape((-1, 1, 2))
                    cv2.polylines(frame, [points], isClosed=False, color=(255, 0, 255), thickness=3)

    def showEraser(self, frame):
        """
        Draws the eraser on the frame.

        Parameters:
        frame (numpy.ndarray): The input image frame.
        """
        cv2.rectangle(frame,(self.posEraser[0],self.posEraser[1]),(self.posEraser[2],self.posEraser[3]), (0, 255, 255), cv2.FILLED)

    def moveEraser(self, lmsList, contactFingers):
        """
        Moves the eraser based on the position of the thumb and index finger.

        Parameters:
        lmsList (list): List of hand landmarks.
        contactFingers (bool): Whether the thumb and index finger are touching.
        """
        if len(lmsList) != 0:
            pos_x = int((lmsList[4][2] + lmsList[8][2]) / 2)
            pos_y = int((lmsList[4][3] + lmsList[8][3]) / 2)
            insEraser = pos_x >= self.posEraser[0] and pos_x <= self.posEraser[2] and pos_y >= self.posEraser[1] and pos_y <= self.posEraser[3]
            if contactFingers and insEraser:
                self.movingEraser = True
                self.posEraser = [pos_x-25,pos_y-25, pos_x+25, pos_y+25]
            else:
                self.movingEraser = False
    
    def eraseDraw(self):
        """
        Erases the drawn points if the eraser is moving.
        """
        if self.movingEraser:
            drawpoints = self.drawpoints
            self.drawpoints = []
            for hand in drawpoints:
                for points in hand:
                    if points != []:
                        cpoints = np.array(points)
                        
                        x_min, y_min, x_max, y_max = self.posEraser[0], self.posEraser[1], self.posEraser[2], self.posEraser[3]

                        insEraser = cpoints[
                            (cpoints[:, 0] < x_min) | (cpoints[:, 0] > x_max) | (cpoints[:, 1] < y_min) | (cpoints[:, 1] > y_max)
                        ]
                        insEraser = insEraser.tolist()

                        self.drawpoints.append(insEraser)
                    else:
                        self.drawpoints.append([])

def main():
    """
    The main function to run the hand tracking and dynamic gesture recognition.
    """
    cap = cv2.VideoCapture(0)  # Open the webcam
    detector = HandTrackingDynamic()  # Initialize the hand tracking detector
    draw = HandTrackingDraw()  # Initialize the drawing class
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # Set the width of the frame
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)  # Set the height of the frame
    if not cap.isOpened():
        print("Cannot open camera")
        exit()

    while True:
        ret, frame = cap.read()  # Read a frame from the webcam
        if not ret:
            break

        frame = cv2.flip(frame, 1)  # Flip the frame horizontally
        frame = detector.findFingers(frame)  # Find and draw hand landmarks
        handDetails = detector.findHand(frame) # Get hand landmarks and bounding box
        for hand in handDetails:
            lmsList, bbox, handedness = hand
            if lmsList != []:
                cfingers = detector.touchFingers(*hand, frame=frame)  # Check if thumb and index finger are touching
            #     draw.moveEraser(lmsList, cfingers)  # Move the eraser if necessary
            #     draw.eraseDraw() # Erase the drawn points if the eraser is moving
            #     draw.draw(lmsList, cfingers)  # Draw the gesture
        # draw.showEraser(frame)  # Show the eraser on the frame
        # draw.showDrawPoints(frame)  # Show the drawn points on the frame
        cv2.imshow('frame', frame)  # Display the frame
        if cv2.waitKey(1) == 27:  # Exit if the 'Esc' key is pressed
            break

    cap.release()  # Release the webcam
    cv2.destroyAllWindows()  # Close all OpenCV windows

if __name__ == "__main__":
    main()