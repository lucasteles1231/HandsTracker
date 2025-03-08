import cv2
import mediapipe as mp
import numpy as np
import math

class HandTrackingDynamic:
    """
    A class to perform hand tracking and dynamic gesture recognition using MediaPipe.
    """
    def __init__(self, mode=False, maxHands=2, detectionCon=0.5, trackCon=0.5):
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
        self.drawpoints = []
        self.posEraser = []
        self.drawNum = 0

    def findFingers(self, frame, draw=True):
        """
        Processes the frame to find hand landmarks and optionally draws them.

        Parameters:
        frame (numpy.ndarray): The input image frame.
        draw (bool): Whether to draw the hand landmarks on the frame.

        Returns:
        numpy.ndarray: The frame with hand landmarks drawn (if draw=True).
        """
        imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(frame, handLms, self.handsMp.HAND_CONNECTIONS)
        return frame

    def findPosition(self, frame, handNo=0, draw=True):
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
        self.lmsList = []
        bbox = []
        if self.results.multi_hand_landmarks:
            for hid, myHand in enumerate(self.results.multi_hand_landmarks):
                xList = []
                yList = []
                for id, lm in enumerate(myHand.landmark):
                    h, w, c = frame.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    xList.append(cx)
                    yList.append(cy)
                    id = str(hid)+'_'+str(id)
                    self.lmsList.append([id, cx, cy])
                    if draw:
                        cv2.circle(frame, (cx, cy), 5, (255, 0, 255), cv2.FILLED)
                        cv2.putText(frame, id, (cx+10, cy-10), cv2.FONT_HERSHEY_PLAIN, 0.5, (0, 0, 255))

                xmin, xmax = min(xList), max(xList)
                ymin, ymax = min(yList), max(yList)
                bbox = xmin, ymin, xmax, ymax
                if draw:
                    cv2.rectangle(frame, (xmin - 20, ymin - 20), (xmax + 20, ymax + 20), (0, 255, 0), 2)

        return self.lmsList, bbox

    def _contactFingers(self, lmsList, bbox, frame):
        """
        Draws the dynamic gesture based on the distance between thumb and index finger.

        Parameters:
        lmsList (list): List of hand landmarks.
        bbox (tuple): Bounding box coordinates (xmin, ymin, xmax, ymax).
        frame (numpy.ndarray): The input image frame.
        """
        if len(lmsList) != 0:
            # Calculate the width and height of the bounding box (bbox)
            bbox_width = bbox[2] - bbox[0]
            bbox_height = bbox[3] - bbox[1]
            
            # Calculate the area of the bounding box
            bbox_area = bbox_width * bbox_height
            
            # Calculate the area of the display window
            frame_width = frame.shape[1]
            frame_height = frame.shape[0]
            frame_area = frame_width * frame_height
            
            # Calculate the ratio between the bbox area and the window area
            area_ratio = bbox_area / frame_area
            
            # Create a dynamic threshold based on the area ratio
            threshold = 15 + (area_ratio * 100)  # Adjust the multiplication as needed
            
            #print(f"Área BBox: {bbox_area}, Área Janela: {frame_area}, Razão da Área: {area_ratio:.4f}, Limiar Dinâmico: {threshold:.2f}")
            
            # Get the coordinates of the thumb (point 4) and index finger (point 8)
            thumb_x, thumb_y = lmsList[4][1], lmsList[4][2]
            index_x, index_y = lmsList[8][1], lmsList[8][2]
            
            # Calculate the Euclidean distance between the thumb and index finger points
            distance = math.sqrt((index_x - thumb_x)**2 + (index_y - thumb_y)**2)
            
            # Check if the distance is less than the dynamic threshold
            if distance < threshold:
                # If the distance is less than the threshold, draw the line or perform some action
                #print("Polegar e indicador estão encostando!")
                
                # Add the points to draw the path, if necessary
                return True
            else:
                # If the distance is greater than the threshold, do nothing
                #print("Polegar e indicador estão afastados!")
                return False

    def draw(self, lmsList, bbox, frame):
        """
        Draws the dynamic gesture based on the distance between thumb and index finger.

        Parameters:
        lmsList (list): List of hand landmarks.
        bbox (tuple): Bounding box coordinates (xmin, ymin, xmax, ymax).
        frame (numpy.ndarray): The input image frame.
        """
        if len(lmsList) != 0:
            contact = self._contactFingers(lmsList, bbox, frame)
            thumb_x, thumb_y = lmsList[4][1], lmsList[4][2]
            index_x, index_y = lmsList[8][1], lmsList[8][2]
            if self.drawpoints == []:
                self.drawpoints.append([])
            if contact:
                self.drawpoints[self.drawNum].append([int((thumb_x + index_x) / 2), int((thumb_y + index_y) / 2)])
            elif self.drawpoints[self.drawNum] != []:
                self.drawNum += 1
                self.drawpoints.append([])

    def showDrawPoints(self, frame):
        """
        Draws the points collected for the dynamic gesture on the frame.

        Parameters:
        frame (numpy.ndarray): The input image frame.
        """
        if self.drawpoints != []:
            for points in self.drawpoints:
                points = np.array(points, np.int32)
                points = points.reshape((-1, 1, 2))
                cv2.polylines(frame, [points], isClosed=False, color=(255, 0, 255), thickness=3)

    # def showEraser(self, frame):
    #     if self.posEraser != []:
    #         cv2.rectangle(frame,(posEraser[0],posEraser[1]),(posEraser[2],posEraser[3]), (0, 255, 255), cv2.FILLED)
    #     cv2.rectangle(frame,(50,50),(125,125), (0, 255, 255), cv2.FILLED)
    #     return [50,50,125,125]

    # def moveEraser(self, bbox, lmsList, posEraser):
    #     if len(lmsList) != 0:
    #         # Calculate the width of the bounding box
    #         bbox_width = bbox[1] - bbox[0]
    #         bbox_height = bbox[2] - bbox[1]
    #         coeficiente = (bbox_width + bbox_height)*0.1
    #         print(coeficiente)
            
    #         # Calculate the difference between the coordinates of points 8 and 4
    #         f1x = lmsList[8][1]
    #         f2x = lmsList[4][1]
    #         f1y = lmsList[8][2]
    #         f2y = lmsList[4][2]
    #         subx = abs(f1x - f2x)
    #         suby = abs(f1y - f2y)
            
    #         # Calculate the midpoint between points 8 and 4
    #         medx = (f1x + f2x) // 2
    #         medy = (f1y + f2y) // 2
            
    #         # Check if the average of the differences is within the coefficient
    #         if (subx + suby) / 2 <= coeficiente:
    #             self.drawpoints.append([medx, medy])

def main():
    """
    The main function to run the hand tracking and dynamic gesture recognition.
    """
    cap = cv2.VideoCapture(0)
    detector = HandTrackingDynamic()
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    if not cap.isOpened():
        print("Cannot open camera")
        exit()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = detector.findFingers(frame)
        lmsList, bbox = detector.findPosition(frame, handNo=0)
        #print(bbox)
        detector.draw(lmsList, bbox, frame)
        detector.showDrawPoints(frame)
        #detector.clearDraw(lmsList, bbox)
        frame = cv2.flip(frame, 1)
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()