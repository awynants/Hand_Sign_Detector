import cv2
import mediapipe as mp

class HandSignDetector:
    def __init__(self):
        """Initialize the MediaPipe Hands and Drawing utilities."""
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands()
        self.mp_drawing = mp.solutions.drawing_utils
        self.current_sign = "None"
        self.landmarks_history = []

    def get_finger_states(self, hand_landmarks):
        """
        Determine the state of each finger (extended or not).
        
        Args:
            hand_landmarks: MediaPipe hand landmarks.

        Returns:
            list: A list of boolean values indicating if each finger is extended.
        """
        finger_states = []

        # Thumb (based on x-coordinates)
        if hand_landmarks.landmark[self.mp_hands.HandLandmark.THUMB_TIP].x < hand_landmarks.landmark[self.mp_hands.HandLandmark.THUMB_IP].x:
            finger_states.append(True)
        else:
            finger_states.append(False)

        # Index finger
        if hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP].y < hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_PIP].y:
            finger_states.append(True)
        else:
            finger_states.append(False)

        # Middle finger
        if hand_landmarks.landmark[self.mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y < hand_landmarks.landmark[self.mp_hands.HandLandmark.MIDDLE_FINGER_PIP].y:
            finger_states.append(True)
        else:
            finger_states.append(False)

        # Ring finger
        if hand_landmarks.landmark[self.mp_hands.HandLandmark.RING_FINGER_TIP].y < hand_landmarks.landmark[self.mp_hands.HandLandmark.RING_FINGER_PIP].y:
            finger_states.append(True)
        else:
            finger_states.append(False)

        # Pinky finger
        if hand_landmarks.landmark[self.mp_hands.HandLandmark.PINKY_TIP].y < hand_landmarks.landmark[self.mp_hands.HandLandmark.PINKY_PIP].y:
            finger_states.append(True)
        else:
            finger_states.append(False)

        # Handle thumb inversion
        if hand_landmarks.landmark[self.mp_hands.HandLandmark.THUMB_TIP].x > hand_landmarks.landmark[self.mp_hands.HandLandmark.THUMB_IP].x:
            finger_states.append(True)
        else:
            finger_states.append(False)

        return finger_states

    def detect_hand_sign(self, image):
        """
        Detect hand signs in the given image and annotate it.

        Args:
            image (ndarray): The input image.

        Returns:
            ndarray: The annotated image with detected hand sign.
        """
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_image)
        self.current_sign = "None"
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                self.landmarks_history.append(hand_landmarks.landmark)
                self.mp_drawing.draw_landmarks(image, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                finger_states = self.get_finger_states(hand_landmarks)

                # Determine the sign based on finger states
                if all(finger_states[1:5]):
                    self.current_sign = "Open Hand"
                elif (finger_states[0] or finger_states[5]) and not any(finger_states[1:4]) and finger_states[4]:
                    self.current_sign = "Surfing Sign"
                elif not finger_states[0] and finger_states[1] and finger_states[2] and not any(finger_states[3:5]):
                    self.current_sign = "Peace Sign"
                elif all(finger_states[2:5]) and not finger_states[1] and \
                     hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP].y < hand_landmarks.landmark[self.mp_hands.HandLandmark.THUMB_TIP].y:
                    self.current_sign = "OK Sign"
                else:
                    self.current_sign = "Unknown Sign"
                
                self.detect_waving()

                # Draw the sign text on the image
                cv2.putText(image, self.current_sign, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
        
        return image

    def detect_waving(self):
        if len(self.landmarks_history) < 10:  # Ensure at least 10 frames are stored
            return

        # Check for back-and-forth motion
        x_positions = [landmarks[0].x for landmarks in self.landmarks_history[-10:]]
        if max(x_positions) - min(x_positions) > 0.1:  # Threshold for waving motion
            self.current_sign = "Waving"

        # Maintain a limited history
        self.landmarks_history = self.landmarks_history[-20:]  # Keep the last 20 frames

    def get_detected_sign(self):
        return self.current_sign

def main():
    """Main function to capture video and detect hand signs."""
    detector = HandSignDetector()
    cap = cv2.VideoCapture(0)
    
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        annotated_image = detector.detect_hand_sign(frame)
        cv2.imshow('Hand Sign Detection', annotated_image)

        if cv2.waitKey(5) & 0xFF == 27:  # Press 'Esc' to exit
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()