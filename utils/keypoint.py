import mediapipe as mp

from pydantic import BaseModel


class HandKeypoint(BaseModel):
    WRIST:              int = 0
    THUMB_CMC:          int = 1
    THUMB_MCP:          int = 2
    THUMB_IP:           int = 3
    THUMB_TIP:          int = 4
    INDEX_FINGER_MCP:   int = 5
    INDEX_FINGER_PIP:   int = 6
    INDEX_FINGER_DIP:   int = 7
    INDEX_FINGER_TIP:   int = 8
    MIDDLE_FINGER_MCP:  int = 9
    MIDDLE_FINGER_PIP:  int = 10
    MIDDLE_FINGER_DIP:  int = 11
    MIDDLE_FINGER_TIP:  int = 12
    RING_FINGER_MCP:    int = 13
    RING_FINGER_PIP:    int = 14
    RING_FINGER_DIP:    int = 15
    RING_FINGER_TIP:    int = 16
    PINKY_FINGER_MCP:   int = 17
    PINKY_FINGER_PIP:   int = 18
    PINKY_FINGER_DIP:   int = 19
    PINKY_FINGER_TIP:   int = 20
    
    
class DetectHandKeypoint:
    def __init__(self):    
        self.mpModel = mp.solutions.hands.Hands(max_num_hands=2)
        self.mpDraw = mp.solutions.drawing_utils
        self.get_keypoint = HandKeypoint()

    # extract function keypoint
    def get_keypoint_list(self, landmark):
        # wrist
        wrist_x, wrist_y = landmark[self.get_keypoint.WRIST].x, landmark[self.get_keypoint.WRIST].y
        # thumb
        thumb_mcp_x, thumb_mcp_y = landmark[self.get_keypoint.THUMB_MCP].x, landmark[self.get_keypoint.THUMB_MCP].y
        thumb_tip_x, thumb_tip_y = landmark[self.get_keypoint.THUMB_TIP].x, landmark[self.get_keypoint.THUMB_TIP].y
        # index finger
        index_finger_pip_x, index_finger_pip_y = landmark[self.get_keypoint.INDEX_FINGER_PIP].x, landmark[self.get_keypoint.INDEX_FINGER_PIP].y
        index_finger_tip_x, index_finger_tip_y = landmark[self.get_keypoint.INDEX_FINGER_TIP].x, landmark[self.get_keypoint.INDEX_FINGER_TIP].y
        # middle finger
        middle_finger_pip_x, middle_finger_pip_y = landmark[self.get_keypoint.MIDDLE_FINGER_PIP].x, landmark[self.get_keypoint.MIDDLE_FINGER_PIP].y
        middle_finger_tip_x, middle_finger_tip_y = landmark[self.get_keypoint.MIDDLE_FINGER_TIP].x, landmark[self.get_keypoint.MIDDLE_FINGER_TIP].y
        # ring finger
        ring_finger_pip_x, ring_finger_pip_y = landmark[self.get_keypoint.RING_FINGER_PIP].x, landmark[self.get_keypoint.RING_FINGER_PIP].y
        ring_finger_tip_x, ring_finger_tip_y = landmark[self.get_keypoint.RING_FINGER_TIP].x, landmark[self.get_keypoint.RING_FINGER_TIP].y
        # pinky finger
        pinky_finger_pip_x, pinky_finger_pip_y = landmark[self.get_keypoint.PINKY_FINGER_PIP].x, landmark[self.get_keypoint.PINKY_FINGER_PIP].y
        pinky_finger_tip_x, pinky_finger_tip_y = landmark[self.get_keypoint.PINKY_FINGER_TIP].x, landmark[self.get_keypoint.PINKY_FINGER_TIP].y

        return [
            wrist_x, wrist_y,
            thumb_mcp_x, thumb_mcp_y, thumb_tip_x, thumb_tip_y,
            index_finger_pip_x, index_finger_pip_y, index_finger_tip_x, index_finger_tip_y,
            middle_finger_pip_x, middle_finger_pip_y, middle_finger_tip_x, middle_finger_tip_y,
            ring_finger_pip_x, ring_finger_pip_y, ring_finger_tip_x, ring_finger_tip_y,
            pinky_finger_pip_x, pinky_finger_pip_y, pinky_finger_tip_x, pinky_finger_tip_y,
        ]
        
    def plot(self, frame, hand):
        self.mpDraw.draw_landmarks(frame, hand, mp.solutions.hands.HAND_CONNECTIONS)
        return frame
    
    def __call__(self, img):
        results = self.mpModel.process(img)
        return results