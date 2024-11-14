import cv2
import mediapipe as mp
import torch
from interception import ffi, lib


def get_keyboard_id():
    for device in range(1, 20):
        if lib.interception_is_keyboard(device):
            return device
    raise Exception("Keyboard device not found")


def get_mouse_id():
    for device in range(1, 20):
        if lib.interception_is_mouse(device):
            return device
    raise Exception("Mouse device not found")


def is_finger_up(tip_id, pip_id, lm_list):
    return lm_list[tip_id][2] < lm_list[pip_id][2]


def is_finger_down(tip_id, pip_id, lm_list):
    return lm_list[tip_id][2] > lm_list[pip_id][2]


class HandController:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        self.hands = mp.solutions.hands.Hands(max_num_hands=2, min_detection_confidence=0.7)
        self.mp_draw = mp.solutions.drawing_utils
        self.interception_context = lib.interception_create_context()
        if self.interception_context == ffi.NULL:
            raise Exception("Failed to initialize interception context.")
        self.device = get_keyboard_id()
        self.mouse_device = get_mouse_id()
        self.key_mappings = {'w': False, 'shift': False, 'a': False, 'd': False, 'space': False}
        self.previous_key_mappings = self.key_mappings.copy()
        self.finger_tips = {'thumb': 4, 'index': 8, 'middle': 12, 'ring': 16, 'pinky': 20}
        self.finger_pips = {'thumb': 3, 'index': 6, 'middle': 10, 'ring': 14, 'pinky': 18}
        self.key_codes = {
            'w': 0x11,
            'a': 0x1E,
            'd': 0x20,
            'space': 0x39,
            'shift': 0x2A,
        }
        self.walk_threshold = 5
        self.run_threshold = 30
        self.prev_right_hand_pos = None
        self.mouse_sensitivity = 16

    def run(self):
        try:
            while True:
                success, img = self.cap.read()
                if not success:
                    break
                img = cv2.flip(img, 1)
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                results = self.hands.process(img_rgb)
                h, w, _ = img.shape
                self.left_hand_detected = False
                self.right_hand_detected = False
                if results.multi_hand_landmarks:
                    for idx, handLms in enumerate(results.multi_hand_landmarks):
                        hand_label = results.multi_handedness[idx].classification[0].label
                        self.mp_draw.draw_landmarks(img, handLms, mp.solutions.hands.HAND_CONNECTIONS)
                        lm_list = [(id, int(lm.x * w), int(lm.y * h)) for id, lm in enumerate(handLms.landmark)]
                        lm_tensor = torch.tensor(lm_list)
                        if hand_label == 'Left':
                            self.left_hand_detected = True
                            self.check_fingers(lm_tensor)
                        elif hand_label == 'Right':
                            self.right_hand_detected = True
                            self.move_mouse(lm_list)
                if self.left_hand_detected:
                    self.perform_actions()
                else:
                    self.release_all_keys()
                if not self.right_hand_detected:
                    self.prev_right_hand_pos = None
                cv2.imshow("Hand Controller", img)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        finally:
            self.release_all_keys()
            self.cap.release()
            cv2.destroyAllWindows()
            lib.interception_destroy_context(self.interception_context)

    def check_fingers(self, lm_list):
        self.key_mappings = dict.fromkeys(self.key_mappings, False)
        if is_finger_down(self.finger_tips['index'], self.finger_pips['index'], lm_list):
            self.key_mappings['d'] = True
        if is_finger_down(self.finger_tips['ring'], self.finger_pips['ring'], lm_list):
            self.key_mappings['a'] = True
        if is_finger_down(self.finger_tips['thumb'], self.finger_pips['thumb'], lm_list):
            self.key_mappings['space'] = True
        if is_finger_down(self.finger_tips['middle'], self.finger_pips['middle'], lm_list):
            y_tip = lm_list[self.finger_tips['middle']][2]
            y_pip = lm_list[self.finger_pips['middle']][2]
            vertical_distance = y_tip - y_pip
            if vertical_distance > self.run_threshold:
                self.key_mappings['w'] = True
                self.key_mappings['shift'] = True
            elif vertical_distance > self.walk_threshold:
                self.key_mappings['w'] = True

    def perform_actions(self):
        for key in self.key_mappings:
            if self.key_mappings[key] != self.previous_key_mappings[key]:
                key_code = self.key_codes[key]
                state = lib.INTERCEPTION_KEY_DOWN if self.key_mappings[key] else lib.INTERCEPTION_KEY_UP
                stroke = ffi.new("InterceptionKeyStroke *", {'code': key_code, 'state': state, 'information': 0})
                lib.interception_send(self.interception_context, self.device,
                                      ffi.cast("InterceptionStroke *", stroke), 1)
        self.previous_key_mappings = self.key_mappings.copy()

    def release_all_keys(self):
        for key, pressed in self.previous_key_mappings.items():
            if pressed:
                key_code = self.key_codes[key]
                stroke = ffi.new("InterceptionKeyStroke *", {'code': key_code, 'state': lib.INTERCEPTION_KEY_UP, 'information': 0})
                lib.interception_send(self.interception_context, self.device,
                                      ffi.cast("InterceptionStroke *", stroke), 1)
        self.previous_key_mappings = dict.fromkeys(self.previous_key_mappings, False)

    def is_hand_open(self, lm_list):
        fingers_up = 0
        for finger in ['index', 'middle', 'ring', 'pinky']:
            tip_id = self.finger_tips[finger]
            pip_id = self.finger_pips[finger]
            if is_finger_up(tip_id, pip_id, lm_list):
                fingers_up += 1
        return fingers_up > 0

    def move_mouse(self, lm_list):
        if self.is_hand_open(lm_list):
            current_x = lm_list[0][1]
            current_y = lm_list[0][2]
            if self.prev_right_hand_pos is not None:
                delta_x = int((current_x - self.prev_right_hand_pos[0]) * self.mouse_sensitivity)
                delta_y = int((current_y - self.prev_right_hand_pos[1]) * self.mouse_sensitivity)
                stroke = ffi.new("InterceptionMouseStroke *", {
                    'state': 0,
                    'flags': lib.INTERCEPTION_MOUSE_MOVE_RELATIVE,
                    'rolling': 0,
                    'x': delta_x,
                    'y': delta_y,
                    'information': 0
                })
                lib.interception_send(self.interception_context, self.mouse_device,
                                      ffi.cast("InterceptionStroke *", stroke), 1)
            self.prev_right_hand_pos = (current_x, current_y)
        else:
            # Hand is making a fist; do not move the mouse
            self.prev_right_hand_pos = None


if __name__ == "__main__":
    HandController().run()
