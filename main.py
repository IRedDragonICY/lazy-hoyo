import cv2
import mediapipe as mp
import time
import threading
from interception import ffi, lib

INTERCEPTION_MOUSE_LEFT_BUTTON_DOWN = 0x001
INTERCEPTION_MOUSE_LEFT_BUTTON_UP = 0x002

def get_device_id(is_device_func):
    for device in range(1, 20):
        if is_device_func(device):
            return device
    raise Exception("Device not found")

def get_keyboard_id():
    return get_device_id(lib.interception_is_keyboard)

def get_mouse_id():
    return get_device_id(lib.interception_is_mouse)

def is_finger_up(tip_y, pip_y):
    return tip_y < pip_y

def is_finger_down(tip_y, pip_y):
    return tip_y > pip_y

class HandController:
    def __init__(self):
        self.right_hand_detected = False
        self.left_hand_detected = False
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise Exception("Camera not accessible")
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        self.hands = mp.solutions.hands.Hands(
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        self.mp_draw = mp.solutions.drawing_utils

        self.interception_context = lib.interception_create_context()
        if self.interception_context == ffi.NULL:
            raise Exception("Failed to initialize interception context.")
        self.keyboard_device = get_keyboard_id()
        self.mouse_device = get_mouse_id()

        self.left_key_mappings = {'w': False, 'shift': False, 'a': False, 'd': False, 'space': False}
        self.right_key_mappings = {'e': False, 'q': False}
        self.previous_left_key_mappings = self.left_key_mappings.copy()
        self.previous_right_key_mappings = self.right_key_mappings.copy()

        self.finger_tips = {'thumb': 4, 'index': 8, 'middle': 12, 'ring': 16, 'pinky': 20}
        self.finger_pips = {'thumb': 3, 'index': 6, 'middle': 10, 'ring': 14, 'pinky': 18}

        self.key_codes = {
            'w': 0x11, 'a': 0x1E, 'd': 0x20, 'space': 0x39, 'shift': 0x2A,
            '1': 0x02, '2': 0x03, '3': 0x04, '4': 0x05, 'e': 0x12, 'q': 0x10
        }

        self.walk_threshold = 5
        self.run_threshold = 30

        self.prev_right_hand_pos = None
        self.mouse_sensitivity = 16

        self.current_key = 1
        self.prev_index_closed = False
        self.prev_thumb_closed = False

        self.pinky_clicking = False
        self.pinky_thread = None

    def run(self):
        try:
            while self.cap.isOpened():
                success, img = self.cap.read()
                if not success:
                    print("Ignoring empty camera frame.")
                    continue

                img = cv2.flip(img, 1)  # Mirror image
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                results = self.hands.process(img_rgb)

                h, w, _ = img.shape

                if results.multi_hand_landmarks:
                    for idx, handLms in enumerate(results.multi_hand_landmarks):
                        hand_label = results.multi_handedness[idx].classification[0].label
                        self.mp_draw.draw_landmarks(img, handLms, mp.solutions.hands.HAND_CONNECTIONS)
                        lm = {id: (int(lm.x * w), int(lm.y * h)) for id, lm in enumerate(handLms.landmark)}
                        self.process_hand_landmarks(hand_label, lm)

                if self.left_hand_detected or self.right_hand_detected:
                    self.perform_actions()
                else:
                    self.release_all_keys()

                if not self.right_hand_detected:
                    self.prev_right_hand_pos = None
                    self.stop_pinky_clicking()

                cv2.imshow("Hand Controller", img)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        finally:
            self.release_all_keys()
            self.stop_pinky_clicking()
            self.cap.release()
            cv2.destroyAllWindows()
            lib.interception_destroy_context(self.interception_context)

    def process_hand_landmarks(self, hand_label, lm):
        if hand_label == 'Left':
            self.left_hand_detected = True
            self.check_left_hand_gestures(lm)
        elif hand_label == 'Right':
            self.right_hand_detected = True
            self.move_mouse(lm)
            self.handle_right_hand_gestures(lm)

    def check_left_hand_gestures(self, lm):
        self.left_key_mappings = {key: False for key in self.left_key_mappings}

        finger_actions = {
            'd': 'index',
            'a': 'ring',
            'space': 'thumb'
        }
        for key, finger in finger_actions.items():
            if self.is_finger_closed(finger, lm):
                self.left_key_mappings[key] = True

        # Check for 'w' and 'shift' (middle finger vertical distance)
        y_tip = lm[self.finger_tips['middle']][1]
        y_pip = lm[self.finger_pips['middle']][1]
        vertical_distance = y_tip - y_pip
        if vertical_distance > self.run_threshold:
            self.left_key_mappings['w'] = True
            self.left_key_mappings['shift'] = True
        elif vertical_distance > self.walk_threshold:
            self.left_key_mappings['w'] = True

    def perform_actions(self):
        self.update_key_mappings(self.left_key_mappings, self.previous_left_key_mappings)
        self.update_key_mappings(self.right_key_mappings, self.previous_right_key_mappings)
        self.previous_left_key_mappings = self.left_key_mappings.copy()
        self.previous_right_key_mappings = self.right_key_mappings.copy()

    def update_key_mappings(self, current_mappings, previous_mappings):
        for key, pressed in current_mappings.items():
            if pressed != previous_mappings.get(key, False):
                key_code = self.key_codes[key]
                state = lib.INTERCEPTION_KEY_DOWN if pressed else lib.INTERCEPTION_KEY_UP
                self.send_key_stroke(key_code, state)

    def release_all_keys(self):
        self.release_keys(self.previous_left_key_mappings)
        self.release_keys(self.previous_right_key_mappings)
        self.previous_left_key_mappings = {key: False for key in self.left_key_mappings}
        self.previous_right_key_mappings = {key: False for key in self.right_key_mappings}

    def release_keys(self, mappings):
        for key, pressed in mappings.items():
            if pressed:
                key_code = self.key_codes[key]
                self.send_key_stroke(key_code, lib.INTERCEPTION_KEY_UP)

    def is_hand_open(self, lm):
        fingers_up = sum(is_finger_up(lm[self.finger_tips[finger]][1], lm[self.finger_pips[finger]][1]) for finger in self.finger_tips)
        return fingers_up > 2

    def is_fist(self, lm):
        fingers_down = sum(is_finger_down(lm[self.finger_tips[finger]][1], lm[self.finger_pips[finger]][1]) for finger in self.finger_tips)
        return fingers_down >= 4

    def is_finger_closed(self, finger_name, lm):
        tip_id = self.finger_tips[finger_name]
        pip_id = self.finger_pips[finger_name]
        return is_finger_down(lm[tip_id][1], lm[pip_id][1])

    def move_mouse(self, lm):
        if self.is_hand_open(lm):
            current_x, current_y = lm[0]
            if self.prev_right_hand_pos is not None:
                delta_x = int((current_x - self.prev_right_hand_pos[0]) * self.mouse_sensitivity)
                delta_y = int((current_y - self.prev_right_hand_pos[1]) * self.mouse_sensitivity)
                self.send_mouse_stroke(0, flags=lib.INTERCEPTION_MOUSE_MOVE_RELATIVE, x=delta_x, y=delta_y)
            self.prev_right_hand_pos = (current_x, current_y)
        else:
            self.prev_right_hand_pos = None

    def handle_right_hand_gestures(self, lm):
        if not self.is_fist(lm):
            index_closed = self.is_finger_closed('index', lm)
            thumb_closed = self.is_finger_closed('thumb', lm)
            middle_closed = self.is_finger_closed('middle', lm)
            pinky_closed = self.is_finger_closed('pinky', lm)

            if index_closed and not self.prev_index_closed:
                self.increment_key()
                self.prev_index_closed = True
            elif not index_closed:
                self.prev_index_closed = False

            if thumb_closed and not self.prev_thumb_closed:
                self.decrement_key()
                self.prev_thumb_closed = True
            elif not thumb_closed:
                self.prev_thumb_closed = False

            self.right_key_mappings['e'] = middle_closed

            if pinky_closed and not self.pinky_clicking:
                self.start_pinky_clicking()
            elif not pinky_closed and self.pinky_clicking:
                self.stop_pinky_clicking()
        else:
            self.prev_index_closed = False
            self.prev_thumb_closed = False
            self.right_key_mappings['e'] = False
            self.stop_pinky_clicking()

    def start_pinky_clicking(self):
        self.pinky_clicking = True
        self.pinky_thread = threading.Thread(target=self.spam_left_click)
        self.pinky_thread.start()

    def stop_pinky_clicking(self):
        if self.pinky_clicking:
            self.pinky_clicking = False
            if self.pinky_thread is not None:
                self.pinky_thread.join()
                self.pinky_thread = None

    def spam_left_click(self):
        while self.pinky_clicking:
            self.send_mouse_stroke(INTERCEPTION_MOUSE_LEFT_BUTTON_DOWN)
            self.send_mouse_stroke(INTERCEPTION_MOUSE_LEFT_BUTTON_UP)
            time.sleep(0.25)

    def increment_key(self):
        old_key = self.current_key
        self.current_key = self.current_key % 4 + 1
        self.update_number_key(old_key, self.current_key)

    def decrement_key(self):
        old_key = self.current_key
        self.current_key = (self.current_key - 2) % 4 + 1
        self.update_number_key(old_key, self.current_key)

    def update_number_key(self, old_key, new_key):
        self.press_and_release_key(str(old_key))
        self.press_and_release_key(str(new_key))

    def press_and_release_key(self, key_str):
        key_code = self.key_codes[key_str]
        self.send_key_stroke(key_code, lib.INTERCEPTION_KEY_DOWN)
        self.send_key_stroke(key_code, lib.INTERCEPTION_KEY_UP)

    def send_key_stroke(self, key_code, state):
        stroke = ffi.new("InterceptionKeyStroke *", {'code': key_code, 'state': state, 'information': 0})
        lib.interception_send(self.interception_context, self.keyboard_device, ffi.cast("InterceptionStroke *", stroke), 1)

    def send_mouse_stroke(self, state, flags=0, x=0, y=0):
        stroke = ffi.new("InterceptionMouseStroke *", {
            'state': state,
            'flags': flags,
            'rolling': 0,
            'x': x,
            'y': y,
            'information': 0
        })
        lib.interception_send(self.interception_context, self.mouse_device, ffi.cast("InterceptionStroke *", stroke), 1)

if __name__ == "__main__":
    HandController().run()