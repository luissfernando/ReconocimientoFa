import cv2
import mediapipe as mp
import numpy as np
import pyautogui

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# Inicializar captura de video
cap = cv2.VideoCapture(0, cv2.CAP_V4L2)

# Colores y configuración de la pantalla de juego
color_mouse_pointer = (255, 0, 255)
SCREEN_GAME_X_INI = 150
SCREEN_GAME_Y_INI = 160
SCREEN_GAME_X_FIN = 150 + 780
SCREEN_GAME_Y_FIN = 160 + 450
aspect_ratio_screen = (SCREEN_GAME_X_FIN - SCREEN_GAME_X_INI) / (SCREEN_GAME_Y_FIN - SCREEN_GAME_Y_INI)
print("aspect_ratio_screen:", aspect_ratio_screen)
X_Y_INI = 100

def calculate_distance(x1, y1, x2, y2):
    p1 = np.array([x1, y1])
    p2 = np.array([x2, y2])
    return np.linalg.norm(p1 - p2)

def detect_finger_down(hand_landmarks, width, height, output):
    finger_down = False
    color_base = (255, 0, 112)
    color_index = (255, 198, 82)

    x_base1 = int(hand_landmarks.landmark[0].x * width)
    y_base1 = int(hand_landmarks.landmark[0].y * height)
    x_base2 = int(hand_landmarks.landmark[9].x * width)
    y_base2 = int(hand_landmarks.landmark[9].y * height)
    x_index = int(hand_landmarks.landmark[8].x * width)
    y_index = int(hand_landmarks.landmark[8].y * height)

    d_base = calculate_distance(x_base1, y_base1, x_base2, y_base2)
    d_base_index = calculate_distance(x_base1, y_base1, x_index, y_index)

    if d_base_index < d_base:
        finger_down = True
        color_base = (255, 0, 255)
        color_index = (255, 0, 255)

    cv2.circle(output, (x_base1, y_base1), 5, color_base, 2)
    cv2.circle(output, (x_index, y_index), 5, color_index, 2)
    cv2.line(output, (x_base1, y_base1), (x_base2, y_base2), color_base, 3)
    cv2.line(output, (x_base1, y_base1), (x_index, y_index), color_index, 3)

    return finger_down

def blur_background(frame, mask):
    blurred_frame = cv2.GaussianBlur(frame, (21, 21), 0)
    mask_inv = cv2.bitwise_not(mask)
    return cv2.bitwise_and(frame, frame, mask=mask) + cv2.bitwise_and(blurred_frame, blurred_frame, mask=mask_inv)

def draw_status(output, status_message, position=(10, 30)):
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(output, status_message, position, font, 1, (0, 255, 0), 2, cv2.LINE_AA)

def draw_instructions(output):
    instructions = [
        "Presiona 'ESC' para salir",
        "Cierra la mano para hacer clic"
    ]
    for i, text in enumerate(instructions):
        cv2.putText(output, text, (10, 30 + i * 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

with mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5) as hands:

    # Inicializar grabador de video
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('output.avi', fourcc, 20.0, (640, 480))

    while True:
        ret, frame = cap.read()
        if not ret:
            print("No se pudo acceder a la cámara")
            break

        height, width, _ = frame.shape
        frame = cv2.flip(frame, 1)

        # Dibujar un área proporcional a la del juego
        area_width = width - X_Y_INI * 2
        area_height = int(area_width / aspect_ratio_screen)
        aux_image = np.zeros(frame.shape, np.uint8)
        aux_image = cv2.rectangle(aux_image, (X_Y_INI, X_Y_INI), (X_Y_INI + area_width, X_Y_INI + area_height), (255, 0, 0), -1)
        mask = np.zeros(frame.shape[:2], np.uint8)
        cv2.rectangle(mask, (X_Y_INI, X_Y_INI), (X_Y_INI + area_width, X_Y_INI + area_height), 255, -1)
        output = blur_background(frame, mask)

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                x = int(hand_landmarks.landmark[9].x * width)
                y = int(hand_landmarks.landmark[9].y * height)
                xm = np.interp(x, (X_Y_INI, X_Y_INI + area_width), (SCREEN_GAME_X_INI, SCREEN_GAME_X_FIN))
                ym = np.interp(y, (X_Y_INI, X_Y_INI + area_height), (SCREEN_GAME_Y_INI, SCREEN_GAME_Y_FIN))
                pyautogui.moveTo(int(xm), int(ym))
                if detect_finger_down(hand_landmarks, width, height, output):
                    pyautogui.click()
                cv2.circle(output, (x, y), 10, color_mouse_pointer, 3)
                cv2.circle(output, (x, y), 5, color_mouse_pointer, -1)

        draw_instructions(output)
        draw_status(output, "Gestos detectados")

        # Grabar el frame en el archivo de video
        out.write(output)

        cv2.imshow('output', output)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
