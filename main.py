from ultralytics import YOLO

import cv2


model = YOLO("model/face-detection/yolov8n-face.pt")

img_path = "images/people.jpg"

cam = cv2.VideoCapture(1)

while cam.isOpened():
    succes, frame = cam.read()

    if succes:
        results = model(frame)

        annotated_frame = results[0].plot()
        boxes = results[0].boxes

        for box in boxes:
            top_left_x = int(box.xyxy.tolist()[0][0])
            top_left_y = int(box.xyxy.tolist()[0][1])
            bottom_right_x = int(box.xyxy.tolist()[0][2])
            bottom_right_y = int(box.xyxy.tolist()[0][3])

            cv2.rectangle(frame, (top_left_x, top_left_y), (bottom_right_x, bottom_right_y), (0, 255, 0), 2)
            cv2.imshow("detection", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    else:
        break

cam.release()
cv2.destroyAllWindows()


