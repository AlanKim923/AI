from ultralytics import YOLO

import cv2


# face_model = YOLO("model/face-detection/yolov8n-face.pt")
# model = YOLO("model/yolov8n.pt")
#
# cam = cv2.VideoCapture(1)
#
# while cam.isOpened():
#     success, frame = cam.read()
#
#     if success:
#         face_results = face_model(frame)
#         results = model(frame)
#
#         annotated_frame = results[0].plot()
#         boxes = face_results[0].boxes
#
#         for box in boxes:
#             top_left_x = int(box.xyxy.tolist()[0][0])
#             top_left_y = int(box.xyxy.tolist()[0][1])
#             bottom_right_x = int(box.xyxy.tolist()[0][2])
#             bottom_right_y = int(box.xyxy.tolist()[0][3])
#
#             cv2.rectangle(frame, (top_left_x, top_left_y), (bottom_right_x, bottom_right_y), (0, 255, 0), 2)
#             cv2.putText(frame, "test", (top_left_x + 5, top_left_y - 10), cv2.FONT_HERSHEY_DUPLEX, 2, (0, 0, 255), 2)
#             cv2.imshow("detection", frame)
#
#         # cv2.imshow("detection", annotated_frame)
#
#         if cv2.waitKey(1) & 0xFF == ord("q"):
#             break
#
#     else:
#         break
#
# cam.release()
# cv2.destroyAllWindows()

model = YOLO("model/yolov10n.pt")

results = model("images/people.jpg")

# print(type(results))
# print(results)

var = model.__dict__
print(var.keys())

results[0].show()

