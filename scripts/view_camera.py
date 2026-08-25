import sys

import cv2

device = sys.argv[1] if len(sys.argv) > 1 else "/dev/video4"

cap = cv2.VideoCapture(device, cv2.CAP_V4L2)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

print("q 누르면 종료")
while True:
    ok, frame = cap.read()
    if not ok:
        continue
    cv2.imshow(f"camera ({device})", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
