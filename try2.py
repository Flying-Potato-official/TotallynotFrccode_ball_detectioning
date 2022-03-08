import numpy as np
import cv2

cap = cv2.VideoCapture(0)

lower_blue = np.array([100, 120, 30])
upper_blue = np.array([120, 255, 255])

lower_red = np.array([0, 100, 70])
upper_red = np.array([10, 200, 255])

# w, h
rectangle_threshold = (90, 90)

while True:
    ret, frame = cap.read(0)
    frame = cv2.flip(frame, 1)
    width = int(cap.get(3))
    height = int(cap.get(4))

    # ksize = int(2 * round(10) + 1)
    # frame_blur = cv2.blur(frame, (ksize, ksize))

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    mask2 = cv2.inRange(hsv, lower_red, upper_red)

    cv2.imshow("blue", mask)
    cv2.imshow("red", mask2)

    result = cv2.bitwise_and(frame, frame, mask=mask)
    result2 = cv2.bitwise_and(frame, frame, mask=mask2)

    bluecnts = cv2.findContours(mask.copy(),
                                cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)[-2]

    redcnts = cv2.findContours(mask2.copy(),
                               cv2.RETR_EXTERNAL,
                               cv2.CHAIN_APPROX_SIMPLE)[-2]

    if len(bluecnts) > 15:
        blue_area = max(bluecnts, key=cv2.contourArea)
        (xg, yg, wg, hg) = cv2.boundingRect(blue_area)
        if not (wg < rectangle_threshold[0] or hg < rectangle_threshold[1]):
            # print("ball detected")
            cv2.putText(frame, "Blue Ball", (xg, yg), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 3)
            cv2.circle(frame, (xg + wg // 2, yg + hg // 2), 2, (255, 0, 0), 4)
            # cv2.rectangle(frame, (xg, yg), (xg + wg, yg + hg), (255, 0, 0), 2)

    if len(redcnts) > 15:
        red_area = max(redcnts, key=cv2.contourArea)
        (xg, yg, wg, hg) = cv2.boundingRect(red_area)
        if not (wg < rectangle_threshold[0] or hg < rectangle_threshold[1]):
            # print("ball detected")
            cv2.putText(frame, "Red Ball", (xg, yg), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)
            cv2.circle(frame, (xg + wg // 2, yg + hg // 2), 2, (0, 0, 255), 4)
            # cv2.rectangle(frame, (xg, yg), (xg + wg, yg + hg), (0, 0, 255), 2)

    cv2.imshow("frame", frame)

    # close
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
