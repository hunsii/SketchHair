import cv2
import numpy as np

WINDOW_NAME = 'painter'
ALPHA = 0.25
THICKNESS = 2
drawing = False # true if mouse is pressed
pt1_x , pt1_y = None , None
r, g, b = None, None, None
# mouse callback function

def line_drawing(event,x,y,flags,param):
    global pt1_x,pt1_y,drawing
    global r,g,b

    if event==cv2.EVENT_LBUTTONDOWN:
        drawing=True
        pt1_x,pt1_y=x,y

    elif event==cv2.EVENT_MOUSEMOVE:
        if drawing==True:
            cv2.line(img,(pt1_x,pt1_y),(x,y),color=(r,g,b),thickness=THICKNESS)
            pt1_x,pt1_y=x,y
    elif event==cv2.EVENT_LBUTTONUP:
        drawing=False
        cv2.line(img,(pt1_x,pt1_y),(x,y),color=(r,g,b),thickness=THICKNESS)        

def onChange(pos):
    pass

white_img = np.zeros((512,512,3), np.uint8)
white_img[:, :, :] = 255

img = cv2.imread('default.png', cv2.IMREAD_COLOR)
img = cv2.addWeighted(img, ALPHA, white_img, 1 - ALPHA, 0)
# 그릴 위치, 텍스트 내용, 시작 위치, 폰트 종류, 크기, 색깔, 두께
cv2.putText(img, "q: exit", (0, 20), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 0), THICKNESS)
cv2.putText(img, "r: reset", (0, 40), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 0), THICKNESS)
backup_img = img.copy()


cv2.namedWindow(WINDOW_NAME)
cv2.setMouseCallback(WINDOW_NAME,line_drawing)

cv2.createTrackbar("R", WINDOW_NAME, 0, 255, onChange)
cv2.createTrackbar("G", WINDOW_NAME, 0, 255, onChange)
cv2.createTrackbar("B", WINDOW_NAME, 0, 255, onChange)

cv2.setTrackbarPos("R", WINDOW_NAME, 255)
cv2.setTrackbarPos("G", WINDOW_NAME, 255)
cv2.setTrackbarPos("B", WINDOW_NAME, 255)

while(1):
    r = cv2.getTrackbarPos("R", WINDOW_NAME)
    g = cv2.getTrackbarPos("G", WINDOW_NAME)
    b = cv2.getTrackbarPos("B", WINDOW_NAME)

    cv2.imshow(WINDOW_NAME,img)
    input_key = cv2.waitKey(1)
    if input_key == ord('q'):
        break
    elif input_key == ord('r'):
        img = backup_img.copy()
cv2.destroyAllWindows()


