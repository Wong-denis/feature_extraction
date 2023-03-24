import cv2

image = cv2.imread("chess/board.png")
image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
ret,thresh1 = cv2.threshold(image,127,255,cv2.THRESH_BINARY_INV)
cv2.imwrite("chess/board_btow.png", thresh1)
