import numpy as np
import cv2

def keyboard_event():
    img = cv2.imread('D:/python/Image/lenna.bmp')

    if img is None:
        print("Image load failed!")
        exit()

    cv2.namedWindow('img')
    cv2.imshow('img', img)

    while True:
        keycode = cv2.waitKey()
        if keycode == ord('i') or keycode == ord('I'):
            img = ~img
            cv2.imshow('img', img)
        elif keycode == 27 or keycode == ord('q') or keycode == ord('Q'):
            break

    cv2.destroyAllWindows()

def mouse_event():
    def on_mouse(event, x, y, flags, param):
        global oldx, oldy

        if event == cv2.EVENT_LBUTTONDOWN:
            oldx, oldy = x, y
            print('EVENT_LBUTTONDOWN: %d, %d' % (x, y))
    
        elif event == cv2.EVENT_LBUTTONUP:
            print('EVENT_LBUTTONUP:, %d, %d' % (x, y))
    
        elif event == cv2.EVENT_MOUSEMOVE:
            if flags & cv2.EVENT_FLAG_LBUTTON:
                cv2.line(img, (oldx, oldy), (x, y), (0, 255, 255), 2)
                cv2.imshow('img', img)
                oldx, oldy = x, y
    img = cv2.imread('D:/python/Image/lenna.bmp')
    
    if img is None:
        print("Image load failed!")
        exit()
    cv2.namedWindow('img')
    cv2.setMouseCallback('img', on_mouse)

    cv2.imshow('img', img)
    cv2.waitKey()
    cv2.destroyAllWindows()

def trackbar():
    def saturated(value):
        if value > 255:
            value = 255
        elif value < 0:
            value = 0
        return value
    
    def on_level_change(pos):
        img[:] = saturated(pos * 16)
        cv2.imshow('img', img)
    
    img = np.zeros((400, 400), np.uint8)
    cv2.namedWindow('image')
    cv2.createTrackbar('level', 'image', 0, 16, on_level_change)

    cv2.imshow('image', img)
    cv2.waitKey()
    cv2.destroyAllWindows()    

if __name__ == '__main__':
    #keyboard_event()
    #mouse_event()
    trackbar()