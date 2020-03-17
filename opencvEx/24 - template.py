import numpy as np
import cv2
import random

def template():
    img = cv2.imread('D:/python/Image/circuit.bmp', cv2.IMREAD_COLOR)
    templ = cv2.imread('D:/python/Image/crystal.bmp', cv2.IMREAD_COLOR)

    if img is None or templ is None:
        print('Image load failed!')
        exit()

    img = img + (50, 50, 50)

    noise = np.zeros(img.shape, np.int32)
    cv2.randn(noise, 0, 10)
    img = cv2.add(img, noise, dtype=cv2.CV_8UC3)

    res = cv2.matchTemplate(img, templ, cv2.TM_CCOEFF_NORMED)
    res_norm = cv2.normalize(res, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

    _, maxv, _, maxloc = cv2.minMaxLoc(res)
    print('maxv:', maxv)

    (th, tw) = templ.shape[:2]
    cv2.rectangle(img, maxloc, (maxloc[0] + tw, maxloc[1] + th), (0, 0, 255), 2)

    cv2.imshow('templ', templ)
    cv2.imshow('res_norm', res_norm)
    cv2.imshow('img', img)
    cv2.waitKey()
    cv2.destroyAllWindows()

def detect_face():
    src = cv2.imread('D:/python/Image/kids.png')

    if src is None:
        print('Image load failed!')
        return

    classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    if classifier.empty():
        print('XML load failed!')
        return

    faces = classifier.detectMultiScale(src)

    for (x, y, w, h) in faces:
        cv2.rectangle(src, (x, y), (x + w, y + h), (255, 0, 255), 2)

    cv2.imshow('src', src)
    cv2.waitKey()
    cv2.destroyAllWindows()

def detect_eyes():
    src = cv2.imread('D:/python/Image/kids.png')

    if src is None:
        print('Image load failed!')
        return

    face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    eye_classifier = cv2.CascadeClassifier('haarcascade_eye.xml')

    if face_classifier.empty() or eye_classifier.empty():
        print('XML load failed')
        return
    
    faces = face_classifier.detectMultiScale(src)

    for (x1, y1, w1, h1) in faces:
        cv2.rectangle(src, (x1, y1), (x1 + w1, y1 + h1), (255, 0, 255), 2)

        faceROI = src[y1:y1 + h1, x1:x1 + w1]
        eyes = eye_classifier.detectMultiScale(faceROI)

        for (x2, y2, w2, h2) in eyes:
            center = (int(x2 + w2 / 2), int(y2 + h2 /2))
            cv2.circle(faceROI, center, int(w2 /2) , (255, 0, 0), 2, cv2.LINE_AA)
    
    cv2.imshow('src', src)
    cv2.waitKey()
    cv2.destroyAllWindows()

def  hog():
    cap = cv2.VideoCapture('D:/python/Image/vtest.avi')

    if not cap.isOpened():
        print('Video open failed')
        exit()
    
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        detected, _ = hog.detectMultiScale(frame)

        for (x, y, w, h) in detected:
            c = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
            cv2.rectangle(frame, (x, y), (x + w, y+ h), c, 3)

        cv2.imshow('frame', frame)
        if cv2.waitKey(10) == 27:
            break
    
    cv2.destroyAllWindows()

def qrcode():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print('Video open failed!')
        exit()
    
    detector = cv2.QRCodeDetector()

    while True:
        ret, frame = cap.read()

        if not ret:
            print('Frame load failed!')
            break
        
        info, points, _ = detector.detectAndDecode(frame)

        if points is not None:
            prints = np.array(points, dtype=np.int32).reshape(4, 2)
            cv2.polylines(frame, [points], True, (0, 0, 255), 2)

        if len(info) > 0:
            cv2.putText(frame, info, (10, 30), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 255), lineType=cv2.LINE_AA)
        
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) == 27:
            break
    
    cv2.destroyAllWindows()

if __name__ == '__main__':
    template()
    #detect_face()
    #detect_eyes()
    #hog()
    #qrcode()