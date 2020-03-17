import numpy as np
import cv2

def emboss_filter():
    src = cv2.imread('D:/python/Image/rose.bmp', cv2.IMREAD_GRAYSCALE)

    if src is None:
        print('Image load failed!')
        return

    emboss = np.array([[-1, -1, 0], [-1, 0, 1], [0, 1, 1]], np.float32)
    dst = cv2.filter2D(src, -1, emboss, delta=128)

    cv2.imshow('src', src)
    cv2.imshow('dst', dst)

    cv2.waitKey()
    cv2.destroyAllWindows()

def blurring_mean():
    src = cv2.imread('D:/python/Image/rose.bmp', cv2.IMREAD_GRAYSCALE)

    if src is None:
        print('Image load failed!')
        return

    cv2.imshow('src', src)

    for ksize in range(3, 9, 2):
        dst = cv2.blur(src, (ksize, ksize))

        desc = "Mean : %dx%d" % (ksize, ksize)
        cv2.putText(dst, desc, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, 255, 1, cv2.LINE_AA)

        cv2.imshow('dst', dst)
        cv2.waitKey()
    
    cv2.destroyAllWindows()

def blurring_gaussian():
    src = cv2.imread('D:/python/Image/rose.bmp', cv2.IMREAD_GRAYSCALE)

    if src is None:
        print('Image load failed!')
        return

    cv2.imshow('src', src)
    
    for sigma in range(1, 6):
        dst = cv2.GaussianBlur(src, (0, 0), sigma)

        desc = "Gaussian: sigma = %d" % (sigma)
        cv2.putText(dst, desc, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, 255, 1, cv2.LINE_AA)

        cv2.imshow('dst', dst)
        cv2.waitKey()
    
    cv2.destroyAllWindows()

def sharpen():
    src = cv2.imread('D:/python/Image/rose.bmp', cv2.IMREAD_GRAYSCALE)

    if src is None:
        print('Image load failed!')
        exit()
    
    cv2.imshow('src', src)

    for sigma in range(1, 6):
        blurred = cv2.GaussianBlur(src, (0, 0), sigma)

        alpha = 1.0
        dst = cv2.addWeighted(src, 1+alpha, blurred, -alpha, 0.0)

        desc = "singma : %d" % sigma
        cv2.putText(dst, desc, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, 255, 1, cv2.LINE_AA)

        cv2.imshow('dst', dst)
        cv2.waitKey()
    
    cv2.destroyAllWindows()

if __name__ == '__main__':
    #emboss_filter()
    #blurring_mean()
    #blurring_gaussian()
    sharpen()