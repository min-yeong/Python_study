import os
import cv2

searchimg = cv2.imread('D:/python/histogram_test/searchimage/wahoo_test.jpg', cv2.IMREAD_COLOR)
cropping = searchimg[:, 20:150]
findlist_x = {}

def histogram(searchimg):
    searchHLS = cv2.cvtColor(searchimg, cv2.COLOR_BGR2HLS)
    searchhistogram = cv2.calcHist([searchHLS], [0], None, [256], [0, 256])
    findlist = {}

    for root, dirs, files in os.walk('D:/python/histogram_test/image'):
        for fname in files:
            full_fname = os.path.join(root, fname)
            #print(full_fname) 
        
            imgNames = cv2.imread(full_fname)
            imgsHLS = cv2.cvtColor(imgNames, cv2.COLOR_BGR2HLS)
            #cv2.imshow('', imgsHLS)
            #cv2.waitKey(0)

            histogram = cv2.calcHist([imgsHLS], [0], None, [256], [0, 256])
            #print(histogram)

            matching_score = cv2.compareHist(histogram, searchhistogram, cv2.HISTCMP_CORREL)
            if (matching_score > 0.3):
                findlist[full_fname] = matching_score 
                #print(findlist)
        findlist_x = sorted(findlist.items(), key=(lambda x:x[1]), reverse=True)
        for k in findlist_x:
            print(k)
        return findlist_x

def matching(searchimg, findlist_x):

    sift = cv2.xfeatures2d.SIFT_create()
    findlist={}

    for fname in findlist_x:
        #print(fname)
        imgNames = cv2.imread(fname[0])

        kp1, des1 = sift.detectAndCompute(searchimg, None)
        kp2, des2 = sift.detectAndCompute(imgNames, None)

        bf = cv2.BFMatcher()
        matches = bf.knnMatch(des1, des2, k=2)
        good = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good.append([m])

        img3 = cv2.drawMatchesKnn(searchimg, kp1, imgNames, kp2, good, None, flags=2)
        count = len(good)
        if (count > 0):
            findlist[fname] = count
            # print(count)
        findlist_x = sorted(findlist.items(), key=(lambda x:x[1]), reverse=True)
    for k in findlist_x:
        print(k)
    return findlist_x

if __name__ == '__main__':
    findlist_x = histogram(searchimg)
    cv2.imshow("img", cropping)
    cv2.waitKey(0) 
    matching(searchimg, findlist_x)

