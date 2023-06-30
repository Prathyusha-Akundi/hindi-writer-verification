import cv2
import numpy as np

class TextDetector:
    def __init__(self):
        pass
    def reject_outliers(self, data, axis = 0, m = 3.0):
        d = np.abs(data[:,axis] - np.median(data[:, axis]))
        mdev = np.median(d)
        s = d/mdev if mdev else np.zeros(len(d))
        return data[s<m]
    
    def get_text_box(self, image):
        # Load image, grayscale, Gaussian blur, adaptive threshold

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (3,3), 0)
        # thresh = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,11,30)
        _, thresh = cv2.threshold(blur,0,255,cv2.THRESH_OTSU)

        # Dilate to combine adjacent text contours
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,1))
        dilate = cv2.dilate(thresh, kernel, iterations=2)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1,3))
        dilate = cv2.dilate(dilate, kernel, iterations=1)
        # Find contours, highlight text areas, and extract ROIs
        cnts = cv2.findContours(dilate, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]

        xmin = 3000
        ymin = 3000
        xmax = 0
        ymax = 0
        rect_list = []
        for c in cnts:
            area = cv2.contourArea(c)
            if area < 1000 :
                rect = cv2.boundingRect(c)
                rect_list.append(rect)
        rect_list= np.array(rect_list)


        filtered_data = self.reject_outliers(rect_list)
        filtered_data = self.reject_outliers(filtered_data, axis=1)
        for r in filtered_data:
            x,y, w, h = r
            xmin = min(x, xmin)
            xmax = max(x + w, xmax)
            ymin = min(y, ymin)
            ymax = max(y +h, ymax)
        return [xmin, ymin, xmax, ymax]
            
class LineDetector:
    def __init__(self):
        pass

    def get_lines(self, image):        
        minLineLength = 400
        edges = cv2.Canny(image, 5, 200)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # ret, thresh1 = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)
        rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1,5))
        dilation = cv2.dilate(edges, rect_kernel, iterations = 3)
        
        lines = cv2.HoughLinesP(image=dilation,rho=1,theta=np.pi/180, threshold=800,lines=np.array([]), minLineLength=minLineLength,maxLineGap=0)

        return np.array(lines)