import cv2
import numpy as np

class DashLaneDet():
    def region_of_interest(self, img, vertices, color3=(255, 255, 255), color1=255):
        mask = np.zeros_like(img)
        if len(img.shape) > 2:
            color = color3
        else:
            color = color1
        cv2.fillPoly(mask, vertices, color)
        ROI_image = cv2.bitwise_and(img, mask)
        return ROI_image

    def detect_stoplineB(self, x, line_koefs):
        frame = x.copy()
        k_koef = line_koefs[0]
        b_koef = line_koefs[1]

        # gray
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # blur
        kernel_size = 5
        blur_frame = cv2.GaussianBlur(gray, (kernel_size, kernel_size), 0)
        
        down_border = frame.shape[0]*0.73
        up_border = frame.shape[0]*0.51
        x_leftdown =  max((down_border - b_koef) / k_koef - 50, 0)
        x_leftup =    max((up_border   - b_koef) / k_koef - 20, 0)
        x_rightup =   min((up_border   - b_koef) / k_koef + 20, frame.shape[1])
        x_rightdown = min((down_border - b_koef) / k_koef + 50, frame.shape[1])

        vertices = np.array([[
            (x_leftdown, down_border),
            (x_leftup, up_border),
            (x_rightup, up_border),
            (x_rightdown, down_border)
        ]], dtype=np.int32)

        roi = self.region_of_interest(blur_frame, vertices)
        # cv2.imshow("roi:", roi)
        # filter
        img_mask = cv2.inRange(roi, 100, 400) ## default 160, 220
        img_result = cv2.bitwise_and(roi, roi, mask=img_mask)

        # binary
        ret, dest = cv2.threshold(img_result, 140, 155, cv2.THRESH_BINARY) ## default 160, 255
        # canny
        low_threshold, high_threshold = 70, 210 #70, 210
        edge_img = cv2.Canny(np.uint8(dest), low_threshold, high_threshold)
        # cv2.imshow('edge_img', edge_img)
        # find contours, opencv4
        contours, hierarchy = cv2.findContours(edge_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        approx_max:any = None
        approxes = []

        if contours:
            stopline_info = [0, 0, 0, 0]
            for contour in contours:
                epsilon = 0.01 * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)

                (x, y), (w, h), theta = cv2.minAreaRect(contour)
                if w > 100 or h > 100:
                    return False

        return True

