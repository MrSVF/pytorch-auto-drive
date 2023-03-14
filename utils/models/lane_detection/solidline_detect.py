import cv2
import numpy as np

class SolidLaneDet():
    def region_of_interest(self, img, vertices, color3=(255, 255, 255), color1=255):
        mask = np.zeros_like(img)
        if len(img.shape) > 2:
            color = color3
        else:
            color = color1
        cv2.fillPoly(mask, vertices, color)
        ROI_image = cv2.bitwise_and(img, mask)
        return ROI_image

    def detect_stoplineB(self, x, line_koefs, down_border, up_border):
        frame = x.copy()
        k_koef = line_koefs[0]
        b_koef = line_koefs[1]

        # gray
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # blur
        kernel_size = 5
        blur_frame = gray#cv2.GaussianBlur(gray, (kernel_size, kernel_size), 0)
        
        # down_border = frame.shape[0]*0.73
        # up_border = frame.shape[0]*0.51
        x_leftdown =  max((down_border - b_koef) / k_koef - 50, 0)
        x_leftup =    max((up_border   - b_koef) / k_koef - 20, 0)
        x_rightup =   min((up_border   - b_koef) / k_koef + 20, frame.shape[1]-1)
        x_rightdown = min((down_border - b_koef) / k_koef + 50, frame.shape[1]-1)
        print('POINTS:', x_leftdown, x_leftup, x_rightup, x_rightdown, k_koef, b_koef)

        vertices = np.array([[
            (x_leftdown, down_border),
            (x_leftup, up_border),
            (x_rightup, up_border),
            (x_rightdown, down_border)
        ]], dtype=np.int32)

        roi = self.region_of_interest(blur_frame, vertices)
        # cv2.imshow("roi:", roi)
        # filter
        img_mask = cv2.inRange(roi, 160, 255) ## default 160, 220
        img_result = cv2.bitwise_and(roi, roi, mask=img_mask)

        # binary
        ret, dest = cv2.threshold(img_result, 150, 255, cv2.THRESH_BINARY) ## default 160, 255
        # canny
        low_threshold, high_threshold = 70, 210 #70, 210
        edge_img = cv2.Canny(np.uint8(dest), low_threshold, high_threshold)
        # cv2.imshow('edge_img', edge_img)
        # find contours, opencv4
        contours, hierarchy = cv2.findContours(edge_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # approx_max:any = None
        # approxes = []
        dashes = []

        if contours:
            for contour in contours:
                # epsilon = 0.01 * cv2.arcLength(contour, True)
                # approx = cv2.approxPolyDP(contour, epsilon, True)
                (x, y), (w, h), theta = cv2.minAreaRect(contour)
                # print('(w, h):', w, h)
                if w > 110 or h > 110:
                    return True
                elif max(w, h) > 30 and min(w, h) > 7:
                    dashes.append([x, y, w, h])
            
            dashes_sorted = sorted(dashes, key=lambda x: x[1], reverse=True)
            dashes_sorted_ok = [dashes_sorted[0]] if len(dashes_sorted) !=0 else []
            for i in range(len(dashes_sorted)-1):
                dash_len_i = max(dashes_sorted[i][2], dashes_sorted[i][3])
                if dashes_sorted[i][1]-dash_len_i/2 > dashes_sorted[i+1][1]:
                    dashes_sorted_ok.append(dashes_sorted[i+1])
            
            if len(dashes_sorted_ok) == 2:
                print('dashes_sorted:', dashes_sorted)
                print('dashes_sorted_ok:', dashes_sorted_ok)
                center_distance = np.sqrt((dashes_sorted_ok[1][0]-dashes_sorted_ok[0][0])**2 + \
                                          (dashes_sorted_ok[1][1]-dashes_sorted_ok[0][1])**2)
                print('center_distance:', center_distance)
                dash_len1 = max(dashes_sorted_ok[0][2], dashes_sorted_ok[0][3])
                dash_len2 = max(dashes_sorted_ok[1][2], dashes_sorted_ok[1][3])
                print('dash_len1/2:', dash_len1/2)
                print('dash_len2/2:', dash_len2/2)
                print('dist:', center_distance - dash_len2/2 - dash_len1/2)
                # dash_angle1 = theta if dashes_sorted[0][2] > dashes_sorted[0][3] else theta + 90
                # dash_angle2 = theta if dashes_sorted[1][2] > dashes_sorted[1][3] else theta + 90

                if center_distance - dash_len2/2 - dash_len1/2 < 23:
                    print('return True')
                    return True

        return False

