import cv2
import numpy as np

img_color = cv2.imread('20200916_181404_cr_0000001480.png')

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

    def detect_stoplineB(self, x):
        frame = x.copy()
        img = frame.copy()
        min_stopline_length = 200 #330 #defualt 250
        #max_stopline_length = 250
        max_distance = 500 #120 #defualt 70
        min_distance = 80

        # gray
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # blur
        kernel_size = 5
        blur_frame = cv2.GaussianBlur(gray, (kernel_size, kernel_size), 0)
        
        # roi
        # vertices = np.array([[
        #     (80, frame.shape[0]),
        #     (120, frame.shape[0] - 120),
        #     (frame.shape[1] - 80, frame.shape[0] - 120),
        #     (frame.shape[1] - 120, frame.shape[0])
        # ]], dtype=np.int32)
        vertices = np.array([[
            (350, frame.shape[0]*0.73), #*0.63
            (400, frame.shape[0]*0.59), #*0.58
            (frame.shape[1] - 300, frame.shape[0]*0.59), #*0.58
            (frame.shape[1] - 250, frame.shape[0]*0.73)  #*0.63
        ]], dtype=np.int32)

        roi = self.region_of_interest(blur_frame, vertices)
        cv2.imshow("roi:", roi)
        # filter
        img_mask = cv2.inRange(roi, 100, 400) ## default 160, 220
        img_result = cv2.bitwise_and(roi, roi, mask=img_mask)

        # cv2.imshow('bin', img_result)

        # binary
        ret, dest = cv2.threshold(img_result, 160, 255, cv2.THRESH_BINARY)
        # cv2.imshow('dest', dest)
        # canny
        low_threshold, high_threshold = 70, 210
        edge_img = cv2.Canny(np.uint8(dest), low_threshold, high_threshold)
        cv2.imshow('edge_img', edge_img)
        # find contours, opencv4
        contours, hierarchy = cv2.findContours(edge_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # print('contours:', len(contours))
        # find contours, opencv3
        #_, contours, hierarchy = cv2.findContours(edge_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        approx_max:any = None

        if contours:
            stopline_info = [0, 0, 0, 0]
            for contour in contours:
                epsilon = 0.01 * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)
                # result = cv2.drawContours(frame, [approx], 0, (0,255,0), 4)
                # print('result:', frame)
                # cv2.imshow('result', result)
                x, y, w, h = cv2.boundingRect(contour)
                print('x, y, w, h:', x, y, w, h)
                if stopline_info[2] < w and h < 40:
                    stopline_info = [x, y, w, h]
                    approx_max = approx
            # cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 3)
            rect = cv2.minAreaRect(approx_max)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            result = cv2.drawContours(frame, [box], 0, (0, 255, 0), 3)
            # cv2.imshow('result', result)
            # print('x, y, w, h:', x, y, w, h)
            
            cx, cy = stopline_info[0] + 0.5 * stopline_info[2], stopline_info[1] + 0.5 * stopline_info[3]
            center = np.array([cx, cy])
            stopline_length = stopline_info[2]
            bot_point = np.array([frame.shape[1] // 2, frame.shape[0]])
            distance = np.sqrt(np.sum(np.square(center - bot_point)))

            # OUTPUT
            print('length : {},  distance : {}'.format(stopline_length, distance))
            # red_color = (0,0,255)
            # cv2.rectangle(img, vertices, red_color, 3)
            if stopline_length > min_stopline_length and min_distance <distance < max_distance:
            #if min_stopline_length <= stopline_length <= max_stopline_length and min_distance < distance < max_distance:
                cv2.imshow('stopline', result)
                cv2.waitKey(1)
                print('STOPLINE Detected')
                # self.stopline_detection_flag = True
                return True

        cv2.imshow('stopline', img)
        cv2.waitKey(1)
        # print('No STOPLINE.')
        return False


    detect_stoplineB(img_color)
    cv2.waitKey(0)
