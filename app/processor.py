import cv2
import numpy as np


class Processor:
    def __init__(self, img_path:str):
        self.img_path = img_path
        self.stripe_width = 10
        self.coef_tol = 3.0
        self.approx_epsilon = 0.5
        self.min_area_size = 10000
    
    @staticmethod
    def _calculate_bg_hue(image:np.ndarray, stripe_width:int, koef_tol:float) -> tuple:
        '''Calculates background hue and tolerance based on the image borders.'''
        
        h, w = image.shape[:2]
        
        l_stripe = image[:, 0:stripe_width].reshape(-1, 3)
        r_stripe = image[:, w-stripe_width:w].reshape(-1, 3)
        up_stripe = image[0:stripe_width, :].reshape(-1, 3)
        bt_stripe = image[h-stripe_width:h, :].reshape(-1, 3)
        
        un_stripe = np.concatenate((l_stripe, r_stripe, up_stripe, bt_stripe), axis=0)
        
        # color statistics
        bg_median = np.median(un_stripe, axis=0)
        bg_std = np.std(un_stripe, axis=0)
        
        h_tol = max(10, koef_tol * bg_std[0])
        s_tol = max(20, koef_tol * bg_std[1])
        v_tol = max(20, koef_tol * bg_std[2])
        
        # thresholds
        tr_lower = np.array([
            max(0, bg_median[0] - h_tol),
            max(30, bg_median[1] - s_tol),
            max(30, bg_median[2] - v_tol)
        ], dtype=np.uint8)

        tr_upper = np.array([
            min(180, bg_median[0] + h_tol),
            min(255, bg_median[1] + s_tol),
            min(255, bg_median[2] + v_tol)
        ], dtype=np.uint8)
        
        return tr_lower, tr_upper
    
    def get_contours(self):
        '''Returns contours of items in the image.'''
        
        image = cv2.imdecode(
            np.fromfile(self.img_path, dtype=np.uint8),
            cv2.IMREAD_COLOR
        )
        
        image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        
        tr_lower, tr_upper = self._calculate_bg_hue(image, self.stripe_width, self.coef_tol)
        
        bg_mask = cv2.inRange(image, tr_lower, tr_upper)
        item_mask = cv2.bitwise_not(bg_mask)
        
        kernelSize = (9, 9)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernelSize)
        
        item_mask = cv2.morphologyEx(
            item_mask, cv2.MORPH_CLOSE, kernel, iterations=1,
        )
        
        contours, _ = cv2.findContours(
            item_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
        )
        
        if len(contours) == 0:
            print("[WARNING] No contours found!")
            return
        
        contours_lst = []
        
        for cnt in contours:
            area = cv2.contourArea(contour=cnt)
            if area < self.min_area_size:
                continue
            
            approx_cnt = cv2.approxPolyDP(cnt, self.approx_epsilon, True)
            approx_cnt = approx_cnt.astype(np.int32)
            
            contours_lst.append(approx_cnt)
            
        if len(contours_lst) == 0:
            print("[WARNING] No contours found after filtering!")
            return
        
        print(f"[INFO] Found {len(contours_lst)} contours after filtering.")
        return contours_lst
                