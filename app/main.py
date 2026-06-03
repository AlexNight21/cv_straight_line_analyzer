import cv2
import numpy as np
import config as cfg

from processor import Processor


# set params
window_adapt_coef = 1.2

processor = Processor(img_path=cfg.img_path)


def put_image_info(image, straight_len):
    cv2.putText(
        image, 
        f"straight length: {straight_len:.2f} px",
        (15, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
        )


def draw_points(image, A_pt, B_pt, straight_section):
    cv2.circle(image, tuple(A_pt), 8, (0, 0, 255), -1)
    cv2.putText(image, "A", (A_pt[0] + 15, A_pt[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    
    cv2.circle(image, tuple(B_pt), 8, (0, 0, 255), -1)
    cv2.putText(image, "B", (B_pt[0] + 15, B_pt[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    
    cv2.polylines(image, [straight_section], False, (0, 0, 255), 3)

def show_result_image(img_path, cnt_info_dict):
    
    image = cv2.imdecode(
        np.fromfile(img_path, dtype=np.uint8),
        cv2.IMREAD_COLOR
    )
    
    for cnt_idx, cnt_info in cnt_info_dict.items():
        straight_length = cnt_info["straight_length"]
        contour = cnt_info["contour"]
        A_pt = cnt_info["A_pt"]
        B_pt = cnt_info["B_pt"]
        straight_section = cnt_info["straight_section"]
    
        cv2.drawContours(image, [contour], -1, (0, 255, 0), 3)
        draw_points(image, A_pt, B_pt, straight_section)
    
        put_image_info(image, straight_length)
    
    cv2.namedWindow("result", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("result", int(image.shape[1] / window_adapt_coef), int(image.shape[0] / window_adapt_coef))

    cv2.imshow("result", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def find_max_straight_edge_length(
    curvature_threshold_px: float
) -> dict:

    contours_lst = processor.get_contours()
    
    if contours_lst is None:
        return 0.0
    
    cnt_info_dict = {}
    
    for cnt_idx, contour in enumerate(contours_lst):
        cnt_info_dict[cnt_idx] = {
            "straight_length": 0.0,
            "contour": contour,
            "A_pt": None,
            "B_pt": None,
            "straight_section": None,
        }
        
        # contour processing
        Pnts = contour.reshape(-1, 2)
        
        num_pts = len(Pnts)
        if num_pts < 2:
            continue
        
        # expand contour to handle wrap-around
        Pnts_exp = np.concatenate((Pnts, Pnts), axis=0)
        
        straight_length = 0.0
        
        for i in range(num_pts):
            for j in range(i+1, i+num_pts):
                A_pt = Pnts_exp[i]
                B_pt = Pnts_exp[j]
                
                dx = B_pt[0] - A_pt[0]
                dy = B_pt[1] - A_pt[1]
                AB_len = (dx**2 + dy**2)**0.5
                
                if AB_len == 0 or AB_len <= straight_length:
                    continue
                
                if j == i + 1:
                    straight_length = AB_len
                    continue
                
                straight_flag = True
                for k in range(i+1, j):
                    C_pt = Pnts_exp[k]
                    
                    sq_par = abs(dx*(C_pt[1] - A_pt[1]) - dy * (C_pt[0] - A_pt[0]))
                    dist = sq_par / AB_len
                    
                    if dist > curvature_threshold_px:
                        straight_flag = False
                        break
                    
                if straight_flag:
                    # straight_section.append(B_pt)
                    straight_length = AB_len
                    
                    cnt_info_dict[cnt_idx]["straight_length"] = straight_length
                    cnt_info_dict[cnt_idx]["A_pt"] = A_pt
                    cnt_info_dict[cnt_idx]["B_pt"] = B_pt
                    cnt_info_dict[cnt_idx]["straight_section"] = Pnts_exp[i:j+1]
        
        print(f"Contour {cnt_idx}:")
        print(f"  Straight length: {cnt_info_dict[cnt_idx]['straight_length']:.2f} px")
        print(len(cnt_info_dict[cnt_idx]["contour"]))
        print(len(cnt_info_dict[cnt_idx]["straight_section"]))

    return cnt_info_dict            
                    

if __name__ == "__main__":
    cnt_info_dict = find_max_straight_edge_length(curvature_threshold_px=10.0)
    show_result_image(cfg.img_path, cnt_info_dict)
    