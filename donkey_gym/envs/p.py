import cv2
import numpy as np  
def nothing(x):
    pass


cv2.namedWindow("Tracking")
cv2.createTrackbar("LHW", "Tracking", 0, 255, nothing)
cv2.createTrackbar("LSW", "Tracking", 0, 255, nothing)
cv2.createTrackbar("LVW", "Tracking", 0, 255, nothing)
cv2.createTrackbar("UHW", "Tracking", 255, 255, nothing)
cv2.createTrackbar("USW", "Tracking", 255, 255, nothing)
cv2.createTrackbar("UVW", "Tracking", 255, 255, nothing)

cv2.createTrackbar("LHY", "Tracking", 0, 255, nothing)
cv2.createTrackbar("LSY", "Tracking", 0, 255, nothing)
cv2.createTrackbar("LVY", "Tracking", 0, 255, nothing)
cv2.createTrackbar("UHY", "Tracking", 255, 255, nothing)
cv2.createTrackbar("USY", "Tracking", 255, 255, nothing)
cv2.createTrackbar("UVY", "Tracking", 255, 255, nothing)
def filter_region(image, vertices):
    """
    Create the mask using the vertices and apply it to the input image
    """
    mask = np.zeros_like(image)
    if len(mask.shape)==2:
        cv2.fillPoly(mask, vertices, 255)
    else:
        cv2.fillPoly(mask, vertices, (255,)*mask.shape[2]) # in case, the input image has a channel dimension        
    return cv2.bitwise_and(image, mask)

    
def select_region(image):
    """
    It keeps the region surrounded by the `vertices` (i.e. polygon).  Other area is set to 0 (black).
    """
    # first, define the polygon by vertices
    rows, cols = image.shape[:2]
    bottom_left  = [cols*.00000001, rows*0.95]
    top_left     = [cols*0.1, rows*0.6]
    bottom_right = [cols*0.9, rows*0.95]
    top_right    = [cols*0.9, rows*0.2] 
    # the vertices are an array of polygons (i.e array of arrays) and the data type must be integer
    vertices = np.array([[bottom_left, top_left, top_right, bottom_right]], dtype=np.int32)
    return vertices


# images showing the region of interest only

def select_rgb_white_yellow(image):
    l_h_w = cv2.getTrackbarPos("LHW", "Tracking")
    l_s_w = cv2.getTrackbarPos("LSW", "Tracking")
    l_v_w = cv2.getTrackbarPos("LVW", "Tracking")
   
    u_h_w = cv2.getTrackbarPos("UHW", "Tracking")
    u_s_w = cv2.getTrackbarPos("USW", "Tracking")
    u_v_w = cv2.getTrackbarPos("UVW", "Tracking")


    l_h_y = cv2.getTrackbarPos("LHY", "Tracking")
    l_s_y= cv2.getTrackbarPos("LSY", "Tracking")
    l_v_y= cv2.getTrackbarPos("LVY", "Tracking")
   
    u_h_y= cv2.getTrackbarPos("UHY", "Tracking")
    u_s_y= cv2.getTrackbarPos("USY", "Tracking")
    u_v_y= cv2.getTrackbarPos("UVY", "Tracking")

    
    
    lower1 = np.uint8([0, 0, 0])
    upper1 = np.uint8([255, 255, 255])

    #lower = np.uint8([200, 200, 200])
    #upper = np.uint8([255, 255, 255])
    white_mask = cv2.inRange(image, lower1, upper1)
    # yellow color mask
    lower2 = np.uint8([0, 0, 0])
    upper2 = np.uint8([255, 255, 255])

    #lower = np.uint8([190, 190,   0])
    #upper = np.uint8([255, 255, 255])

    yellow_mask = cv2.inRange(image, lower2, upper2)
    # combine the mask
    mask = cv2.bitwise_or(white_mask, yellow_mask)
    masked = cv2.bitwise_and(image, image, mask = mask)
    #masked=cv2.GaussianBlur(masked, (15, 15), 0)

    masked=cv2.Canny(masked[:,:], 200, 300)

    return masked

'''
while(1):
    
    frame = cv2.imread('50.jpg')
    res=select_rgb_white_yellow(frame)

    print(res.shape)
    #res = select_region(frame)
    cv2.imshow("frame", frame)
    
    cv2.imshow("res", res)

    key = cv2.waitKey(1)

    if(key == 27):
        break
'''

def segment(image):

    print(image)
	frame = cv2.imread(image)
    res=select_rgb_white_yellow(frame)

    return res