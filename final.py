import cv2
import numpy as np

def checkforobstacle(image):
    count0,count1=0,0
    for i in range(image.shape[0]):
        for j in range(56,image.shape[1]):
            count0+=1
            if (image[i,j][2]>=128):
                count1+=1
    if((count1/count0)>=0.1):
        return True
    else:
        return False
                
            
left = cv2.VideoCapture(0)
left.set(3, 320);
left.set(4, 200);
right = cv2.VideoCapture(1)
right.set(3, 320);
right.set(4, 200);


# SGBM Parameters -----------------
window_size = 3                     # wsize default 3; 5; 7 for SGBM reduced size image; 15 for SGBM full size image (1300px and above); 5 Works nicely
left_matcher = cv2.StereoSGBM_create(
    minDisparity=8,
    numDisparities=48,             # max_disp has to be dividable by 16 f. E. HH 192, 256
    blockSize=5,
    P1=8 * 3 * window_size ** 2,    # wsize default 3; 5; 7 for SGBM reduced size image; 15 for SGBM full size image (1300px and above); 5 Works nicely
    P2=32 * 3 * window_size ** 2,
    disp12MaxDiff=1,
    uniquenessRatio=15,
    speckleWindowSize=0,
    speckleRange=2,
    preFilterCap=63,
    mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
)

right_matcher = cv2.ximgproc.createRightMatcher(left_matcher)

# FILTER Parameters
lmbda = 80000
sigma = 1.2
visual_multiplier = 1.0

wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left=left_matcher)
wls_filter.setLambda(lmbda)
wls_filter.setSigmaColor(sigma)


while(True):
    if not (left.grab() and right.grab()):
        print("No more frames")
        break
    _, leftFrame = left.read()
    _, rightFrame = right.read()
    cv2.imshow('leftFrame', leftFrame)
    cv2.imshow('rightFrame', rightFrame)

    displ = left_matcher.compute(leftFrame, rightFrame)  #.astype(np.float32)/16
    dispr = right_matcher.compute(rightFrame, leftFrame) # .astype(np.float32)/16
    displ = np.int16(displ)
    dispr = np.int16(dispr)
    filteredImg = wls_filter.filter(displ, leftFrame, None, dispr)  # important to put "imgL" here!!!

    filteredImg = cv2.normalize(src=filteredImg, dst=filteredImg, beta=0, alpha=255, norm_type=cv2.NORM_MINMAX);
    filteredImg = np.uint8(filteredImg)
    filteredImg=cv2.applyColorMap(filteredImg, cv2.COLORMAP_JET)
    cv2.imshow('Disparity Map', filteredImg)
    if(checkforobstacle(filteredImg)):
        print("Obstacle detected")
    else:
        print("Obstacle not detected")
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cv2.destroyAllWindows()
