import cv2
import numpy as np
def imgClip(forOrb,cx,cy):
	clippedImage=forOrb[cy-100:cy+100,cx-100:cx+100]
	return clippedImage


def initOrb(clipForOrb):
	referenceImage = cv2.imread("reference.jpeg")
	orb = cv2.ORB_create(400)
	#sift = cv2.xfeatures2d.SIFT_create()
	kpRef,desRef=orb.detectAndCompute(referenceImage,None)
	kpAct,desAct=orb.detectAndCompute(clipForOrb,None)
	
	bf = cv2.BFMatcher()
	matches = bf.knnMatch(desRef,desAct, k=2)

	good = []
	for m,n in matches:
		if m.distance < 0.94*n.distance:    	
			good.append([m])        	
	print len(good)
	img3 = cv2.drawMatchesKnn(referenceImage,kpRef,clipForOrb,kpAct,good,None)    
	return img3    



	


frame = cv2.imread("Test  99.jpeg")
b,g,r = cv2.split(frame) 														#Splitting channels for clahe
forOrb=frame
clahe = cv2.createCLAHE(clipLimit=0.1, tileGridSize=(8,8))						# Applying Clahe filter
b1 = clahe.apply(b)
b2 = clahe.apply(g)
b3 = clahe.apply(r)

#frame = cv2.merge((b1,b2,b3))													#MAIN IMAGE
frame=cv2.medianBlur(frame,5)
hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)  									#HSV IMAGE FOR PROCESSING THE BLUE COLOR
lower_blue = np.array([82,80,120])												#LOWER BLUE COLOR RANGE 
upper_blue = np.array([140,250,250])		 									#HIGHER BLUE COLOR RANGE
in_range = cv2.inRange(hsv, lower_blue, upper_blue)  							#FINDING THE BLUE MASK
img,cnts,hi = cv2.findContours(in_range,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)	#FINDING IF BLUE COLOR IS PRESENT OR NOT

maxArea=0 																		#FOR FINDING THE MAXIMUM AREA CONTOUR
index=0

if len(cnts)==0:
	pass
else:
	for i in range(len(cnts)):
			a = cv2.contourArea(cnts[i])
			if a >maxArea:
				cnt = cnts[i]

	cv2.drawContours(frame,cnt,-1,(0,255,0),3)
	M = cv2.moments(cnt)	
	cx = int(M['m10']/M['m00'])
	cy = int(M['m01']/M['m00'])
	cv2.circle(frame,(cx,cy),3,(0,0,255),-1)									#DRAWING THE CENTROID 
	clipForOrb=imgClip(forOrb,cx,cy)											#CLIPPING THE IMAGE FOR ORB FEATURE MATCHING
	final=initOrb(clipForOrb)															#INITIALIZING THE ORB FUNCTION
	
	cv2.imshow("clippedImage",clipForOrb)
	cv2.imshow("Final_matched",final)
	cv2.imshow("inRange",in_range)
	cv2.imshow("ImageWithCentroid",frame)


cv2.waitKey(0)
cv2.destroyAllWindows()



