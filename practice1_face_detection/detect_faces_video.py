'''
使用摄像头实现人脸检测
'''
import cv2
import detect_faces_image
 
vs = cv2.VideoCapture(0) # 用笔记本自带摄像头，请选0

while True:
	ret , frame = vs.read()

	resImage = detect_faces_image.face_detector(frame)

	cv2.imshow("Frame", resImage)
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break
 
vs.release()
cv2.destroyAllWindows()
