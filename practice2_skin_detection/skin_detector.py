'''
皮肤检测：利用皮肤颜色的HSV范围，去掉其他的区域，实现皮肤检测
'''
import numpy as np
import argparse
import cv2
 
# 定义皮肤的范围，具体自己调整
lower = np.array([0, 50, 100], dtype = "uint8")
upper = np.array([25, 255, 255], dtype = "uint8")

camera = cv2.VideoCapture(0) # 开启摄像头


while True:
	
	(grabbed, frame) = camera.read() #grabbed为布尔类型，true表示获取到帧数

	frame = cv2.resize(frame, (400,400))
	# 将BGR格式转为HSV颜色空间
	# HSV在用于指定颜色分割时，有比较大的作用
	hsvImage = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV) 
   
	# 创建掩码，q低于lower或高于upper的，设置为0，在范围内设置为255
	# 这样便可以得到皮肤的像素范围
	skinMask = cv2.inRange(hsvImage, lower, upper)

	# 将掩码和原图进行“与”运算，这样原图便只保留皮肤区域
	skin = cv2.bitwise_and(frame, frame, mask = skinMask)
	
	# np.hstack是将原图和检测的图捆绑显示（将矩阵横向并行）
	cv2.imshow("images", np.hstack([frame,hsvImage, skin]))

	if cv2.waitKey(1) & 0xFF == ord("q"):
		break

# 释放资源
camera.release()
cv2.destroyAllWindows()


