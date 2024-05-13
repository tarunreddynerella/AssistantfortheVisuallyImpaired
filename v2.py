import numpy as np
import time
import cv2
import pyttsx3

engine = pyttsx3.init('sapi5')
voices = engine.getProperty('voices')

engine.setProperty('voice', voices[1].id)

def speak(audio):
    engine.say(audio)
    engine.runAndWait()

Names = open("coco.names").read().strip().split("\n")
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]


print("loading YOLO ...")
network = cv2.dnn.readNetFromDarknet("yolov3.cfg", "yolov3.weights")
font = cv2.FONT_HERSHEY_DUPLEX

l = network.getLayerNames()
l = [l[i[0] - 1] for i in network.getUnconnectedOutLayers()]


img = cv2.VideoCapture(0)

ins_count = 0

inss = []
flag=1

while True:
	ins_count += 1

	ret, ins = img.read()
	cv2.imshow("aj", ins)
	inss.append(ins)


	if cv2.waitKey(25) & 0xFF == ord('w'):
		break
	if ret:
		if ins_count %60 == 0:


			(boxheight, boxwidth) = ins.shape[:2]
			instance = cv2.dnn.blobFromImage(ins, 1 / 255.0, (416, 416),
				swapRB=True, crop=False)
			network.setInput(instance)
			Outputlayers = network.forward(l)

			boundingboxes = []
			probabilities = []
			objectnums = []
			midpoints = []


			for output in Outputlayers:

				for foundobject in output:
					scores = foundobject[5:]
					objectnum = np.argmax(scores)
					probability = scores[objectnum]

					if probability > 0.6:
						box = foundobject[0:4] * np.array([boxwidth,boxheight, boxwidth,boxheight])
						(midpointX, midpointY, w, h) = box.astype("int")

						cordinate_x = int(midpointX - (w / 2))
						cordinate_y = int(midpointY - (h / 2))

						boundingboxes.append([cordinate_x, cordinate_y, int(w), int(h)])
						probabilities.append(float(probability))
						objectnums.append(objectnum)
						midpoints.append((midpointX, midpointY))

			IDS = cv2.dnn.NMSBoxes(boundingboxes, probabilities, 0.5, 0.3)





			sttr = "Try changing orientation"


			if len(IDS) > 0:
				sttr = "The environment has following objects"

				for i in IDS.flatten():

					midpointX, midpointY = midpoints[i][0], midpoints[i][1]
					
					if midpointX <= boxwidth/3:
						finalhorizontal_pos = " left "
					elif midpointX <= (boxwidth/3 * 2):
						finalhorizontal_pos = " center "
					else:
						finalhorizontal_pos = " right "
					
					if midpointY <= boxheight/3:
						finalvertical_pos = " top "
					elif midpointY <= (boxheight/3 * 2):
						finalvertical_pos = " mid "
					else:
						finalvertical_pos = " bottom "

					sttr += finalvertical_pos + finalhorizontal_pos + Names[objectnums[i]]
					flag=0

			print(sttr)
			
			if (flag==0):
				speak(sttr)


img.release()
cv2.destroyAllWindows()