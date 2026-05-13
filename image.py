import numpy as np
import cv2
import matplotlib.pyplot as plt
import time
labels = open('coco.names').read().strip().split('\n')  # list of names

# # Check point
print(labels)
weights_path = 'yolov3.weights'
configuration_path = 'yolov3.cfg'
probability_minimum = 0.6
network = cv2.dnn.readNetFromDarknet(configuration_path, weights_path)
# Setting threshold for non maximum suppression
threshold = 0.3
layers_names_all = network.getLayerNames() 
print(layers_names_all) # list of layers' names
print(labels)
classes=[]

with open("coco.names",'r') as f:
    classes=f.read().splitlines()
print(classes)

image_input = cv2.imread('aa.png')


# Getting image shape
image_input_shape = image_input.shape

# Check point
print(image_input_shape)  # tuple of (917, 1222, 3)

plt.rcParams['figure.figsize'] = (10.0, 10.0)
plt.imshow(cv2.cvtColor(image_input, cv2.COLOR_BGR2RGB))
plt.show()
blob = cv2.dnn.blobFromImage(image_input, 1 / 255.0, (416, 416), swapRB=True, crop=False)
height, width = image_input_shape[:2]
print(height, width)
print(image_input.shape)  # (806, 1084, 3)
print(blob.shape)  # (1, 3, 416, 416)
network.setInput(blob)
output_layers_name=network.getUnconnectedOutLayersNames()
#output_layers_name
layeroutput=network.forward(output_layers_name)
boxes=[]
confidences=[]
class_ids=[]
for output in layeroutput:
    for detection in output:
        score=detection[5:]
        class_id=np.argmax(score)
        confidence=score[class_id]
        if confidence>0.5:
            center_x=int(detection[0]*width)
            center_y=int(detection[1]*height)
            w=int(detection[2]*width)
            h=int(detection[3]*height)
        
            x=int(center_x-w/2)
            y=int(center_y-h/2)

            boxes.append([x,y,w,h])
            confidences.append(float(confidence))
            class_ids.append(class_id)

len(boxes)
indexes=cv2.dnn.NMSBoxes(boxes,confidences,0.5,0.4)
font=cv2.FONT_HERSHEY_PLAIN
colors=np.random.uniform(0,255,size=(len(boxes),3))
for i in indexes.flatten():
    x,y,w,h=boxes[i]
    
    label=str(classes[class_ids[i]])
    confi=str(round(confidences[i],2))
    color=colors[i]
    
    cv2.rectangle(image_input,(x,y),(x+w,y+h),color,5)
    cv2.putText(image_input,label+" "+confi,(x,y+20),font,2,(255,255,255),1)
plt.imshow(image_input)
plt.show()
