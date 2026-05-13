from tkinter import messagebox
from tkinter import *
from tkinter import simpledialog
import tkinter
from tkinter import filedialog
from tkinter.filedialog import askopenfilename
import imutils
import time
import cv2
import matplotlib.pyplot as plt
import numpy as np


main = tkinter.Tk()
main.title("Object Tracking Using Python")
main.geometry("1300x1200")

labels = open('coco.names').read().strip().split('\n')
weights_path = 'yolov3.weights'
configuration_path = 'yolov3.cfg'
probability_minimum = 0.6
network = cv2.dnn.readNetFromDarknet(configuration_path, weights_path)
net = cv2.dnn.readNetFromCaffe("MobileNetSSD_deploy.prototxt.txt","MobileNetSSD_deploy.caffemodel")
    

global filename
global train
global ga_acc, bat_acc, bee_acc
global classifier

CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
	"bottle", "bus", "car", "cat", "chair", "cow", "diningtable","cup"
	"dog", "horse", "motorbike", "person", "pottedplant", "sheep",
	"sofa", "train", "tvmonitor","mobile","bicycle","car","motorbike","bus","truck", "train",
    "person","bicycle","car","motorbike","aeroplane","bus","train","truck","boat","traffic light","fire hydrant","stop sign","parking meter","bench","bird","cat",
    "dog","horse","sheep","cow","elephant","bear","zebra","giraffe","backpack","umbrella","handbag","tie","suitcase","frisbee","skis","snowboard","sports ball","kite",
    "baseball bat","baseball glove","skateboard","surfboard","tennis racket","bottle","wine glass","cup","fork","knife","spoon","bowl","banana","apple","sandwich","orange",
    "broccoli","carrot","hot dog","pizza","donut","cake","chair","sofa","pottedplant","bed","diningtable","toilet","tvmonitor","laptop","mouse","remote",
    "keyboard","cell phone","microwave","oven","toaster","sink","refrigerator","book","clock","vase","scissors","teddy bear","hair drier","toothbrush"]
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))


def imagesVideo():
    print("image")
    global filename
    filename = filedialog.askopenfilename(initialdir="images")
    pathlabel.config(text=filename)
    text.delete('1.0', END)
    text.insert(END,filename+" loaded\n");
    image = cv2.imread(filename)
    frame = imutils.resize(image, width=500)
    (h, w) = frame.shape[:2]
    threshold = 0.3
    layers_names_all = network.getLayerNames()
    classess=[]
    with open("coco.names",'r') as f:
        classess=f.read().splitlines()
    image_input1 = cv2.imread(filename)
    image_input = imutils.resize(image_input1, width=500)
    image_input_shape = image_input.shape
    # print(image_input_shape)
    plt.rcParams['figure.figsize'] = (10.0, 10.0)
    plt.imshow(cv2.cvtColor(image_input, cv2.COLOR_BGR2RGB))
    plt.show()
    # cv2.imshow(cv2.cvtColor(image_input, cv2.COLOR_BGR2RGB))
    blob = cv2.dnn.blobFromImage(image_input, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    height, width = image_input_shape[:2]
    network.setInput(blob)
    output_layers_name=network.getUnconnectedOutLayersNames() 
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
        
        label=str(classess[class_ids[i]])
        confi=str(round(confidences[i],2))
        color=colors[i]
        
        cv2.rectangle(image_input,(x,y),(x+w,y+h),color,5)
        cv2.putText(image_input,label+" "+confi,(x,y+20),font,2,(255,255,255),1)
    # plt.imshow(image_input)
    # plt.show()
    cv2.imshow("Frame", image_input)

def uploadVideo():
    global filename
    filename = filedialog.askopenfilename(initialdir="videos")
    pathlabel.config(text=filename)
    text.delete('1.0', END)
    text.insert(END,filename+" loaded\n");
    vc = cv2.VideoCapture(filename)
    while True:
        frame = vc.read()
        frame = frame if filename is None else frame[1] 
        if frame is None:
            break
        frame = imutils.resize(frame, width=500)
        (h, w) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)),0.007843, (300, 300), 127.5)
        net.setInput(blob)
        detections = net.forward()
        for i in np.arange(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.2:
                idx = int(detections[0, 0, i, 1])
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                if (confidence * 100) > 50:
                    label = "{}: {:.2f}%".format(CLASSES[idx],confidence * 100)
                    cv2.rectangle(frame, (startX, startY), (endX, endY),COLORS[idx], 2)
                    y = startY - 15 if startY - 15 > 15 else startY + 15
                    cv2.putText(frame, label, (startX, y),cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)
                    text.insert(END,"\n ",label," "+" \n")
                    
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
             break
        
    vc.stop() if filename is None else vc.release()
    cv2.destroyAllWindows()
                        


def webcamVideo():
    text.delete('1.0', END)
    webcamera = cv2.VideoCapture(0)
    time.sleep(0.25)
    oldFrame = None
    while True:
        (grab, frame) = webcamera.read()
        if not grab:
            break
        frame = imutils.resize(frame, width=500)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)

        if oldFrame is None:
            oldFrame = gray
            continue
        (h, w) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)),0.007843, (300, 300), 127.5)
        net.setInput(blob)
        detections = net.forward()
        for i in np.arange(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.2:
                idx = int(detections[0, 0, i, 1])
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                if (confidence * 100) > 50:
                    labels = "{}: {:.2f}%".format(CLASSES[idx],confidence * 100)
                    print("lable",labels)
                    print(type(labels))
                    cv2.rectangle(frame, (startX, startY), (endX, endY),COLORS[idx], 2)
                    y = startY - 15 if startY - 15 > 15 else startY + 15
                    cv2.putText(frame,labels, (startX, y),cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)
                    text.insert(END,"\n ",labels," "+"\n")
                    
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
             break
        
    #webcamera.stop()
    webcamera.release()
    cv2.destroyAllWindows()
    
	


def exit():
    main.destroy()

    
font = ('times', 16, 'bold')
title = Label(main, text='Object Tracking Using Python')
title.config(bg='light cyan', fg='pale violet red')  
title.config(font=font)           
title.config(height=3, width=120)       
title.place(x=0,y=5)

font1 = ('times', 14, 'bold')
uploadButton = Button(main, text="Browse System Videos", command=uploadVideo)
uploadButton.place(x=50,y=100)
uploadButton.config(font=font1)  



font1 = ('times', 14, 'bold')
uploadButton = Button(main, text="Browse System Images", command=imagesVideo)
uploadButton.place(x=50,y=150)
uploadButton.config(font=font1)  


pathlabel = Label(main)
pathlabel.config(bg='light cyan', fg='pale violet red')  
pathlabel.config(font=font1)           
pathlabel.place(x=460,y=100)

webcamButton = Button(main, text="Start Webcam Video Tracking", command=webcamVideo)
webcamButton.place(x=50,y=200)
webcamButton.config(font=font1) 

exitButton = Button(main, text="Exit", command=exit)
exitButton.place(x=330,y=200)
exitButton.config(font=font1) 


font1 = ('times', 12, 'bold')
text=Text(main,height=20,width=250)
scroll=Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=10,y=250)
text.config(font=font1)


main.config(bg='snow3')
main.mainloop()
