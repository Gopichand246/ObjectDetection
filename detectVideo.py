import cv2
import numpy as np

# Pretrained Network
net=cv2.dnn.readNet("yolov3.weights","yolov3.cfg")

# Making Objects List
classes=[]
with open("coco.names") as f:
    classes=f.read().splitlines()


class VideoCamera(object):
    def __init__(self):   
        #Capturing Video 
        self.video=cv2.VideoCapture(0)

    def __del__(self):
        #Releasing Video
        self.video.release()

    def get_frame(self):
        #Capturing frame-by-frame
        _, img=self.video.read()
        height,width,_=img.shape    

        # Image input to network
        blob=cv2.dnn.blobFromImage(img, 1/255, (320,320), (0,0,0), swapRB=True, crop=False)
        net.setInput(blob)
        output_layers_names=net.getUnconnectedOutLayersNames()
        layerOutputs=net.forward(output_layers_names)

        #Bounding box pedictions and confidence levels
        boxes=[]
        confidences=[]
        class_ids=[]
        for output in layerOutputs:
            for detections in output:
                scores=detections[5:]
                class_id=np.argmax(scores)
                confidence=scores[class_id]
                if confidence > 0.5:
                    center_x=int(detections[0]*width)
                    center_y=int(detections[1]*height)
                    wid=int(detections[2]*width)
                    ht=int(detections[3]*height)

                    x=int(center_x-wid/2)
                    y=int(center_y-ht/2)

                    boxes.append([x, y, wid, ht])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        #   Non Max Suppression
        indexes= cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

        #   Drawing bounding boxes
        font=cv2.FONT_HERSHEY_PLAIN
        colors= np.random.uniform(0,255,size=(len(boxes), 3))

        #print("Number of objects detected = " + str(len(indexes.flatten())))
        for i in indexes.flatten():
            x, y, wid, ht= boxes[i]
            label= str(classes[class_ids[i]])
            confidence= str(round(confidences[i],2))
            #print(" Object detected: "+ label +" With prob = "+confidence)
            color=colors[i]
            cv2.rectangle(img, (x,y), (x+wid, y+ht), color, 2)
            cv2.putText(img, label + " "+ confidence, (x,y+20), font, 2, (0,0,0), 2)
        
        # encode OpenCV raw frame to jpg and displaying it
        ret, jpeg = cv2.imencode('.jpg', img)
        return jpeg.tobytes()
    