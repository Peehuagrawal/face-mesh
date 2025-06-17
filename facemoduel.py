import cv2
import mediapipe as mp
import time

class FaceDetector():
    def __init__(self, minDetectionCon=0.5):
        self.minDetectionCon=minDetectionCon
        self.mpFace= mp.solutions.face_detection
        self.mpDraw= mp.solutions.drawing_utils
        self.faceDetection= self.mpFace.FaceDetection(0.75)

    def findFaces(self,img,draw=True):

        imgRGB= cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results= self.faceDetection.process(imgRGB)
        #print(results)
        bboxs=[]
        if self.results.detections:
            for id, detection in enumerate(self.results.detections):
                self.mpDraw.draw_detection(img, detection)

                #print(id,detection)
                #print(detection.score)
                #print(detection.location_data.relative_bounding_box)

                bboxC=detection.location_data.relative_bounding_box
                ih,iw,ic= img.shape
                bbox= int(bboxC.xmin *iw), int(bboxC.ymin*ih), \
                int(bboxC.width*iw), int(bboxC.height*ih)

                bboxs.append((id, bbox, detection.score))
                img=self.fancyDraw(img,bbox)
                cv2.rectangle(img, bbox,(255,0,255),2)
                cv2.putText(img, f'{int(detection.score[0]*100)}%', (bbox[0],bbox[1]-20), cv2.FONT_HERSHEY_PLAIN,3,(255,0,0),2)

        return img, bboxs
    
    def fancyDraw(self,img,bbox,t=10):
        x,y,w,h= bbox
        x1,y1= x+w, y+h

        cv2.line(img,(x,y),(x+3,y),(255,0,255),t)
        cv2.line(img,(x,y),(x,y+3),(255,0,255),t)
        cv2.line(img,(x1,y),(x1-3,y),(255,0,255),t)
        cv2.line(img,(x1,y),(x1,y-3),(255,0,255),t)
        cv2.line(img,(x1,y1),(x1+3,y1),(255,0,255),t)
        cv2.line(img,(x1,y1),(x1,y1+3),(255,0,255),t)
        cv2.line(img,(x,y1),(x-3,y1),(255,0,255),t)
        cv2.line(img,(x,y1),(x,y1-3),(255,0,255),t)
        return img
   
def main():
    cap= cv2.VideoCapture("facevideos/5.mp4")
    pTime=0
    detector= FaceDetector()
    while True:
        success, img = cap.read()
        img, bboxs= detector.findFaces(img)

        cTime= time.time()
        fps= 1/(cTime-pTime)
        pTime = cTime
        cv2.putText(img, f'FPS:{int(fps)}', (28,78), cv2.FONT_HERSHEY_PLAIN,3,(255,0,0),2)
        img = cv2.resize(img, (1280, 720))  # or (960, 540) for smaller display
        cv2.imshow("Image", img)
        key = cv2.waitKey(20)
        if key == ord('q'):
            break



if __name__=="__main__":
    main()