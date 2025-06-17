import cv2
import mediapipe as mp
import time

cap= cv2.VideoCapture("videos/4.mp4")
pTime=0

mpFace= mp.solutions.face_mesh
mpDraw= mp.solutions.drawing_utils
facemesh= mpFace.FaceMesh(max_num_faces=2)
drawSpec= mpDraw.DrawingSpec(thickness=2,circle_radius=5)
while True:
    success, img= cap.read()
    
    imgRGB= cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results= facemesh.process(imgRGB)
    #print(results)

    if results.multi_face_landmarks:
        for faceLms in results.multi_face_landmarks:
            mpDraw.draw_landmarks(img, faceLms, mpFace.FACEMESH_CONTOURS, drawSpec,drawSpec)
            ih,iw,ic= img.shape
            

    cTime= time.time()
    fps= 1/(cTime-pTime)
    pTime = cTime
    cv2.putText(img, f'FPS:{int(fps)}', (28,78), cv2.FONT_HERSHEY_PLAIN,3,(255,0,0),2)
    img = cv2.resize(img, (1280, 720))  # or (960, 540) for smaller display
    cv2.imshow("Image", img)
    key = cv2.waitKey(20)
    if key == ord('q'):
        break
