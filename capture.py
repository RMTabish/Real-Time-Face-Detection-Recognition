import cv2


#insitalize the webcam 
cap =cv2.VideoCapture(0)
name =input("Please enter your name: ")
#read all the frame 
while True:
    succec, frames=cap.read()

    if succec:
        cv2.imshow("framse",frames)
    
    if cv2.waitKey(1) & 0xFF==ord('q'):
        break

    if cv2.waitKey(1)==ord('c'):
        filename='faces/'+name+'.jpg'
        cv2.imwrite(filename,frames)
        print("image saved")
        break

cap.release()
cv2.destroyAllWindows()