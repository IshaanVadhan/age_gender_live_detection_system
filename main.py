import cv2

def faceBox(faceNet, frame):
    frameHeight = frame.shape[0]
    frameWidth = frame.shape[1]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300,300), [104,117,123], swapRB=False)
    faceNet.setInput(blob)
    detection = faceNet.forward()
    bounding_boxes = []
    for i in range(detection.shape[2]):
        confidence = detection[0,0,i,2]
        if confidence > 0.7:
            x1 = int(detection[0,0,i,3] * frameWidth)
            y1 = int(detection[0,0,i,4] * frameHeight)
            x2 = int(detection[0,0,i,5] * frameWidth)
            y2 = int(detection[0,0,i,6] * frameHeight)
            bounding_boxes.append([x1, y1, x2, y2])
            cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 1)
    return frame, bounding_boxes

faceProto = "opencv_face_detector.pbtxt"
faceModel = "opencv_face_detector_uint8.pb"

ageProto = "age_deploy.prototxt"
ageModel = "age_net.caffemodel"

genderProto = "gender_deploy.prototxt"
genderModel = "gender_net.caffemodel"

faceNet = cv2.dnn.readNet(faceModel, faceProto)
ageNet = cv2.dnn.readNet(ageModel,ageProto)
genderNet = cv2.dnn.readNet(genderModel,genderProto)

MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
genderList = ['Male', 'Female']

video = cv2.VideoCapture(0)

padding = 20

# Initialize variables for smoothing
smoothing_window_size = 10
gender_preds_buffer = []
age_preds_buffer = []

while True:
    ret, frame = video.read()
    frame, bboxs = faceBox(faceNet, frame)

    for bbox in bboxs:
        face = frame[max(0,bbox[1]-padding):min(bbox[3]+padding,frame.shape[0]-1), max(0,bbox[0]-padding):min(bbox[2]+padding, frame.shape[1]-1)]
        blob_face = cv2.dnn.blobFromImage(face, 1.0, (227,227), MODEL_MEAN_VALUES, swapRB=False)

        # Get gender prediction
        genderNet.setInput(blob_face)
        genderPred = genderNet.forward()
        gender = genderList[genderPred[0].argmax()]
        gender_preds_buffer.append(gender)

        # Get age prediction
        ageNet.setInput(blob_face)
        agePred = ageNet.forward()
        age = ageList[agePred[0].argmax()]
        age_preds_buffer.append(age)

        # Apply temporal smoothing
        if len(gender_preds_buffer) > smoothing_window_size:
            gender_preds_buffer.pop(0)
            age_preds_buffer.pop(0)

        # Get smoothed predictions
        smoothed_gender = max(set(gender_preds_buffer), key = gender_preds_buffer.count)
        smoothed_age = max(set(age_preds_buffer), key = age_preds_buffer.count)

        label = "{},{}".format(smoothed_gender, smoothed_age)
        cv2.rectangle(frame, (bbox[0], bbox[1]-30), (bbox[2], bbox[1]), (0,255,0), -1) 
        cv2.putText(frame, label, (bbox[0], bbox[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2, cv2.LINE_AA)

    cv2.imshow("Age-Gender", frame)
    k = cv2.waitKey(1)
    if k == ord('q'):
        break

video.release()
cv2.destroyAllWindows()
