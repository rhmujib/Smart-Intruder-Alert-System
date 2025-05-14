import cv2

def detect_intruder(frame, proto_path, model_path):
    """
    Detects intruders using the MobileNet-SSD model.
    Returns a list of bounding boxes for all detected intruders.
    """
    net = cv2.dnn.readNetFromCaffe(proto_path, model_path)
    h, w = frame.shape[:2]
    
    # Create a blob from the input frame
    blob = cv2.dnn.blobFromImage(frame, 0.007843, (300, 300), (127.5, 127.5, 127.5), swapRB=True)
    net.setInput(blob)
    detections = net.forward()

    boxes = []  # List to store bounding boxes for all detected intruders

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:  # Increased confidence threshold for better accuracy
            idx = int(detections[0, 0, i, 1])
            if idx == 15:  # Person class index in MobileNet SSD
                box = detections[0, 0, i, 3:7] * [w, h, w, h]
                box = box.astype("int")
                boxes.append(box)

    return boxes