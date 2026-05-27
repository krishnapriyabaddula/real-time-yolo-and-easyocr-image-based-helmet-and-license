import cv2
import os
from ultralytics import YOLO
import easyocr
from matplotlib import pyplot as plt

# -----------------------------
# LOAD MODELS
# -----------------------------
helmet_model = YOLO(r"C:\Users\DELL\Downloads\3-2 1st stage\helmet_model.pt")
plate_model = YOLO(r"C:\Users\DELL\Downloads\3-2 1st stage\plate_model.pt")

reader = easyocr.Reader(['en'], gpu=False)

# -----------------------------
# IMAGE FOLDER
# -----------------------------
image_folder = r"C:\Users\DELL\Downloads\3-2 1st stage\image"

# Create output folder
os.makedirs("output", exist_ok=True)

# -----------------------------
# PROCESS ALL IMAGES
# -----------------------------
for image_name in os.listdir(image_folder):

    image_path = os.path.join(image_folder, image_name)

    img = cv2.imread(image_path)

    if img is None:
        print("Image not loaded:", image_name)
        continue

    output = img.copy()

    # -----------------------------
    # AUTO TEXT SIZE BASED ON IMAGE
    # -----------------------------
    h, w = output.shape[:2]

    font_scale = min(w, h) / 800
    thickness = max(1, int(font_scale * 2))

    # -----------------------------
    # HELMET DETECTION
    # -----------------------------
    helmet_results = helmet_model.predict(img, conf=0.4)

    helmet_present = False
    no_helmet = False

    for r in helmet_results:
        for box in r.boxes:

            cls_id = int(box.cls[0])
            label = helmet_model.names[cls_id]

            x1,y1,x2,y2 = map(int, box.xyxy[0])

            cv2.rectangle(output,(x1,y1),(x2,y2),(0,255,0),thickness)

            cv2.putText(output,label,(x1,y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        font_scale,
                        (0,255,0),
                        thickness)

            if label.lower() == "helmet":
                helmet_present = True

            if label.lower() == "head":
                no_helmet = True


    # -----------------------------
    # LICENSE PLATE DETECTION
    # -----------------------------
    plate_results = plate_model.predict(img, conf=0.5)

    plate_text = ""

    for r in plate_results:
        for box in r.boxes:

            x1,y1,x2,y2 = map(int,box.xyxy[0])

            cv2.rectangle(output,(x1,y1),(x2,y2),(0,0,255),thickness)

            plate_crop = img[y1:y2, x1:x2]

            if plate_crop.size != 0:

                gray = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)

                result = reader.readtext(gray)

                if len(result) > 0:
                    plate_text = result[0][1]


    # -----------------------------
    # HELMET STATUS BANNER
    # -----------------------------
    if no_helmet:
        status = "NO HELMET"
        color = (0,0,255)

    elif helmet_present:
        status = "HELMET PRESENT"
        color = (0,255,0)

    else:
        status = "NO RIDER DETECTED"
        color = (255,255,0)

    banner_top = int(h * 0.04)
    banner_bottom = int(h * 0.12)

    cv2.rectangle(output,(int(w*0.25),banner_top),(int(w*0.75),banner_bottom),color,-1)

    cv2.putText(output,status,
                (int(w*0.30),int(h*0.09)),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale*1.2,
                (255,255,255),
                thickness+1)

    # -----------------------------
    # PLATE TEXT BANNER
    # -----------------------------
    if plate_text != "":

        cv2.rectangle(output,
                      (int(w*0.15),h-int(h*0.12)),
                      (int(w*0.85),h-int(h*0.03)),
                      (255,255,255),-1)

        cv2.rectangle(output,
                      (int(w*0.15),h-int(h*0.12)),
                      (int(w*0.85),h-int(h*0.03)),
                      (0,0,0),2)

        cv2.putText(output,
                    "Plate: " + plate_text,
                    (int(w*0.20),h-int(h*0.06)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale*1.2,
                    (0,0,0),
                    thickness+1)

    # -----------------------------
    # SAVE OUTPUT IMAGE
    # -----------------------------
    save_path = os.path.join("output", image_name)

    cv2.imwrite(save_path, output)

    # -----------------------------
    # SHOW OUTPUT IMAGE
    # -----------------------------
    plt.figure(figsize=(10,7))
    plt.imshow(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))
    plt.title("Helmet + Number Plate Detection Result")
    plt.axis("off")
    plt.show()