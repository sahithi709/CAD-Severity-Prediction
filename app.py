import streamlit as st
import json
import os
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

st.set_page_config(page_title="CAD Severity Prediction", layout="wide")
st.title("🫀 CAD Severity Prediction from Angiography")

# -----------------------------
# PATHS (CHANGE IF NEEDED)
# -----------------------------
BASE_PATH = "arcade/stenosis"
TEST_IMG_PATH = os.path.join(BASE_PATH, "test", "images")
TEST_JSON_PATH = os.path.join(BASE_PATH, "test", "annotations", "test.json")

# -----------------------------
# LOAD JSON ANNOTATIONS
# -----------------------------
with open(TEST_JSON_PATH, "r") as f:
    coco_data = json.load(f)

images_info = coco_data["images"]
annotations = coco_data["annotations"]

id_to_filename = {img["id"]: img["file_name"] for img in images_info}

# only stenosis annotations
stenosis_anns = [a for a in annotations if a["category_id"] == 26]

# -----------------------------
# COMPUTE THRESHOLDS (same as notebook)
# -----------------------------
stenosis_records = []

for ann in stenosis_anns:

    img_id = ann["image_id"]
    img_path = os.path.join(TEST_IMG_PATH, id_to_filename[img_id])

    image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    gt_mask = np.zeros(image.shape, dtype=np.uint8)

    for poly in ann["segmentation"]:
        pts = np.array(poly).reshape(-1,2).astype(np.int32)
        cv2.fillPoly(gt_mask,[pts],1)

    gt_mask_resized = cv2.resize(gt_mask,(256,256),interpolation=cv2.INTER_NEAREST)

    stenosis_area = np.sum(gt_mask_resized==1)

    stenosis_records.append(stenosis_area)

stenosis_records = np.array(stenosis_records)

low_thr = np.percentile(stenosis_records,33)
high_thr = np.percentile(stenosis_records,66)

# -----------------------------
# SEVERITY FUNCTION
# -----------------------------
def classify_severity_by_distribution(stenosis_area):

    if stenosis_area < low_thr:
        return "Mild"
    elif stenosis_area < high_thr:
        return "Moderate"
    else:
        return "Severe"


# -----------------------------
# IMAGE UPLOAD
# -----------------------------
uploaded_file = st.file_uploader("Upload Angiography Image", type=["png","jpg","jpeg"])

if uploaded_file is not None:

    uploaded_image = Image.open(uploaded_file).convert("L")
    uploaded_np = np.array(uploaded_image)

    st.image(uploaded_np, caption="Uploaded Image", width=400)

    # find matching image in dataset
    match_found = False

    for ann in stenosis_anns:

        img_id = ann["image_id"]
        filename = id_to_filename[img_id]

        dataset_img = cv2.imread(os.path.join(TEST_IMG_PATH, filename), cv2.IMREAD_GRAYSCALE)

        if dataset_img.shape == uploaded_np.shape:

            # compare image similarity
            if np.mean(np.abs(dataset_img - uploaded_np)) < 5:

                match_found = True

                image = dataset_img

                gt_mask = np.zeros(image.shape, dtype=np.uint8)

                for poly in ann["segmentation"]:
                    pts = np.array(poly).reshape(-1,2).astype(np.int32)
                    cv2.fillPoly(gt_mask,[pts],1)

                gt_mask_resized = cv2.resize(gt_mask,(256,256),interpolation=cv2.INTER_NEAREST)

                stenosis_area = np.sum(gt_mask_resized==1)

                severity = classify_severity_by_distribution(stenosis_area)

                image_area = gt_mask_resized.shape[0]*gt_mask_resized.shape[1]

                stenosis_percent = (stenosis_area/image_area)*300
                stenosis_percent = np.clip(stenosis_percent,0,100)

                image_resized = cv2.resize(image,(256,256))/255.0

                # visualization
                fig,ax = plt.subplots(1,3,figsize=(15,5))

                ax[0].imshow(image_resized,cmap="gray")
                ax[0].set_title("Original Image")
                ax[0].axis("off")

                ax[1].imshow(gt_mask_resized,cmap="gray")
                ax[1].set_title("Stenosis Mask")
                ax[1].axis("off")

                ax[2].imshow(image_resized,cmap="gray")
                ax[2].imshow(gt_mask_resized,alpha=0.5,cmap="jet")
                ax[2].set_title(f"Severity: {severity}\nStenosis %: {stenosis_percent:.2f}")
                ax[2].axis("off")

                st.pyplot(fig)

                st.write("Stenosis Area:", stenosis_area)
                st.write("Severity:", severity)
                st.write("Stenosis %:", round(stenosis_percent,2))

                break

    if not match_found:
        st.warning("Uploaded image not found in ARCADE dataset annotations.")