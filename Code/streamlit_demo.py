import streamlit as st
import pandas as pd
import torch
from cVAE_train_revised import CVAE, CLASS_NAMES, device
from torchvision.utils import make_grid
import torch.nn.functional as F
from classifier_train import CNN_BiLSTM
from PIL import Image
from torchvision import transforms
import os
import cv2
import numpy as np

# Reference: 07_01-layouts_sidebar.py
# Uses sidebar for navigation and app information.
st.set_page_config(
    page_title="Face Attribute Generation & Classification",
    layout="wide"
)

# --------------------------------------------------
# Global UI Styling (Custom Theme)
# --------------------------------------------------
# This block overrides default Streamlit styles using CSS
# to create a consistent product-like UI with a pink-red theme (#f94367)

st.markdown("""
<style>
/* Main color variables */
:root {
    --primary: #f94367;
    --black: #000000;
    --white: #ffffff;
    --line: #222222;
}

/* Sidebar background */
[data-testid="stSidebar"] {
    background-color: #000000;
}

/* Sidebar text */
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
    color: #ffffff;
}

/* Main buttons */
.stButton > button {
    background-color: #f94367;
    color: white;
    border-radius: 8px;
    border: none;
    font-weight: 600;
}

/* Main button hover */
.stButton > button:hover {
    background-color: #d73757;
    color: white;
}

/* Headings */
h1, h2, h3 {
    color: #000000;
}

/* Centered intro card */
.intro-card {
    max-width: 780px;
    margin: 0 auto 28px auto;
    padding: 16px 24px;
    background-color: #ffffff;
    border: 1px solid #eeeeee;
    border-radius: 12px;
    text-align: left;
}

/* Short vertical divider between two home modules */
.vertical-divider {
    height: 180px;
    width: 1px;
    background-color: #222222;
    margin: 20px auto 0 auto;
}

/* Intro list */
.intro-list {
    margin-left: 36px;
    line-height: 1.6;
}

/* Sidebar inactive buttons */
section[data-testid="stSidebar"] div[data-testid="stButton"] button[kind="secondary"] {
    background-color: #ffffff !important;
    color: #000000 !important;
    border: none !important;
    border-radius: 8px;
    font-weight: 600;
}

/* Force inactive button inner text to black */
section[data-testid="stSidebar"] div[data-testid="stButton"] button[kind="secondary"] * {
    color: #000000 !important;
}

/* Sidebar active button */
section[data-testid="stSidebar"] div[data-testid="stButton"] button[kind="primary"] {
    background-color: #f94367 !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 8px;
    font-weight: 700;
}

/* Force active button inner text to white */
section[data-testid="stSidebar"] div[data-testid="stButton"] button[kind="primary"] * {
    color: #ffffff !important;
}

/* HOME button emphasis */
section[data-testid="stSidebar"] div[data-testid="stButton"]:first-of-type button {
    font-weight: 800 !important;
    letter-spacing: 1px;
}

</style>
""", unsafe_allow_html=True)

# # cVAE path (one model version)
# CVAE_PATH = "/home/ubuntu/Final-Project-Group3/Models/model_cVAE_Group3_final.pt"

# cVAE path (several models version)
CVAE_MODEL_OPTIONS = {
    "Final Model": "/home/ubuntu/Final-Project-Group3/Models/model_cVAE_Group3_final.pt",
    "Old Model": "/home/ubuntu/Final-Project-Group3/Models/model_cVAE_Group3.pt",
}

# # Load cVAE (one model version)
# @st.cache_resource
# def load_cvae():
#     model = CVAE().to(device)
#     model.load_state_dict(torch.load(CVAE_PATH, map_location=device))
#     model.eval()
#     return model
#
# cvae = load_cvae()

# Load cVAE (several models version)
@st.cache_resource
def load_cvae(model_path):
    model = CVAE().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

# Load classifier
CLASSIFIER_PATH = "/home/ubuntu/Final-Project-Group3/Models/model_Group3.pt"
THRESHOLD_PATH = "/home/ubuntu/Final-Project-Group3/per_attr_f1_Group3.csv"
@st.cache_resource
def load_classifier():
    model = CNN_BiLSTM().to(device)
    model.load_state_dict(torch.load(CLASSIFIER_PATH, map_location=device))
    model.eval()
    return model

clf = load_classifier()

@st.cache_data
def load_thresholds():
    if os.path.exists(THRESHOLD_PATH):
        thr_df = pd.read_csv(THRESHOLD_PATH)
        threshold_map = dict(zip(thr_df["attribute"], thr_df["threshold"]))
        return [threshold_map.get(attr, 0.5) for attr in CLASS_NAMES]
    return [0.5] * len(CLASS_NAMES)

thresholds = torch.tensor(load_thresholds(), dtype=torch.float32, device=device)

# ---------- Session State ----------
if "page" not in st.session_state:
    st.session_state.page = "home"

if "selected_attrs" not in st.session_state:
    st.session_state.selected_attrs = {}

if "uploaded_image" not in st.session_state:
    st.session_state.uploaded_image = None


# ---------- Attribute Groups ----------
ATTRIBUTE_GROUPS = {
    "Hair": [
        "Bald", "Bangs", "Black_Hair", "Blond_Hair", "Brown_Hair",
        "Gray_Hair", "Straight_Hair", "Wavy_Hair"
    ],
    "Face Structure": [
        "Chubby", "Double_Chin", "High_Cheekbones", "Oval_Face",
        "Pointy_Nose", "Big_Nose"
    ],
    "Facial Hair": [
        "No_Beard", "Mustache", "Goatee", "Sideburns", "5_o_Clock_Shadow"
    ],
    "Eyes & Eyebrows": [
    "Arched_Eyebrows", "Bushy_Eyebrows", "Narrow_Eyes",
    "Bags_Under_Eyes"
    ],
    "Mouth & Expression": [
        "Smiling", "Mouth_Slightly_Open", "Big_Lips"
    ],
    "Makeup & Skin": [
        "Heavy_Makeup", "Wearing_Lipstick", "Pale_Skin",
        "Rosy_Cheeks", "Blurry"
    ],
    "Accessories": [
    "Eyeglasses",
    "Wearing_Hat", "Wearing_Earrings",
    "Wearing_Necklace", "Wearing_Necktie"
    ],
    "Other": ["Attractive", "Receding_Hairline"]
}

ATTRIBUTE_TO_GROUP = {
    attr: group
    for group, attrs in ATTRIBUTE_GROUPS.items()
    for attr in attrs
}

ATTRIBUTE_TO_GROUP["Male"] = "Gender"
ATTRIBUTE_TO_GROUP["Young"] = "Age"

DISPLAY_NAME = {
    "Young": "Young",

    "Bald": "Bald",
    "Bangs": "Bangs",
    "Black_Hair": "Black",
    "Blond_Hair": "Blond",
    "Brown_Hair": "Brown",
    "Gray_Hair": "Gray",
    "Straight_Hair": "Straight",
    "Wavy_Hair": "Wavy",
    "Receding_Hairline": "Receding Hairline",

    "Chubby": "Chubby",
    "Double_Chin": "Double Chin",
    "High_Cheekbones": "High Cheekbones",
    "Oval_Face": "Oval Face",
    "Pointy_Nose": "Pointy Nose",
    "Big_Nose": "Big Nose",

    "No_Beard": "No Beard",
    "Mustache": "Mustache",
    "Goatee": "Goatee",
    "Sideburns": "Sideburns",
    "5_o_Clock_Shadow": "5 O'Clock Shadow",

    "Arched_Eyebrows": "Arched Eyebrows",
    "Bushy_Eyebrows": "Bushy Eyebrows",
    "Narrow_Eyes": "Narrow Eyes",
    "Bags_Under_Eyes": "Bags Under Eyes",

    "Smiling": "Smiling",
    "Mouth_Slightly_Open": "Mouth Slightly Open",
    "Big_Lips": "Big Lips",

    "Heavy_Makeup": "Heavy Makeup",
    "Wearing_Lipstick": "Lipstick",
    "Pale_Skin": "Pale Skin",
    "Rosy_Cheeks": "Rosy Cheeks",
    "Blurry": "Blurry",

    "Eyeglasses": "Eyeglasses",
    "Wearing_Hat": "Hat",
    "Wearing_Earrings": "Earrings",
    "Wearing_Necklace": "Necklace",
    "Wearing_Necktie": "Necktie",

    "Attractive": "Attractive",
}

# Hard conflict pairs: logically do not make sense
HARD_CONFLICT_PAIRS = [
    # Bald conflicts with visible hair
    ("Bald", "Bangs"),
    ("Bald", "Black_Hair"),
    ("Bald", "Blond_Hair"),
    ("Bald", "Brown_Hair"),
    ("Bald", "Gray_Hair"),
    ("Bald", "Straight_Hair"),
    ("Bald", "Wavy_Hair"),

    # No beard conflicts with facial hair
    ("No_Beard", "Mustache"),
    ("No_Beard", "Goatee"),
    ("No_Beard", "Sideburns"),
    ("No_Beard", "5_o_Clock_Shadow"),

    # Main hair colors
    ("Black_Hair", "Blond_Hair"),
    ("Black_Hair", "Brown_Hair"),
    ("Black_Hair", "Gray_Hair"),
    ("Blond_Hair", "Brown_Hair"),
    ("Blond_Hair", "Gray_Hair"),
    ("Brown_Hair", "Gray_Hair"),
]

# ---------- Helper Functions ----------
def go_to(page_name):
    st.session_state.page = page_name
    st.rerun()


def flatten_selected_attrs(selected_dict):
    attrs = []
    for values in selected_dict.values():
        attrs.extend(values)
    return attrs


def detect_conflicts(target_attrs):
    hard_conflicts = []
    soft_conflicts = []

    selected_set = set(target_attrs.keys())

    for a, b in HARD_CONFLICT_PAIRS:
        if a in selected_set and b in selected_set:
            hard_conflicts.append((
                DISPLAY_NAME.get(a, a),
                DISPLAY_NAME.get(b, b),
                ATTRIBUTE_TO_GROUP.get(b) or ATTRIBUTE_TO_GROUP.get(a) or "General"
            ))

    # Soft conflict: Male + makeup/lipstick
    if target_attrs.get("Male") == 1:
        for attr in ["Heavy_Makeup", "Wearing_Lipstick"]:
            if target_attrs.get(attr) == 1:
                soft_conflicts.append((
                    "Male",
                    DISPLAY_NAME.get(attr, attr),
                    ATTRIBUTE_TO_GROUP.get(attr, "Makeup & Skin")
                ))

    # Soft conflict: Female + facial hair
    if target_attrs.get("Male") == 0:
        for attr in ["Mustache", "Goatee", "Sideburns", "5_o_Clock_Shadow"]:
            if target_attrs.get(attr) == 1:
                soft_conflicts.append((
                    "Female",
                    DISPLAY_NAME.get(attr, attr),
                    "Facial Hair"
                ))

    # Soft conflict: Female + Bald
    if target_attrs.get("Male") == 0 and target_attrs.get("Bald") == 1:
        soft_conflicts.append(("Female", "Bald", "Hair"))

    return hard_conflicts, soft_conflicts


# cVAE generation function
def generate_faces(cvae_model, target_attrs, n_images=8):
    # ---------- build attribute vector ----------
    attr_vec = torch.zeros(len(CLASS_NAMES), device=device)

    for attr, val in target_attrs.items():
        if attr in CLASS_NAMES:
            attr_vec[CLASS_NAMES.index(attr)] = float(val)

    attr_vec = attr_vec.unsqueeze(0).repeat(n_images, 1)

    # ---------- generate ----------
    with torch.no_grad():
        imgs = cvae_model.generate(attr_vec)

    imgs = torch.clamp(imgs, 0, 1)

    # ---------- make grid ----------
    grid = make_grid(imgs.cpu(), nrow=4)

    return grid, imgs

# Preprocess
def preprocess_generated_for_classifier(imgs):
    processed = []

    for img_tensor in imgs:
        # tensor: (3, 64, 64), values in [0, 1]
        img_np = img_tensor.detach().cpu().permute(1, 2, 0).numpy()
        img_np = (img_np * 255).astype(np.uint8)

        image = Image.fromarray(img_np)

        x = preprocess_for_classifier_pil(image)  # shape: (1, 3, 224, 224)
        processed.append(x.squeeze(0))

    return torch.stack(processed).to(device)

def preprocess_for_classifier_pil(image):
    img = np.array(image.convert("RGB"))

    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    lab[:, :, 0] = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    ).apply(lab[:, :, 0])

    img = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    img = cv2.GaussianBlur(img, (3, 3), 0)
    img = cv2.resize(img, (224, 224))

    x = torch.FloatTensor(img / 255.0).permute(2, 0, 1)

    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )

    return normalize(x).unsqueeze(0).to(device)

# Classifier evaluation function
def evaluate_generated_images(imgs, target_attrs):
    x_clf = preprocess_generated_for_classifier(imgs)

    with torch.no_grad():
        logits = clf(x_clf)
        probs = torch.sigmoid(logits)
        preds = (probs >= thresholds).int().cpu().numpy()
        probs_np = probs.cpu().numpy()

    rows = []
    for attr, target_val in target_attrs.items():
        idx = CLASS_NAMES.index(attr)
        pred_rate = preds[:, idx].mean()
        avg_prob = probs_np[:, idx].mean()

        rows.append({
            "attribute": (
                "Female" if attr == "Male" and target_val == 0 else
                "Older-looking" if attr == "Young" and target_val == 0 else
                DISPLAY_NAME.get(attr, attr)
            ),
            "target": target_val,
            "avg_classifier_probability": round(float(avg_prob), 3),
            "predicted_positive_rate": round(float(pred_rate), 3),
            "match": "Yes" if (pred_rate >= 0.5) == bool(target_val) else "No"
        })

    return pd.DataFrame(rows)


# Uploaded image classification function
def classify_uploaded_image(uploaded_file):
    image = Image.open(uploaded_file).convert("RGB")

    x = preprocess_for_classifier_pil(image)

    with torch.no_grad():
        logits = clf(x)
        probs = torch.sigmoid(logits).squeeze(0)
        preds = (probs >= thresholds).int()

    rows = []
    for i, attr in enumerate(CLASS_NAMES):
        if preds[i].item() == 1:
            rows.append({
                "attribute": attr,
                "probability": round(float(probs[i].item()), 3),
                "prediction": 1
            })

    result_df = pd.DataFrame(rows).sort_values(
        "probability",
        ascending=False
    ).reset_index(drop=True)

    return image, result_df

# Add a face detect function
def detect_face(uploaded_file):
    # Reset pointer
    uploaded_file.seek(0)

    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    return len(faces) > 0


# ---------- Sidebar ----------
with st.sidebar:
    st.title("Demo Navigation")

    # Reference: 05_01-input_button.py
    # Uses st.button to navigate between pages.
    home_active = st.session_state.page == "home"
    generate_active = st.session_state.page in ["generate", "generate_result"]
    upload_active = st.session_state.page in ["upload", "upload_result"]

    if st.button("HOME", type="primary"):
        go_to("home")

    # White divider between HOME and the other two buttons
    st.markdown("<hr style='border: 1px solid white; margin: 10px 0 16px 0;'>", unsafe_allow_html=True)

    if st.button("Generate Face", type="primary" if generate_active else "secondary"):
        go_to("generate")

    if st.button("Upload & Classify", type="primary" if upload_active else "secondary"):
        go_to("upload")

    st.divider()

# ---------- Home Page ----------
if st.session_state.page == "home":
    # App title and description
    st.title("Face Attribute Generation & Classification Demo")
    st.caption("Customize facial attributes and generate realistic faces using cVAE")

    st.markdown(
        """
    <div class="intro-card">
    <p>This demo has two main modules:</p>
    <div class="intro-list">
    <p><b>1.</b>&nbsp;&nbsp;Generate your desired face using selected CelebA attributes.</p>
    <p><b>2.</b>&nbsp;&nbsp;Upload an existing face image and classify its attributes using the trained classifier.</p>
    </div>
    </div>
        """,
        unsafe_allow_html=True
    )

    # Reference: 07_03-layouts_tabs.py
    # Uses tab-like organization idea; here we use two columns as entry modules.
    col1, divider_col, col2 = st.columns([1, 0.08, 1])

    with col1:
        st.subheader("Generate Your Desired Face")
        st.write("Select facial attributes and generate images using the trained cVAE model.")
        if st.button("Start Generation", type="primary"):
            go_to("generate")

    with divider_col:
        st.markdown("<div class='vertical-divider'></div>", unsafe_allow_html=True)

    with col2:
        st.subheader("Already Have a Face Image?")
        st.write("Upload a local image and classify its facial attributes.")
        if st.button("Upload Image", type="primary"):
            go_to("upload")


# ---------- Module 1: Generate Face ----------
elif st.session_state.page == "generate":
    st.title("Generate Your Desired Face")

    st.write("""
    Select attributes from the categories below. Some attributes may conflict
    with each other in real-world data. If such conflicts are detected, a warning
    will be shown, but generation is still allowed.
    """)

    # Placeholder
    conflict_boxes = {}

    # Select boxes
    model_choice = st.selectbox(
        "Choose a cVAE model",
        list(CVAE_MODEL_OPTIONS.keys())
    )

    if "prev_cvae_model_choice" not in st.session_state:
        st.session_state.prev_cvae_model_choice = model_choice

    if st.session_state.prev_cvae_model_choice != model_choice:
        st.session_state.pop("generated_grid", None)
        st.session_state.pop("generated_imgs", None)
        st.session_state.prev_cvae_model_choice = model_choice

    st.session_state.cvae_model_choice = model_choice
    st.session_state.cvae_model_path = CVAE_MODEL_OPTIONS[model_choice]

    gender_choice = st.selectbox(
        "Gender",
        ["Skip", "Male", "Female"]
    )
    conflict_boxes["Gender"] = st.container()

    age_choice = st.selectbox(
        "Age",
        ["Skip", "Young", "Older-looking"]
    )
    conflict_boxes["Age"] = st.container()

    selected_by_group = {}

    # Reference: 05_07-input_multiselect.py
    # Uses st.multiselect for selecting multiple attributes in each category.
    for group_name, attrs in ATTRIBUTE_GROUPS.items():
        selected_by_group[group_name] = st.multiselect(
            f"{group_name}",
            attrs,
            default=[],
            key=f"select_{group_name}",
            format_func=lambda x: DISPLAY_NAME.get(x, x)
        )
        conflict_boxes[group_name] = st.container()

    selected_attrs = flatten_selected_attrs(selected_by_group)

    target_attrs = {attr: 1 for attr in selected_attrs}

    if gender_choice == "Male":
        target_attrs["Male"] = 1
    elif gender_choice == "Female":
        target_attrs["Male"] = 0

    if age_choice == "Young":
        target_attrs["Young"] = 1
    elif age_choice == "Older-looking":
        target_attrs["Young"] = 0

    if target_attrs:
        display_attrs = [
            "Female" if attr == "Male" and val == 0 else
            "Older-looking" if attr == "Young" and val == 0 else
            DISPLAY_NAME.get(attr, attr)
            for attr, val in target_attrs.items()
        ]
        st.write("Selected attributes:", display_attrs)

    hard_conflicts, soft_conflicts = detect_conflicts(target_attrs)

    for group_name in conflict_boxes:
        box = conflict_boxes[group_name]
        box.empty()

        group_hard = [c for c in hard_conflicts if c[2] == group_name]
        group_soft = [c for c in soft_conflicts if c[2] == group_name]

        messages = []

        if group_hard:
            messages.append((
                "error",
                "Strong conflicts detected: "
                + ", ".join([f"{a} + {b}" for a, b, _ in group_hard])
                + ". You can still continue generation, but results may be unstable."
            ))

        if group_soft:
            messages.append((
                "warning",
                "Unusual combinations detected: "
                + ", ".join([f"{a} + {b}" for a, b, _ in group_soft])
                + ". These are rare in the dataset, but generation is still allowed."
            ))

        for msg_type, msg in messages:
            if msg_type == "error":
                conflict_boxes[group_name].error(msg)
            else:
                conflict_boxes[group_name].warning(msg)

    # Reference: 05_08-input_slider.py
    # Uses st.slider to control number of generated images.
    n_images = st.slider("Number of generated images", 4, 16, 8, step=4)

    if st.button("Generate Images", type="primary"):
        # avoid all 0 selection
        if len(target_attrs) == 0:
            st.warning("Please select at least one attribute.")
            st.stop()

        st.session_state.selected_attrs = target_attrs
        st.session_state.n_images = n_images
        go_to("generate_result")


# ---------- Module 1 Result Page ----------
elif st.session_state.page == "generate_result":
    st.title("Generated Images & Classifier Evaluation")
    st.caption(f"cVAE model: {st.session_state.cvae_model_choice}")

    # Results section
    target_attrs = st.session_state.selected_attrs

    # Reference: 09_02-status_spinner.py
    # Uses st.spinner while generation/evaluation is running.
    if "generated_grid" not in st.session_state or "generated_imgs" not in st.session_state:
        with st.spinner("Generating images with cVAE..."):
            selected_cvae = load_cvae(st.session_state.cvae_model_path)
            grid, imgs = generate_faces(selected_cvae, target_attrs, st.session_state.n_images)
            st.session_state.generated_grid = grid
            st.session_state.generated_imgs = imgs
    else:
        grid = st.session_state.generated_grid
        imgs = st.session_state.generated_imgs

    # Reference: 07_02-layouts_columns.py style from layout examples
    # Uses columns to show generated images on the left and evaluation on the right.
    left_col, right_col = st.columns([1.2, 1])

    with left_col:
        st.subheader("Selected Attributes")
        display_attrs = [
            "Female" if attr == "Male" and val == 0 else
            "Older-looking" if attr == "Young" and val == 0 else
            DISPLAY_NAME.get(attr, attr)
            for attr, val in target_attrs.items()
        ]
        st.write(display_attrs)

        grid_np = grid.permute(1, 2, 0).detach().cpu().numpy()
        st.image(grid_np, caption="Generated Faces")

        st.success("Generation finished.")

    with right_col:
        st.subheader("Classifier Evaluation Results")

        eval_df = evaluate_generated_images(imgs, target_attrs)

        # Reference: 02_01-dataframe_basic.py
        # Uses st.dataframe to display evaluation table.
        st.dataframe(eval_df)

        # Reference: 03_03-data_elements_metrics.py
        # Uses st.metric to summarize evaluation.
        if len(eval_df) > 0:
            match_rate = (eval_df["match"] == "Yes").mean()
            st.metric("Attribute Match Rate", f"{match_rate:.2f}")

    if st.button("Back to Attribute Selection"):
        st.session_state.pop("generated_grid", None)
        st.session_state.pop("generated_imgs", None)
        go_to("generate")


# ---------- Module 2: Upload Image ----------
elif st.session_state.page == "upload":
    st.title("Upload a Face Image for Attribute Classification")

    st.write("""
    Upload a local face image. The classifier will predict which CelebA
    attributes are present in the image.
    """)

    # Reference: 05_13-input_file_uploader.py
    # Uses st.file_uploader for uploading local image files.
    uploaded_file = st.file_uploader(
        "Upload a JPEG or PNG image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:
        st.session_state.uploaded_image = uploaded_file

        # Reference: 06_01-media_image.py
        # Uses st.image to display uploaded images.
        st.image(uploaded_file, caption="Uploaded image", width=300)

        if st.button("Classify Attributes", type="primary"):
            if detect_face(uploaded_file):
                go_to("upload_result")
            else:
                st.warning(
                    "No face detected. This classifier is trained on face images only. "
                    "Please upload a clear face image."
                )

# ---------- Module 2 Result Page ----------
elif st.session_state.page == "upload_result":
    st.title("Uploaded Image Attribute Classification")

    left_col, right_col = st.columns([1, 1.2])

    with left_col:
        st.subheader("Uploaded Image")
        if st.session_state.uploaded_image is not None:

            # Reset file pointer to the beginning before reading again.
            # Streamlit's uploaded file is a file-like object, and after one read,
            # the pointer moves to the end, so we need seek(0) to reuse it.
            st.session_state.uploaded_image.seek(0)

            image = Image.open(st.session_state.uploaded_image).convert("RGB")
            st.image(image, caption="Uploaded image", width=300)

    with right_col:
        st.subheader("Predicted Attributes")

        if st.session_state.uploaded_image is None:
            st.warning("No uploaded image found. Please upload an image first.")
        else:
            with st.spinner("Running classifier..."):
                # Reset file pointer to the beginning for re-reading the uploaded file
                st.session_state.uploaded_image.seek(0)
                image, result_df = classify_uploaded_image(st.session_state.uploaded_image)

            # Reference: 02_01-dataframe_basic.py
            # Displays predicted attributes and probabilities.
            st.dataframe(result_df)

            if len(result_df) > 0:
                # Reference: 04_03-charts_bar.py
                # Uses st.bar_chart to visualize classifier probabilities.
                chart_df = result_df.set_index("attribute")[["probability"]]
                st.bar_chart(chart_df)
            else:
                st.info("No attributes were predicted as positive.")

    if st.button("Upload Another Image"):
        st.session_state.uploaded_image = None
        go_to("upload")