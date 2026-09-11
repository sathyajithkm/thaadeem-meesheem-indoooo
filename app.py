from flask import (
    Flask,
    render_template,
    request,
    send_from_directory
)

import os
import cv2
import numpy as np
import random

from werkzeug.utils import secure_filename

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# ============================================================
# FLASK SETUP
# ============================================================

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

ALLOWED_EXTENSIONS = {
    "jpg",
    "jpeg",
    "png",
    "webp"
}

os.makedirs(
    UPLOAD_FOLDER,
    exist_ok=True
)


# ============================================================
# MEDIAPIPE FACE LANDMARKER
# ============================================================

MODEL_PATH = os.path.join(
    "models",
    "face_landmarker.task"
)

if not os.path.exists(MODEL_PATH):

    raise FileNotFoundError(
        "\nERROR: face_landmarker.task was not found.\n"
        "Make sure this file exists:\n"
        "models/face_landmarker.task\n"
    )


base_options = python.BaseOptions(
    model_asset_path=MODEL_PATH
)


face_landmarker_options = (
    vision.FaceLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.IMAGE,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5
    )
)


FACE_LANDMARKER = (
    vision.FaceLandmarker.create_from_options(
        face_landmarker_options
    )
)


# ============================================================
# IMAGE SIZE
# ============================================================

MAX_IMAGE_WIDTH = 1200
MAX_IMAGE_HEIGHT = 1200


def resize_keep_aspect_ratio(image):

    height, width = image.shape[:2]

    scale = min(
        MAX_IMAGE_WIDTH / width,
        MAX_IMAGE_HEIGHT / height,
        1.0
    )

    if scale >= 1.0:
        return image

    new_width = int(
        width * scale
    )

    new_height = int(
        height * scale
    )

    return cv2.resize(
        image,
        (
            new_width,
            new_height
        ),
        interpolation=cv2.INTER_AREA
    )


# ============================================================
# FILE VALIDATION
# ============================================================

def allowed_file(filename):

    return (
        "." in filename
        and
        filename.rsplit(
            ".",
            1
        )[1].lower()
        in ALLOWED_EXTENSIONS
    )


# ============================================================
# LANDMARK TO PIXEL
# ============================================================

def landmark_to_pixel(
    landmark,
    width,
    height
):

    x = int(
        landmark.x * width
    )

    y = int(
        landmark.y * height
    )

    x = max(
        0,
        min(
            width - 1,
            x
        )
    )

    y = max(
        0,
        min(
            height - 1,
            y
        )
    )

    return x, y


# ============================================================
# DRAW LANDMARK
# ============================================================

def draw_landmark(
    image,
    point,
    label=None
):

    cv2.circle(
        image,
        point,
        5,
        (255, 0, 255),
        -1
    )

    if label:

        cv2.putText(
            image,
            label,
            (
                point[0] + 8,
                point[1] - 8
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 0, 255),
            1
        )


# ============================================================
# DRAW BOX
# ============================================================

def draw_box(
    image,
    x1,
    y1,
    x2,
    y2,
    color,
    label
):

    height, width = image.shape[:2]

    x1 = max(
        0,
        min(
            width - 1,
            int(x1)
        )
    )

    x2 = max(
        0,
        min(
            width - 1,
            int(x2)
        )
    )

    y1 = max(
        0,
        min(
            height - 1,
            int(y1)
        )
    )

    y2 = max(
        0,
        min(
            height - 1,
            int(y2)
        )
    )

    cv2.rectangle(
        image,
        (x1, y1),
        (x2, y2),
        color,
        3
    )

    label_y = max(
        25,
        y1 - 10
    )

    cv2.putText(
        image,
        label,
        (
            x1,
            label_y
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        color,
        2
    )


# ============================================================
# GET REGION
# ============================================================

def get_region(
    image,
    x1,
    y1,
    x2,
    y2
):

    height, width = image.shape[:2]

    x1 = max(
        0,
        min(
            width - 1,
            int(x1)
        )
    )

    x2 = max(
        0,
        min(
            width,
            int(x2)
        )
    )

    y1 = max(
        0,
        min(
            height - 1,
            int(y1)
        )
    )

    y2 = max(
        0,
        min(
            height,
            int(y2)
        )
    )

    if x2 <= x1:
        return None

    if y2 <= y1:
        return None

    return image[
        y1:y2,
        x1:x2
    ]


# ============================================================
# HAIR DENSITY
# ============================================================

def calculate_hair_density(region):

    if region is None:
        return 0

    if region.size == 0:
        return 0

    gray = cv2.cvtColor(
        region,
        cv2.COLOR_BGR2GRAY
    )

    gray = cv2.GaussianBlur(
        gray,
        (5, 5),
        0
    )

    threshold = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        21,
        7
    )

    kernel = np.ones(
        (2, 2),
        np.uint8
    )

    threshold = cv2.morphologyEx(
        threshold,
        cv2.MORPH_OPEN,
        kernel
    )

    dark_pixels = np.sum(
        threshold == 255
    )

    total_pixels = threshold.size

    if total_pixels == 0:
        return 0

    density = (
        dark_pixels /
        total_pixels
    ) * 100

    return round(
        max(
            0,
            min(
                100,
                density
            )
        ),
        1
    )


# ============================================================
# COMPLETELY RANDOM BEARD LOADING TIME 😂
# ============================================================

def generate_random_beard_loading_time():

    # --------------------------------------------------------
    # Completely random numbers
    # --------------------------------------------------------

    years = random.randint(
        1,
        999
    )

    months = random.randint(
        0,
        11
    )

    days = random.randint(
        0,
        30
    )


    # --------------------------------------------------------
    # Sometimes make it even more ridiculous
    # --------------------------------------------------------

    ridiculous_mode = random.randint(
        1,
        10
    )


    if ridiculous_mode == 1:

        # Century mode

        centuries = random.randint(
            1,
            9
        )

        years = random.randint(
            0,
            99
        )

        time_text = (
            f"{centuries} centuries, "
            f"{years} years, "
            f"{months} months"
        )


    elif ridiculous_mode == 2:

        # Millennium mode 💀

        millennium = random.randint(
            1,
            3
        )

        years = random.randint(
            0,
            999
        )

        time_text = (
            f"{millennium} millennium, "
            f"{years} years, "
            f"{months} months"
        )


    elif ridiculous_mode == 3:

        # Extremely broken calculation 😂

        time_text = (
            f"{years} years, "
            f"{months + 12} months, "
            f"{days + 31} days"
        )


    elif ridiculous_mode == 4:

        # Fast beard miracle

        hours = random.randint(
            1,
            999
        )

        minutes = random.randint(
            1,
            59
        )

        time_text = (
            f"{hours} hours, "
            f"{minutes} minutes"
        )


    elif ridiculous_mode == 5:

        # Scientific nonsense

        seconds = random.randint(
            100000,
            99999999
        )

        time_text = (
            f"{seconds:,} seconds"
        )


    else:

        # Normal-looking but still ridiculous

        time_text = (
            f"{years} years, "
            f"{months} months, "
            f"{days} days"
        )


    # --------------------------------------------------------
    # Completely random loading percentage
    # --------------------------------------------------------

    loading_progress = random.randint(
        1,
        99
    )


    # --------------------------------------------------------
    # Random funny messages
    # --------------------------------------------------------

    messages = [

        "🧔 Your beard is downloading... very slowly.",

        "🐌 Facial hair connection detected. Speed: terrible.",

        "📡 Searching for beard signal from your ancestors...",

        "🧬 Contacting the Beard Council...",

        "💀 The beard server appears to be overloaded.",

        "🚀 Beard installation has encountered unexpected delays.",

        "🪒 Please do NOT shave while the download is running.",

        "😂 Your beard has requested more processing power.",

        "🧔‍♂️ Your facial hair is currently thinking about it.",

        "⚠️ Beard.exe is taking longer than expected.",

        "🌱 Your beard is still in beta testing.",

        "🔮 The Beard-O-Meter has consulted the future.",

        "📥 Downloading follicles from the cloud...",

        "🛰️ Attempting to establish connection with future beard.",

        "🔥 Beard levels are behaving suspiciously.",

        "👨‍🔬 Our scientists are confused but optimistic.",

        "💻 Facial Hair OS is installing updates.",

        "🧪 Results have been peer-reviewed by absolutely nobody.",

        "🧔 The beard has entered the queue.",

        "⏳ Please wait. Your beard refuses to hurry."

    ]


    loading_message = random.choice(
        messages
    )


    return {

        "text":
            time_text,

        "progress":
            loading_progress,

        "message":
            loading_message

    }


# ============================================================
# ANALYZE FACE
# ============================================================

def analyze_face(image_path):

    image = cv2.imread(
        image_path
    )

    if image is None:
        return None


    # --------------------------------------------------------
    # RESIZE
    # --------------------------------------------------------

    image = resize_keep_aspect_ratio(
        image
    )

    height, width = image.shape[:2]


    # --------------------------------------------------------
    # RGB
    # --------------------------------------------------------

    rgb_image = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )


    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb_image
    )


    # --------------------------------------------------------
    # FACE LANDMARKS
    # --------------------------------------------------------

    result = FACE_LANDMARKER.detect(
        mp_image
    )


    if not result.face_landmarks:
        return None


    landmarks = result.face_landmarks[0]


    # ========================================================
    # IMPORTANT LANDMARKS
    # ========================================================

    nose = landmark_to_pixel(
        landmarks[1],
        width,
        height
    )

    upper_lip = landmark_to_pixel(
        landmarks[13],
        width,
        height
    )

    lower_lip = landmark_to_pixel(
        landmarks[14],
        width,
        height
    )

    chin = landmark_to_pixel(
        landmarks[152],
        width,
        height
    )

    mouth_left = landmark_to_pixel(
        landmarks[61],
        width,
        height
    )

    mouth_right = landmark_to_pixel(
        landmarks[291],
        width,
        height
    )

    face_left = landmark_to_pixel(
        landmarks[234],
        width,
        height
    )

    face_right = landmark_to_pixel(
        landmarks[454],
        width,
        height
    )


    # ========================================================
    # FACE BOX
    # ========================================================

    all_points = []

    for landmark in landmarks:

        px, py = landmark_to_pixel(
            landmark,
            width,
            height
        )

        all_points.append(
            (px, py)
        )


    xs = [
        point[0]
        for point in all_points
    ]

    ys = [
        point[1]
        for point in all_points
    ]


    face_x1 = min(xs)
    face_x2 = max(xs)

    face_y1 = min(ys)
    face_y2 = max(ys)


    face_width = (
        face_x2 -
        face_x1
    )

    face_height = (
        face_y2 -
        face_y1
    )


    # ========================================================
    # MOUSTACHE BOX
    # ========================================================

    mouth_width = abs(
        mouth_right[0] -
        mouth_left[0]
    )

    moustache_width = (
        mouth_width *
        1.18
    )

    moustache_center_x = (
        mouth_left[0] +
        mouth_right[0]
    ) / 2

    moustache_x1 = (
        moustache_center_x -
        moustache_width / 2
    )

    moustache_x2 = (
        moustache_center_x +
        moustache_width / 2
    )


    nose_to_lip = (
        upper_lip[1] -
        nose[1]
    )


    moustache_y1 = (
        nose[1] +
        nose_to_lip * 0.48
    )

    moustache_y2 = (
        upper_lip[1] +
        max(
            5,
            face_height * 0.018
        )
    )


    if moustache_y2 <= moustache_y1:

        moustache_y1 = (
            upper_lip[1] -
            face_height * 0.08
        )

        moustache_y2 = (
            upper_lip[1] +
            face_height * 0.02
        )


    # ========================================================
    # BEARD BOX
    # ========================================================

    beard_x1 = (
        face_left[0] +
        face_width * 0.08
    )

    beard_x2 = (
        face_right[0] -
        face_width * 0.08
    )

    beard_y1 = (
        lower_lip[1] +
        face_height * 0.06
    )

    beard_y2 = (
        chin[1] -
        face_height * 0.015
    )


    # ========================================================
    # DRAW FACE
    # ========================================================

    cv2.rectangle(
        image,
        (
            face_x1,
            face_y1
        ),
        (
            face_x2,
            face_y2
        ),
        (0, 255, 0),
        3
    )


    cv2.putText(
        image,
        "FACE",
        (
            face_x1 + 5,
            max(
                face_y1 + 30,
                30
            )
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (0, 255, 0),
        2
    )


    # ========================================================
    # LANDMARKS
    # ========================================================

    draw_landmark(
        image,
        nose,
        "nose"
    )

    draw_landmark(
        image,
        upper_lip,
        "upper lip"
    )

    draw_landmark(
        image,
        lower_lip,
        "lower lip"
    )

    draw_landmark(
        image,
        chin,
        "chin"
    )

    draw_landmark(
        image,
        mouth_left,
        "L"
    )

    draw_landmark(
        image,
        mouth_right,
        "R"
    )


    # ========================================================
    # DRAW MOUSTACHE
    # ========================================================

    draw_box(
        image,

        moustache_x1,
        moustache_y1,

        moustache_x2,
        moustache_y2,

        (255, 0, 0),

        "MOUSTACHE"
    )


    # ========================================================
    # DRAW BEARD
    # ========================================================

    draw_box(
        image,

        beard_x1,
        beard_y1,

        beard_x2,
        beard_y2,

        (0, 165, 255),

        "BEARD"
    )


    # ========================================================
    # GET REGIONS
    # ========================================================

    moustache_region = get_region(
        image,

        moustache_x1,
        moustache_y1,

        moustache_x2,
        moustache_y2
    )


    beard_region = get_region(
        image,

        beard_x1,
        beard_y1,

        beard_x2,
        beard_y2
    )


    # ========================================================
    # DENSITY
    # ========================================================

    moustache_density = (
        calculate_hair_density(
            moustache_region
        )
    )

    beard_density = (
        calculate_hair_density(
            beard_region
        )
    )


    # ========================================================
    # ESTIMATED HAIR COUNT
    # ========================================================

    moustache_count = int(
        moustache_density * 12
    )

    beard_count = int(
        beard_density * 35
    )


    # ========================================================
    # CURRENT STATUS
    # ========================================================

    if moustache_density >= 15:

        moustache_status = "Detected"

    elif moustache_density >= 8:

        moustache_status = "Light"

    else:

        moustache_status = (
            "Low / Not Clearly Detected"
        )


    if beard_density >= 15:

        beard_status = "Detected"

    elif beard_density >= 8:

        beard_status = "Light"

    else:

        beard_status = (
            "Low / Not Clearly Detected"
        )


    # ========================================================
    # FUTURE POTENTIAL
    # ========================================================

    moustache_potential = int(
        30 +
        moustache_density * 0.75
    )

    beard_potential = int(
        30 +
        beard_density * 0.75
    )


    moustache_potential = max(
        5,
        min(
            99,
            moustache_potential
        )
    )

    beard_potential = max(
        5,
        min(
            99,
            beard_potential
        )
    )


    # ========================================================
    # RANDOM BEARD LOADING TIME
    # ========================================================

    beard_loading = (
        generate_random_beard_loading_time()
    )


    # ========================================================
    # SAVE IMAGE
    # ========================================================

    original_name = os.path.basename(
        image_path
    )

    name, extension = os.path.splitext(
        original_name
    )

    result_filename = (
        f"{name}_analysis{extension}"
    )

    result_path = os.path.join(
        UPLOAD_FOLDER,
        result_filename
    )

    cv2.imwrite(
        result_path,
        image
    )


    # ========================================================
    # RETURN ANALYSIS
    # ========================================================

    return {

        "filename":
            result_filename,

        "beard_density":
            beard_density,

        "moustache_density":
            moustache_density,

        "beard_count":
            beard_count,

        "moustache_count":
            moustache_count,

        "beard_status":
            beard_status,

        "moustache_status":
            moustache_status,

        "beard_potential":
            beard_potential,

        "moustache_potential":
            moustache_potential,

        "beard_loading_time":
            beard_loading["text"],

        "beard_loading_progress":
            beard_loading["progress"],

        "beard_loading_message":
            beard_loading["message"]
    }


# ============================================================
# HOME
# ============================================================

@app.route("/")
def home():

    return render_template(
        "index.html"
    )


# ============================================================
# UPLOAD
# ============================================================

@app.route(
    "/upload",
    methods=["POST"]
)
def upload():

    if "photo" not in request.files:

        return (
            "No photo was uploaded."
        )


    file = request.files["photo"]


    if file.filename == "":

        return (
            "No photo was selected."
        )


    if not allowed_file(
        file.filename
    ):

        return (
            "Invalid image type. "
            "Please upload JPG, JPEG, PNG "
            "or WEBP."
        )


    filename = secure_filename(
        file.filename
    )


    filepath = os.path.join(
        app.config["UPLOAD_FOLDER"],
        filename
    )


    file.save(
        filepath
    )


    analysis = analyze_face(
        filepath
    )


    if analysis is None:

        return """
        <!DOCTYPE html>

        <html>

        <head>

            <title>Face Not Detected</title>

        </head>

        <body>

            <h1>😕 Face not detected</h1>

            <p>
                Please upload a clear,
                front-facing photograph.
            </p>

            <a href="/">
                Try Again
            </a>

        </body>

        </html>
        """


    return render_template(
        "result.html",
        analysis=analysis
    )


# ============================================================
# SERVE UPLOADED FILE
# ============================================================

@app.route(
    "/uploads/<filename>"
)
def uploaded_file(filename):

    return send_from_directory(
        app.config["UPLOAD_FOLDER"],
        filename
    )


# ============================================================
# START APPLICATION
# ============================================================

if __name__ == "__main__":

    print("")
    print("======================================")
    print("       BEARD-O-METER STARTING")
    print("======================================")
    print("")
    print("MediaPipe Face Landmarker: READY")
    print("Model:", MODEL_PATH)
    print("")
    print("Open:")
    print("http://127.0.0.1:5000")
    print("")

    app.run(
        debug=True
    )