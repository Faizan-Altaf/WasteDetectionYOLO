
from pathlib import Path
import tempfile

import cv2
import pandas as pd
import streamlit as st
from PIL import Image
from ultralytics import YOLO


# =========================================================
# CONFIGURATION
# =========================================================

PROJECT = Path(__file__).resolve().parent
MODEL_PATH = PROJECT / "models" / "best.pt"

CLASS_NAMES = {
    0: "Plastic Bottle",
    1: "Paper",
    2: "Can",
    3: "Plastic Bag",
}

st.set_page_config(
    page_title="AI Waste Detection Website",
    page_icon="♻️",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# =========================================================
# STYLING
# =========================================================

st.markdown(
    """
    <style>
    .stApp { background: #f4f7fb; }
    .block-container {
        max-width: 1500px;
        padding-top: 1.5rem;
        padding-bottom: 2rem;
    }
    .main-header {
        background: linear-gradient(120deg, #102a43, #176b87);
        padding: 26px 30px;
        border-radius: 16px;
        margin-bottom: 24px;
        color: white;
    }
    .main-header h1 {
        color: white;
        font-size: 32px;
        font-weight: 750;
        margin: 0;
    }
    .main-header p { color: #d8e8f2; margin-top: 8px; }
    .section-title {
        font-size: 22px;
        font-weight: 750;
        color: #16324f;
        margin-bottom: 5px;
    }
    .section-subtitle {
        font-size: 13px;
        color: #63788c;
        margin-bottom: 16px;
    }
    div[data-testid="stFileUploader"] {
        background: white;
        border: 2px dashed #8dbbd0;
        border-radius: 12px;
        padding: 10px;
    }
    div[data-testid="stRadio"] label,
    div[data-testid="stRadio"] p {
        color: #16324f !important;
        opacity: 1 !important;
        font-weight: 600 !important;
    }
    div[data-testid="stMetric"] {
        background: white;
        border: 1px solid #e0e7ef;
        border-radius: 12px;
        padding: 12px;
    }
    .footer-text {
        text-align: center;
        color: #718096;
        font-size: 12px;
        padding-top: 24px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# LOAD MODEL
# =========================================================

if not MODEL_PATH.exists():
    st.error(f"Model file not found: {MODEL_PATH}")
    st.stop()


@st.cache_resource
def load_model(path):
    return YOLO(str(path))


try:
    model = load_model(str(MODEL_PATH))
except Exception as exc:
    st.error(f"Could not load model: {exc}")
    st.stop()


# =========================================================
# HEADER AND SESSION STATE
# =========================================================

st.markdown(
    """
    <div class="main-header">
        <h1>♻️ AI Waste Detection Website</h1>
        <p>
            AI-powered waste identification, object tracking,
            and waste composition analysis using YOLOv8n.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

defaults = {
    "image_original": None,
    "image_annotated": None,
    "image_records": [],
    "video_original": None,
    "video_annotated": None,
    "video_records": [],
    "video_summary": [],
    "video_diagnostics": {},
    "video_processed": False,
}

for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value


# =========================================================
# MAIN TWO-COLUMN LAYOUT
# =========================================================

upload_col, preview_col = st.columns([1, 1], gap="large")


# =========================================================
# LEFT: UPLOAD AND CONTROLS
# =========================================================

with upload_col:
    st.markdown(
        '<div class="section-title">Upload Media</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="section-subtitle">'
        'Upload an image or video for waste detection.'
        '</div>',
        unsafe_allow_html=True,
    )

    mode = st.radio(
        "Choose detection mode",
        ["🖼️ Image Detection", "🎬 Video Detection"],
        horizontal=True,
        key="detection_mode",
    )

    is_image = mode == "🖼️ Image Detection"
    is_video = mode == "🎬 Video Detection"

    confidence = st.slider(
        "Minimum detection confidence",
        0.05,
        0.90,
        0.15,
        0.05,
        help="Lower values may find more objects but can add false detections.",
    )

    # -----------------------------------------------------
    # IMAGE DETECTION
    # -----------------------------------------------------

    if is_image:
        uploaded_image = st.file_uploader(
            "Drag and drop an image, or browse files",
            type=["jpg", "jpeg", "png", "bmp", "webp"],
            key="image_upload",
        )

        if uploaded_image is not None:
            if st.button(
                "🔍 Detect Waste Objects",
                type="primary",
                key="detect_image",
            ):
                try:
                    image = Image.open(
                        uploaded_image
                    ).convert("RGB")

                    with st.spinner("Detecting waste objects..."):
                        result = model.predict(
                            source=image,
                            conf=confidence,
                            imgsz=512,
                            verbose=False,
                        )[0]

                    annotated_bgr = result.plot(
                        boxes=True,
                        labels=True,
                        conf=True,
                    )

                    annotated_rgb = cv2.cvtColor(
                        annotated_bgr,
                        cv2.COLOR_BGR2RGB,
                    )

                    records = []

                    for box in result.boxes:
                        class_id = int(box.cls[0].item())
                        class_name = result.names[class_id]
                        score = float(box.conf[0].item())
                        x1, y1, x2, y2 = box.xyxy[0].tolist()

                        records.append({
                            "Waste Class": class_name,
                            "Confidence (%)": round(score * 100, 2),
                            "X1": round(x1, 1),
                            "Y1": round(y1, 1),
                            "X2": round(x2, 1),
                            "Y2": round(y2, 1),
                        })

                    st.session_state.image_original = image.copy()
                    st.session_state.image_annotated = annotated_rgb
                    st.session_state.image_records = records

                    st.session_state.video_original = None
                    st.session_state.video_annotated = None
                    st.session_state.video_records = []
                    st.session_state.video_summary = []
                    st.session_state.video_diagnostics = {}
                    st.session_state.video_processed = False

                    st.success("Image detection completed.")

                except Exception as exc:
                    st.error(f"Image detection failed: {exc}")

    # -----------------------------------------------------
    # VIDEO DETECTION
    # -----------------------------------------------------

    else:
        uploaded_video = st.file_uploader(
            "Drag and drop a video, or browse files",
            type=["mp4", "avi", "mov", "mkv"],
            key="video_upload",
        )

        if uploaded_video is not None:
            st.caption(f"Selected video: {uploaded_video.name}")

            if st.button(
                "🎬 Process Video",
                type="primary",
                key="process_video",
            ):
                progress = st.progress(0.0)
                status = st.empty()

                try:
                    original_bytes = uploaded_video.getvalue()

                    with tempfile.TemporaryDirectory() as temp_dir:
                        temp_dir = Path(temp_dir)

                        suffix = (
                            Path(uploaded_video.name).suffix.lower()
                            or ".mp4"
                        )

                        input_path = temp_dir / f"input{suffix}"
                        output_path = temp_dir / "annotated_video.mp4"
                        sample_path = temp_dir / "sample_frame.jpg"

                        input_path.write_bytes(original_bytes)

                        cap = cv2.VideoCapture(str(input_path))

                        if not cap.isOpened():
                            raise RuntimeError(
                                "Could not open the uploaded video."
                            )

                        fps = cap.get(cv2.CAP_PROP_FPS)
                        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        total_frames = int(
                            cap.get(cv2.CAP_PROP_FRAME_COUNT)
                        )

                        if width <= 0 or height <= 0:
                            cap.release()
                            raise RuntimeError(
                                "The uploaded video has invalid dimensions."
                            )

                        if not fps or fps <= 0:
                            fps = 25.0

                        # Make an even-sized frame for MP4 compatibility.
                        output_width = width if width % 2 == 0 else width - 1
                        output_height = height if height % 2 == 0 else height - 1

                        writer = cv2.VideoWriter(
                            str(output_path),
                            cv2.VideoWriter_fourcc(*"mp4v"),
                            fps,
                            (output_width, output_height),
                        )

                        if not writer.isOpened():
                            cap.release()
                            raise RuntimeError(
                                "Could not initialize video output writer."
                            )

                        frame_count = 0
                        frames_with_detections = 0
                        total_boxes = 0
                        unique_tracks = set()
                        confidence_by_track = {}
                        sample_annotated = None
                        sample_original = None

                        # Process each frame and save the annotated version.
                        try:
                            while True:
                                ok, frame = cap.read()

                                if not ok:
                                    break

                                if (
                                    frame.shape[1] != output_width
                                    or frame.shape[0] != output_height
                                ):
                                    frame = cv2.resize(
                                        frame,
                                        (output_width, output_height),
                                    )

                                result = model.track(
                                    source=frame,
                                    conf=confidence,
                                    imgsz=512,
                                    persist=True,
                                    tracker="bytetrack.yaml",
                                    verbose=False,
                                )[0]

                                # Annotated BGR frame includes boxes,
                                # predicted classes and confidence values.
                                annotated = result.plot(
                                    boxes=True,
                                    labels=True,
                                    conf=True,
                                )

                                boxes = result.boxes
                                frame_has_detections = (
                                    boxes is not None and len(boxes) > 0
                                )

                                if frame_has_detections:
                                    frames_with_detections += 1
                                    total_boxes += len(boxes)

                                    for i, box in enumerate(boxes):
                                        class_id = int(box.cls[0].item())
                                        class_name = result.names[class_id]
                                        score = float(box.conf[0].item())

                                        if boxes.id is not None:
                                            track_id = int(
                                                boxes.id[i].item()
                                            )
                                            track_key = (
                                                class_name,
                                                track_id,
                                            )
                                            unique_tracks.add(track_key)

                                            confidence_by_track.setdefault(
                                                track_key, []
                                            ).append(score)

                                # Save the first frame containing detections
                                # for a visible preview of actual predictions.
                                if (
                                    sample_annotated is None
                                    and frame_has_detections
                                ):
                                    sample_annotated = annotated.copy()
                                    sample_original = frame.copy()

                                writer.write(annotated)
                                frame_count += 1

                                if total_frames > 0:
                                    progress.progress(
                                        min(
                                            frame_count / total_frames,
                                            1.0,
                                        )
                                    )

                                status.write(
                                    f"Frames processed: {frame_count} | "
                                    f"Frames with detections: "
                                    f"{frames_with_detections} | "
                                    f"Bounding boxes: {total_boxes}"
                                )

                        finally:
                            cap.release()
                            writer.release()

                        if (
                            not output_path.exists()
                            or output_path.stat().st_size == 0
                        ):
                            raise RuntimeError(
                                "No valid annotated video was generated."
                            )

                        annotated_video_bytes = output_path.read_bytes()

                        # Save a JPEG sample frame so users can immediately
                        # see boxes even if browser video playback is blank.
                        if sample_annotated is not None:
                            ok, encoded = cv2.imencode(
                                ".jpg", sample_annotated
                            )

                            if ok:
                                sample_path.write_bytes(encoded.tobytes())
                                sample_image_bytes = sample_path.read_bytes()
                            else:
                                sample_image_bytes = None
                        else:
                            sample_image_bytes = None

                    # Build per-track results.
                    records = []

                    for class_name, track_id in sorted(unique_tracks):
                        scores = confidence_by_track.get(
                            (class_name, track_id), []
                        )

                        records.append({
                            "Waste Class": class_name,
                            "Track ID": track_id,
                            "Average Confidence (%)": round(
                                sum(scores) / len(scores) * 100, 2
                            ) if scores else 0.0,
                            "Frames Detected": len(scores),
                        })

                    if records:
                        track_df = pd.DataFrame(records)
                        counts = (
                            track_df["Waste Class"]
                            .value_counts()
                            .to_dict()
                        )
                    else:
                        counts = {}

                    total_tracks = len(records)
                    summary = []

                    for class_name in CLASS_NAMES.values():
                        count = int(counts.get(class_name, 0))

                        summary.append({
                            "Waste Class": class_name,
                            "Unique Tracked Objects": count,
                            "Composition (%)": round(
                                count / total_tracks * 100, 2
                            ) if total_tracks else 0.0,
                        })

                    st.session_state.video_original = original_bytes
                    st.session_state.video_annotated = annotated_video_bytes
                    st.session_state.video_records = records
                    st.session_state.video_summary = summary
                    st.session_state.video_processed = True
                    st.session_state.video_diagnostics = {
                        "frames_processed": frame_count,
                        "frames_with_detections": frames_with_detections,
                        "total_boxes": total_boxes,
                        "sample_image": sample_image_bytes,
                    }

                    st.session_state.image_original = None
                    st.session_state.image_annotated = None
                    st.session_state.image_records = []

                    progress.progress(1.0)
                    status.success(
                        f"Completed: {frame_count} frames processed."
                    )

                    if total_boxes:
                        st.success(
                            f"Found {total_boxes} detections across "
                            f"{frames_with_detections} frames."
                        )
                    else:
                        st.warning(
                            "No objects detected. Try lowering confidence "
                            "to 0.05–0.10, or test a clearer video."
                        )

                except Exception as exc:
                    st.error(f"Video processing failed: {exc}")


# =========================================================
# RIGHT: MEDIA PREVIEW
# =========================================================

with preview_col:
    st.markdown(
        '<div class="section-title">Media Preview</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="section-subtitle">'
        'See the original media and annotated AI predictions.'
        '</div>',
        unsafe_allow_html=True,
    )

    # IMAGE PREVIEW
    if is_image and st.session_state.image_annotated is not None:
        detected_tab, original_tab = st.tabs(
            ["🎯 AI Detection", "Original Image"]
        )

        with detected_tab:
            st.image(
                st.session_state.image_annotated,
                caption="Bounding boxes show predicted class and confidence",
                use_container_width=True,
            )

            records = st.session_state.image_records

            if records:
                st.success(f"{len(records)} object(s) detected.")

                for i, record in enumerate(records, start=1):
                    st.markdown(
                        f"**{i}. {record['Waste Class']}** — "
                        f"{record['Confidence (%)']:.2f}% confidence"
                    )
            else:
                st.warning(
                    "No detections at this threshold. "
                    "Try a lower confidence value."
                )

        with original_tab:
            st.image(
                st.session_state.image_original,
                caption="Original uploaded image",
                use_container_width=True,
            )

    # VIDEO PREVIEW
    elif is_video and st.session_state.video_processed:
        diagnostics = st.session_state.video_diagnostics

        # Show an annotated still image first. This lets users verify
        # that bounding boxes exist independently of video playback.
        sample_bytes = diagnostics.get("sample_image")

        if sample_bytes:
            st.subheader("Detected Objects — Frame Preview")
            st.image(
                sample_bytes,
                caption="An actual processed video frame with predictions",
                use_container_width=True,
            )
        else:
            st.warning(
                "No annotated frame is available because the model "
                "did not detect any objects in the video."
            )

        video_tab, original_tab = st.tabs(
            ["🎯 Annotated Video", "Original Video"]
        )

        with video_tab:
            st.video(st.session_state.video_annotated)

        with original_tab:
            st.video(st.session_state.video_original)

        st.download_button(
            "⬇️ Download Annotated Video",
            data=st.session_state.video_annotated,
            file_name="annotated_waste_video.mp4",
            mime="video/mp4",
            key="download_annotated_video",
        )

    else:
        st.info(
            "Upload an image or video on the left and run detection. "
            "Your prediction preview will appear here."
        )


# =========================================================
# RESULTS UNDER BOTH PANELS
# =========================================================

st.divider()

st.markdown(
    '<div class="section-title">Detection Results</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="section-subtitle">'
    'Predicted classes, confidence percentages, object counts, and composition.'
    '</div>',
    unsafe_allow_html=True,
)


# IMAGE RESULTS
if is_image and st.session_state.image_annotated is not None:
    records = st.session_state.image_records
    total = len(records)

    counts = {
        class_name: sum(
            1 for record in records
            if record["Waste Class"] == class_name
        )
        for class_name in CLASS_NAMES.values()
    }

    c1, c2, c3 = st.columns(3)
    c1.metric("Objects Detected", total)
    c2.metric(
        "Categories Found",
        sum(1 for count in counts.values() if count > 0),
    )
    c3.metric("Available Classes", len(CLASS_NAMES))

    composition = []

    for class_name, count in counts.items():
        composition.append({
            "Waste Class": class_name,
            "Object Count": count,
            "Composition (%)": round(
                count / total * 100, 2
            ) if total else 0.0,
        })

    st.subheader("Waste Composition")
    st.dataframe(
        pd.DataFrame(composition),
        use_container_width=True,
        hide_index=True,
    )

    st.subheader("Individual Predictions")

    if records:
        results_df = pd.DataFrame(records)
        st.dataframe(
            results_df,
            use_container_width=True,
            hide_index=True,
        )

        st.download_button(
            "⬇️ Download Image Results CSV",
            data=results_df.to_csv(index=False).encode("utf-8"),
            file_name="image_detection_results.csv",
            mime="text/csv",
        )
    else:
        st.warning("No objects detected in the image.")


# VIDEO RESULTS
elif is_video and st.session_state.video_processed:
    diagnostics = st.session_state.video_diagnostics
    records = st.session_state.video_records
    summary = st.session_state.video_summary

    total_tracks = len(records)
    categories = sum(
        1 for row in summary
        if row["Unique Tracked Objects"] > 0
    )

    c1, c2, c3 = st.columns(3)
    c1.metric("Unique Tracked Objects", total_tracks)
    c2.metric("Categories Found", categories)
    c3.metric(
        "Frames with Detections",
        diagnostics.get("frames_with_detections", 0),
    )

    st.subheader("Video Detection Summary")
    st.dataframe(
        pd.DataFrame(summary),
        use_container_width=True,
        hide_index=True,
    )

    st.subheader("Detected Objects and Confidence")

    if records:
        results_df = pd.DataFrame(records)
        st.dataframe(
            results_df,
            use_container_width=True,
            hide_index=True,
        )

        st.download_button(
            "⬇️ Download Video Detection Results CSV",
            data=results_df.to_csv(index=False).encode("utf-8"),
            file_name="video_detection_results.csv",
            mime="text/csv",
        )
    else:
        st.warning(
            "No track IDs were recorded. See the diagnostic counts below. "
            "Some detections may not receive a tracking ID."
        )

    d1, d2, d3 = st.columns(3)
    d1.metric(
        "Frames Processed",
        diagnostics.get("frames_processed", 0),
    )
    d2.metric(
        "Total Bounding Boxes",
        diagnostics.get("total_boxes", 0),
    )
    d3.metric(
        "Frames with Detections",
        diagnostics.get("frames_with_detections", 0),
    )

    if diagnostics.get("total_boxes", 0) == 0:
        st.warning(
            "The model returned zero boxes at the selected threshold. "
            "Lower the confidence slider and process the video again."
        )

    st.caption(
        "Average confidence is calculated from detections assigned to "
        "each track. Tracking may occasionally split one physical object "
        "into multiple IDs or miss objects."
    )

else:
    st.info(
        "Run detection to view object names, confidence percentages, "
        "and waste composition."
    )


# =========================================================
# FOOTER
# =========================================================

st.markdown(
    """
    <div class="footer-text">
        AI Waste Detection Website · YOLOv8n ·
        Plastic Bottle · Paper · Can · Plastic Bag
    </div>
    """,
    unsafe_allow_html=True,
)
