import json
from pathlib import Path
import cv2
import numpy as np
from shapely.geometry import Polygon
import torch
import torch.nn as nn
from torchvision import transforms, models
import csv
import shutil

# These are the file paths and the runtime configurations.
CLASSIFIER = "occupancy_classifier.pt"
DEVICE = "cpu"
VIDEO = "data/lot.mp4"
LIST = "spots_video.json"
REFERENCE = "data/frame_video.jpg"
INTERVAL = 5
ADJACENT = 9
ALIGNMENT = True
SAVE = True
OUT_DIR = Path("output")
CSV_PATH = OUT_DIR / "occupancy_over_time.csv"
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

# Here, I transform the images to match the needed training input format.
image_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])

# This function filters through and returns the assigned spots in spots_video.json.
def load_spots(path: str):
    list = json.loads(Path(path).read_text())
    final = []
    for spot in list:
        points = spot.get("points")
        if not points or len(points) != 4:
            continue
        shape = Polygon(points)
        if not shape.is_valid:
            shape = shape.buffer(0)
        if shape.is_empty or shape.area <= 0:
            continue
        final.append({"id": spot.get("id", len(final) + 1), "poly": shape})
    return final

# Here, I initialize the ResNet18 architecture and restore learned occupancy classification weights.
def deploy():
    model = models.resnet18(weights = None)
    model.fc = nn.Linear(model.fc.in_features, 2)
    model.load_state_dict(torch.load(CLASSIFIER, map_location = DEVICE))
    model.to(DEVICE)
    model.eval()
    return model

# This function identifies whether a spot is occupied or not using the trained classifier.
def classify(frame, spot, model):
    threshold = 0.7
    # I crop out the specific image region here.
    minx, miny, maxx, maxy = map(int, spot.bounds)
    minx = max(minx, 0)
    miny = max(miny, 0)
    maxx = min(maxx, frame.shape[1] - 1)
    maxy = min(maxy, frame.shape[0] - 1)
    crop = frame[miny:maxy, minx:maxx]
    if crop.size == 0:
        return False, 0.0
    x = image_transform(crop).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        logits = model(x)
    # The classifier is run to determine whether an image meets the confidence threshold of being occupied.
        probabilities = torch.softmax(logits, dim = 1)
        predictions = int(probabilities.argmax(dim = 1).item())
        confidence = float(probabilities.max(dim = 1).values.item())
    occupied = (predictions == 0) and (confidence >= threshold)
    return occupied, confidence

# Here, I check the occupancy in multiple adjacent frames to ensure an accurate prediction.
def predict (frames, spots, model):
    # Here, each frame votes on whether each parking spot is occupied.
    votes = {s["id"]: 0 for s in spots}
    for frame in frames:
        for s in spots:
            occupied, _ = classify(frame, s["poly"], model)
            if occupied:
                votes[s["id"]] += 1
    # If a spot was occupied in the majority of frames, it will be returned as occupied.
    majority = len(frames) // 2
    occupied_ids = []
    for spot, vote_count in votes.items():
        if vote_count > majority:
            occupied_ids.append(spot)

    return occupied_ids

# I draw out the parking spot outlines, then display a summary count.
def draw_overlay(frame, spots, occupied_ids):
    annotated = frame.copy()
    # Parking spot outlines are drawn here, with a red outline representing occupied and a green one representing unoccupied.
    for s in spots:
        pts = [(int(x), int(y)) for x, y in s["poly"].exterior.coords]
        if s["id"] in occupied_ids:
            color = (0, 0, 255)
        else:
            color = (0, 255, 0)
        for i in range(len(pts) - 1):
            cv2.line(annotated, pts[i], pts[i + 1], color, 2)

        c = s["poly"].centroid
        cv2.putText(
            annotated,
            str(s["id"]),
            (int(c.x), int(c.y)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2,
        )
    # A summary of the occupied and unoccupied parking spots is displayed.
    unoccupied = len(spots) - len(occupied_ids)
    cv2.putText(
        annotated,
        f"Occupied: {len(occupied_ids)}  Unoccupied: {unoccupied}  Total: {len(spots)}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (255, 255, 255),
        2,
    )
    return annotated

def main():
    # The list of parking spots is loaded here, as well as reference data needed for frame alignment.
    spots = load_spots(LIST)
    if not spots:
        raise ValueError("The parking spots list is empty")

    ref = cv2.imread(REFERENCE)
    if ref is None:
        raise FileNotFoundError(f"The reference frame is missing: {REFERENCE}")

    height, width = ref.shape[:2]
    grayscale, orb, bf, keypoints, descriptors = align(ref)
    # Here, I opened the video stream and determined how many frames correspond to each snapshot interval
    cap = cv2.VideoCapture(VIDEO)
    if not cap.isOpened():
        raise IOError("The video was not opened")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(INTERVAL * fps)
    # Here, the trained model is loaded and the outputs are initialized.
    clf = deploy()
    if SAVE:
        if OUT_DIR.exists():
            shutil.rmtree(OUT_DIR)
        OUT_DIR.mkdir(parents = True, exist_ok = True)
        csv_file = open(CSV_PATH, "w", newline="")
        writer = csv.writer(csv_file)
        writer.writerow(["time (sec)", "occupied", "unoccupied", "total"])

    snap_idx = 0
    frame_idx = 0
    print(f"FPS = {fps:.2f} | snapshot every {INTERVAL}s ({total} frames) | alignment = {ALIGNMENT}")

    while True:
        # I created an array that stores adjacent image frames to a target time.
        images = []
        half = ADJACENT // 2
        start = max(frame_idx - half, 0)

        # This section iterates over the adjacent frames, aligning each from with the main reference frame.
        for j in range(ADJACENT):
            cap.set(cv2.CAP_PROP_POS_FRAMES, start + j)
            ok, fr = cap.read()
            if not ok:
                continue

            if ALIGNMENT:
                fr_aligned, ok_align = warp(
                    fr,
                    (height, width),
                    grayscale,
                    orb,
                    bf,
                    keypoints,
                    descriptors,
                )
                images.append(fr_aligned)
            else:
                images.append(fr)

        if not images:
            break

        # This section is where each parking spot as classified and recorded as occupied or unoccupied based on the majority of adjacent frames.
        occupied_ids = predict(images, spots, clf)
        annotated = draw_overlay(images[-1], spots, occupied_ids)
        time = frame_idx / fps
        unoccupied = len(spots) - len(occupied_ids)
        print(f"time  ={time:6.1f}s | occupied ={len(occupied_ids):2d} unoccupied ={unoccupied:2d} total ={len(spots):2d}")

        if SAVE:
            output_image = OUT_DIR / f"snapshot_{snap_idx:03d}_t{time:05.1f}s.jpg"
            cv2.imwrite(str(output_image), annotated)
            writer.writerow([f"{time:.1f}", len(occupied_ids), unoccupied, len(spots)])

        # Here, I gave users the ability to use certain keys to switch between images and close the image window.
        title = "Parking Space Occupancy Classifier: (n/space = next image, q = quit)"
        cv2.imshow(title, annotated)
        key = cv2.waitKey(0) & 0xFF

        if key == ord("q"):
            break

        if key in (ord("n"), 32):
            pass

        snap_idx += 1
        frame_idx += total

    if SAVE:
        csv_file.close()

    cap.release()
    cv2.destroyAllWindows()

# This function computes keypoints, descriptors, and matchers from the fixed reference frame.
def align(ref):
    grayscale = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY)
    orb = cv2.ORB_create(nfeatures = 2500)
    keypoints, descriptors = orb.detectAndCompute(grayscale, None)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck = False)
    return grayscale, orb, bf, keypoints, descriptors

# This function warps each video frame to the reference image so that parking spot locations stay consistent even as the camera slightly shifts.
def warp(frame, ref_shape_hw, orb, bf, keypoints, descriptors):
    if descriptors is None or keypoints is None or len(keypoints) < 10:
        return frame, False

    # Here, I check for the visual features in the current frame and compare them to the reference view.
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    current_keypoints, current_descriptors = orb.detectAndCompute(gray, None)
    if current_descriptors is None or current_keypoints is None or len(current_keypoints) < 10:
        return frame, False

    matches = bf.knnMatch(current_descriptors, descriptors, k = 2)
    # Only the strong feature matches are kept to avoid false alignment.
    good = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good.append(m)

    if len(good) < 12:
        return frame, False

    current_points = []
    for match in good:
        idx = match.queryIdx
        point = current_keypoints[idx].pt
        current_points.append(point)
    current_pts = np.float32(current_points).reshape(-1, 1, 2)

    reference_points = []
    for match in good:
        idx = match.trainIdx
        point = keypoints[idx].pt
        reference_points.append(point)
    reference_pts = np.float32(reference_points).reshape(-1, 1, 2)

    # Here, the matched features are used to estimate a geometric transform and warp the frame to the reference layout.
    homography, _ = cv2.findHomography(current_pts, reference_pts, cv2.RANSAC, 5.0)
    if homography is None:
        return frame, False

    height, width = ref_shape_hw
    warped = cv2.warpPerspective(frame, homography, (width, height), flags=cv2.INTER_LINEAR)
    return warped, True

if __name__ == "__main__":
    main()