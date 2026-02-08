import json
from pathlib import Path
import cv2
import numpy as np

#These are the paths and configurations.
REFERENCE = "data/frame_video.jpg"
LABELED = "spots_video.json"
POINTS_PER_SPOT = 4

# I created this function to sort the points chosen by the user into clockwise order.
def order_points_clockwise(points: list[list[int]]) -> list[list[int]]:
    # First, the angle of each chosen point around the center is calculated to order the points circularly.
    array = np.array(points, dtype = np.float32)
    c = array.mean(axis = 0)
    angles = np.arctan2(array[:, 1] - c[1], array[:, 0] - c[0])
    order = np.argsort(angles)
    sort = array[order]
    # Then, the ordered points are rotated so the polygon always starts from the top-left corner.
    sums = sort[:, 0] + sort[:, 1]
    start = int(np.argmin(sums))
    sort = np.roll(sort, -start, axis=0)
    ans = []
    for x, y in sort:
        ans.append([int(x), int(y)])
    return ans

# This function checks if a set of labeled points make up a valid quadrilateral.
def check(points: list[list[int]]) -> bool:
    if len(points) != 4:
        return False
    if len({(p[0], p[1]) for p in points}) != 4:
        return False
    contour = np.array(points, dtype = np.int32)
    return bool(cv2.isContourConvex(contour))

points: list[list[int]] = []
spots: list[dict] = []
spot_id = 1

# I created this function to load the existing points in spots_path.json and check their validity.
try:
    exists = json.loads(Path(LABELED).read_text())

    if isinstance(exists, list):
        valid = []
        for spot in exists:
            if not isinstance(spot, dict):
                continue
            pts = spot.get("points")
            if not isinstance(pts, list):
                continue
            if len(pts) != POINTS_PER_SPOT:
                continue
            valid.append(spot)

        invalid = len(exists) - len(valid)
        if invalid:
            print(f"{invalid} invalid entries in {LABELED}.")

        spots = valid
        if spots:
            ids = []
            for s in spots:
                value = s.get("id")
                if isinstance(value, int):
                    ids.append(value)

            if ids:
                spot_id = max(ids) + 1
            else:
                spot_id = len(spots) + 1
    else:
        spots = []
        spot_id = 1

except FileNotFoundError:
    pass
except Exception:
    print(f"The entries in {LABELED} could not be read.")
    spots = []
    spot_id = 1

# The mouse callback function is used to capture user-clicked corner points when labeling parking spots.
def mouse_callback(event, x, y, flags, param):
    global points, spots, spot_id
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append([int(x), int(y)])
        print(f"Clicked: {x}, {y} (spot {spot_id}, point {len(points)}/{POINTS_PER_SPOT})")

# Here, I load the reference image, initialize the labeling window, and display user instructions.
img = cv2.imread(REFERENCE)
if img is None:
    raise FileNotFoundError(f"Could not read {REFERENCE}")

cv2.namedWindow("Label Spots")
cv2.setMouseCallback("Label Spots", mouse_callback)

print("Instructions:")
print(f"1. Click 4 corners of a parking spot.")
print("2. Once done, press 'n' to save the spot and move on to the next one.")
print("3. Press 'u' to undo the last click.")
print(f"4. Press 'q' to quit the screen and write at {LABELED}.")

while True:
    copy = img.copy()
    for p in points:
        cv2.circle(copy, (p[0], p[1]), 5, (0, 255, 255), -1)
    # Here, all previously saved parking spots are drawn out.
    for s in spots:
        sp = s.get("points")
        if not isinstance(sp, list) or len(sp) != POINTS_PER_SPOT:
            continue
        pts = sp + [sp[0]]
        for i in range(POINTS_PER_SPOT):
            cv2.line(copy, tuple(pts[i]), tuple(pts[i+1]), (0, 255, 0), 2)
        cv2.putText(copy, str(s.get("id", "?")), tuple(sp[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

    cv2.putText(copy, f"Current spot: {spot_id} (points: {len(points)}/{POINTS_PER_SPOT})", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255), 2)

    cv2.imshow("Label Spots", copy)
    key = cv2.waitKey(20) & 0xFF
    # If the user presses "u", the point they most recently clicked will be removed.
    if key == ord('u') and points:
        points.pop()
    # If the user presses "n", the spot they have labeled will be saved, and they can enter points for a new spot.
    if key == ord('n'):
        if len(points) != POINTS_PER_SPOT:
            print(f"Click {POINTS_PER_SPOT} corners before saving.")
            continue

        # This ensures each spot is stored clockwise from the top-left.
        points = order_points_clockwise(points)
        if not check(points):
            print("Invalid spot shape.")
            continue

        # This rejects spots that are too small.
        area = abs(cv2.contourArea(np.array(points, dtype = np.int32)))
        if area < 1000.0:
            print(f"Spot area too small ({area:.1f}px).")
            continue
        # Here, I finalize the current parking spot by copying its points, assigning an ID, and saving it.
        coordinates = []
        for p in points:
            coordinates.append(p[:])

        spot_entry = {"id": spot_id, "points": coordinates}
        spots.append(spot_entry)
        print(f"Saved spot {spot_id}. Total spots: {len(spots)}")
        spot_id += 1
        points = []
    # If the user presses "q", the window closes.
    if key == ord('q'):
        if points:
            print("There are unsaved clicks for the current spot. Press 'n' to save it.")
        break

cv2.destroyAllWindows()
# Here, all saved spots are re-validated.
valid_spots = []
for spot in spots:
    if not isinstance(spot, dict):
        continue
    points = spot.get("points")
    if not isinstance(points, list):
        continue
    if len(points) != POINTS_PER_SPOT:
        continue
    valid_spots.append(spot)

if len(valid_spots) != len(spots):
    print(f"Dropping {len(spots) - len(valid_spots)} invalid spot(s).")

Path(LABELED).write_text(json.dumps(valid_spots, indent = 2))
print(f"Wrote {len(valid_spots)} spots to {LABELED}")