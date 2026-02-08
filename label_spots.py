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

# This function loads the existing points in spots_path.json if it already exists and if the points are valid.
try:
    exists = json.loads(Path(LABELED).read_text())
    if isinstance(exists, list):
        validity = [s for s in exists if isinstance(s, dict) and isinstance(s.get("points"), list) and len(s.get("points")) == POINTS_PER_SPOT]
        invalid_count = len(exists) - len(validity)
        if invalid_count:
            print(f"{invalid_count} entries in {LABELED} are not valid and will be ignored.")
        spots = validity
        # This sets the function to the next id.
        if spots:
            try:
                spot_id = max(int(s.get("id", 0)) for s in spots) + 1
            except Exception:
                spot_id = len(spots) + 1
except FileNotFoundError:
    pass
except Exception as e:
    print(f"The entries in {LABELED} could not be determined. Restarting.")
    spots = []
    spot_id = 1

def mouse_callback(event, x, y, flags, param):
    global points, spots, spot_id
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append([int(x), int(y)])
        print(f"Clicked: {x}, {y} (spot {spot_id}, point {len(points)}/{POINTS_PER_SPOT})")

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
    vis = img.copy()
    for p in points:
        cv2.circle(vis, (p[0], p[1]), 5, (0, 255, 255), -1)

    # Here, the spots are drawn out.
    for s in spots:
        sp = s.get("points")
        if not isinstance(sp, list) or len(sp) != POINTS_PER_SPOT:
            continue
        pts = sp + [sp[0]]
        for i in range(POINTS_PER_SPOT):
            cv2.line(vis, tuple(pts[i]), tuple(pts[i+1]), (0, 255, 0), 2)
        cv2.putText(vis, str(s.get("id", "?")), tuple(sp[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

    cv2.putText(vis, f"Current spot: {spot_id} (points: {len(points)}/{POINTS_PER_SPOT})", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255), 2)

    cv2.imshow("Label Spots", vis)
    key = cv2.waitKey(20) & 0xFF

    if key == ord('u') and points:
        points.pop()

    if key == ord('n'):
        if len(points) != POINTS_PER_SPOT:
            print(f"Need exactly {POINTS_PER_SPOT} points before saving. Click {POINTS_PER_SPOT} corners first.")
            continue

        # This normalizes the point order.
        points = order_points_clockwise(points)
        if not check(points):
            print("Invalid spot shape. Re-label the spot.")
            continue

        # This uses contour area to ensure that the polygon is drawn to be the proper shape.
        area = abs(cv2.contourArea(np.array(points, dtype = np.int32)))
        if area < 1000.0:
            print(f"Spot area too small ({area:.1f}px). Re-label this spot.")
            continue

        # Deep-copy points so later edits can't affect saved spots.
        spots.append({"id": spot_id, "points": [p[:] for p in points]})
        print(f"Saved spot {spot_id}. Total spots: {len(spots)}")
        spot_id += 1
        points = []

    if key == ord('q'):
        if points:
            print("WARNING: You have unsaved clicks for the current spot. Press 'n' to save it or 'u' to undo.")
        break

cv2.destroyAllWindows()

valid_spots = [s for s in spots if isinstance(s, dict) and isinstance(s.get("points"), list) and len(s.get("points")) == POINTS_PER_SPOT]
if len(valid_spots) != len(spots):
    print(f"WARNING: Dropping {len(spots) - len(valid_spots)} invalid spot(s) before writing.")

Path(LABELED).write_text(json.dumps(valid_spots, indent = 2))
print(f"Wrote {len(valid_spots)} spots to {LABELED}")