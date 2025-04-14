import argparse
import os
import cv2
from ultralytics import YOLO
from acw2_utils import CLASS_MAPPING, normalize_bbox



# # Dictionary mapping class index to (sign_number, sign_name)
# CLASS_MAPPING = {
#     0: (1, "Roundabout"),
#     1: (4, "Traffic lights"),
#     2: (5, "Roadworks"),
#     3: (13, "No entry"),
#     4: (16, "30MPH"),
#     5: (19, "National speed limit")
# }

# def normalize_bbox(xyxy, image_shape):
#     x1, y1, x2, y2 = xyxy
#     h, w = image_shape[:2]
#     x_centre = ((x1 + x2) / 2) / w
#     y_centre = ((y1 + y2) / 2) / h
#     width = (x2 - x1) / w
#     height = (y2 - y1) / h
#     return x_centre, y_centre, width, height

def process_image(image_path, output_path, model):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image from {image_path}")
        return

    image_name = os.path.basename(image_path)
    
    results = model(image)[0]  # Run inference on image (NOT path)
    
    print(f"[DEBUG] Writing output to: {output_path}")

    with open(output_path, 'w') as f:
        for box in results.boxes:
            cls = int(box.cls[0].item())
            conf = float(box.conf[0].item())
            xyxy = box.xyxy[0].tolist()  # [x1, y1, x2, y2]

            if cls not in CLASS_MAPPING:
                continue

            sign_number, sign_name = CLASS_MAPPING[cls]
            x_c, y_c, w, h = normalize_bbox(xyxy, image.shape)
            line = f"{image_name},{sign_number},{sign_name},{x_c:.3f},{y_c:.3f},{w:.3f},{h:.3f},0,0,{conf:.3f}\n"
            f.write(line)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', type=str, required=True, help="Input image path")
    parser.add_argument('--output', type=str, default="output.txt", help="Output text file")
    parser.add_argument('--interactive', action='store_true', help="Enable optional visualization mode")
    args = parser.parse_args()

    model = YOLO("runs/detect/train72/weights/best.pt")

    process_image(args.image, args.output, model)

    if args.interactive:
        img = cv2.imread(args.image)
        results = model(args.image)[0]
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            _, sign_name = CLASS_MAPPING.get(cls, ("?", "?"))
            label = f"{sign_name} {conf:.2f}"
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.imshow("Detections", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
