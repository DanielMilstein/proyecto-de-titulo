from pathlib import Path
from huggingface_hub import hf_hub_download
import torch

from PIL import Image, ImageDraw

repo_id = "Javiai/3dprintfails-yolo5vs"
filename = "model_torch.pt"

model_path = hf_hub_download(repo_id=repo_id, filename=filename)

model = torch.hub.load('Ultralytics/yolov5', 'custom', model_path, verbose = False)


output_dir = Path("outputs")
output_dir.mkdir(parents=True, exist_ok=True)

# load images from imagenes directory
valid_suffixes = {".jpg", ".jpeg", ".png", ".webp"}

for image_path in Path("imagenes").iterdir():
    if image_path.suffix.lower() not in valid_suffixes:
        continue

    image = Image.open(image_path)

    draw = ImageDraw.Draw(image)

    detections = model(image)

    categories = [        
    {'name': 'error', 'color': (0,0,255)},
    {'name': 'extrusor', 'color': (0,255,0)},
    {'name': 'part', 'color': (255,0,0)},
    {'name': 'spaghetti', 'color': (0,0,255)}
    ]


    for detection in detections.xyxy[0]:
        x1, y1, x2, y2, p, category_id = detection
        x1, y1, x2, y2, category_id = int(x1), int(y1), int(x2), int(y2), int(category_id)
        draw.rectangle((x1, y1, x2, y2), 
                        outline=categories[category_id]['color'], 
                        width=1)
        draw.text((x1, y1), categories[category_id]['name'], 
                    categories[category_id]['color'])

    output_path = output_dir / f"{image_path.stem}_detections{image_path.suffix}"
    image.save(output_path)
    print(f"Saved detections to {output_path}")