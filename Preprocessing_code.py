import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

def resize_with_aspect_ratio(image, target_width, target_height):
    h, w = image.shape[:2]
    scale = min(target_width / w, target_height / h)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    canvas = np.full((target_height, target_width, 3), 255, dtype=np.uint8)  # White background
    x_offset = (target_width - new_w) // 2
    y_offset = (target_height - new_h) // 2
    canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
    return canvas

def preprocess_image(image_path, target_width=224, target_height=224):
    image = cv2.imread(image_path)
    if image is None:
        print(f"⚠️ Failed to load image: {image_path}")
        return None

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 11, 2)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        padding = 10
        x = max(0, x - padding)
        y = max(0, y - padding)
        w = min(image.shape[1] - x, w + 2 * padding)
        h = min(image.shape[0] - y, h + 2 * padding)
        cropped_image = image[y:y+h, x:x+w]
        resized_image = resize_with_aspect_ratio(cropped_image, target_width, target_height)
        return resized_image
    else:
        print(f"⚠️ No contours found in image: {image_path}")
        return None

def preprocess_and_split_dataset(input_base_folder, output_base_folder, train_ratio=0.85, val_ratio=0.15):
    if not os.path.exists(output_base_folder):
        os.makedirs(output_base_folder)
    
    for split in ['train', 'test']:
        split_path = os.path.join(input_base_folder, split)
        if not os.path.isdir(split_path):
            print(f"⚠️ {split_path} does not exist.")
            continue

        for category in os.listdir(split_path):
            category_path = os.path.join(split_path, category)
            if not os.path.isdir(category_path):
                continue

            images = [f for f in os.listdir(category_path) if f.lower().endswith(('png', 'jpg', 'jpeg', 'bmp', 'tiff'))]
            if not images:
                continue

            if split == 'train':
                train_images, val_images = train_test_split(images, test_size=val_ratio/(train_ratio + val_ratio), random_state=42)
                for dataset_type, img_list in zip(['train', 'val'], [train_images, val_images]):
                    output_dir = os.path.join(output_base_folder, dataset_type, category)
                    os.makedirs(output_dir, exist_ok=True)
                    for img in img_list:
                        img_path = os.path.join(category_path, img)
                        preprocessed_img = preprocess_image(img_path)
                        if preprocessed_img is not None:
                            save_path = os.path.join(output_dir, img)
                            cv2.imwrite(save_path, preprocessed_img)
                            print(f"✅ Saved: {save_path}")
            else:
                test_output_dir = os.path.join(output_base_folder, 'test', category)
                os.makedirs(test_output_dir, exist_ok=True)
                for img in images:
                    img_path = os.path.join(category_path, img)
                    preprocessed_img = preprocess_image(img_path)
                    if preprocessed_img is not None:
                        save_path = os.path.join(test_output_dir, img)
                        cv2.imwrite(save_path, preprocessed_img)
                        print(f"✅ Saved: {save_path}")

def preview_image(image_path, new_width, new_height):
    image = cv2.imread(image_path)
    if image is None:
        print(f"⚠️ Failed to load image: {image_path}")
        return

    resized_image = preprocess_image(image_path, new_width, new_height)
    if resized_image is None:
        return

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title('Original Image')
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.title('Preprocessed Image')
    plt.imshow(cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()

# === Run Preprocessing ===
input_base_folder = 'dataset_images'  # Change to your dataset location
output_base_folder = 'ecg-classification-project/data/preprocessed'
preprocess_and_split_dataset(input_base_folder, output_base_folder)

# === Preview an Example Image ===
image_path = 'dataset_images/test/ECG Images of Patient that have History of MI (172x12=2064)/test (1).jpg'
preview_image(image_path, 960, 540)