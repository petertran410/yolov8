import json
import os


def fix_coco_format(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Kiểm tra xem file có đúng COCO format không
    if "images" not in data or "annotations" not in data or "categories" not in data:
        print(f"⚠️ Lỗi: {json_path} không đúng định dạng COCO!")
        return

    # Tạo một từ điển để hợp nhất categories theo tên
    category_mapping = {}
    new_categories = []
    new_category_id = 1  # COCO index bắt đầu từ 1

    for cat in data["categories"]:
        # Convert về lowercase để tránh sai khác do chữ hoa/thường
        cat_name = cat["name"].lower()
        if cat_name not in category_mapping:
            category_mapping[cat_name] = new_category_id
            new_categories.append({"id": new_category_id, "name": cat_name})
            new_category_id += 1

    # Cập nhật lại category_id trong annotations theo mapping mới
    for ann in data["annotations"]:
        old_id = ann["category_id"]  # Lấy ID cũ
        cat_name = next((c["name"]
                        for c in data["categories"] if c["id"] == old_id), None)
        if cat_name:
            ann["category_id"] = category_mapping[cat_name.lower()]

    # Gán lại danh mục đã hợp nhất
    data["categories"] = new_categories

    # Tạo file JSON mới
    output_path = json_path.replace(
        "_annotations.coco.json", "_annotations_fixed.coco.json")

    # Lưu file JSON đã sửa
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)

    print(f"✅ Đã chuẩn hóa {json_path} và lưu tại: {output_path}")


# Chạy script cho train, valid, test
dataset_paths = [
    r"D:/DATASET/recycle-trash-yolo-v8.v43i.coco/test/_annotations.coco.json",
    r"D:/DATASET/recycle-trash-yolo-v8.v43i.coco/train/_annotations.coco.json",
    r"D:/DATASET/recycle-trash-yolo-v8.v43i.coco/valid/_annotations.coco.json"
]

for path in dataset_paths:
    fix_coco_format(path)
