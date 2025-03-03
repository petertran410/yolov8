import json


def check_annotations(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    categories = {cat["id"]: cat["name"] for cat in data["categories"]}
    category_counts = {cat["name"]: 0 for cat in data["categories"]}

    for ann in data["annotations"]:
        cat_name = categories.get(ann["category_id"], "Unknown")
        category_counts[cat_name] += 1

    print(f"📊 Thống kê số lượng annotations theo danh mục trong {json_path}:")
    for cat, count in category_counts.items():
        print(f" - {cat}: {count} annotations")


# Chạy kiểm tra cho từng tập dữ liệu
for dataset_type in ["train", "valid", "test"]:
    check_annotations(
        f"D://DATASET//recycle-trash-yolo-v8.v43i.coco//{dataset_type}//_annotations.coco.json")
