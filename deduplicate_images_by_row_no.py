"""
按截图中的 row_no 列对图片做初步去重。

判断规则：
1. 只 OCR 图片中 x=270~432 的 row_no 列。
2. 一张图片的所有 row_no 数值组成一个元组，例如 ("0", "1", ..., "26")。
3. 两张图片的元组完全相同时，只保留按文件名排序后的第一张。
4. 默认把重复图片移动到 duplicate_images 文件夹；使用 --delete 时永久删除。
5. OCR 失败、结果为空或含非数字内容时，为避免误删，该图片不会参与去重。

示例：
    python deduplicate_images_by_row_no.py images_test
    python deduplicate_images_by_row_no.py images_test --dry-run
    python deduplicate_images_by_row_no.py images_test --delete
"""

import argparse
import csv
import re
import shutil
import sys
from pathlib import Path

import cv2
from paddleocr import PaddleOCR


# 与 new_data_timestamp.py 中的表格位置一致。
ROW_NO_X1 = 270
ROW_NO_X2 = 432
START_Y = 3
ROW_HEIGHT = 27
NUM_ROWS_PER_IMAGE = 27

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

ocr = None


def get_ocr():
    """延迟加载 OCR 模型，整个程序只加载一次。"""
    global ocr
    if ocr is None:
        ocr = PaddleOCR(
            lang="en",
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
        )
    return ocr


def collect_images(image_folder):
    """按文件名排序，确保重复组中总是保留第一张截图。"""
    folder = Path(image_folder)
    return sorted(
        (
            path
            for path in folder.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        ),
        key=lambda path: path.name,
    )


def extract_texts(result):
    """兼容 PaddleOCR 新、旧版结果结构，提取所有识别文本。"""
    texts = []

    for item in result or []:
        getter = getattr(item, "get", None)
        if callable(getter):
            try:
                rec_texts = getter("rec_texts", [])
            except Exception:
                rec_texts = []
            if rec_texts:
                texts.extend(str(text) for text in rec_texts)
                continue

            try:
                text = getter("text", "")
            except Exception:
                text = ""
            if text:
                texts.append(str(text))
                continue

        # 兼容 PaddleOCR 旧版：[[box, (text, score)], ...]
        if isinstance(item, list):
            for line in item:
                try:
                    texts.append(str(line[1][0]))
                except (IndexError, TypeError):
                    pass

    return texts


def recognize_row_no_sequence(image_path):
    """
    只识别一张图片的 row_no 列。

    将整列一次性送入 OCR，比逐个单元格调用 OCR 更适合大批量预去重。
    返回 (row_no 元组, 错误原因)。元组为 None 时禁止该图片参与自动去重。
    """
    image = cv2.imread(str(image_path))
    if image is None:
        return None, "无法读取图片"

    height, width = image.shape[:2]
    x1 = max(0, min(ROW_NO_X1, width))
    x2 = max(0, min(ROW_NO_X2, width))
    y1 = max(0, min(START_Y, height))
    y2 = max(0, min(START_Y + ROW_HEIGHT * NUM_ROWS_PER_IMAGE, height))

    if x1 >= x2 or y1 >= y2:
        return None, f"row_no 裁剪区域无效，图片尺寸为 {width}x{height}"

    row_no_column = image[y1:y2, x1:x2]

    try:
        result = get_ocr().predict(row_no_column)
    except Exception as exc:
        return None, f"OCR 失败：{exc}"

    raw_texts = extract_texts(result)
    row_numbers = []

    for raw_text in raw_texts:
        # 单次识别结果偶尔可能包含换行，因此提取其中的每一段数字。
        parts = re.findall(r"\d+", raw_text.replace(" ", ""))
        row_numbers.extend(parts)

    if not row_numbers:
        return None, "未识别到任何 row_no"

    # 样例截图固定有 27 行。少识别或多识别都可能造成错误的重复判断，因此
    # 只有完整识别出 27 个 row_no 时才允许自动去重。
    if len(row_numbers) != NUM_ROWS_PER_IMAGE:
        return (
            None,
            f"只识别到 {len(row_numbers)}/{NUM_ROWS_PER_IMAGE} 个 row_no，"
            "为避免误删已跳过",
        )

    return tuple(row_numbers), ""


def unique_destination(destination_folder, filename):
    """避免隔离目录中已有同名文件时被覆盖。"""
    destination = destination_folder / filename
    if not destination.exists():
        return destination

    stem = Path(filename).stem
    suffix = Path(filename).suffix
    index = 1
    while True:
        destination = destination_folder / f"{stem}_duplicate_{index}{suffix}"
        if not destination.exists():
            return destination
        index += 1


def write_report(report_path, rows):
    """输出 CSV 报告，方便核对每张图片的识别结果和处理动作。"""
    with report_path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "image",
                "row_no_sequence",
                "status",
                "kept_image",
                "message",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def show_progress(current, total, duplicate_count, skipped_count):
    """在同一行刷新进度，避免为大量图片逐张打印日志。"""
    bar_width = 30
    ratio = current / total if total else 1
    completed_width = int(bar_width * ratio)
    bar = "█" * completed_width + "░" * (bar_width - completed_width)
    kept_count = current - duplicate_count - skipped_count

    sys.stdout.write(
        f"\r[{bar}] {current}/{total} ({ratio:6.2%}) "
        f"保留:{kept_count} 重复:{duplicate_count} 跳过:{skipped_count}"
    )
    sys.stdout.flush()

    if current == total:
        sys.stdout.write("\n")


def deduplicate_images(image_folder, action="move"):
    """
    action:
        dry-run - 只报告，不改动图片
        move    - 把重复图片移入 duplicate_images（默认）
        delete  - 永久删除重复图片
    """
    folder = Path(image_folder).resolve()
    if not folder.is_dir():
        raise NotADirectoryError(f"图片文件夹不存在：{folder}")

    image_paths = collect_images(folder)
    if not image_paths:
        print(f"未找到图片：{folder}")
        return

    duplicate_folder = folder / "duplicate_images"
    if action == "move":
        duplicate_folder.mkdir(exist_ok=True)

    first_image_by_sequence = {}
    report_rows = []
    duplicate_count = 0
    skipped_count = 0

    total_images = len(image_paths)
    print(f"共找到 {total_images} 张图片，开始识别 row_no 列……")
    show_progress(0, total_images, duplicate_count, skipped_count)

    for index, image_path in enumerate(image_paths, start=1):
        sequence, error = recognize_row_no_sequence(image_path)

        if sequence is None:
            skipped_count += 1
            report_rows.append(
                {
                    "image": image_path.name,
                    "row_no_sequence": "",
                    "status": "skipped",
                    "kept_image": "",
                    "message": error,
                }
            )
            show_progress(
                index, total_images, duplicate_count, skipped_count
            )
            continue

        sequence_text = ",".join(sequence)
        kept_image = first_image_by_sequence.get(sequence)

        if kept_image is None:
            first_image_by_sequence[sequence] = image_path
            report_rows.append(
                {
                    "image": image_path.name,
                    "row_no_sequence": sequence_text,
                    "status": "kept",
                    "kept_image": image_path.name,
                    "message": "",
                }
            )
            show_progress(
                index, total_images, duplicate_count, skipped_count
            )
            continue

        duplicate_count += 1
        if action == "delete":
            image_path.unlink()
            status = "deleted"
            message = "重复图片已永久删除"
        elif action == "move":
            destination = unique_destination(duplicate_folder, image_path.name)
            shutil.move(str(image_path), str(destination))
            status = "moved"
            message = f"已移动到 {destination}"
        else:
            status = "duplicate-dry-run"
            message = "仅预览，文件未改动"

        report_rows.append(
            {
                "image": image_path.name,
                "row_no_sequence": sequence_text,
                "status": status,
                "kept_image": kept_image.name,
                "message": message,
            }
        )
        show_progress(
            index, total_images, duplicate_count, skipped_count
        )

    report_path = folder / "row_no_dedup_report.csv"
    write_report(report_path, report_rows)

    print("\n处理完成")
    print(f"  图片总数：{len(image_paths)}")
    print(f"  重复图片：{duplicate_count}")
    print(f"  安全跳过：{skipped_count}")
    print(f"  处理报告：{report_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="只 OCR row_no 列，并删除或隔离 row_no 完全相同的重复截图。"
    )
    parser.add_argument("image_folder", help="待去重图片所在文件夹")

    action_group = parser.add_mutually_exclusive_group()
    action_group.add_argument(
        "--dry-run",
        action="store_true",
        help="只识别和生成报告，不移动或删除图片",
    )
    action_group.add_argument(
        "--delete",
        action="store_true",
        help="永久删除重复图片（默认行为是移动到 duplicate_images）",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    selected_action = "delete" if args.delete else "dry-run" if args.dry_run else "move"
    deduplicate_images(args.image_folder, action=selected_action)
