import cv2
import re
import time
import pandas as pd
import multiprocessing as mp
from pathlib import Path
from paddleocr import PaddleOCR


# =========================
# 1. 初始化 OCR
# =========================
# 延迟初始化：串行模式只加载一份模型；并行模式中由两个子进程各自加载一份，
# 主进程不额外占用模型内存。
ocr = None


def get_ocr():
    global ocr

    if ocr is None:
        ocr = PaddleOCR(
            lang="en",
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False
        )

    return ocr


# =========================
# 2. 表格切割参数
# =========================
# 当前图片结构：
# 左侧序号（跳过）/ ID / ...（跳过）/ row_no / 空白分隔列（跳过）/
# value_1 / value_2 / value_3 / value_4 / value_5
#
# COLS 只包含需要 OCR 并写入 Excel 的列；不识别左侧序号、... 和空白分隔列。
COLS = [
    (71, 234),      # ID
    (270, 432),     # row_no
    (445, 607),     # 数值列1
    (608, 770),     # 数值列2
    (771, 935),     # 数值列3
    (936, 1098),    # 数值列4
    (1099, 1261),   # 数值列5
]

HEADERS = [
    "ID",
    "row_no",
    "value_1",
    "value_2",
    "value_3",
    "value_4",
    "value_5",
]

ROW_NO_COL_INDEX = 1
ID_COL_INDEX = 0
VALUE_COL_NAMES = ["value_1", "value_2", "value_3", "value_4", "value_5"]

START_Y = 3
ROW_HEIGHT = 27
NUM_ROWS_PER_IMAGE = 27

IMAGE_EXTENSIONS = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp"]


# =========================
# 3. 文本清洗
# =========================
def normalize_text(text):
    if text is None:
        return ""

    text = str(text).strip()
    text = text.replace(" ", "")
    text = text.replace("\n", "")
    text = text.replace("\t", "")

    # OCR 有时会把负号识别为 Unicode 减号、短横或破折号；统一为 ASCII 负号，
    # 使负数能够通过数值格式校验并正确写入 Excel。
    text = text.translate(str.maketrans({
        "−": "-",  # Unicode minus
        "–": "-",  # en dash
        "—": "-",  # em dash
        "﹣": "-",  # small hyphen-minus
        "－": "-",  # fullwidth hyphen-minus
    }))

    return text


def normalize_id_prefix(value):
    """
    ID 固定以 ZY 开头。

    OCR 将 ZY 误读为 7Z、7Y、ZV 等任意两个字符时，保留后续编号，
    并统一修正为 ZY 前缀。
    """
    value = normalize_text(value)

    if value == "" or value.startswith("ZY"):
        return value, False

    corrected_value = "ZY" + value[2:]
    return corrected_value, True


def normalize_numeric_comma(value):
    """
    数值列只应包含小数。将 OCR 误读的小数逗号转换为小数点。
    """
    value = normalize_text(value)

    if re.fullmatch(r"-?\d+,\d+", value):
        return value.replace(",", "."), True

    return value, False


# =========================
# 4. 单元格 OCR 识别
# =========================
def summarize_ocr_result(result):
    """
    生成适合写入复核表的 OCR 原始结果摘要。

    只保留识别文本和置信度等关键信息，避免将图像数组等大对象写入 Excel。
    """
    summaries = []

    for item in result:
        getter = getattr(item, "get", None)

        if callable(getter):
            item_summary = []

            for key in ("rec_texts", "rec_scores", "text"):
                try:
                    raw_value = getter(key, None)
                except Exception:
                    raw_value = None

                if raw_value is not None:
                    item_summary.append(f"{key}={raw_value!r}")

            if item_summary:
                summaries.append(", ".join(item_summary))
                continue

        # 兼容 PaddleOCR 旧版返回的列表结构。
        if isinstance(item, list):
            summaries.append(repr(item))
        else:
            summaries.append(repr(item))

    return " | ".join(summaries)


def recognize_cell(cell_img, return_debug=False):
    """
    识别一个单元格。

    return_debug=True 时，同时返回原始 OCR 文本，供排查“最终为空但 OCR
    实际返回过字符”的情况。
    """
    debug_info = {
        "raw_text": "",
        "raw_result_summary": "",
        "corrections": [],
    }

    if cell_img is None or cell_img.size == 0:
        return ("", debug_info) if return_debug else ""

    try:
        result = get_ocr().predict(cell_img)
    except Exception as e:
        print("OCR识别失败：", e)
        return ("", debug_info) if return_debug else ""

    if not result:
        return ("", debug_info) if return_debug else ""

    debug_info["raw_result_summary"] = summarize_ocr_result(result)

    texts = []

    for item in result:
        if isinstance(item, dict):
            if "rec_texts" in item and item["rec_texts"]:
                texts.extend(item["rec_texts"])
            elif "text" in item and item["text"]:
                texts.append(item["text"])

        elif isinstance(item, list):
            for line in item:
                try:
                    texts.append(line[1][0])
                except Exception:
                    pass

        else:
            try:
                if hasattr(item, "get"):
                    rec_texts = item.get("rec_texts", [])
                    if rec_texts:
                        texts.extend(rec_texts)
                    else:
                        text = item.get("text", "")
                        if text:
                            texts.append(text)
            except Exception:
                pass

    if not texts:
        return ("", debug_info) if return_debug else ""

    debug_info["raw_text"] = "".join(str(t) for t in texts)
    value = normalize_text(debug_info["raw_text"])

    return (value, debug_info) if return_debug else value


# =========================
# 5. 格式校验
# =========================
def validate_cell(value, col_index):
    """
    True  = 可疑，需要人工检查
    False = 暂时正常
    """
    value = normalize_text(value)

    # 空白值允许存在
    if value == "":
        return False

    # row_no：只允许数字
    if col_index == ROW_NO_COL_INDEX:
        return not re.fullmatch(r"\d+", value)

    # 数值列：允许小数、负数；整数需要人工复核。
    if HEADERS[col_index] in VALUE_COL_NAMES:
        return not re.fullmatch(r"-?\d+\.\d+", value)

    return False


# =========================
# 6. 判断空行
# =========================
def is_effective_empty_row(row_data):
    """
    判断是否为空行。
    """
    for value in row_data:
        if normalize_text(value) != "":
            return False

    return True


# =========================
# 7. 修复漏识别 row_no
# =========================
def repair_missing_row_no(rows):
    """
    修复偶发漏识别的 row_no。

    只修复单行漏识别：
    1. 第一行 row_no 空，第二行是 2，则第一行补 1。
    2. 中间某行 row_no 空，上一行是 10，下一行是 12，则补 11。

    不处理大范围缺失。
    大范围缺失会在最终去重后由 check_row_no_gaps() 写入复核表。
    """
    repaired_rows = [row.copy() for row in rows]

    for i, row in enumerate(repaired_rows):
        current = normalize_text(row[ROW_NO_COL_INDEX])

        if current != "":
            continue

        prev_no = None
        next_no = None

        # 找上一行有效 row_no
        if i > 0:
            prev_value = normalize_text(repaired_rows[i - 1][ROW_NO_COL_INDEX])
            if prev_value.isdigit():
                prev_no = int(prev_value)

        # 找下一行有效 row_no
        if i < len(repaired_rows) - 1:
            next_value = normalize_text(repaired_rows[i + 1][ROW_NO_COL_INDEX])
            if next_value.isdigit():
                next_no = int(next_value)

        # 第一行漏识别，下一行是 2，则补 1
        if prev_no is None and next_no is not None:
            inferred = next_no - 1
            if inferred > 0:
                row[ROW_NO_COL_INDEX] = str(inferred)

        # 中间一行漏识别，前后刚好差 2
        elif prev_no is not None and next_no is not None:
            if next_no - prev_no == 2:
                row[ROW_NO_COL_INDEX] = str(prev_no + 1)

    return repaired_rows


# =========================
# 8. 只删除末尾空白行
# =========================
def trim_trailing_empty_rows(rows):
    """
    只删除末尾多切出来的空白行。
    不删除开头或中间的空行。
    """
    trimmed_rows = [row.copy() for row in rows]

    while trimmed_rows and is_effective_empty_row(trimmed_rows[-1]):
        trimmed_rows.pop()

    return trimmed_rows


# =========================
# 9. 收集图片路径
# =========================
def collect_images(image_folder):
    image_folder = Path(image_folder)

    image_paths = []
    for ext in IMAGE_EXTENSIONS:
        image_paths.extend(image_folder.glob(ext))

    image_paths = sorted(image_paths, key=lambda p: str(p))

    return image_paths


# =========================
# 10. 处理单张图片
# =========================
def process_image(image_path):
    start_time = time.perf_counter()

    img = cv2.imread(str(image_path))

    if img is None:
        raise ValueError(f"无法读取图片：{image_path}")

    rows = []
    row_debug_infos = []
    review_rows = []
    ocr_call_count = 0

    image_height, image_width = img.shape[:2]

    for r in range(NUM_ROWS_PER_IMAGE):
        y1 = START_Y + r * ROW_HEIGHT
        y2 = y1 + ROW_HEIGHT

        if y1 >= image_height:
            break

        y2 = min(y2, image_height)

        row_data = []
        row_debug_data = []

        for c, (x1, x2) in enumerate(COLS):
            x1_safe = max(0, min(x1, image_width))
            x2_safe = max(0, min(x2, image_width))

            cell = img[y1:y2, x1_safe:x2_safe]

            ocr_call_count += 1
            value, debug_info = recognize_cell(cell, return_debug=True)

            if c == ID_COL_INDEX:
                corrected_value, was_corrected = normalize_id_prefix(value)
                if was_corrected:
                    debug_info["corrections"].append({
                        "reason": "id_prefix_corrected_to_ZY",
                        "original_value": value,
                        "corrected_value": corrected_value,
                    })
                    value = corrected_value

            elif HEADERS[c] in VALUE_COL_NAMES:
                corrected_value, was_corrected = normalize_numeric_comma(value)
                if was_corrected:
                    debug_info["corrections"].append({
                        "reason": "numeric_comma_converted_to_decimal_point",
                        "original_value": value,
                        "corrected_value": corrected_value,
                    })
                    value = corrected_value

            row_data.append(value)
            row_debug_data.append(debug_info)

        # 不提前删除空行，避免误删 row_no=1 且数值为空的有效行
        rows.append(row_data)
        row_debug_infos.append(row_debug_data)

    # 先修复单行 row_no 漏识别
    rows = repair_missing_row_no(rows)

    # 再只删除末尾多切出来的空白行
    rows = trim_trailing_empty_rows(rows)
    row_debug_infos = row_debug_infos[:len(rows)]

    # 修复和裁剪后再做格式校验
    for r, row_data in enumerate(rows):
        for c, value in enumerate(row_data):
            debug_info = row_debug_infos[r][c]

            # 自动修正的 ID 前缀和数值小数点保留原始值，写入复核表。
            for correction in debug_info["corrections"]:
                review_rows.append({
                    "image": image_path.name,
                    "image_path": str(image_path),
                    "image_row_index": r + 1,
                    "column_index": c + 1,
                    "column_name": HEADERS[c],
                    "value": value,
                    "original_value": correction["original_value"],
                    "corrected_value": correction["corrected_value"],
                    "reason": correction["reason"],
                })

            if validate_cell(value, c):
                reason = "format_suspicious"
                if (
                    HEADERS[c] in VALUE_COL_NAMES
                    and re.fullmatch(r"-?\d+", normalize_text(value))
                ):
                    reason = "integer_value_detected"

                review_rows.append({
                    "image": image_path.name,
                    "image_path": str(image_path),
                    "image_row_index": r + 1,
                    "column_index": c + 1,
                    "column_name": HEADERS[c],
                    "value": value,
                    "reason": reason
                })

            # 当前文本清洗会删除空格、换行和制表符。若 OCR 原始文本非空，
            # 但清洗后的最终值为空，将原始返回写入复核表，便于直接检查。
            if value == "" and debug_info["raw_text"] != "":
                review_rows.append({
                    "image": image_path.name,
                    "image_path": str(image_path),
                    "image_row_index": r + 1,
                    "column_index": c + 1,
                    "column_name": HEADERS[c],
                    "value": value,
                    "raw_ocr_text": debug_info["raw_text"],
                    "raw_ocr_output": debug_info["raw_result_summary"],
                    "reason": "empty_after_text_normalization"
                })

    elapsed_time = time.perf_counter() - start_time

    return rows, review_rows, ocr_call_count, elapsed_time


def process_image_worker(task):
    """
    在独立进程中处理一张图片。

    只返回可序列化的数据；主进程按任务输入顺序汇总结果，以保证后续去重仍以
    图片原始排序为准，而不是以子进程完成顺序为准。
    """
    index, image_path_str = task
    image_path = Path(image_path_str)

    try:
        rows, review_rows, ocr_call_count, image_elapsed_time = process_image(image_path)
        return {
            "index": index,
            "image_path": image_path_str,
            "success": True,
            "rows": rows,
            "review_rows": review_rows,
            "ocr_call_count": ocr_call_count,
            "image_elapsed_time": image_elapsed_time,
            "error": "",
        }
    except Exception as e:
        return {
            "index": index,
            "image_path": image_path_str,
            "success": False,
            "rows": [],
            "review_rows": [],
            "ocr_call_count": 0,
            "image_elapsed_time": 0.0,
            "error": str(e),
        }


# =========================
# 11. 按 row_no 去重
# =========================
def deduplicate_by_row_no(df):
    """
    按 row_no 去重：

    规则：
    1. row_no 为空的行不参与去重，全部保留。
    2. row_no 不为空的行参与去重。
    3. 同一个 row_no 出现多次时，直接保留最先出现的那一行；
       其他重复行写入 duplicate_df。
    """
    df = df.copy()

    df["row_no"] = df["row_no"].astype(str).str.strip()

    # 记录原始顺序，保留每个 row_no 首次出现的记录。
    df["_original_order"] = range(len(df))

    non_empty_mask = df["row_no"] != ""

    empty_df = df[~non_empty_mask].copy()
    non_empty_df = df[non_empty_mask].copy()

    if non_empty_df.empty:
        final_df = df.drop(columns=["_original_order"])
        duplicate_df = pd.DataFrame(columns=final_df.columns)
        return final_df, duplicate_df

    keep_indexes = []
    duplicate_indexes = []

    for row_no, group in non_empty_df.groupby("row_no", sort=False):
        # groupby(sort=False) 保持 row_no 首次出现的分组顺序；
        # 组内按原始顺序取第一行即可。
        keep_index = group.index[0]
        keep_indexes.append(keep_index)

        for idx in group.index[1:]:
            duplicate_indexes.append(idx)

    unique_non_empty_df = non_empty_df.loc[keep_indexes].copy()
    duplicate_df = non_empty_df.loc[duplicate_indexes].copy()

    # 最终结果按原始出现顺序排序
    final_df = pd.concat([unique_non_empty_df, empty_df], ignore_index=False)
    final_df = final_df.sort_values("_original_order").reset_index(drop=True)

    duplicate_df = duplicate_df.sort_values("_original_order").reset_index(drop=True)

    # 删除辅助列
    final_df = final_df.drop(columns=["_original_order"], errors="ignore")
    duplicate_df = duplicate_df.drop(columns=["_original_order"], errors="ignore")

    return final_df, duplicate_df


# =========================
# 13. 检查最终 row_no 是否连续
# =========================
def check_row_no_gaps(df):
    """
    检查去重后的 final_df 中 row_no 是否连续。

    只检查，不自动补行。

    如果发现：
    100
    101
    130

    会写入 review：
    row_no_gap_detected: missing 28 row_no(s), range 102-129

    注意：
    这个函数应该在 deduplicate_by_row_no(raw_df) 之后执行，
    即先去重，再检查 final_df。
    """
    review_rows = []

    if df.empty or "row_no" not in df.columns:
        return review_rows

    valid_rows = []

    for idx, row in df.iterrows():
        row_no = normalize_text(row.get("row_no", ""))

        if row_no.isdigit():
            valid_rows.append({
                "df_index": idx,
                "row_no": int(row_no)
            })

    if len(valid_rows) < 2:
        return review_rows

    # 按 row_no 数字大小排序后检查连续性
    valid_rows = sorted(valid_rows, key=lambda x: x["row_no"])

    for i in range(1, len(valid_rows)):
        prev_no = valid_rows[i - 1]["row_no"]
        curr_no = valid_rows[i]["row_no"]

        gap = curr_no - prev_no

        if gap > 1:
            missing_start = prev_no + 1
            missing_end = curr_no - 1
            missing_count = gap - 1

            review_rows.append({
                "image": "",
                "image_path": "",
                "image_row_index": "",
                "column_index": "",
                "column_name": "row_no",
                "value": f"{prev_no} -> {curr_no}",
                "reason": (
                    f"row_no_gap_detected: missing {missing_count} row_no(s), "
                    f"range {missing_start}-{missing_end}"
                )
            })

    return review_rows


# =========================
# 14. 批量处理
# =========================
def batch_process(
    image_folder,
    output_excel,
    review_excel,
    duplicate_excel,
    raw_excel,
    save_raw=True,
    use_parallel=False,
    parallel_workers=2
):
    total_start_time = time.perf_counter()

    image_paths = collect_images(image_folder)

    print(f"共发现 {len(image_paths)} 张图片")

    if not image_paths:
        print("没有找到图片，请检查 images 文件夹路径。")
        return

    all_rows = []
    all_review_rows = []

    def collect_image_result(result):
        image_path = Path(result["image_path"])

        if result["success"]:
            all_rows.extend(result["rows"])
            all_review_rows.extend(result["review_rows"])
            print(
                f"{image_path.name} | OCR调用: {result['ocr_call_count']} 次 | "
                f"耗时: {result['image_elapsed_time']:.3f} 秒"
            )
            return

        print(f"处理失败：{image_path}")
        print("原因：", result["error"])
        all_review_rows.append({
            "image": image_path.name,
            "image_path": str(image_path),
            "image_row_index": "",
            "column_index": "",
            "column_name": "",
            "value": "",
            "reason": f"image_process_failed: {result['error']}"
        })

    if use_parallel:
        if parallel_workers != 2:
            raise ValueError("并行模式固定使用 2 个独立进程，请将 parallel_workers 设为 2。")

        print("已开启双进程并行处理")

        tasks = [(index, str(image_path)) for index, image_path in enumerate(image_paths, start=1)]

        # spawn 可避免子进程继承父进程已加载的 OCR 模型；imap 采用有界任务队列，
        # 不会为五万张图片一次性创建五万个 Future 对象。
        context = mp.get_context("spawn")
        with context.Pool(processes=2) as pool:
            for completed_count, result in enumerate(
                pool.imap(process_image_worker, tasks, chunksize=1),
                start=1
            ):
                image_path = Path(result["image_path"])
                print(f"[{completed_count}/{len(image_paths)}] 已完成: {image_path.name}")
                # imap 按 tasks 的输入顺序返回，确保首次出现行的语义不变。
                collect_image_result(result)

    else:
        print("使用串行处理")

        for index, image_path in enumerate(image_paths, start=1):
            print(f"[{index}/{len(image_paths)}] 正在处理: {image_path.name}")
            result = process_image_worker((index, str(image_path)))
            collect_image_result(result)

    if not all_rows:
        print("没有识别到数据")
        return

    # 原始识别结果
    raw_df = pd.DataFrame(all_rows, columns=HEADERS)
    raw_df = raw_df.fillna("")
    raw_df = raw_df.astype(str)
    raw_df = raw_df.replace("nan", "")

    # 按 row_no 去重：重复 row_no 时，保留最先出现的那一行。
    final_df, duplicate_df = deduplicate_by_row_no(raw_df)

    # 再检查去重后的 final_df 是否存在 row_no 缺号
    gap_review_rows = check_row_no_gaps(final_df)
    all_review_rows.extend(gap_review_rows)

    # 异常复核表
    review_df = pd.DataFrame(all_review_rows)

    # 写去重后的最终结果
    with pd.ExcelWriter(output_excel, engine="openpyxl") as writer:
        final_df.to_excel(writer, index=False, sheet_name="deduplicated_result")

    # 写可疑单元格、处理失败、row_no 缺号记录
    with pd.ExcelWriter(review_excel, engine="openpyxl") as writer:
        review_df.to_excel(writer, index=False, sheet_name="review")

    # 写重复数据
    with pd.ExcelWriter(duplicate_excel, engine="openpyxl") as writer:
        duplicate_df.to_excel(writer, index=False, sheet_name="duplicates")

    # 写未去重原始结果
    if save_raw:
        with pd.ExcelWriter(raw_excel, engine="openpyxl") as writer:
            raw_df.to_excel(writer, index=False, sheet_name="raw_result")

    total_elapsed_time = time.perf_counter() - total_start_time

    print("\n处理完成：")
    print("原始识别总行数：", len(raw_df))
    print("去重后总行数：", len(final_df))
    print("重复行数量：", len(duplicate_df))
    print("可疑单元格数量：", len(review_df))
    print(f"总耗时： {total_elapsed_time:.3f} 秒")

    print("\n输出文件：")
    print("去重结果：", output_excel)
    print("重复数据：", duplicate_excel)
    print("复核文件：", review_excel)

    if save_raw:
        print("原始结果：", raw_excel)


# =========================
# 15. 程序入口
# =========================
if __name__ == "__main__":
    batch_process(
        image_folder="/Users/villanelle/Documents/excel_ocr/images_test",
        output_excel="ocr_result_deduplicated.xlsx",
        review_excel="ocr_review.xlsx",
        duplicate_excel="ocr_duplicates.xlsx",
        raw_excel="ocr_raw_result.xlsx",
        save_raw=True,
        # False：串行处理，便于调试；True：使用两个独立进程并行处理图片。
        use_parallel=True,
        parallel_workers=2
    )
