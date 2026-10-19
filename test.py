import hashlib
import os

def calculate_image_sha256(image_path):
    """计算单张图片的 SHA-256 哈希值"""
    # 检查文件是否存在
    if not os.path.exists(image_path):
        return f"错误：文件 {image_path} 不存在"
    
    sha256_hash = hashlib.sha256()
    
    # 以二进制只读模式 ('rb') 打开图片
    with open(image_path, "rb") as f:
        # 分块读取，防止大图片占满内存
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
            
    return sha256_hash.hexdigest()

def compare_images(image_paths):
    """对比多张图片并输出结果"""
    print("--- 开始计算图片 SHA-256 数字身份证 ---")
    results = {}
    
    for path in image_paths:
        sha256_val = calculate_image_sha256(path)
        results[path] = sha256_val
        print(f"图片: {path}\n👉 SHA-256: {sha256_val}\n")
        
    print("--- 对比结论 ---")
    # 用哈希值作为 key，反向分组成群
    seen_hashes = {}
    for path, sha_val in results.items():
        if "错误" in sha_val:
            continue
        seen_hashes.setdefault(sha_val, []).append(path)
        
    # 输出对比结果
    if len(seen_hashes) == 1 and len(image_paths) > 1:
        print("🎉 恭喜！所有输入的图片【完全相同】，无任何像素差异！")
    else:
        group_idx = 1
        for sha_val, paths in seen_hashes.items():
            if len(paths) > 1:
                print(f"👥 【相同图片组 {group_idx}】以下图片完全一样：")
                for p in paths:
                    print(f"  - {p}")
                group_idx += 1
            else:
                print(f"⚠️ 【独一无二】图片没有任何重复：{paths[0]}")

# ================= 运行测试 =================
if __name__ == "__main__":
    # 💡 请在这里替换成你想对比的图片路径（支持 png, jpg, jpeg 等任何格式）
    images_to_test = [
        "/Users/villanelle/Documents/excel_ocr/images_test/00013.jpg",
        "/Users/villanelle/Documents/excel_ocr/images_test/00023.jpg",
        "/Users/villanelle/Documents/excel_ocr/images_test/00026.jpg",
        "/Users/villanelle/Documents/excel_ocr/images_test/00043.jpg",
        "/Users/villanelle/Documents/excel_ocr/images_test/00049.jpg",
        "/Users/villanelle/Documents/excel_ocr/images_test/00063.jpg",
        "/Users/villanelle/Documents/excel_ocr/images_test/00083.jpg",
        "/Users/villanelle/Documents/excel_ocr/images_test/00088.jpg",
        "/Users/villanelle/Documents/excel_ocr/images_test/00092.jpg",
        "/Users/villanelle/Documents/excel_ocr/images_test/00095.jpg",
        "/Users/villanelle/Documents/excel_ocr/images_test/00023_副本.jpg"

    ]
    
    compare_images(images_to_test)
