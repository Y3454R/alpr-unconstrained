import pytesseract
import cv2
import os


def preprocess_image(image):
    # gray = get_grayscale(image)
    # contrast_img = increase_contrast(gray)
    # threshold_img = adaptive_threshold(contrast_img)
    # smoothed_img = smooth_image(threshold_img)
    # final_img = dilate_then_erode(smoothed_img, kernel_size=2)

    thresh = cv2.threshold(image, 150, 255, cv2.THRESH_BINARY_INV)[1]
    # kernel = np.ones((2,2),np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    final_img = 255 - opening

    # # erosion = cv2.erode(final_img, kernel, iterations = 1)
    # # final_img = erosion

    final_img = cv2.resize(final_img, (0, 0), fx=1.0, fy=0.70)

    return final_img


def read_single_bangla_character(image_path):
    # print(f'image path: {image_path}')
    image = cv2.imread(image_path)
    # processed_image = image
    processed_image = preprocess_image(image)

    # Tesseract configuration for single character recognition
    # custom_config = r'--oem 3 --psm 10 -c tessedit_char_whitelist=০১২৩৪৫৬৭৮৯'
    custom_config = r"--oem 3 --psm 13 numbers"
    character = pytesseract.image_to_string(
        processed_image, lang="ben", config=custom_config
    )

    return character.strip()


def remove_hyphen(text):
    text = text.replace("-", "")
    return text


# def read_images_from_folder(image_path):
#     combined_text = ""
#     character = read_single_bangla_character(image_path)
#     if len(character) > 6:
#         character = character[:2] + character[3:]
#     combined_text += character
#     return combined_text


# def evaluate(base_folder_path, gt_dic, how_many):
#     accurate_count = 0
#     total_count = 0
#     total_cer = 0

#     counter = 0

#     for folder_name in os.listdir(base_folder_path):
#         folder_path = os.path.join(base_folder_path, folder_name)
#         gt_file_name = folder_name.replace('.png', '')
#         if gt_file_name in gt_dic:
#             total_count += 1
#             ocr_text = read_images_from_folder(folder_path)
#             gt_txt = remove_hyphen(gt_dic[gt_file_name])
#             is_matched = '>>> MISMATCH'
#             if ocr_text == gt_txt:
#                 accurate_count += 1
#                 is_matched = ''

#             current_cer = cer(gt_txt, ocr_text)
#             total_cer += current_cer

#             accuracy = (accurate_count / total_count) if total_count > 0 else 0
#             avg_cer = (total_cer/total_count) if total_count > 0 else 0
#             print(f'{folder_path} -> {ocr_text} | {gt_txt} {is_matched} | Accuracy: {accuracy:.2f}% | CER: {avg_cer:.2f}')

#         else:
#             print(f"Skipping {folder_name}: no ground truth found or not a directory.")

#         counter += 1
#         if counter >= how_many:
#             break

#     accuracy = (accurate_count / total_count) if total_count > 0 else 0
#     avg_cer = (total_cer/total_count) if total_count > 0 else 0
#     print(f"Total: {total_count}, Accurate: {accurate_count}, Accuracy: {accuracy:.4f}, CER: {avg_cer:.4f}")


if __name__ == "__main__":

    image_folder = "plates"
    gt_folder = "gt_files"
    dic = create_gt_dic(image_folder, gt_folder)
    # print(dic)
    evaluate(image_folder, dic, 500)
