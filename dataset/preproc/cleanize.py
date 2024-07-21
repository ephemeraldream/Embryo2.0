import json
import os
import re


def clean_non_resized_annotations(input_json_path, output_json_path):
    with open(input_json_path, 'r', encoding='utf-8') as file:
        data = json.load(file)

    cleaned_data = []
    pattern = re.compile(r'\d+resized\.jpg')

    for item in data:
        image_info = item.get('data', {})
        image_path = image_info.get('image', '').split('=')[-1]
        image_name = os.path.basename(image_path)

        if pattern.search(image_name):
            cleaned_data.append(item)

    with open(output_json_path, 'w', encoding='utf-8') as outfile:
        json.dump(cleaned_data, outfile, indent=4)

    print(f"Cleaned data saved to {output_json_path}")


def main():
    input_json_path = 'C:\\work\\routine_projects\\NetworkFirst\\dataset\\labels\\labels.json'
    output_json_path = 'C:\\work\\routine_projects\\NetworkFirst\\dataset\\labels\\cleaned_labels.json'

    clean_non_resized_annotations(input_json_path, output_json_path)


if __name__ == "__main__":
    main()
