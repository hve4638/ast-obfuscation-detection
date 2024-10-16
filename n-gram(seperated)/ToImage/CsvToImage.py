#CsvToImage : 민들어진 .csv를 이미지로(멘토님 데이터셋 기준)
import os
import csv
import numpy as np
from PIL import Image

# CSV 파일이 있는 디렉토리와 저장할 이미지 파일 디렉토리 설정
csv_dir = r'C:\Users\...\csv_files'  # CSV 파일들이 저장된 디렉토리
output_img_dir = r'C:\...\images'  # 이미지 파일이 저장될 디렉토리

# 이미지 크기 변환을 위한 설정
image_scale_factor = 10  # 이미지 확대를 위한 스케일 factor

# 디렉토리 없을 시 생성
if not os.path.exists(output_img_dir):
    os.makedirs(output_img_dir)

# CSV 파일을 읽어서 이미지로 변환하는 함수
def csv_to_image(csv_file_path, output_image_path, scale_factor=1):
    # CSV 파일 읽기
    data = []
    with open(csv_file_path, 'r') as csv_file:
        reader = csv.reader(csv_file)
        for row in reader:
            data.append([int(val) for val in row])
    
    # numpy 배열로 변환
    data_array = np.array(data)
    
    # 이미지 크기를 원본보다 확대 (스케일 factor 적용)
    height, width = data_array.shape
    scaled_height = height * scale_factor
    scaled_width = width * scale_factor
    data_array_scaled = np.kron(data_array, np.ones((scale_factor, scale_factor)))

    # 이미지 생성 (0은 검정, 1은 흰색으로 매핑)
    image = Image.fromarray(np.uint8(data_array_scaled * 255), 'L')  # 'L'은 흑백 이미지

    # 이미지 파일 저장
    image.save(output_image_path)
    print(f"{output_image_path}로 이미지 저장 완료.")

# CSV 디렉토리에서 모든 CSV 파일을 처리하여 이미지를 생성
for csv_filename in os.listdir(csv_dir):
    if csv_filename.endswith('.csv'):
        csv_file_path = os.path.join(csv_dir, csv_filename)
        image_filename = csv_filename.replace('.csv', '.png')  # 이미지 파일 이름 설정
        output_image_path = os.path.join(output_img_dir, image_filename)
        
        # CSV를 이미지로 변환
        csv_to_image(csv_file_path, output_image_path, scale_factor=image_scale_factor)
