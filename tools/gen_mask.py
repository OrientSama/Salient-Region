import os
from multiprocessing import Pool
from functools import partial
from PIL import Image, ImageDraw


# Generate a mask on the image
def G(imgs, coord):
    draw = ImageDraw.Draw(imgs)
    draw.polygon(coord, outline="white", fill="white")


# Analyze the coordinates from the text
def analyze_coord(str):
    split_str = str.split(' ')[:8]
    coord = [(float(split_str[2 * i]), float(split_str[2 * i + 1])) for i in range(len(split_str) // 2)]

    return coord


def mask_single_img(pic_name, labels_path, mask_imgs_path):
    imgs = Image.new("L", (1024, 1024))
    assert os.path.exists(os.path.join(labels_path, pic_name))
    with open(os.path.join(labels_path, pic_name), 'r') as f:
        strlines = f.readlines()
        for str in strlines:
            coord = analyze_coord(str)
            G(imgs, coord)

    imgs.save(os.path.join(mask_imgs_path, pic_name.replace('.txt', '.jpg')))


def main():
    # 进程数量
    num_process = 24
    pool = Pool(num_process)
    base_path = r"E:\Dataset\DOTA-Classifier-1.5\val"
    mask_imgs_path = os.path.join(base_path, 'mask_images')
    if not os.path.exists(mask_imgs_path):
        os.mkdir(mask_imgs_path)
    labels_path = os.path.join(base_path, 'annfiles')
    pic_name_list = os.listdir(labels_path)
    print("Found {} Labels".format(len(pic_name_list)))
    worker = partial(mask_single_img, labels_path=labels_path, mask_imgs_path=mask_imgs_path)
    pool.map(worker, pic_name_list)


if __name__ == "__main__":
    main()
