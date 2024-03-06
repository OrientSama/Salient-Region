import os
from PIL import Image, ImageDraw

def main():
    fn_path = '/home/ubuntu/Dataset/DOTA-Split-classification/hard_pred/FN'
    fn_img_list = os.listdir(fn_path)
    for fn_img in fn_img_list:
        ann_path = os.path.join('/home/ubuntu/Dataset/DOTA-Split-classification/val/annfiles/', fn_img.replace('.png', '.txt'))
        img_path = os.path.join(fn_path, fn_img)
        img = Image.open(img_path)
        draw = ImageDraw.Draw(img)
        with open(ann_path, 'r') as ann:
            ann_content = ann.readlines()
        for line in ann_content:
            line = line.split(' ')[:8]
            coord = lambda li: [(float(li[2 * i]), float(li[2 * i + 1])) for i in range(int(len(li)/2))]
            draw.polygon(coord(line), outline="red")
        img.save(img_path, format='PNG')


if __name__ == '__main__':
    main()