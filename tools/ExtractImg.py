import os

def main():
    img_path = r"E:\Dataset\DOTA1.5\val\images"
    ann_path = r"E:\Dataset\DOTA1.5\val\labelTxt-v1.5"
    # img_list = os.listdir(img_path)
    # img_list.sort()
    ann_list = os.listdir(ann_path)
    ann_list.sort()
    for ann in ann_list:
        path = os.path.join(ann_path, ann)
        with open(path, 'r') as f:
            if f is None:
                pass
            else:
                line = next(iter(f))
                src_type = line.split(':')[-1]
                if src_type != 'GoogleEarth\n':
                    img = os.path.join(img_path, ann.replace('.txt', '.png'))
                    os.rename(img, img + '.disabled')
                    f.close()
                    os.rename(path, path + '.disabled')
    pass

if __name__ == '__main__':
    main()