"""Download CUB-200-2011 from the fast.ai mirror and rearrange into an
ImageFolder layout so load_imagenet_val can read it.

Source:  https://s3.amazonaws.com/fast-ai-imageclas/CUB_200_2011.tgz  (~1.1 GB)

Target tree:
    ./cub_imagefolder/
        001.Black_footed_Albatross/<image>.jpg
        002.Laysan_Albatross/<image>.jpg
        ...

We use the test split (per train_test_split.txt) so the eval is not on
training data, ~5,794 images across 200 classes.

Run on a login node (compute nodes have no internet).
"""
import os, sys, tarfile, urllib.request, shutil

ROOT     = '.'
TGZ      = os.path.join(ROOT, 'CUB_200_2011.tgz')
EXTRACT  = ROOT  # tgz extracts to ./CUB_200_2011/...
TARGET   = os.path.join(ROOT, 'cub_imagefolder')
URL      = 'https://s3.amazonaws.com/fast-ai-imageclas/CUB_200_2011.tgz'


def download():
    if os.path.exists(TGZ):
        print(f"[skip] tarball exists: {TGZ} ({os.path.getsize(TGZ)//1024//1024} MB)")
        return
    print(f"[dl ] {URL} -> {TGZ}")
    with urllib.request.urlopen(URL) as r, open(TGZ, 'wb') as f:
        total = int(r.headers.get('Content-Length', 0))
        got = 0
        while True:
            chunk = r.read(8 * 1024 * 1024)
            if not chunk:
                break
            f.write(chunk)
            got += len(chunk)
            if total:
                print(f"\r  {got/1024/1024:7.1f} / {total/1024/1024:7.1f} MB", end='', flush=True)
        print()


def extract():
    src = os.path.join(EXTRACT, 'CUB_200_2011')
    if os.path.isdir(src):
        print(f"[skip] already extracted: {src}")
        return
    print(f"[tar] extracting {TGZ} -> {EXTRACT}")
    with tarfile.open(TGZ) as tf:
        tf.extractall(EXTRACT)


def build_test_imagefolder():
    if os.path.isdir(TARGET) and len(os.listdir(TARGET)) > 0:
        n = sum(len(os.listdir(os.path.join(TARGET, d)))
                for d in os.listdir(TARGET)
                if os.path.isdir(os.path.join(TARGET, d)))
        print(f"[skip] imagefolder exists: {TARGET} ({n} files)")
        return

    src = os.path.join(EXTRACT, 'CUB_200_2011')
    images_dir = os.path.join(src, 'images')
    images_txt = os.path.join(src, 'images.txt')          # id -> relpath
    split_txt  = os.path.join(src, 'train_test_split.txt')  # id is_train

    id2path = {}
    with open(images_txt) as f:
        for line in f:
            i, p = line.strip().split(' ', 1)
            id2path[int(i)] = p

    test_ids = set()
    with open(split_txt) as f:
        for line in f:
            i, t = line.strip().split()
            if int(t) == 0:  # 0 == test
                test_ids.add(int(i))

    print(f"[build] {len(test_ids)} test images")
    os.makedirs(TARGET, exist_ok=True)
    n_copied = 0
    for img_id in sorted(test_ids):
        relpath = id2path[img_id]              # e.g. 001.Black_footed_Albatross/foo.jpg
        cls_name = os.path.dirname(relpath)
        src_file = os.path.join(images_dir, relpath)
        dst_dir  = os.path.join(TARGET, cls_name)
        os.makedirs(dst_dir, exist_ok=True)
        dst_file = os.path.join(dst_dir, os.path.basename(relpath))
        if not os.path.exists(dst_file):
            shutil.copy2(src_file, dst_file)
            n_copied += 1
    print(f"[build] copied {n_copied} files into {TARGET}")
    n_classes = len([d for d in os.listdir(TARGET)
                     if os.path.isdir(os.path.join(TARGET, d))])
    print(f"[done] {n_classes} class dirs under {TARGET}")


def main():
    os.makedirs(ROOT, exist_ok=True)
    download()
    extract()
    build_test_imagefolder()


if __name__ == '__main__':
    main()
