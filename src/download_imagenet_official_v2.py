"""Download ONLY the validation parquet shards from ILSVRC/imagenet-1k.

The HF datasets API was pre-fetching all 294 parquet shards including the
~150 GB training split when called even with split='validation'. We bypass
that by using hf_hub_download() to grab the 14 validation parquet files
directly (~6.3 GB total), then iterate them with pyarrow to extract JPEG
images per class to ImageFolder layout.

Output: ~/scratch/ca2gradcam/imagenet_val_official/<synset>/img_NNNN.jpeg
        15 images per class × 1000 classes = 15,000 jpegs.

HF cache lives under HF_HOME=$HF_HOME (set in env at call site).
"""
import os, io, json
from huggingface_hub import hf_hub_download
import pyarrow.parquet as pq
from PIL import Image

REPO = 'ILSVRC/imagenet-1k'
OUT_DIR = './imagenet_val_official'
TARGET_PER_CLASS = 15
N_SHARDS = 14

# Cached label-int to synset mapping. The ILSVRC labels in the parquet are
# integer indices 0-999. We need the synset string (n01440764 etc.) for
# directory names. Read these from the dataset metadata via classes.py.
SYNSETS_FALLBACK = None  # filled lazily below if classes.py read fails

os.makedirs(OUT_DIR, exist_ok=True)


def get_synset_mapping():
    """Return dict int -> synset_id like {0: 'n01440764', ...}."""
    # The dataset bundles a classes.py file at the repo root. Easiest way:
    # download and exec it. It defines IMAGENET2012_CLASSES = {synset: name}
    # in canonical synset order (0..999).
    path = hf_hub_download(repo_id=REPO, filename='classes.py', repo_type='dataset')
    ns = {}
    with open(path) as f:
        exec(f.read(), ns)
    cls = ns.get('IMAGENET2012_CLASSES')
    if not cls:
        raise RuntimeError('IMAGENET2012_CLASSES not found in classes.py')
    synsets = list(cls.keys())  # preserves insertion order
    assert len(synsets) == 1000, f'expected 1000 synsets, got {len(synsets)}'
    return {i: s for i, s in enumerate(synsets)}


def main():
    print('[STEP] downloading classes.py for label→synset map...', flush=True)
    label_to_synset = get_synset_mapping()
    print(f'  loaded {len(label_to_synset)} synset names; first three: '
          f'{[label_to_synset[i] for i in range(3)]}', flush=True)

    counts = {}
    total_written = 0
    for shard_idx in range(N_SHARDS):
        fname = f'data/validation-{shard_idx:05d}-of-{N_SHARDS:05d}.parquet'
        print(f'\n[STEP] downloading {fname} ...', flush=True)
        local = hf_hub_download(repo_id=REPO, filename=fname, repo_type='dataset')
        sz = os.path.getsize(local)
        print(f'  {sz/1e9:.2f} GB at {local}', flush=True)

        # Iterate by row group to keep memory bounded
        table = pq.read_table(local)
        n_rows = table.num_rows
        print(f'  shard has {n_rows} rows', flush=True)
        image_col = table.column('image').to_pylist()
        label_col = table.column('label').to_pylist()
        for ex_image, lbl in zip(image_col, label_col):
            synset = label_to_synset[int(lbl)]
            if counts.get(synset, 0) >= TARGET_PER_CLASS:
                continue
            # image column is a struct {bytes, path}
            img_bytes = ex_image.get('bytes')
            if not img_bytes:
                continue
            try:
                img = Image.open(io.BytesIO(img_bytes))
                if img.mode != 'RGB':
                    img = img.convert('RGB')
            except Exception as e:
                print(f'    decode fail: {e}', flush=True)
                continue
            cls_dir = os.path.join(OUT_DIR, synset)
            os.makedirs(cls_dir, exist_ok=True)
            n = counts.get(synset, 0)
            path = os.path.join(cls_dir, f'img_{n:04d}.jpeg')
            try:
                img.save(path, format='JPEG', quality=95)
                counts[synset] = n + 1
                total_written += 1
            except Exception as e:
                print(f'    save fail: {e}', flush=True)

        # Free the shard memory + don't carry parquet table forward
        del table, image_col, label_col

        # Progress
        full = sum(1 for v in counts.values() if v >= TARGET_PER_CLASS)
        print(f'  [PROGRESS] shard {shard_idx+1}/{N_SHARDS}: '
              f'{total_written} written, {len(counts)} classes seen, '
              f'{full}/1000 at target', flush=True)
        if full == 1000:
            print(f'\n[DONE] all classes at {TARGET_PER_CLASS} images after shard {shard_idx+1}', flush=True)
            break

    full = sum(1 for v in counts.values() if v >= TARGET_PER_CLASS)
    print(f'\n[FINAL] wrote {total_written} images, '
          f'{len(counts)} classes seen, {full}/1000 at target. '
          f'output: {OUT_DIR}', flush=True)

    # Save manifest for downstream verification
    with open(os.path.join(OUT_DIR, '_manifest.json'), 'w') as f:
        json.dump({
            'source': 'ILSVRC/imagenet-1k validation (HuggingFace, gated)',
            'target_per_class': TARGET_PER_CLASS,
            'classes_at_target': full,
            'classes_seen': len(counts),
            'total_images': total_written,
            'per_class_counts': counts,
        }, f, indent=2)


if __name__ == '__main__':
    main()
