"""Export cold-start data splits to a unified JSON format.

For every category of every dataset we instantiate the *same* Dataset class
that the training scripts consume, then dump the enumerated file lists into
``splits/<dataset>/<category>.json``. Reusing the Dataset classes guarantees
the released JSON splits are byte-for-byte consistent with the splits that
produced the reported numbers.

Output schema (one file per category)::

    {
      "meta": {
        "dataset": "mvtec",
        "category": "bottle",
        "num_labeled": 121,
        "num_test": 223
      },
      "labeled": [ {"image": "<rel-path>", "mask": "<rel-path-or-''>",
                    "label": 0|1, "anomaly_class": "good"}, ... ],
      "test":    [ ... ]
    }

All paths are relative to the dataset root (the ``--<dataset>_root`` argument).
``mask`` is "" for normal samples. ``label`` is 0 (normal) / 1 (anomaly).

Usage::

    python prepare_data/gen_splits.py --dataset mvtec   --mvtec_root   /path/to/mvtec_CD
    python prepare_data/gen_splits.py --dataset visa    --visa_root    /path/to/VisA_CD
    python prepare_data/gen_splits.py --dataset realiad --realiad_root /path/to/Real-IAD_CD
    python prepare_data/gen_splits.py --dataset manta   --manta_root   /path/to/MANTA_CD
"""
import os
import sys
import json
import argparse

# allow running from repo root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataset import (
    MVTecDataset, VisADataset, RealIADDataset, MANTADataset, AnomalyDataset,
)


def _rel(path, root):
    """Make *path* relative to *root*; fall back to the original string."""
    try:
        return os.path.relpath(path, root)
    except ValueError:
        return path


def _entry(image, mask, label, cls, root):
    return {
        "image": _rel(image, root),
        "mask": _rel(mask, root) if (mask is not None and mask != "" and str(mask) != "") else "",
        "label": int(bool(label)),
        "anomaly_class": str(cls),
    }


def dump(out_dir, dataset, category, labeled, test, root):
    obj = {
        "meta": {
            "dataset": dataset,
            "category": category,
            "num_labeled": len(labeled),
            "num_test": len(test),
        },
        "labeled": labeled,
        "test": test,
    }
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{category}.json")
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)
    n_lab_norm = sum(1 for e in labeled if e["label"] == 0)
    n_lab_anom = sum(1 for e in labeled if e["label"] == 1)
    n_tst_anom = sum(1 for e in test if e["label"] == 1)
    print(f"  [{dataset}/{category}] labeled={len(labeled)} "
          f"(normal {n_lab_norm} / anomaly {n_lab_anom})  "
          f"test={len(test)} (anomaly {n_tst_anom})  -> {path}")


# ---- per-dataset builders -------------------------------------------------

def build_mvtec(root, categories, out_dir):
    """MVTec cold-start. Labeled = <cat>/train/label (AnomalyDataset);
    test = <cat>/test with masks under <cat>/ground_truth (MVTecDataset)."""
    for cat in categories:
        labeled_ds = AnomalyDataset(root_dir=os.path.join(root, cat, "train", "label"),
                                    transform=None, mask_transform=None)
        labeled = [_entry(im, m, l, "good" if l == 0 else "defect", root)
                   for im, m, l in zip(labeled_ds.image_paths,
                                       labeled_ds.mask_paths, labeled_ds.labels)]
        test_ds = MVTecDataset(root=os.path.join(root, cat), transform=None,
                               gt_transform=None, phase="test")
        test = [_entry(im, gt, l, t, root)
                for im, gt, l, t in zip(test_ds.img_paths, test_ds.gt_paths,
                                        test_ds.labels, test_ds.types)]
        dump(out_dir, "mvtec", cat, labeled, test, root)


def build_visa(root, categories, out_dir):
    """VisA cold-start. Labeled = <cat>/train (AnomalyDataset, masks under
    train/ground_truth/bad); test = <cat>/test (VisADataset, masks under
    ground_truth/<defect>)."""
    for cat in categories:
        labeled_ds = AnomalyDataset(root_dir=os.path.join(root, cat, "train"),
                                    transform=None, mask_transform=None)
        labeled = [_entry(im, m, l, "good" if l == 0 else "defect", root)
                   for im, m, l in zip(labeled_ds.image_paths,
                                       labeled_ds.mask_paths, labeled_ds.labels)]
        test_ds = VisADataset(root=os.path.join(root, cat), transform=None,
                              gt_transform=None, phase="test")
        test = [_entry(im, gt, l, t, root)
                for im, gt, l, t in zip(test_ds.img_paths, test_ds.gt_paths,
                                        test_ds.labels, test_ds.types)]
        dump(out_dir, "visa", cat, labeled, test, root)


def build_realiad(root, categories, out_dir):
    """Real-IAD supervised split. Splits come from realiad_jsons/sup/<cat>.json
    (phases: labeled / test). Images under realiad_1024/<cat>/."""
    for cat in categories:
        labeled_ds = RealIADDataset(root=root, category=cat, transform=None,
                                    gt_transform=None, phase="labeled")
        test_ds = RealIADDataset(root=root, category=cat, transform=None,
                                 gt_transform=None, phase="test")
        labeled = [_entry(im, gt, l, t, root)
                   for im, gt, l, t in zip(labeled_ds.img_paths, labeled_ds.gt_paths,
                                           labeled_ds.labels, labeled_ds.types)]
        test = [_entry(im, gt, l, t, root)
                for im, gt, l, t in zip(test_ds.img_paths, test_ds.gt_paths,
                                        test_ds.labels, test_ds.types)]
        dump(out_dir, "realiad", cat, labeled, test, root)


def build_manta(root, categories, out_dir):
    """MANTA supervised split. Splits come from sup_cropped/<cat>.json
    (phases: labeled / test). Images under MANTA_TINY_256_cropped/<cat>/."""
    for cat in categories:
        labeled_ds = MANTADataset(root=root, category=cat, transform=None,
                                  gt_transform=None, phase="labeled")
        test_ds = MANTADataset(root=root, category=cat, transform=None,
                               gt_transform=None, phase="test")
        labeled = [_entry(im, gt, l, t, root)
                   for im, gt, l, t in zip(labeled_ds.img_paths, labeled_ds.gt_paths,
                                           labeled_ds.labels, labeled_ds.types)]
        test = [_entry(im, gt, l, t, root)
                for im, gt, l, t in zip(test_ds.img_paths, test_ds.gt_paths,
                                        test_ds.labels, test_ds.types)]
        dump(out_dir, "manta", cat, labeled, test, root)


CATEGORIES = {
    "mvtec": ["carpet", "grid", "leather", "tile", "wood", "bottle", "cable",
              "capsule", "hazelnut", "metal_nut", "pill", "screw", "toothbrush",
              "transistor", "zipper"],
    "visa": ["candle", "capsules", "cashew", "chewinggum", "fryum", "macaroni1",
             "macaroni2", "pcb1", "pcb2", "pcb3", "pcb4", "pipe_fryum"],
    "realiad": ["audiojack", "bottle_cap", "button_battery", "end_cap", "eraser",
                "fire_hood", "mint", "mounts", "pcb", "phone_battery", "plastic_nut",
                "plastic_plug", "porcelain_doll", "regulator", "rolled_strip_base",
                "sim_card_set", "switch", "tape", "terminalblock", "toothbrush",
                "toy", "toy_brick", "transistor1", "usb", "usb_adaptor", "u_block",
                "vcpill", "wooden_beads", "woodstick", "zipper"],
    "manta": ["block_inductor", "button", "capsule", "coated_tablet", "coffee_beans",
              "copper_standoff", "embossed_tablet", "flat_nut", "gear", "goji_berries",
              "led", "led_pad", "lettered_tablet", "long_button", "maize", "nut",
              "nut_cap", "oblong_tablet", "paddy", "pink_tablet", "pistachios",
              "power_inductor", "red_tablet", "red_washer", "round_button_cap",
              "screw", "short_button", "soybean", "square_button_cap", "terminal",
              "thin_resistor", "type_c", "wafer_resistor", "wheat", "white_tablet",
              "wire_cap", "yellow_green_washer", "yellow_tablet"],
}

BUILDERS = {
    "mvtec": build_mvtec,
    "visa": build_visa,
    "realiad": build_realiad,
    "manta": build_manta,
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export ArcAD cold-start split JSONs.")
    parser.add_argument("--dataset", required=True, choices=list(BUILDERS.keys()))
    parser.add_argument("--mvtec_root", default="/home/hanningning/AD/mvtec_CD")
    parser.add_argument("--visa_root", default="/home/hanningning/AD/VisA_CD")
    parser.add_argument("--realiad_root", default="/home/hanningning/AD/Real-IAD_CD")
    parser.add_argument("--manta_root", default="/home/hanningning/AD/MANTA_CD")
    parser.add_argument("--out_dir", default=None,
                        help="output dir (default: splits/<dataset>)")
    args = parser.parse_args()

    root = getattr(args, f"{args.dataset}_root")
    out_dir = args.out_dir or os.path.join("splits", args.dataset)
    print(f"Exporting {args.dataset} splits -> {out_dir}  (root={root})")
    BUILDERS[args.dataset](root, CATEGORIES[args.dataset], out_dir)
    print("Done.")
