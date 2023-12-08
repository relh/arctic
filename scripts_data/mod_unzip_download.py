#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import os.path as op
import zipfile
from glob import glob
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm


def unzip(zip_p, out_dir):
    # Check if output directory exists and is not empty
    if os.path.exists(out_dir) and os.listdir(out_dir):
        print(f"Skipping {zip_p}, already unzipped.")
        return

    os.makedirs(out_dir, exist_ok=True)
    with zipfile.ZipFile(zip_p, "r") as zip_ref:
        zip_ref.extractall(out_dir)
    print(f"Unzipped {zip_p} to {out_dir}")


def process_zip_files(zip_files, out_dir, description):
    with ThreadPoolExecutor() as executor:
        futures = {
            executor.submit(unzip, zip_p, op.join(
                out_dir,
                zip_p.replace("downloads/data/", "")
                .replace(".zip", "")
                .replace(description, description.replace("_zips", "")),
            )): zip_p for zip_p in zip_files
        }

        for future in tqdm(as_completed(futures), total=len(zip_files), desc=f"Unzipping {description}"):
            future.result()


def main():
    fnames = glob(op.join("downloads/data/", "**/*"), recursive=True)

    full_img_zips = []
    cropped_images_zips = []
    misc_zips = []
    models_zips = []
    # Categorize zip files
    for fname in fnames:
        if ".zip" in fname:
            if "/images_zips/" in fname:
                full_img_zips.append(fname)
            elif "/cropped_images_zips/" in fname:
                cropped_images_zips.append(fname)
            elif "models.zip" in fname:
                models_zips.append(fname)
            else:
                misc_zips.append(fname)
        else:
            print(f"Unknown file type: {fname}")

    out_dir = "./unpack/arctic_data/data"
    os.makedirs(out_dir, exist_ok=True)

    # Process different categories of zip files
    #process_zip_files(misc_zips, out_dir, "misc_zips")
    #process_zip_files(models_zips, out_dir.replace("/data", ""), "models_zips")
    #process_zip_files(cropped_images_zips, out_dir, "cropped_images_zips")
    process_zip_files(full_img_zips, out_dir, "images_zips")


if __name__ == "__main__":
    main()



