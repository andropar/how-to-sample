import argparse
from pathlib import Path

import numpy as np
from annoy import AnnoyIndex
from tqdm.auto import tqdm

from how_to_sample.data import (
    LAIONSampleHelper,
    get_random_samples_from_LAION,
    load_pickle_file,
    save_pickle_file,
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true")
    args = parser.parse_args()

    output_dir = Path("laion2b/") / "annoy_index"
    output_dir.mkdir(parents=True, exist_ok=True)

    feature_fps_fp = output_dir / "used_feature_fps.pkl"
    included_indices_fp = output_dir / "included_indices.pkl"
    if not feature_fps_fp.exists() or args.reset:
        used_feature_fps, features, included_indices = get_random_samples_from_LAION(
            5e6, clf_thresh=0.75, n_jobs=32
        )
        save_pickle_file(used_feature_fps, feature_fps_fp)
        save_pickle_file(included_indices, included_indices_fp)
    else:
        used_feature_fps = load_pickle_file(feature_fps_fp)
        included_indices = load_pickle_file(included_indices_fp)
        laion_sample_helper = LAIONSampleHelper(used_feature_fps, included_indices)
        features = laion_sample_helper.get_sample_features()

    features = features / np.linalg.norm(features, axis=1, keepdims=True)

    annoy_index = AnnoyIndex(features.shape[1], metric="angular")
    for i, feature in tqdm(enumerate(features)):
        annoy_index.add_item(i, feature)
    annoy_index.build(n_trees=200)
    annoy_index.save(str(output_dir / "laion_annoy_index.ann"))
