import argparse
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import clip
import numpy as np
import torch
from annoy import AnnoyIndex
from tqdm import tqdm

from how_to_sample.data import LAIONSampleHelper, load_pickle_file, save_pickle_file
from how_to_sample.plotting import display_images_in_grid


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Analyze concept appearance in LAION dataset"
    )
    parser.add_argument(
        "--ecoset_path",
        type=str,
        default="data/ecoset",
        help="Path to Ecoset dataset",
    )
    parser.add_argument(
        "--mscoco_path",
        type=str,
        default="data/mscoco",
        help="Path to MSCOCO dataset",
    )
    parser.add_argument(
        "--imagenet_path",
        type=str,
        default="data/imagenet",
        help="Path to ImageNet dataset",
    )
    parser.add_argument(
        "--laion_path",
        type=str,
        default="data/laion2b",
        help="Path to LAION dataset",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory (defaults to script_dir/outputs/test_concept_appearance)",
    )
    return parser.parse_args()


def setup_logging(output_dir: Path) -> None:
    """Set up logging configuration."""
    log_file = (
        output_dir
        / f"concept_appearance_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    )
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
    )


def gather_concepts(
    ecoset_path: str, mscoco_path: str, imagenet_path: str
) -> List[str]:
    """Gather and deduplicate concepts from multiple datasets."""
    # Import Ecoset concepts
    import sys

    sys.path.insert(0, ecoset_path)
    from concepts import labels as ecoset_labels

    # Load MSCOCO concepts
    with open(Path(mscoco_path) / "annotations/instances_train2017.json", "r") as f:
        coco_train = json.load(f)
    mscoco_concepts = [cat["name"] for cat in coco_train["categories"]]

    # Load ImageNet concepts
    with open(Path(imagenet_path) / "class_labels.txt", "r") as f:
        imagenet_dict = eval(f.read())
    imagenet_concepts = list(imagenet_dict.values())

    # Combine and deduplicate concepts
    all_concepts = list(
        set(
            s.lower().strip()
            for s in (ecoset_labels + mscoco_concepts + imagenet_concepts)
        )
    )
    all_concepts.sort()
    return all_concepts


def load_laion_data(laion_path: str) -> Tuple[LAIONSampleHelper, AnnoyIndex]:
    """Load LAION data and Annoy index."""
    annoy_index_dir = Path(laion_path) / "annoy_index"
    annoy_index_fp = annoy_index_dir / "laion_annoy_index.ann"
    used_feature_fps_fp = annoy_index_dir / "used_feature_fps.pkl"
    included_indices_fp = annoy_index_dir / "included_indices.pkl"

    used_feature_fps = load_pickle_file(str(used_feature_fps_fp))
    included_indices = load_pickle_file(str(included_indices_fp))
    laion_sample_helper = LAIONSampleHelper(used_feature_fps, included_indices)

    laion_features = laion_sample_helper.get_sample_features()
    dim = laion_features.shape[1]

    annoy_index = AnnoyIndex(dim, metric="angular")
    annoy_index.load(str(annoy_index_fp))
    return laion_sample_helper, annoy_index


def process_concepts(
    concepts: List[str],
    model: Any,
    device: torch.device,
    annoy_index: AnnoyIndex,
    output_dir: Path,
) -> Dict[str, Dict[str, Any]]:
    """Process concepts and find nearest neighbors in LAION dataset."""
    results_fp = output_dir / "concept_appearance_results.pkl"
    results = {}

    for i, concept in tqdm(enumerate(concepts), total=len(concepts)):
        try:
            # Compute CLIP text embedding
            tokenized = clip.tokenize([concept]).to(device)
            with torch.no_grad():
                text_embedding = model.encode_text(tokenized)
            text_embedding = text_embedding.cpu().numpy().squeeze()
            text_embedding = text_embedding / np.linalg.norm(text_embedding)

            # Find nearest neighbors
            nns, distances = annoy_index.get_nns_by_vector(
                text_embedding, 25, include_distances=True
            )
            closeness = np.mean(distances)
            results[concept] = {
                "nns": nns,
                "distances": distances,
                "closeness": closeness,
            }

            # Save intermediate results
            save_pickle_file(results, str(results_fp))

            if (i + 1) % 100 == 0:
                logging.info(f"Processed {i + 1}/{len(concepts)} concepts")
        except Exception as e:
            logging.error(f"Error processing concept '{concept}': {str(e)}")
            continue

    return results


def generate_image_grids(
    results: Dict[str, Dict[str, Any]],
    laion_sample_helper: LAIONSampleHelper,
    output_dir: Path,
) -> None:
    """Generate and save image grids for concepts."""
    sorted_concepts = sorted(results.items(), key=lambda x: x[1]["closeness"])

    for idx, (concept, data) in tqdm(
        enumerate(sorted_concepts, start=1), total=len(sorted_concepts)
    ):
        try:
            images, _ = laion_sample_helper.get_sample_info(indices=data["nns"])
            output_fp = output_dir / f"{idx:02d}_{concept.replace(' ', '_')}_grid.png"
            display_images_in_grid(images, saveto=str(output_fp))
            logging.info(
                f"Saved grid for concept: {concept} (closeness: {data['closeness']:.4f})"
            )
        except Exception as e:
            logging.error(f"Failed to generate grid for concept '{concept}': {str(e)}")
            continue


def main():
    args = parse_args()

    # Set up output directory
    script_dir = Path(__file__).parent
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else script_dir / "outputs" / "test_concept_appearance"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    setup_logging(output_dir)
    logging.info("Starting concept appearance analysis")

    # Setup CLIP
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")
    try:
        model, _ = clip.load("ViT-B/32", device=device)
        logging.info("Successfully loaded CLIP model")
    except Exception as e:
        logging.error(f"Failed to load CLIP model: {str(e)}")
        raise

    # Gather concepts
    all_concepts = gather_concepts(
        args.ecoset_path, args.mscoco_path, args.imagenet_path
    )
    logging.info(f"Gathered {len(all_concepts)} unique concepts from datasets")

    # Load LAION data
    try:
        laion_sample_helper, annoy_index = load_laion_data(args.laion_path)
        logging.info("Successfully loaded LAION data and Annoy index")
    except Exception as e:
        logging.error(f"Failed to load LAION data: {str(e)}")
        raise

    # Process concepts
    results = process_concepts(all_concepts, model, device, annoy_index, output_dir)

    # Generate image grids
    generate_image_grids(results, laion_sample_helper, output_dir)

    logging.info("Completed concept appearance analysis")


if __name__ == "__main__":
    main()
