import argparse
import json
import logging
import pickle
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Tuple

import google.generativeai as genai
import numpy as np
import typing_extensions as typing
from tqdm.auto import tqdm

from how_to_sample.data import LAIONSampleHelper, get_random_samples_from_LAION


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate keywords for LAION images using Gemini"
    )
    parser.add_argument("--api-key", type=str, required=True, help="Gemini API key")
    parser.add_argument(
        "--n-samples", type=int, default=100000, help="Number of samples to process"
    )
    parser.add_argument(
        "--clf-thresh", type=float, default=0.75, help="Classification threshold"
    )
    parser.add_argument(
        "--max-cost", type=float, default=25.0, help="Maximum cost in USD"
    )
    parser.add_argument(
        "--n-workers", type=int, default=32, help="Number of parallel workers"
    )
    parser.add_argument(
        "--save-interval", type=int, default=1000, help="Save interval in samples"
    )
    return parser.parse_args()


class ImageAnswer(typing.TypedDict):
    """Type definition for Gemini API response."""

    keywords: List[str]


def setup_model(api_key: str) -> genai.GenerativeModel:
    """Initialize and configure the Gemini model."""
    genai.configure(api_key=api_key)

    generation_config = {
        "temperature": 1,
        "top_p": 0.95,
        "top_k": 40,
        "max_output_tokens": 8192,
        "response_schema": ImageAnswer,
        "response_mime_type": "application/json",
    }

    return genai.GenerativeModel(
        model_name="gemini-1.5-flash-8b",
        generation_config=generation_config,
    )


def get_cost(response: Any) -> float:
    """Calculate the cost of a Gemini API call in USD."""
    input_token_price = 0.0375 / 1e6
    output_token_price = 0.15 / 1e6
    input_tokens = response.usage_metadata.prompt_token_count
    output_tokens = response.usage_metadata.candidates_token_count
    return input_tokens * input_token_price + output_tokens * output_token_price


def process_image(
    model: genai.GenerativeModel, image: Any
) -> Tuple[Dict[str, List[str]], float]:
    """Process a single image to generate keywords using Gemini."""
    response = model.generate_content(
        [
            image,
            "Describe these images in as many keywords as you like. Return as a list of keywords.",
        ]
    )
    cost = get_cost(response)
    response_data = json.loads(response.text)
    return response_data, cost


def main():
    args = parse_args()

    # Setup output directory and logging
    output_dir = Path(__file__).parent / "outputs" / "get_LAION_image_keywords"
    output_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(message)s",
        level=logging.INFO,
        filename=output_dir / "log.log",
    )
    logger = logging.getLogger(__name__)

    # Initialize model
    model = setup_model(args.api_key)

    # Sample LAION features
    logger.info("Sampling LAION features...")
    used_feature_fps, features, included_indices = get_random_samples_from_LAION(
        n_samples=args.n_samples, clf_thresh=args.clf_thresh
    )

    # Save features and sample info
    features_fp = output_dir / "features.npy"
    np.save(features_fp, features)
    logger.info(f"Saved raw features to {features_fp}")

    sample_info_fp = output_dir / "random_sample_info.pkl"
    with open(sample_info_fp, "wb") as f:
        pickle.dump((used_feature_fps, included_indices), f)
    logger.info(f"Saved random sample info to {sample_info_fp}")

    # Load images
    logger.info("Loading sample images...")
    sample_helper = LAIONSampleHelper(used_feature_fps, included_indices)
    images, _ = sample_helper.get_sample_info(indices=range(args.n_samples))
    logger.info(f"Loaded {len(images)} images.")

    # Process images in parallel
    image_keywords: Dict[str, List[str]] = {}
    cumulative_cost = 0.0
    processed_count = 0

    logger.info("Starting keyword generation...")
    with ProcessPoolExecutor(max_workers=args.n_workers) as executor:
        future_to_index = {
            executor.submit(process_image, model, image): idx
            for idx, image in enumerate(images)
        }

        with tqdm(total=len(future_to_index), desc="Processing images") as pbar:
            for future in as_completed(future_to_index):
                idx = future_to_index[future]
                try:
                    response_data, cost = future.result()
                except Exception as exc:
                    logger.exception(f"Image {idx} generated an exception: {exc}")
                    continue

                keywords = response_data.get("keywords", [])
                image_keywords[str(idx)] = keywords
                cumulative_cost += cost
                processed_count += 1

                pbar.set_postfix(cost=cumulative_cost)
                pbar.update(1)

                # Save intermediate results
                if processed_count % args.save_interval == 0:
                    interim_fp = output_dir / "image_keywords_partial.json"
                    with open(interim_fp, "w") as f:
                        json.dump(image_keywords, f)
                    logger.info(
                        f"Saved intermediate results after {processed_count} samples, "
                        f"cumulative cost: ${cumulative_cost:.2f}."
                    )

                # Check cost threshold
                if cumulative_cost > args.max_cost:
                    logger.info(
                        f"Cumulative cost exceeds threshold of ${args.max_cost}. Stopping."
                    )
                    break

            # Cancel remaining futures
            for future in future_to_index:
                if not future.done():
                    future.cancel()

    # Save final results
    final_fp = output_dir / "image_keywords.json"
    with open(final_fp, "w") as f:
        json.dump(image_keywords, f)
    logger.info(
        f"Processing complete. Processed {processed_count} images with "
        f"cumulative cost ${cumulative_cost:.2f}."
    )
    logger.info(f"Final keywords saved to {final_fp}")


if __name__ == "__main__":
    main()
