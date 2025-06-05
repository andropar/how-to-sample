# How to Sample the World for Understanding the Visual System
This repository contains the code and experiments for the paper **"How to sample the world for understanding the visual system"**, published as a full paper in the [Computational and Cognitive Neuroscience (CCN) 2025](https://ccneuro.org/) proceedings.

## Abstract
Understanding vision requires capturing the vast diversity of the visual world we experience. How can we sample this diversity in a manner that supports robust, generalizable inferences? While widely-used, massive neuroimaging datasets have strongly contributed to our understanding of brain function, their ability to comprehensively capture the diversity of visual and semantic experiences has remained largely untested. 

Here we used a subset of 120 million natural photographs filtered from LAION-2B - **LAION-natural** - as a proxy of the visual world in assessing visual-semantic coverage. Our analysis showed significant representational gaps in existing datasets (NSD, THINGS), demonstrating that they cover only a subset of the space spanned by LAION-natural. Importantly, our results suggest that even moderately sized stimulus sets can achieve strong generalization if they are sampled from a diverse stimulus pool, and that this diversity is more important than the specific sampling strategy employed.

## Key Findings
1. **Coverage Gaps**: Existing large-scale neuroimaging datasets (NSD, THINGS) cover only ~50% of the visual-semantic (CLIP) space defined by LAION-natural
2. **Diversity Matters**: Dataset diversity is more critical for out-of-distribution generalization than the specific sampling strategy used
3. **Practical Scale**: Diverse stimulus sets of moderate size (5,000-10,000 images) can achieve strong generalization performance
4. **Sampling Strategy**: While Core-Set sampling shows slight advantages, most strategies perform similarly when the underlying stimulus pool is diverse

## Installation
1. Create new conda environment

```bash
conda create -n how_to_sample python=3.10
conda activate how_to_sample
```

2. Install package and dependencies
```bash
pip install -e .
pip install -r requirements.txt
```

## Data Requirements

To reproduce the results, you will need to download the following datasets:
- **LAION-2B**: Download from [LAION](https://laion.ai/blog/laion-5b/) (~2B image-text pairs)
- **Natural Scenes Dataset (NSD)**: Download from [NSD](http://naturalscenesdataset.org/) (~70k images + fMRI responses)
- **THINGS Database**: Download from [THINGS](https://things-initiative.org/) (~26k object images)

The experiment scripts expect all datasets to be in the `/data` directory (`data/laion2b`, `data/nsd`, `data/things`). 

LAION-2B is required for some of the experiments - these are marked with a ⚠️. You can download LAION-2B using the img2dataset tool (see [here](https://github.com/rom1504/img2dataset) for instructions) with `output_format='files'`. After extracting CLIP features, you can then use the classifier `data/laion_natural_img_clf.pkl` to filter the images and obtain LAION-natural (this is done automatically in the scripts). 

## Reproducing Paper Results
First, extract CLIP features from all datasets:

```bash
cd experiments/coverage/extract_features

# Extract NSD features
python extract_nsd_features.py --hdf5_path /path/to/nsd_stimuli.hdf5 --output_fp outputs/nsd_clip_features.npz

# Extract THINGS features  
python extract_things_features.py --image_glob "/path/to/things/images/*/*.jpg" --output_fp outputs/things_clip_features.npz

# Extract LAION features ⚠️
python extract_laion_features.py --tar_glob "/path/to/laion/tar_files/*.tar"
```

### Visual-semantic coverage of LAION-natural ⚠️
Generate k-means cluster centers from LAION-natural samples for cluster-based analysis:
```bash
cd experiments/coverage
python cluster_laion.py
```

Calculate coverage metrics:
```bash
# Calculate NSD and THINGS coverage of LAION-natural
python calculate_coverage.py --task datasets_vs_laion_natural 

# Calculate LAION-natural coverage of LAION-2B  
python calculate_coverage.py --task laion_natural_vs_laion_2b 
```

Visualize results:
```bash
jupyter notebook notebooks/plot_coverage_metrics.ipynb
jupyter notebook notebooks/cluster_comparison.ipynb  
```

### Effect of dataset diversity on out-of-distribution generalization

#### Simulated data (GMM + LAION-2B ⚠️) 
```bash
cd experiments/ood_accuracy
jupyter notebook OOD_accuracy_GMM_LAION.ipynb
```

#### NSD fMRI validation
```bash
python test_nsd_OOD_accuracy.py 
python plot_nsd_OOD_results.py 
```

### Effect of sampling strategy on generalization

Run sampling experiments:
```bash
cd experiments/sampling
bash scripts/3_test_sampling_gmm.sh
bash scripts/3_test_sampling_laion.sh ⚠️
bash scripts/3_test_sampling_nsd.sh
```

Visualize results:
```bash
jupyter notebook plot_sampling_results.ipynb
```

### Effect of sampling strategy on concept distribution ⚠️
Generate image keywords using Gemini API:
```bash
cd experiments/concept_distribution
python get_LAION_image_keywords.py --api-key YOUR_GEMINI_API_KEY 
```

Apply sampling strategies to keyword-labeled images:
```bash
python get_image_keyword_subsets.py 
```

Compare concept distributions:
```bash
jupyter notebook compare_concept_distributions.ipynb
```
## Citation

If you use this code or our photography classifier in your research, please cite:

```bibtex
@article{roth2025sampling,
  title={How to sample the world for understanding the visual system},
  author={Roth, Johannes and Hebart, Martin N.},
  journal={Computational and Cognitive Neuroscience (CCN)},
  year={2025},
  institution={Max Planck Institute for Human Cognitive and Brain Sciences}
}
```

## Contact

**Johannes Roth**  
Max Planck Institute for Human Cognitive and Brain Sciences  
Email: jroth@cbs.mpg.de

## License

This project is developed for research purposes. The code is available for academic use. Please cite appropriately if using this code or the photography classifier in your work.
