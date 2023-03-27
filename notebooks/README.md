## Notebooks
| Notebook Title | Google Colab Link |
| --- | --- |
| Image Tagging Pipline | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kk-digital/kcg-ml/blob/main/notebooks/example_image_tagging_tools.ipynb)|
| Clip Example | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kk-digital/kcg-ml/blob/main/notebooks/example_clip.ipynb)|
| Clip Cache Example | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kk-digital/kcg-ml/blob/main/notebooks/example_clip_cache.ipynb) |
| Clip Image Similarity | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kk-digital/kcg-ml/blob/main/notebooks/example_image_clip_similarity.ipynb) |
| Ranking App | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kk-digital/kcg-ml/blob/main/notebooks/example_ranking_app.ipynb) |
| Clip Interrogator | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kk-digital/kcg-ml/blob/main/notebooks/clip_interrogator.ipynb) |




## Quick Installation Of OpenClip 
OpenClip model used :: https://github.com/mlfoundations/open_clip

```
pip install open_clip_torch
```
## Installation Of ASCII Graph 
```
pip install ascii_graph
```

## Cleaning Jupyter Notebooks for Version Control
### Installation
First, make sure you have nbstripout and nbconvert installed . You can install them using pip:
```sh
pip install nbstripout nbconvert
```
### Setting up nbstripout

```sh
nbstripout --install
```
Alternative installation to git attributes
```sh
nbstripout --install --attributes .gitattributes
```
### Using nbconvert
```sh
python -m nbconvert --ClearOutputPreprocessor.enabled=True --to notebook *.ipynb --inplace
```

