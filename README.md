# amass_blender
A format for exporting dynamic compensation data to AMASS-style SMPL-X

## download body_models

- https://smpl.is.tue.mpg.de/download.php
- https://smpl-x.is.tue.mpg.de/download.php


## Setup

Thanks for the code sharing of the following open-source repositories:

- [GitHub - nghorbani/amass: Data preparation and loader for AMASS](https://github.com/nghorbani/amass)
- [GitHub - nghorbani/human_body_prior: VPoser: Variational Human Pose Prior](https://github.com/nghorbani/human_body_prior)


```
git clone https://github.com/Beat-in-our-hearts/amass_blender.git
cd amass_blender

conda create -n amass_blender python=3.7

pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html

git clone https://github.com/nghorbani/human_body_prior.git
cd human_body_prior
pip install -r requirements.txt
python setup.py develop
cd ..

git clone https://github.com/nghorbani/amass.git
cd amass
pip install -r requirements.txt
python setup.py develop
cd ..

```