
[![Build Status](https://github.com/jajcayn/re_hippocampal_model/workflows/pytest/badge.svg)](https://github.com/jajcayn/re_hippocampal_model/actions) [![DOI](https://zenodo.org/badge/353683773.svg)](https://zenodo.org/badge/latestdoi/353683773)

# [Re] Sharp wave ripples in the mass model of the hippocampus
*Replication of hippocampal CA3 rate model due to Evangelista et al., J. Neurosci., 2020.*


This repository contains code that replicates findings concerning sharp wave ripples in the CA3 region of the hippocampus using a simplified rate model. The accompanying paper is currently in progress.

## Abstract
*WIP*

## How to run

### Locally
Fastest, complete control, requires python et al. already set up. Currently, python versions 3.6 - 3.8 are supported.
```bash
git clone https://github.com/jajcayn/re_hippocampal_model.git
cd re_hippocampal_model
pip install --upgrade -r requirements.txt
# tests, just to be sure
pytest
jupyter lab
```

### Docker
Easy to use, only docker required. Runs `jupyter` inside a docker container.
```bash
docker run -p XXXX:XXXX ...
```
and navigate to localhost:XXXX

### Binder
Easiest to use, no setup required, slowest.

Just click here >>
