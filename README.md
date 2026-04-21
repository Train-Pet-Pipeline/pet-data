# pet-data

Data acquisition, cleaning, augmentation, and weak supervision pipeline for the Train-Pet-Pipeline project.

## Prerequisites

pet-data depends on `pet-infra` as a peer dependency. Install it first using the tag pinned in
[`pet-infra/docs/compatibility_matrix.yaml`](https://github.com/Train-Pet-Pipeline/pet-infra/blob/main/docs/compatibility_matrix.yaml):

```bash
pip install 'pet-infra @ git+https://github.com/Train-Pet-Pipeline/pet-infra@<matrix_tag>'
```

Then install pet-data:
```bash
pip install -e . --no-deps
```

## Installation

```bash
pip install -e ".[dev]"
```
