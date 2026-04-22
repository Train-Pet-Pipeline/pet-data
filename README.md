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

## License

This project is licensed under the [Business Source License 1.1](LICENSE) (BSL 1.1).
On **2030-04-22** it converts automatically to the Apache License, Version 2.0.

> Note: BSL 1.1 is **source-available**, not OSI-approved open source.
> Production / commercial use requires a separate commercial license.

![License: BSL 1.1](https://img.shields.io/badge/license-BSL%201.1-blue.svg)
