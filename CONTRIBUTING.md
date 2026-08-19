# Contributing

Thank you for helping improve `gsplat-mojo`. By participating, you agree to the
[`CODE_OF_CONDUCT.md`](CODE_OF_CONDUCT.md).

## Before opening a change

1. Search existing issues and pull requests.
2. Use an issue template for bugs, features, or documentation work.
3. Keep changes focused. Discuss API or architecture changes before investing
   in a large implementation.
4. Do not add datasets, models, images, or generated renders without documented
   source, creator/rightsholder, license, modification history, and evidence
   that redistribution is permitted.

## Development environments

- **Tested — Tier 1:** uv manages the Python package and contract tests.
- **Tested — Tier 1:** pixi manages the native Mojo/MAX checks.
- **Best effort — Tier 2:** Other MAX accelerators are accepted for reports but
  are not a substitute for Tier 1 validation.
- **Planned:** Broader platform support requires the promotion evidence in
  [`SUPPORT.md`](SUPPORT.md).

From the repository root:

```bash
uv sync --extra test
uv run pytest
uv run python driver.py  # requires the Tier 1 GPU environment

cd gsplat
pixi run forward
pixi run intersect
pixi run sh
pixi run render-ply
pixi run package
pixi run mojo format .
```

Run only the checks relevant to a documentation-only change, but explain what
you did and did not run in the pull request.

## Pull requests

- Describe the problem and the reason for the approach.
- Label every public capability statement **Tested — Tier 1**,
  **Best effort — Tier 2**, or **Planned**.
- Update the API contract, support matrix, roadmap, and changelog when relevant.
- Add or update tests for behavior changes.
- Complete the pull request checklist honestly; do not mark an unrun check as
  passing.
- Do not include unrelated formatting or generated files.

Contributions are submitted under the repository's MIT license unless a file
clearly states another compatible license. Do not submit material you lack the
right to redistribute.
