# Contributing

Thanks for helping! The most useful contributions are:

- **Reports of wrong results.** Open an issue with the screenshot and what it should have read.
  These are the main way the recognizer improves.
- Bug fixes in the CLI, the HTTP API or the browser package.
- Docs, and examples of using the package in other tools.

For anything bigger, such as a new overlay or a change to how grids, species or shadows are
detected, please open an issue first so we can agree on the approach.

## Setup

```sh
uv sync                              # Python 3.12, the inference dependencies and dev tools (pytest, ruff)
uv sync --extra train                # only if you'll build references or train (adds PyTorch;
                                     # a later plain `uv sync` removes it again)
uv run recognizer write-fixtures     # test fixtures for the browser build
cd web && npm install
```

## Checks

CI runs these on every pull request. Run them locally first:

```sh
uv run ruff check . && uv run ruff format --check .
uv run pytest
cd web && npx tsc -p tsconfig.json && npm run build && npm test
```

## What you can and can't test

Accuracy is measured with `recognizer eval` on labeled screenshots of real broadcasts. Those
screenshots aren't in the repo: they're broadcast frames and show players. So CI can check that
everything runs, but not whether a change reads teams better or worse.

If your change touches how grids, species, shadows or names are found:
- Try it on screenshots you have, and say which ones in the pull request.
- The maintainer runs the full evaluation before merging.
- Changes that lower accuracy on the labeled set won't be merged, even if they help on other
  screenshots. They can often be reworked so they don't.

Please don't include retrained models (`models/*.onnx`, `models/*.npz`) in a pull request.
Propose the training change in code; models are rebuilt and released by the maintainer.

## Python and TypeScript stay in step

`web/src/` is a line-by-line port of the Python pipeline, and both must find the same grids and
give the same answers. A change to one of these files needs the same change in its pair:

| Python | TypeScript |
| --- | --- |
| `recognizer/grid.py` | `web/src/grid.ts` |
| `recognizer/pipeline.py`, `assign.py`, `forms.py` | `web/src/pipeline.ts`, `web/src/assign.ts` |
| `recognizer/shadow.py` | `web/src/shadow.ts` |
| `recognizer/ocr.py` | `web/src/ocr.ts` |

Constants such as thresholds, grid styles and class lists belong in Python. `recognizer
export-web` writes them into `config.json` for the browser, so don't hardcode them in TypeScript.
If you can't port a change yourself, say so in the pull request and the maintainer will.

## Style

- **Python:** ruff's formatter and checks, configured in `pyproject.toml`.
- **TypeScript:** no formatter is enforced. Match the code around you: two-space indents, double
  quotes, semicolons, lines under 120 columns. The numeric code is compact on purpose so it reads
  side by side with the Python.
- **Comments:** explain why, not what. Look at the existing modules for the tone.

## New Pokémon and forms

When Pokémon GO releases new species or forms, run `uv run recognizer update-species` and commit
`data/species.json`. A species still needs reference sprites and a retrained head before it can be
recognized; that part is the maintainer's.

## Releases

The maintainer publishes the npm package (`web/`) and deploys the demo and API. Pull requests
shouldn't bump versions.

A model release, with the corrections tools have shared since the last one (see the README's
"Shared corrections"):

```sh
git -C ../team-recognizer-data pull
uv run recognizer corrections review            # open cache/corrections-review.html
uv run recognizer corrections accept <id>...    # and reject the rest; then commit the data folder
uv run recognizer train --holdout <newest event>
uv run recognizer eval                          # must not regress
uv run recognizer eval --corrections            # are the corrected tiles now right?
uv run recognizer train                         # the release model, nothing held out
uv run recognizer export-web
```

Then commit `models/`, bump `web/package.json`, publish the package and push. The API redeploys
from the push, and tools pick up the new models when they move to the new package version.

## License

By contributing, you agree that your contributions are licensed under the repository's MIT
license. Screenshots you post in issues are public, so crop out anything you'd rather not share.
