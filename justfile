# Recipes run from the repository root, including on just 1.21.

# List the recipes.
default:
    @just --list

# Lint the Python sources.
lint:
    #!/usr/bin/env bash
    set -euo pipefail
    cd "{{justfile_directory()}}"
    uv run ruff check .
    uv run ruff format --check .

# Apply the linter fixes and format the sources.
fmt:
    #!/usr/bin/env bash
    set -euo pipefail
    cd "{{justfile_directory()}}"
    uv run ruff check --fix .
    uv run ruff format .

# Run the tests.
test:
    #!/usr/bin/env bash
    set -euo pipefail
    cd "{{justfile_directory()}}"
    uv run pytest

# Lint and test. Continuous integration runs this.
check: lint test

# Build the HTML documentation into docs/_build/html.
docs:
    #!/usr/bin/env bash
    set -euo pipefail
    cd "{{justfile_directory()}}"
    uv run sphinx-build -W -b html docs docs/_build/html

# Fit the plant and write output/identify.png. An optional path overrides config/config.yaml.
identify config="config/config.yaml":
    #!/usr/bin/env bash
    set -euo pipefail
    cd "{{justfile_directory()}}"
    uv run python app/identify.py --config "{{config}}"

# Tune the firmware PID and write output/tune.png. An optional path overrides config/config.yaml.
tune config="config/config.yaml":
    #!/usr/bin/env bash
    set -euo pipefail
    cd "{{justfile_directory()}}"
    uv run python app/tune.py --config "{{config}}"
