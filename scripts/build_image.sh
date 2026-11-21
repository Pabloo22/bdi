#!/usr/bin/env bash
# Build and push the image for rented GPU machines (Dockerfile), then print its name.
#
#   scripts/build_image.sh
#
# The tag is a digest of the image's only inputs: the Dockerfile, .dockerignore, the two manifests
# and the vendored verl, which the gpu group installs from its folder. So a tag names exactly one
# dependency set, an unchanged input rebuilds to the same tag, and a run can cite the image it
# used. Renting `:latest` instead would be unpinnable, and a host that has already cached that
# name may not re-pull a newer push of it.
set -euo pipefail
cd "$(dirname "$0")/.."

image=pabloo22/bdi

# The verl files exactly as the build sees them: .dockerignore drops __pycache__ and *.egg-info.
tag=$(
  {
    cat Dockerfile .dockerignore pyproject.toml poetry.lock
    find bdi/vendored/oel/verl -type f ! -path '*/__pycache__/*' ! -path '*.egg-info/*' -print0 |
      LC_ALL=C sort -z | xargs -0 sha256sum
  } | sha256sum | cut -c1-12
)

docker build -t "$image:$tag" .
docker push "$image:$tag"
echo "$image:$tag"
