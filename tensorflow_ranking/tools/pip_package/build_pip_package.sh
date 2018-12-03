#!/usr/bin/env bash
# Copyright 2015 The TensorFlow Ranking Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
set -e

function is_absolute {
  [[ "$1" = /* ]] || [[ "$1" =~ ^[a-zA-Z]:[/\\].* ]]
}

function real_path() {
  is_absolute "$1" && echo "$1" || echo "$PWD/${1#./}"
}

function build_wheel() {
  TMPDIR="$1"
  DEST="$2"
  PKG_NAME_FLAG="$3"

  mkdir -p "$TMPDIR"
  echo $(date) : "=== Preparing sources in dir: ${TMPDIR}"

  if [ ! -d bazel-bin/tensorflow_ranking ]; then
    echo "Could not find bazel-bin.  Did you run from the root of the build tree?"
    exit 1
  fi
  cp -r "bazel-bin/tensorflow_ranking/tools/pip_package/build_pip_package.runfiles/org_tensorflow_ranking/tensorflow_ranking" "$TMPDIR"
  cp tensorflow_ranking/tools/pip_package/setup.py "$TMPDIR"

  # Make sure init files exist.
  touch "${TMPDIR}/tensorflow_ranking/__init__.py"
  touch "${TMPDIR}/tensorflow_ranking/python/__init__.py"

  pushd ${TMPDIR} > /dev/null
  echo $(date) : "=== Building wheel"
  "${PYTHON_BIN_PATH:-python}" setup.py bdist_wheel --universal
  mkdir -p ${DEST}
  cp dist/* ${DEST}
  popd > /dev/null
  echo $(date) : "=== Output wheel file is in: ${DEST}"
  rm -rf "${TMPDIR}"
}

function main() {
  PKG_NAME_FLAG="tensorflow_ranking"

  DSTDIR="$(real_path $1)"
  SRCDIR="$(mktemp -d -t tmp.XXXXXXXXXX)"
  if [[ -z "$DSTDIR" ]]; then
    echo "No destination dir provided"
    exit 1
  fi

  build_wheel "$SRCDIR" "$DSTDIR" "$PKG_NAME_FLAG"
}

main "$@"
