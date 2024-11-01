#!/usr/bin/env bash

set -e

export PYTHONPATH=${PWD%/*}:$PYTHONPATH
cd ./tests
pytest -k 'not _slow'
