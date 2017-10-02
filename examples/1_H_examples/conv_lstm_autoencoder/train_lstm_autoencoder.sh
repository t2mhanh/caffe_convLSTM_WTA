#!/usr/bin/env sh
set -e

./build/tools/caffe train --solver=examples/1_H_examples/conv_lstm_autoencoder/solver_patch.prototxt $@
