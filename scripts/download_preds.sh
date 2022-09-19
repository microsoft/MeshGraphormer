# --------------------------------
# Setup
# --------------------------------
export REPO_DIR=$PWD
if [ ! -d $REPO_DIR/models ] ; then
    mkdir -p $REPO_DIR/models
fi
BLOB='https://datarelease.blob.core.windows.net/metro'

# --------------------------------
# Download our model predictions that can be submitted to FreiHAND Leaderboard
# --------------------------------
if [ ! -d $REPO_DIR/predictions ] ; then
    mkdir -p $REPO_DIR/predictions
fi
# Our model + test-time augmentation. It achieves 5.9 PA-MPVPE on FreiHAND Leaderboard
wget -nc $BLOB/graphormer-release-ckpt200-multisc-pred.zip -O $REPO_DIR/predictions/ckpt200-multisc-pred.zip
