# Dataset Download Instructions

This project uses the **StackSample** dataset from Kaggle:
**10% of Stack Overflow Q&A**

## Option 1: Manual Download (Recommended)

1. Go to: https://www.kaggle.com/datasets/stackoverflow/stacksample
2. Click **Download** (requires a free Kaggle account)
3. Extract `archive.zip` into this `data/` directory
4. Verify you have these files:

data/
├── Questions.csv (~1.9 GB)
├── Answers.csv (~1.6 GB)
└── Tags.csv (~10 MB)


## Option 2: Kaggle API

```bash
pip install kaggle
mkdir -p ~/.kaggle
cp kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json

kaggle datasets download stackoverflow/stacksample
unzip stacksample.zip -d data/