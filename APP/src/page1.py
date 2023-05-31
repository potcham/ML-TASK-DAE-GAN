import streamlit as st

introduction = """
Brief details about DAE GAN:

### What is DAE GAN?

DAE GAN is an text to image generation architecture proposed by Shulan Ruan and teammates in ICCV 2021.
Dynamic Aspect-awarE GAN (DAE-GAN) gets a better understanding with text information based on multiple 
granularities, including sentence-level, word-level, and aspect-leve.

![](../images/framework.png)

### Which dataset for this specific project?

DAE-GAN has been trained and evaluated on th CUB-200 and COCO datasets.

Since these a ML homework project, the model has been only trained for only 1 epoch and
1 GPU ysung the CUD-200 dataset and 5 captions for each image, the original implementation has 
been shared in the README file.

### How can I use this LOCAL MODEL DEPLOYMENT?

```
Example usage:
1. Select a page (option box at the top-left):
    * Text to image

2. Input
    * PROMPT: a small red and white bird with a small curved beak'
    * NUMBER OF IMAGES: 1-4
```

"""

# Define the pages
def page1():
    st.title("DAE-GEN: Text to Image")
    st.markdown(introduction)