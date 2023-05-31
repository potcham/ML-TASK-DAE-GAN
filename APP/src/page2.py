import streamlit as st
from src.daegan.inference import DAE_GAN
import torch
import os

def set_up_model() -> torch.nn.Module:
    # 1. set parameters
    cfg_path = 'src/daegan/bird_DAEGAN.yml'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 2. create & load model
    model = DAE_GAN(cfg_file=cfg_path, device=device)
    model.load_models()

    return model
    # 3. generate image from prompt
    # model.predict(prompt=prompt, out_path=out_path)


model = set_up_model()

def page2():
    st.title("DAE-GAN Image Generation")
    st.info("""#### NOTE: you can download image by \
    right clicking on the image and select save image as option""")

    with st.form(key='form'):
        prompt = st.text_input(label='Enter text prompt for image generation')
        num_images = st.selectbox('Enter number of images to be generated', (1,2,3,4))
        submit_button = st.form_submit_button(label='Submit')

    if submit_button:
        if prompt:

            try:

                for idx in range(num_images):
                    out_path = os.path.join('src/output', 'image_{}.png'.format(idx))
                    model.predict(prompt, out_path)

                    st.image(out_path, caption=f"Generated image: {idx+1}",
                            use_column_width=True)
            except:
                st.warning("Make sure input prompt must be related to bird's descrition", icon="⚠️")