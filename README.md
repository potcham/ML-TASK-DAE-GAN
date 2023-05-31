# TASK MACHINE LEARNING

This is a ML Homework.

## TASK 1: CHOOSING A MODEL

    Find a Text to Image model with code and train it for only 1 epoch

1. **MODEL** 

    DAE-GAN: Dynamic Aspect-aware GAN for Text-to-Image Synthesis

    * [DAE-GAN Paper](https://openaccess.thecvf.com/content/ICCV2021/papers/Ruan_DAE-GAN_Dynamic_Aspect-Aware_GAN_for_Text-to-Image_Synthesis_ICCV_2021_paper.pdf)
    * [ORGINAL REPO](https://github.com/hiarsal/DAE-GAN)


        ![DAE-GAN framework](images/framework.png)

    *source: Image from [DAE-GAN Paper](https://openaccess.thecvf.com/content/ICCV2021/papers/Ruan_DAE-GAN_Dynamic_Aspect-Aware_GAN_for_Text-to-Image_Synthesis_ICCV_2021_paper.pdf)*

2. **INSTALL** (Anaconda for Python Environment)

    *Make sure to install Anaconda*

    a. create a conda environment 
    ```bash
    conda create -n dae python=3.9
    ```
    b. activate the `dae` environment
    ```bash
    conda create -n dae python=3.9
    ```
    c. Install dependecies
    ```bash
    pip install -r requeriments.txt
    ```

3. **TRAINING**

4. **SAMPLE INPUT - OUTPUT**

    * INPUT: 
    ```python
    prompt = 'a small red and white bird with a small curved beak'
    ```
    * OUTPUT:

    ![DAE-GAN Ouput](output/output.png)


5. **TRAINING DATASET**

    * [Caltech-UCSD Birds-200-2011 (CUB-200-2011)](http://www.vision.caltech.edu/datasets/cub_200_2011/)


6. **NUMBER OF PARAMETERS**

    * Discriminator:
    * Generator: 

7. **MODEL EVALUATION METRIC**

8. **TRAINED MODEL**

    * [disc.pth]()
    * [gen.pth]()

9. **MODIFICATIONS**



## TASK 2: LOCAL DEPLOY MODEL
