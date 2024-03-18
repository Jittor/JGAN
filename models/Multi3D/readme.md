# Official code for Multi3D: 3D-aware Multimodal Image Synthesis.

![teaser2](https://github.com/Jittor/JGAN/assets/43036573/92cb4e95-0256-4b9a-ab41-ee8413f721e5)

3D-aware image synthesis has attained high quality and robust 3D consistency. Existing 3D controllable generative models are designed to synthesize 3D-aware images through a single modality, such as 2D segmentation or sketches, but lack the ability to finely control generative content, such as texture and age. In pursuit of enhancing user-guided controllability, we propose Multi3D, a 3D-aware controllable image synthesis model that supports multi-modal input. Our model can govern the geometry of the generated image using a 2D label map, such as a segmentation or sketch map, while concurrently regulating the appearance of the generated image through a textual description. To demonstrate the effectiveness of our method, we conduct experiments on multiple datasets, including CelebAMask-HQ, AFHQ-cat, and shapenet-car. Qualitative and quantitative evaluations prove that our method outperforms existing state-of-the-art methods.

## pipeline

<img width="1308" alt="image" src="https://github.com/Jittor/JGAN/assets/43036573/bf975a58-ad2b-4278-8105-f7f76091ee4c">

## reuslts

![seg2face2](https://github.com/Jittor/JGAN/assets/43036573/10b953d5-db61-41b9-b270-45c72a9cc8a9)

![seg2cat2](https://github.com/Jittor/JGAN/assets/43036573/ba4c6911-4f17-4234-a4f7-c98a7cbc12e9)

![edge2car2](https://github.com/Jittor/JGAN/assets/43036573/a51ff843-56a5-4d04-93ab-8dcc44d13ff0)
