# MODELLING THE BRAIN’S RESPONSE TO NATURAL SCENES IN THE BOTTLENECK SPACE.

# Abstract 
Understanding human intelligence through computational models mirroring the brain’s behaviour requires efficient modelling of the brain’s response to visual stimuli. SOTA vision models are inefficient and brain activities are represented by high-dimensional matrices. We propose a novel approach to address this: 1) Projecting image stimuli and brain responses to low-dimensional vectors and learning a non-linear mapping. 2) Incorporating tiny leaks of ground truth during training, replaced with Gaussian noise at inference. 3) Pre-training the non-linear model on all subjects and fine-tuning on each subject. 4) Simultaneously modelling all vertices in the left and right visual cortices using an objective we call "Racing Loss". Our method, based on frozen CLIP-ViT image features, demonstrates significant performance improvements, additionally achieving 12% higher Noise-Normalized Mean Correlation Scores compared to fully fine-tuning CLIP-ViT to the high-dimensional brain responses.

# AUTHORS
Henry Ndubuaku\
ndubuakuhenry@gmail.com\
[Linkedin](https://www.linkedin.com/in/henry-ndubuaku-7b6350b8/)\
Full Paper: 

# Details
![Alt text](/images/bottleneck.png "Diagram")
![Alt text](/images/network.png "Diagram")

# RESULTS
![Alt text](/images/network.png "Diagram")

# USAGE
* Step 1: Ensure you have Jax, Flax, Optax, Haiku and Jax-Dataoader setup, full dependenceies can be found in the provided environment.yaml file. All can be installed using "pip install <package-name>" but for GPU access, please checkout the installation guide on Jax's repository here: https://github.com/google/jax#installation

* Step 2: You can run from scratch, or download the globally pretrained weigthts from here: . preprocessed dataset for each subject can also be be found in the same link.

* Step 3: Place the downloaded files (checkpoints and data) in the outer FBNET folder.

* Step 4: Open the experimentation notebook to run an example usage of the base network.

# NOTE
* The metric reported in the paper require the witheld noise ceilings, this experimentation only use mean correlation without the noise normalisation and at such will yield lower results.

* The clip feature extraction is done separately using the provided scripts.