# Variational AutoEncoders on Faces Dataset

### ** Training Code **
To train pass the following arguments as in the example listed below:
```
CUDA\_VISIBLE\_DEVICES=0 python3 train.py --src\_dir='inference\_dir' --batch\_size=32 --lr=0.0001 --latent_space=512
```

### ** Evaluation Script **
To Run inference on the pretrained network weights:
```
python3 infer.py --src_dir='inference\_dir'
```

# VAE_CAE-Faces-dataset
# VAE_CAE-Faces-dataset
