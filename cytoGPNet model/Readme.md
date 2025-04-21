### Default parameter settings for AE pretrain

| Name         | Description                                       | Default            |
|--------------|---------------------------------------------------|--------------------|
| -bs          | batch size of the model                           | 128                |
| -lrAE        | learning rate of the autoencoder                  | 1e-6               |
| -e           | maximum epochs to train the autoencoder           | 1000               |
| --latent-dims| latent dimension size of the autoencoder          | 2                  |
| -datadir     | data repository                                   | ./HEUvsUE          |
| -sd          | repository to save model                          | ./ae_output        |
| --save-freq  | frequency to save checkpoints                     | 10                 |

### Default parameter settings for whole model training
| Name                 | Description                               | Default            |
|----------------------|-------------------------------------------|--------------------|
| -bs                  | batch size of the model                   | 1                  |
| -lrAE                | learning rate of the autoencoder          | 1e-4               |
| -lrD                 | learning rate of the the other layers     | 1e-2               |
| --num-inducing-points| number of the inducing points in GP layer | 100                |
| -e                   | maximum epochs to train the autoencoder   | 100                |
| --latent-dims        | latent dimension size of the autoencoder  | 2                  |
| -datadir             | data repository                           | ./HEUvsUE          |
| --pretrained-file    | pretrained AE file                        | must input by user |
| -sd                  | repository to save model                  | ./cytogpnet_output |
| --save-freq          | frequency to save checkpoints             | 10                 |
