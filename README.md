## 📖 Project Introduction

Fashion-MNIST is a dataset of Zalando's article images, consisting of 60,000 training examples and 10,000 test examples. Each example is a 28x28 grayscale image associated with one of 10 fashion categories.

This project builds a hybrid model combining CNN and Transformer architectures. Specifically, the CNN is used to extract local features, followed by a Transformer to capture global features. The model contains only about 600k parameters, making it lightweight and suitable for training on most devices. After 90 epochs of training, it achieves an impressive accuracy of 93.1%.

Finally, Grad-CAM is used to visualize and interpret the model’s decision-making process by highlighting the important regions in the input image that influence predictions.


---

## 🛠️ How to implement?

1. Clone this project.
```sh
$ git clone https://github.com/James-sjt/FashionMnist.git
```
2. Install the requirement.
```bash
$ cd FashionMnist
$ pip install -r requirement.txt
```
3. Train this model, using the default parameters.
```bash
$ cd MyModel
$ python train.py
```
```
**************************************************
EPOCH: 97, result on training set:
              precision    recall  f1-score   support

           0       0.89      0.90      0.90      6000
           1       1.00      0.99      1.00      6000
           2       0.90      0.91      0.90      6000
           3       0.93      0.95      0.94      6000
           4       0.90      0.90      0.90      6000
           5       0.99      0.99      0.99      6000
           6       0.83      0.80      0.82      6000
           7       0.96      0.97      0.97      6000
           8       0.99      0.99      0.99      6000
           9       0.98      0.97      0.97      6000

    accuracy                           0.94     60000
   macro avg       0.94      0.94      0.94     60000
weighted avg       0.94      0.94      0.94     60000

Mean of Loss: 0.0839, current lr: 0.0000
**************************************************
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 10000/10000 [00:17<00:00, 577.12it/s]
**************************************************
EPOCH: 97, result on testing set:
              precision    recall  f1-score   support

           0       0.88      0.87      0.88      1000
           1       0.98      0.99      0.99      1000
           2       0.90      0.91      0.91      1000
           3       0.94      0.92      0.93      1000
           4       0.89      0.91      0.90      1000
           5       0.99      0.98      0.99      1000
           6       0.81      0.79      0.80      1000
           7       0.96      0.98      0.97      1000
           8       0.99      0.99      0.99      1000
           9       0.97      0.96      0.97      1000

    accuracy                           0.93     10000
   macro avg       0.93      0.93      0.93     10000
weighted avg       0.93      0.93      0.93     10000

Mean of Loss: 0.2000
Best Accuracy: 0.9310, parameters saved!
**************************************************
```
If you want to change the default parameters, which include dim, depth, heads, dim_head, mlp_dim, batch_size, lr and pre_train, you can run, eg:
```bash
$ python train.py --depth=5 --lr=1e-4
```
And you can also use pre-trained parameters:
```bash
$ python train.py --pre_train=True
```

If you need any help when training the model.
```bash
$ python train.py -h
```
```
Train this model.

Usage:
  Train.py [--dim=<int>] [--depth=<int>] [--heads=<int>] [--dim_head=<int>]
           [--mlp_dim=<int>] [--batch_size=<int>] [--lr=<float>] [--pre_train=<bool>]
  Train.py (-h | --help)

Options:
  -h --help                 Show this screen.
  --dim=<int>              Input feature dimension of Transformer [default: 128]
  --depth=<int>            Number of Transformer blocks [default: 4]
  --heads=<int>            Number of heads in Transformer [default: 4]
  --dim_head=<int>         Dimension of each head [default: 32]
  --mlp_dim=<int>          MLP layer dimension [default: 256]
  --batch_size=<int>       Batch size [default: 2]
  --lr=<float>             Learning rate [default: 1e-3]
  --pre_train=<bool>        Use pre-trained weights [default: False]
```

4. To draw the heatmap. The heatmap will be saved in FashionMnist/HeapImg
```bash
$ mkdir HeapImg
$ python Visulization.py
```

Some examples of visulization:
![b5326dbe7ec4142e7fd14ddb1c51dcc0](https://github.com/user-attachments/assets/b5295e8c-1ea2-435b-afe4-5f7fd29c5114)
![9d1b0261206ab07647162ba44d2c4797](https://github.com/user-attachments/assets/421a2f79-8417-4c8c-9d0e-cde99f286b2c)
![91367f9e5f2eba7ed33c0b90324dd5f1](https://github.com/user-attachments/assets/4ffaebe6-f172-4ac9-b09f-41500ddcfe68)
![919719b166d524266f26ca062e6860b9](https://github.com/user-attachments/assets/0c27846a-1967-4618-b509-10a081df5221)
![4c9c259b6bcad0a7ded406632b495dc2](https://github.com/user-attachments/assets/60c7c55a-1ea3-45cd-a6be-17587b022786)

5. Comparisons to other models, eg. VGG16, ResNet50 and MobileNet
```sh
$ cd ../Experiments
$ python <ModelName>.py
```
The model will use pre-trained parameters by default. If you want to change hyper-parameters, using:
```sh
$ python <ModelName>.py --batch_size=5 --lr=1e-4 --pre_train=False
```
If you need more help when doing experiment, run:
```sh
$ python <ModelName>.py -h
```
```sh
Fine-tuning ResNet50.

Usage:
  Train.py [--batch_size=<int>] [--lr=<float>] [--pre_train=<bool>]
  Train.py (-h | --help)

Options:
  -h --help                 Show this screen.
  --batch_size=<int>       Batch size [default: 2]
  --lr=<float>             Learning rate [default: 1e-3]
  --pre_train=<bool>        Use pre-trained weights [default: True]
```
