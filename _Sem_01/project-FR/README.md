# Проект Face Recognition
 	
Разработка пайплайна для задачи Face Recognition

## Cтруктура проекта

В проекте выполнена обязательная часть, состоящая из:
* [1. FaceAlignment](1_FaceAlignment.ipynb)
* [2. ArcFace](2_ArcFace.ipynb)

## Результаты работы

Оформлены и представлены в Jupiter ноутбуках.

Сохранённые состояния весов моделей в ходе обучения заботливо складывались в директории (перенесены на Google Drive, поскольку не помещаются на github):
* [weightsStackedHourglass (для FaceAlignment)](https://drive.google.com/drive/folders/1Hy0N2sCDc7W6sGWLgwJUWYnDpkHxLZWV?usp=sharing)
* [weights_ce (для ArcFace)](https://drive.google.com/drive/folders/1O35jKuXd5uhAOVFzn3pTGdz40w2W4zVw?usp=sharing)
* [weights_arcface (тоже для ArcFace)](https://drive.google.com/drive/folders/1TI4hXNjcxDTfeLwCNPFlqa8mVtcspyVp?usp=sharing)

## TODO: Чего поделать дальше:

1. Во второй части рассмотреть предообученную модель:
```python
class FaceRecognitionModel(nn.Module):

    def __init__( self, num_classes ):
        super().__init__()
        self.backbone=models.resnet18( weights=None )
        self.embedding=nn.Linear( 512, 256 )
        self.classifier=nn.Linear( 256, num_classes )

    def forward(self,x):
        x=self.backbone(x)
        emb=self.embedding(x)
        logits=self.classifier(emb)
        return logits, emb
```

2. Обучить модели на оригинальном (not aligned) датасете и провести сравнение с уже полученными результатами на aligned dataset.

3. Переделать сохранение состоянии весов в папку "weights" и далее "weights/ce", "weights/arcface" и т.д.