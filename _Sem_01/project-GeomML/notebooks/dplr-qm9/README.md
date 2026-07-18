# **Проект**: 
# Геометрический ML для Физики и Beyond

В этом семестре два параллельных потока:

1) Улучшение предсказаний вектора дипольного момента (и тензора поляризуемости) молекул датасета Alchemy с использованием GeomML + TDA

2) Анализ качества предсказаний сигналов (на примере физики частиц, i.e., HIGGS) при использовании различных аугментаций, построенных на (а) hand-crafted symmetries, (b) automatically extracted symmetries (e.g., AtlasD) + добавление фич из TDA (интересно посмотреть, как они меняются при действии групповых свойств из automatically extracted symmetries)

## **Основной objective**:
познакомится основными идеями геометрического ML (e.g. Equivariant Neural Networks) и топологического анализа данных (e.g. комплексом Вьеториса — Рипса и кучей расширений PCA) с последующем практическим применение для задач физики и/или drug design

## **Программа минимум**: 
(a) подготовить простейший полный пайплайн для построения решения с геометрическим ML на отдельно взятом датасете с известными геометрическими prior-ами, 

(b) обучить геометрическую модель на исходнных данных и с использованием топологических фич,

(с) enjoy!  
## **Программа (super) максимум**: 
Решить обратную задачу экстракции prior-ов из топологического анализа данных и построить пайплайн для выбора оптимальной DL архитекутры автоматически

## **Последующие этапы**.
1. Ознакомится с:
    * [5] [Maurice Weiler | Equivariant neural networks - what, why and how?](https://maurice-weiler.gitlab.io/blog_post/cnn-book_1_equivariant_networks), 
    * [7] [YouTube AMMI Course "Geometric Deep Learning"](https://www.youtube.com/watch?v=PtA0lg_e5nA&list=PLn2-dEmQeTfQ8YVuHBOvAhUlnIPYxkeu3), 
    * понятиями SU/SO(3) симметрий 
    * и зачем вообще нужны геометрические модели 
2. Выбрать понятный физический/химический датасет для работы (e.g. QM9/MD17 для посчета HOMO-LUMO gap/энергии) с заложенными симметриями (e.g. трансляционные и вращательные)
3. Собрать pipeline с топологическим анализом даннных (можно посмотреть набор простых статей на medium или сразу сунуться сюда [2] [Jeff Murugan & Duncan Robertson | An Introduction to Topological Data
Analysis for Physicists: From LGM to
FRBs](https://arxiv.org/pdf/1904.11044), и добавлением новых, глобальных фич 
4. Обучить equivariant-модель (например, [8] [[GitHub]: egnn-pytorch](https://github.com/lucidrains/egnn-pytorch) на решение предсказательной задачи на исходном и новом (с топологическими фичами) датасетах 
5. Провести сравнительный анализ результатов, в том числе против простых бейзлайнов (FCNN без Inductive Biases, возможно таблиные методы)
6. Оформить GitHub с результатами, описанием пайплайнов и табличкой результатов
	
## **Критерии**:
- Максимум по проекту можно будет получить 20 баллов.  
- За части 1-2 можно получить 4 балла суммарно
- За части 3, 4, 5, 6 (каждую по отдельности) можно получить еще по 4 балла

---

# **Список литературы и источников**

## Книги

1. Основная книга по теме: [Michael M. Bronstein et. | Geometric Deep Learning: Grids, Groups, Graphs, Geodesics, and Gauges, 27 Apr 2021](https://arxiv.org/abs/2104.13478)

2. [Jeff Murugan & Duncan Robertson | An Introduction to Topological Data
Analysis for Physicists: From LGM to
FRBs](https://arxiv.org/pdf/1904.11044)

## Статьи
3. Jones A., Brown M. Machine Learning Applications in Finance // *Journal of Data Science*. — 2021. — Vol. 19, No. 4. — P. 123–138.
4. [Про графовые нейросети (для понимания предложенной части 4)](https://arxiv.org/pdf/2412.19419)

## Интернет-ресурсы

5. [Maurice Weiler | Equivariant neural networks - what, why and how?](https://maurice-weiler.gitlab.io/blog_post/cnn-book_1_equivariant_networks)
6. [Topological Data Analysis for Pangenomics](https://carpentries-incubator.github.io/topological-data-analysis/aio/index.html)
7. [Michael Bronstein | YouTube AMMI Course "Geometric Deep Learning"](https://www.youtube.com/watch?v=PtA0lg_e5nA&list=PLn2-dEmQeTfQ8YVuHBOvAhUlnIPYxkeu3)
8. [EGNN - Pytorch GitHub](https://github.com/lucidrains/egnn-pytorch)
9. [NVIDIA cuEST](https://developer.nvidia.com/cuda/cuda-x-libraries/cuest)
10. [Лучшие проекты курса DLS прошлых потоков](https://github.com/DeepLearningSchool/Best_Course_Projects)

## Использованные наборы данных

11. понятный физический/химический датасет для работы (e.g. QM9/MD17 для посчета HOMO-LUMO gap/энергии Kaggle. *Titanic Dataset*. URL: https://www.kaggle.com/c/titanic (дата обращения: 10.06.2026).

---