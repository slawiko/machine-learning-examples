# Метрики качества классификации

### Введение

В задачах классификации может быть много особенностей, влияющих на подсчет качества: различные цены ошибок, несбалансированность классов и т.д. Из-за этого существует большое количество метрик качества — каждая из них рассчитана на определенное сочетание свойств задачи и требований к ее решению.

Меры качества классификации можно разбить на две большие группы: предназначенные для алгоритмов, выдающих номера классов, и для алгоритмов, выдающих оценки принадлежности к классам. К первой группе относятся доля правильных ответов, точность, полнота, F-мера. Ко второй — площади под ROC- или PR-кривой.

### Реализация в sklearn

Различные метрики качества реализованы в пакете [sklearn.metrics](http://scikit-learn.org/stable/modules/classes.html). Конкретные функции указаны в инструкции по выполнению задания.

### Материалы

* [Подробнее о метриках качества](https://github.com/esokolov/ml-course-msu/blob/master/ML15/lecture-notes/Sem05_metrics.pdf)

### Инструкция по выполнению

<ol>
<li>Загрузите файл [classification.csv](../classification.csv). В нем записаны истинные классы объектов выборки (колонка true) и ответы некоторого классификатора (колонка predicted).</li>

<li>Подсчитайте величины TP, FP, FN и TN согласно их определениям. Например, FP — это количество объектов, имеющих класс 0, но отнесенных алгоритмом к классу 1. Ответ в данном вопросе — четыре числа через пробел. Заполните таблицу ошибок классификации:

|                    | Actual Positive | Actual Negative |
|:------------------:|:---------------:|:---------------:|
| Predicted Positive | TP              | FP              |
| Predicted Negative | FN              | TN              |

</li>
<li>Посчитайте основные метрики качества классификатора:

* Accuracy (доля верно угаданных) — [sklearn.metrics.accuracy_score](http://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html)

* Precision (точность) — [sklearn.metrics.precision_score](http://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_score.html)

* Recall (полнота) — [sklearn.metrics.recall_score](http://scikit-learn.org/stable/modules/generated/sklearn.metrics.recall_score.html)

* F-мера — [sklearn.metrics.f1_score](http://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html)

В качестве ответа укажите эти четыре числа через пробел.
</li>
<li>Имеется четыре обученных классификатора. В файле [scores.csv](../scores.csv) записаны истинные классы и значения степени принадлежности положительному классу для каждого классификатора на некоторой выборке:

* для логистической регрессии — вероятность положительного класса (колонка score_logreg),

* для SVM — отступ от разделяющей поверхности (колонка score_svm),

* для метрического алгоритма — взвешенная сумма классов соседей (колонка score_knn),

* для решающего дерева — доля положительных объектов в листе (колонка score_tree).

</li>
<li>Посчитайте площадь под ROC-кривой для каждого классификатора. Какой классификатор имеет наибольшее значение метрики AUC-ROC (укажите название столбца)? Воспользуйтесь функцией [sklearn.metrics.roc_auc_score](http://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html).
</li>
<li>Какой классификатор достигает наибольшей точности (Precision) при полноте (Recall) не менее 70% ? Какое значение точности при этом получается?
</li>
</ol>
Чтобы получить ответ на этот вопрос, найдите все точки precision-recall-кривой с помощью функции [sklearn.metrics.precision_recall_curve](http://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_recall_curve.html). Она возвращает три массива: precision, recall, thresholds. В них записаны точность и полнота при определенных порогах, указанных в массиве thresholds. Найдите максимальной значение точности среди тех записей, для которых полнота не меньше, чем 0.7.
