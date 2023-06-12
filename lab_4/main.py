import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, \
    f1_score
from sklearn.model_selection import ShuffleSplit
from sklearn.neighbors import KNeighborsClassifier

df = pd.read_csv("WQ-R.csv", sep=";")

if __name__ == '__main__':
    '''Визначити та вивести кількість записів'''
    num_rows, num_columns = df.shape
    print("Number of rows:", num_rows, " Number of columns:", num_columns)

    '''Вивести атрибути набору даних'''
    print(list(df.columns))

    '''Отримати десять варіантів перемішування набору даних та розділення 
    його на навчальну (тренувальну) та тестову вибірки, використовуючи
    функцію ShuffleSplit. Сформувати начальну та тестові вибірки на
    основі восьмого варіанту. З’ясувати збалансованість набору даних'''
    X = df.drop('quality', axis=1)  # features
    y = df['quality']  # target

    shuf_spl = ShuffleSplit(n_splits=10, test_size=0.3, random_state=0)
    train_index, test_index = list(shuf_spl.split(X))[7]

    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    print('------------ТРЕНУВАЛЬНА ВИБІРКА------------')
    print(X_train.shape, y_train.shape)
    train_uni, train_count = np.unique(y_train, return_counts=True)
    train = dict(zip(train_uni, train_count))
    print(train)
    print('------------ТЕСТОВА ВИБІРКА------------')
    print(X_test.shape, y_test.shape)
    test_uni, test_count = np.unique(y_test, return_counts=True)
    print(dict(zip(test_uni, test_count)))

    class_labels = list(train.keys())

    '''Використовуючи функцію KNeighborsClassifier бібліотеки scikit-learn, 
    збудувати класифікаційну модель на основі методу k найближчих
    сусідів (значення всіх параметрів залишити за замовчуванням) та
    навчити її на тренувальній вибірці, вважаючи, що цільова
    характеристика визначається стовпчиком quality, а всі інші виступають
    в ролі вихідних аргументів. '''
    kn = KNeighborsClassifier()
    kn.fit(X_train, y_train)
    y_train_pr = kn.predict(X_train)
    y_test_pr = kn.predict(X_test)

    '''Обчислити класифікаційні метрики збудованої моделі для тренувальної
    та тестової вибірки. Представити результати роботи моделі на тестовій
    вибірці графічно. '''
    print('---------МЕТРИКИ---------')
    print('------------ТРЕНУВАЛЬНА ВИБІРКА------------')
    print(classification_report(y_train, y_train_pr, output_dict=True))
    print('------------ТЕСТОВА ВИБІРКА------------')
    print(classification_report(y_test, y_test_pr, output_dict=True))

    conf_matrix = confusion_matrix(y_test, y_test_pr)
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=class_labels, yticklabels=class_labels)
    plt.title('Confusion matrix of test values')
    tick_marks = np.arange(len(class_labels))
    plt.xticks(tick_marks, class_labels)
    plt.yticks(tick_marks, class_labels)
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.show()

    '''З’ясувати вплив кількості сусідів (від 1 до 20) на результати
    класифікації. Результати представити графічно.'''
    acc_train = []
    acc_test = []

    pre_train = []
    pre_test = []

    rec_train = []
    rec_test = []

    f1_train = []
    f1_test = []

    neighbors = range(1, 21)

    for k in neighbors:
        kn = KNeighborsClassifier(n_neighbors=k)
        kn.fit(X_train, y_train)
        train_pr = kn.predict(X_train)
        test_pr = kn.predict(X_test)

        acc_train.append(accuracy_score(y_train, train_pr))
        acc_test.append(accuracy_score(y_test, test_pr))
        pre_train.append(precision_score(y_train, train_pr, average='weighted'))
        pre_test.append(precision_score(y_test, test_pr, average='weighted'))
        rec_train.append(recall_score(y_train, train_pr, average='weighted'))
        rec_test.append(recall_score(y_test, test_pr, average='weighted'))
        f1_train.append(f1_score(y_train, train_pr, average='weighted'))
        f1_test.append(f1_score(y_test, test_pr, average='weighted'))

    plt.plot(neighbors, acc_train, label='Training')
    plt.plot(neighbors, acc_test, label='Test')
    plt.title('Accuracy Score')
    plt.xlabel('Neighbors (k)')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

    plt.plot(neighbors, pre_train, label='Training')
    plt.plot(neighbors, pre_test, label='Test')
    plt.title('Precision Score')
    plt.xlabel('Neighbors (k)')
    plt.ylabel('Precision')
    plt.legend()
    plt.show()

    plt.plot(neighbors, rec_train, label='Training')
    plt.plot(neighbors, rec_test, label='Test')
    plt.title('Recall Score')
    plt.xlabel('Neighbors (k)')
    plt.ylabel('Recall')
    plt.legend()
    plt.show()

    plt.plot(neighbors, f1_train, label='Training')
    plt.plot(neighbors, f1_test, label='Test')
    plt.title('F1 Score')
    plt.xlabel('Neighbors (k)')
    plt.ylabel('F1')
    plt.legend()
    plt.show()









