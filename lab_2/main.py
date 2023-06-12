import numpy as np
import pandas as pd
import graphviz
from matplotlib import pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.tree import export_graphviz, DecisionTreeClassifier

df = pd.read_csv("WQ-R.csv", sep=';')

def tree_influence(depth, min_el_count):
    classifier_dict = {
        'accuracy': [],
        'precision': [],
        'recall': [],
        'f1': []
    }
    for d in depth:
        a = []
        p = []
        r = []
        f = []
        for el in min_el_count:
            dt = DecisionTreeClassifier(max_depth=d, min_samples_leaf=el, random_state=0)
            dt.fit(X_train, y_train)
            y_pred = dt.predict(X_test)
            score = classification_report(y_test, y_pred, output_dict=True)
            a.append(score['accuracy'])
            p.append(score['weighted avg']['precision'])
            r.append(score['weighted avg']['recall'])
            f.append(score['weighted avg']['f1-score'])

        classifier_dict['accuracy'].append(a)
        classifier_dict['precision'].append(p)
        classifier_dict['recall'].append(r)
        classifier_dict['f1'].append(f)

    return classifier_dict

def plot_influence(depth, min_el_count):
    fig = plt.figure()
    classifier_plot = tree_influence(depth, min_el_count)
    X, Y = np.meshgrid(depth, min_el_count)
    Z = [np.array(classifier_plot[i]) for i in classifier_plot.keys()]

    for i in range(0, len(Z)):
        ax = fig.add_subplot(2, 2, i+1, projection='3d')
        ax.plot_surface(X, Y, Z[i], cmap='viridis')
        name = str(list(classifier_plot.keys())[i])
        ax.set_title(name + ' score plot')
        ax.set_xlabel('depth')
        ax.set_ylabel('element')
        ax.set_zlabel(name)

    plt.tight_layout()
    plt.show()

def plot_importance(clf_list, names):
    col = len(clf_list)
    fig, axes = plt.subplots(nrows=1, ncols=col, figsize=(12, 4))
    for i in range(0, col):
        importance = clf_list[i].feature_importances_
        axes[i].bar(range(X_train.shape[1]), importance)
        axes[i].set_xlabel('Features')
        axes[i].set_ylabel('Importance')
        axes[i].set_title(names[i])

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    num_rows, num_columns = df.shape
    print("Number of rows:", num_rows, " Number of columns:", num_columns)
    print("10 ROWS:")
    print(df.head(10).to_string())

    '''Розділити набір даних на навчальну(тренувальну) та тестову вибірки'''
    X = df.drop('quality', axis=1)  # features
    y = df['quality']  # target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

    print(f"X_train: \n{X_train.head()}\n")
    print(f"y_train: \n{y_train.head()}\n")

    '''Збудувати класифікаційну модель дерева прийняття рішень глибини 5 та навчити
    її на тренувальній вибірці'''
    clf = tree.DecisionTreeClassifier(max_depth=5)
    clf.fit(X_train, y_train)

    '''Представити графічно побудоване дерево'''
    dot_data = export_graphviz(clf, out_file=None, feature_names=X.columns, filled=True, rounded=True,
                               special_characters=True)
    tr = graphviz.Source(dot_data)
    tr.view(filename="Decision Tree")
    print('ПОБУДОВАНО DECISION TREE')

    '''Обчислити класифікаційні метрики збудованої моделі для тренувальної
    та тестової вибірки.'''
    y_train_pred = clf.predict(X_train)
    y_test_pred = clf.predict(X_test)

    print('--------ТРЕНУВАЛЬНА ВИБІРКА--------')
    print(classification_report(y_train, y_train_pred))

    print('--------ТЕСТОВА ВИБІРКА--------')
    print(classification_report(y_test, y_test_pred))
    print('Представимо роботу моделі графічно через матрицю помилок')
    print(confusion_matrix(y_test, y_test_pred))

    print('--------МОДЕЛЬ НА ОСНОВІ ЕНТРОПІЇ--------')
    clf_entropy = DecisionTreeClassifier(criterion='entropy', max_depth=5)
    clf_entropy.fit(X_train, y_train)
    y_p_entropy = clf_entropy.predict(X_test)
    print(classification_report(y_test, y_p_entropy))

    dot_data_entropy = export_graphviz(clf_entropy, out_file=None, feature_names=X.columns,
                                       filled=True, rounded=True, special_characters=True)
    tr_entropy = graphviz.Source(dot_data_entropy)
    tr_entropy.view(filename="Tree Decision Entropy")

    print('--------МОДЕЛЬ НА ОСНОВІ НЕОДНОРІДНОСТІ ДЖИНІ--------')
    clf_gini = DecisionTreeClassifier(criterion='gini', max_depth=5)
    clf_gini.fit(X_train, y_train)
    y_p_gini = clf_gini.predict(X_test)
    print(classification_report(y_test, y_p_gini))

    dot_data_gini = export_graphviz(clf_gini, out_file=None, feature_names=X.columns,
                                    filled=True, rounded=True, special_characters=True)
    tr_gini = graphviz.Source(dot_data_gini)
    tr_gini.view(filename="Decision Tree Gini")

    '''З’ясувати вплив глибини дерева та мінімальної кількості елементів в
    листі дерева на результати класифікації. Результати представити
    графічно.'''
    depth = np.arange(1, 10)
    min_el_count = np.arange(1, 10)
    plot_influence(depth, min_el_count)

    '''Навести стовпчикову діаграму важливості атрибутів, які
    використовувалися для класифікації'''
    feature_name = list(df.columns[:-1])
    models = [clf, clf_entropy, clf_gini]
    model_names = ['Initial model', 'Entropy model', 'Gini model']
    plot_importance(models, model_names)