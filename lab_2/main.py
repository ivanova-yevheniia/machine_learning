import pandas as pd
import sklearn.metrics as m
import matplotlib.pyplot as plt
import numpy as np

LINE_MODEL_1 = '-'
LINE_MODEL_2 = '--'

df = pd.read_csv("KM-02-1.csv")
class1 = df[df['GT'] == 1]
class0 = df[df['GT'] == 0]

def get_model_predicate(model_name: str, threshold: float, d):
    '''
    Calculate the values of predicated Yes depending on threshold
    :param model_name: name of chosen model
    :param threshold: step
    :param d: dataframe
    :return: pandas column with values of predicated Yes
    '''
    d['M_pred'] = d[model_name].map(lambda x: 1 if x > threshold else 0)
    return d['M_pred']

def metrics_calculate(y_true, y_pred):
    '''
    Calculate metrics values
    :param y_true: True Positive
    :param y_pred: Predicated Yes (TP+FP)
    :return: dictionary with metrics-value
    '''
    precision, recall, thresh = m.precision_recall_curve(y_true, y_pred)
    return {"accuracy": m.accuracy_score(y_true, y_pred),
            "precision": m.precision_score(y_true, y_pred),
            "recall": m.recall_score(y_true, y_pred),
            "F-Scores": m.f1_score(y_true, y_pred),
            "Matthews Correlation Coefficient": m.matthews_corrcoef(y_true, y_pred),
            "Balanced Accuracy": m.balanced_accuracy_score(y_true, y_pred),
            "Youden’s J statistics": m.balanced_accuracy_score(y_true, y_pred, adjusted=True),
            "Area Under Curve for Precision-Recall Curve": m.auc(recall, precision),
            "Area Under Curve for Receiver Operation Curve": m.roc_auc_score(y_true, y_pred)}

def create_metrics_df(t: float, model: str, d):
    '''
    Plot metrics depending on threshold changes
    :param t: step of threshold
    :param d: dataframe
    :param model: the name of model
    '''
    threshold = np.arange(0, 1 + t, t).tolist()
    m = pd.DataFrame(columns=['accuracy', 'precision', 'recall', 'F-scores', 'MCC', 'balanced accuracy', 'Youden’s J stat',
                              'under PR', 'under ROC', 'threshold'])
    for element in threshold:
        d["M_pred"] = get_model_predicate(model, element, d)
        metrics_t = metrics_calculate(d["GT"], d["M_pred"])
        metrics_t["threshold"] = element
        m.loc[len(m.index)] = list(metrics_t.values())
    return m

def draw_metrics_plots(t: float, d):
    '''
    Plot the metrics results depending on threshold
    :param t: step of threshold
    :param d: dataframe old or new
    :return: dataframe with all metrics values
    '''
    m1 = create_metrics_df(t, "Model_1", d)
    m2 = create_metrics_df(t, "Model_2", d)
    m = pd.merge(m1, m2, on='threshold')
    m.set_index('threshold', inplace=True)
    max_values = list(m.max())
    max_indexes = list(m.idxmax())
    stl = []
    for i in range(0, 9): stl.append(LINE_MODEL_1)
    for i in range(0, 9): stl.append(LINE_MODEL_2)
    ax = m.plot.line(title='Значення метрик', legend=True, style=stl)
    plt.plot(max_indexes, max_values, 'o')
    ax.legend(ncol=2)
    plt.show()
    return m

def create_pr_curve(d):
    '''
    Plot PR-curve
    :param d: dataframe old or new
    '''
    precision, recall, thresholds = m.precision_recall_curve(d['GT'], d['Model_1'])
    f1_scores = 2 * (precision * recall) / (precision + recall)
    optimal_idx = np.argmax(f1_scores)
    plt.plot(recall, precision, label="Model_1")
    plt.scatter(recall[optimal_idx], precision[optimal_idx])

    precision, recall, thresholds = m.precision_recall_curve(d['GT'], d['Model_2'])
    f1_scores = 2 * (precision * recall) / (precision + recall)
    optimal_idx = np.argmax(f1_scores)
    plt.plot(recall, precision, label="Model_2")
    plt.scatter(recall[optimal_idx], precision[optimal_idx])

    plt.title("PR-curve")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend()
    plt.show()


def create_roc_curve(d):
    '''
    Plot ROC-curve
    :param d: dataframe old or new
    '''
    fpr, tpr, thresholds = m.roc_curve(d['GT'], d['Model_1'])
    optimal_idx = np.argmax(tpr - fpr)
    plt.plot(fpr, tpr, label='Model_1')
    plt.scatter(fpr[optimal_idx], tpr[optimal_idx])

    fpr, tpr, thresholds = m.roc_curve(d['GT'], d['Model_2'])
    optimal_idx = np.argmax(tpr - fpr)
    plt.plot(fpr, tpr, label='Model_2')
    plt.scatter(fpr[optimal_idx], tpr[optimal_idx])

    plt.title("ROC curve")
    plt.legend()
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.show()

def create_classifier_plot(t: float, m, d, model: str):
    '''
    Plot classifier score depending on the threshold
    :param t: step of threshold
    :param m: dataframe with metrics values
    :param d: dataframe new or old
    '''
    class_df = pd.DataFrame(columns=['threshold', 'Class 0', 'Class 1'])
    threshold = np.arange(0, 1 + t, t).tolist()

    for element in threshold:
        vals = get_model_predicate(model, element, d).value_counts()
        if list(vals.index)[0]==0 and len(list(vals.index))==2:
            class_df.loc[len(class_df.index)] = [element, vals[0], vals[1]]
        elif list(vals.index)[0]==1: class_df.loc[len(class_df.index)] = [element, 0, vals[1]]
        else: class_df.loc[len(class_df.index)] = [element, vals[0], 0]

    optimal_threshold = list(m.idxmax())
    class_df.set_index('threshold', inplace=True)

    ax = class_df.plot.line()
    for element in optimal_threshold:
        ax.axvline(element, color='g', linestyle='--')

    plt.xticks(np.arange(0, 1+t, t))
    plt.title('Оцінка класифікатора')
    plt.show()



if __name__ == '__main__':
    '''Чи збалансований набір даних?'''
    print('[class 0, class 1]')
    vals = list(df['GT'].value_counts())
    print(vals)
    if vals[0]==vals[1]: print('data is balanced')
    else: print('data is unbalanced')

    '''Обчислити всі метрики для кожної моделі'''
    threshold = float(input('Enter threshold: '))

    df["M1_pred"] = get_model_predicate('Model_1', threshold, df)
    df["M2_pred"] = get_model_predicate('Model_2', threshold, df)

    print('----------METRICS----------')
    print('Model_1: ')
    print(metrics_calculate(df['GT'], df['M1_pred']))
    print('Model_2: ')
    print(metrics_calculate(df['GT'], df['M2_pred']))

    '''Збудувати на одному графіку в одній координатній системі
    (величина порогу; значення метрики) графіки усіх обчислених
    метрик, відмітивши певним чином максимальне значення кожної
    з них'''
    metrs = draw_metrics_plots(threshold, df)
    print('created chart with METRICS VALUES')

    '''Збудувати в координатах (значення оцінки класифікаторів; 
    кількість об’єктів кожного класу) окремі для кожного класу
    графіки кількості об’єктів та відмітити вертикальними лініями
    оптимальні пороги відсічення для кожної метрики'''
    print('created CLASSIFIER GRAPH for MODEL 1')
    create_classifier_plot(threshold, metrs, df, 'Model_1')
    print('created CLASSIFIER GRAPH for MODEL 2')
    create_classifier_plot(threshold, metrs, df, 'Model_2')

    '''Збудувати для кожного класифікатору PR-криву та ROC-криву, 
    показавши графічно на них значення оптимального порогу.'''
    print('created PR-CURVE')
    create_pr_curve(df)
    print('created ROC-CURVE')
    create_roc_curve(df)

    '''Створити новий набір даних, прибравши з початкового набору 
    (50 + 10К)% об’єктів класу 1, вибраних випадковим чином.'''
    print('CREATING NEW DATASET...')
    date = '10-09'
    print('Date: ', date)
    new_date = date.split('-')
    if new_date[1][0] == '0': month = int(new_date[1][1])
    else: month = int(new_date[1])
    k = month % 4
    percent = (50 + 10*k)/100
    print('Remove ', percent, '% of class 1')
    row_num = int(vals[1]*percent)
    print('Total: ', row_num, 'rows')
    df_new = df.copy()
    rows_to_del = df_new[df_new['GT'] == 1].sample(n=row_num).index
    df_new = df_new.drop(rows_to_del)
    print(df_new['GT'].value_counts())
    df_new.to_csv('new.csv', index=False)
    print('Created new.csv with new dataset')

    '''Виконати дії п.3 для нового набору даних.'''
    print('----------METRICS----------')
    df_new["M1_pred"] = get_model_predicate('Model_1', threshold, df_new)
    df_new["M2_pred"] = get_model_predicate('Model_2', threshold, df_new)

    print('Model_1: ')
    print(metrics_calculate(df_new['GT'], df_new['M1_pred']))
    print('Model_2: ')
    print(metrics_calculate(df_new['GT'], df_new['M2_pred']))
    metrs = draw_metrics_plots(threshold, df_new)
    print('created CLASSIFIER GRAPH for MODEL 1')
    create_classifier_plot(threshold, metrs, df_new, 'Model_1')
    print('created CLASSIFIER GRAPH for MODEL 2')
    create_classifier_plot(threshold, metrs, df_new, 'Model_2')
    print('created PR-CURVE')
    create_pr_curve(df_new)
    print('created ROC-CURVE')
    create_roc_curve(df_new)