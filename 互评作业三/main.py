import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import orangecontrib.associate.fpgrowth as oaf
import functools
from statsmodels.tsa.stattools import adfuller as ADF
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.tsa.arima_model import ARMA


# 关联规则挖掘
def associate_rules(data):
    listToAnalysis = []
    listToStore = []
    for i in range(data.iloc[:, 0].size):
        temp = data.iloc[i]['Name']
        listToStore.append(temp)
        temp = data.iloc[i]['Platform']
        listToStore.append(temp)
        temp = data.iloc[i]['Year']
        listToStore.append(temp)
        temp = data.iloc[i]['Genre']
        listToStore.append(temp)
        temp = data.iloc[i]['Publisher']
        listToStore.append(temp)
        temp = data.iloc[i]['NA_Sales']
        if temp < 10:
            temp = 'NA_10'
        elif 10 <= temp < 20:
            temp = 'NA_10_20'
        elif 20 <= temp < 30:
            temp = 'NA_20_30'
        elif 30 <= temp < 40:
            temp = 'NA_30_40'
        else:
            temp = 'NA_40'
        listToStore.append(temp)
        temp = data.iloc[i]['EU_Sales']
        if temp < 10:
            temp = 'EU_10'
        elif 10 <= temp < 15:
            temp = 'EU_10_15'
        elif 15 <= temp < 20:
            temp = 'EU_15_20'
        elif 20 <= temp < 25:
            temp = 'EU_20_25'
        else:
            temp = 'EU_25'
        listToStore.append(temp)
        temp = data.iloc[i]['JP_Sales']
        if temp < 10:
            temp = 'JP_10'
        elif 10 <= temp < 15:
            temp = 'JP_10_15'
        elif 15 <= temp < 20:
            temp = 'JP_15_20'
        elif 20 <= temp < 25:
            temp = 'JP_20_25'
        else:
            temp = 'JP_25'
        listToStore.append(temp)
        temp = data.iloc[i]['Other_Sales']
        if temp < 3:
            temp = 'Other_3'
        elif 3 <= temp < 6:
            temp = 'other_3_6'
        else:
            temp = 'other_6'
        listToStore.append(temp)
        temp = data.iloc[i]['Global_Sales']
        if temp < 10:
            temp = 'Global_10'
        elif 10 <= temp < 20:
            temp = 'Global_10_20'
        elif 20 <= temp < 30:
            temp = 'Global_20_30'
        elif 30 <= temp < 40:
            temp = 'Global_30_40'
        elif 40 <= temp < 50:
            temp = 'Global_40_50'
        elif temp > 50:
            temp = 'Global_50'
        listToStore.append(temp)
        listToAnalysis.append(listToStore.copy())
        listToStore.clear()

    strSet = set(functools.reduce(lambda a, b: a + b, listToAnalysis))
    strEncode = dict(zip(strSet, range(len(strSet))))
    strDecode = dict(zip(strEncode.values(), strEncode.keys()))
    listToAnalysis_int = [list(map(lambda item: strEncode[item], row)) for row in listToAnalysis]
    itemsets = dict(oaf.frequent_itemsets(listToAnalysis_int, .02))  # 支持度
    items = []

    N = len(data)
    for i in itemsets:
        temp = ''
        for j in i:
            temp = temp + str(strDecode[j]) + ' & '
        temp = temp[:-3]
        items.append([temp, round(itemsets[i] / N, 4)])
        temp = temp + ': ' + str(round(itemsets[i] / N, 4))
    pd.set_option('display.max_rows', 500)
    df = pd.DataFrame(items, columns=['频繁项集', '支持度'])
    df = df.sort_values('支持度', ascending=False)
    df.index = range(len(df))

    rules = oaf.association_rules(itemsets, .5)  # 置信度
    rules = list(rules)

    # Rules(规则前项，规则后项，支持度，置信度)
    returnRules = []
    for i in rules:
        temStr = '';
        for j in i[0]:  # 处理第一个frozenset
            temStr = temStr + str(strDecode[j]) + ' & '
        temStr = temStr[:-3]
        temStr = temStr + ' ==> '
        for j in i[1]:
            temStr = temStr + strDecode[j] + ' & '
        temStr = temStr[:-3]
        returnRules.append([temStr, round(i[2] / N, 4), round(i[3], 4)])
        temStr = temStr + ';' + '\t' + str(i[2]) + ';' + '\t' + str(i[3])
    df = pd.DataFrame(returnRules, columns=['关联规则', '支持度', '置信度'])
    df = df.sort_values('置信度', ascending=False)
    df.index = range(len(df))


if __name__ == '__main__':
    path = './game/vgsales.csv'
    data = pd.read_csv(path)
    data.info()

    # 关联规则挖掘
    # associate_rules(data)

    # Genre
    Genre_list = data['Genre'].value_counts().index
    sales_list = []
    for i in Genre_list:
        temp_data = data[data['Genre'] == i]
        sales_list.append([i, temp_data['NA_Sales'].sum(), temp_data['EU_Sales'].sum(), temp_data['JP_Sales'].sum(),
                           temp_data['Other_Sales'].sum(), temp_data['Global_Sales'].sum()])
    Genre_df = pd.DataFrame(sales_list,
                            columns=['Genre', 'NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales', 'Global_Sales'])

    # Platform
    platform_list = data['Platform'].value_counts().index
    sales_list = []
    for i in platform_list:
        temp_data = data[data['Platform'] == i]
        sales_list.append([i, temp_data['NA_Sales'].sum(), temp_data['EU_Sales'].sum(), temp_data['JP_Sales'].sum(),
                           temp_data['Other_Sales'].sum(), temp_data['Global_Sales'].sum()])
    platform_df = pd.DataFrame(sales_list,
                               columns=['Platform', 'NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales', 'Global_Sales'])

    # Publisher
    Publisher_list = data['Publisher'].value_counts().index
    sales_list = []
    for i in Publisher_list:
        temp_data = data[data['Publisher'] == i]
        sales_list.append([i, temp_data['NA_Sales'].sum(), temp_data['EU_Sales'].sum(), temp_data['JP_Sales'].sum(),
                           temp_data['Other_Sales'].sum(), temp_data['Global_Sales'].sum()])
    Publisher_df = pd.DataFrame(sales_list, columns=['Publisher', 'NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales',
                                                     'Global_Sales'])

    # Year
    Year_list = data['Year'].value_counts().index
    sales_list = []
    for i in Year_list:
        temp_data = data[data['Year'] == i]
        sales_list.append([i, temp_data['NA_Sales'].sum(), temp_data['EU_Sales'].sum(), temp_data['JP_Sales'].sum(),
                           temp_data['Other_Sales'].sum(), temp_data['Global_Sales'].sum()])
    Year_df = pd.DataFrame(sales_list,
                           columns=['Year', 'NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales', 'Global_Sales'])

    new_data = pd.DataFrame(data.groupby('Year').agg({'Global_Sales': np.sum}))
    year = data.pivot_table(index='Year', values=['JP_Sales', 'EU_Sales', 'NA_Sales', 'Global_Sales'], aggfunc=np.sum, )
    year = year.drop(year.index[38])

    diff_data = year.diff().dropna()  # 去除NA值
    diff_data.plot()

    plot_acf(new_data, lags=15).show()
    plot_pacf(new_data, lags=15).show()

    model = ARMA(new_data, order=(1, 1)).fit()
    plt.figure()
    plt.plot(new_data, label='Origin_diff')
    plt.plot(model.fittedvalues, label='Predict_diff')
    print(model.fittedvalues)
    plt.legend(loc='best')
    plt.show()

    # 直方图
    # 各游戏类型销量
    plt.figure(figsize=(15, 8))

    x = list(range(len(Genre_list)))
    width = 0.15
    Sales_item = ['Global_Sales', 'NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales']
    color = ['#ff0066', '#ff3399', '#ff6699', '#ff9999', '#ffcc99']
    for i in range(len(Sales_item)):
        if i == 2:
            plt.bar(x, Genre_df[Sales_item[i]], color=color[i], width=width, label=Sales_item[i], tick_label=Genre_list)
        else:
            plt.bar(x, Genre_df[Sales_item[i]], color=color[i], width=width, label=Sales_item[i])
        for j in range(len(x)):
            x[j] = x[j] + width
    plt.legend()
    plt.show()

    # 饼图
    explode = [0.01, 0.05, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01]
    colors = ['#EE2C2C', '#EE4000', '#EE6AA7', '#EE799F', '#EE9572', '#EEA9B8', '#EEB4B4', '#EECFA1', '#EED8AE',
              '#EEE5DE']
    plt.figure(figsize=(20, 30))
    plt.subplot(3, 2, 1)
    plt.title('Global_Sales')
    plt.pie(platform_df.loc[:9, 'Global_Sales'], explode=explode, labels=platform_list[:10], autopct='%1.1f%%',
            colors=colors)
    plt.subplot(3, 2, 2)
    plt.title('NA_Sales')
    plt.pie(platform_df.loc[:9, 'NA_Sales'], explode=explode, labels=platform_list[:10], autopct='%1.1f%%',
            colors=colors)
    plt.subplot(3, 2, 3)
    plt.title('EU_Sales')
    plt.pie(platform_df.loc[:9, 'EU_Sales'], explode=explode, labels=platform_list[:10], autopct='%1.1f%%',
            colors=colors)
    plt.subplot(3, 2, 4)
    plt.title('JP_Sales')
    plt.pie(platform_df.loc[:9, 'JP_Sales'], explode=explode, labels=platform_list[:10], autopct='%1.1f%%',
            colors=colors)
    plt.subplot(3, 2, 5)
    plt.title('Other_Sales')
    plt.pie(platform_df.loc[:9, 'Other_Sales'], explode=explode, labels=platform_list[:10], autopct='%1.1f%%',
            colors=colors)
    plt.show()

    top_p = ['Nintendo', 'Electronic Arts', 'Activision', 'Sony Computer Entertainment', 'Ubisoft']
    top_p_df = data[data['Publisher'].isin(top_p)]
    top5_genre = pd.pivot_table(data=top_p_df, index=['Genre', 'Publisher'],
                                values=['NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales', 'Global_Sales'],
                                aggfunc=np.sum)
    order = ['NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales', 'Global_Sales']  # 调整列的顺序
    new_top5_genre = top5_genre[order]
