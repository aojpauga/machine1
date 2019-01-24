import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.model_selection
import sklearn.preprocessing


def getData(csvFile):
    df = pd.read_csv(csvFile, index_col = 0)
    # df = df[(df.labels > -1500) & (df.labels < 0) ]
    # df = df[(df.x1 > 11.5) & (df.x1 < 13)]
    # df = df[(df.x2 > 10) & (df.x2 < 50)]
    # df = df[ df.x3 < 80]
    # df = df[(df.x5 > 8)]
    # df = df[(df.x7 > 5)& (df.x7 < 120) ]
    # df.to_csv('a-cut.csv')

    return df

def split_data( data, ratio ):
    data_train, data_test = sklearn.model_selection.train_test_split( data, test_size=ratio )
    data_train.to_csv( "data-train.csv" )
    data_test.to_csv( "data-test.csv" )
    return

def displayData(data):
    data.hist( bins = 50, figsize=(13, 7))
    plt.figure(2, figsize=(13, 7))

    plt.subplot(2, 4, 1)
    plt.scatter(data['labels'], data['labels'], s=1)

    plt.subplot(2, 4, 2)
    #plt.ylim(11, 13)
    plt.scatter(data['labels'], data['x1'], s=1)

    plt.subplot(2, 4, 3)
    plt.scatter(data['labels'], data['x2'], s=1)

    plt.subplot(2, 4, 4)
    plt.scatter(data['labels'], data['x3'], s=1)

    plt.subplot(2, 4, 5)
    plt.scatter(data['labels'], data['x4'], s=1)

    plt.subplot(2, 4, 6)
    plt.scatter(data['labels'], data['x5'], s=1)

    plt.subplot(2, 4, 7)
    plt.scatter(data['labels'], data['x6'], s=1)

    plt.subplot(2, 4, 8)
    plt.scatter(data['labels'], data['x7'], s=1)

    plt.show()
    return

def main():
    data = getData('aj.csv')
    displayData(data)
    split_data(data, 0.20)
    print(data)
    return

if __name__=="__main__":
    main()