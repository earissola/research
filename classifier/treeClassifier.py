#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys

import pandas as pd
import numpy as np

from math import ceil
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.metrics import roc_auc_score, roc_curve, auc, confusion_matrix, precision_score

def binData(series, step, bounds=None, binType='int'):
    
    if (bounds == None):
        lowerBound = series.min()
        upperBound = series.max()
    else:
        lowerBound = bounds[0]
        upperBound = bounds[1]

    if (binType == 'int'):
        bins = np.arange(0, upperBound + step, step)
    else: # binType == 'float'
        # bins = np.linspace(lowerBound, upperBound, num=step)
        bins = np.arange(upperBound, step=step)
        # np.arange(1.1, step=0.1)
    # indexData['Li_Len_Binned'] = pd.cut(indexData['Li_Len'], bins, labels=bins[1:])
    # print series[:10]
    # print pd.cut(series, bins)[:10]
    return pd.cut(series, bins, labels=bins[1:])


def main():
    indexData = pd.read_csv('../data/indexData_webbase_uint16.txt', sep='\t', header=None)
    header = ['Term', 'Li_Len', '1s_Qnt', '1rl_Avg', 'Liwo1_Len', 'C_Li_Len',
              'C_Liwo1_Len', 'Bitmap_Bytes', 'Bitmap_BytesOnDisk',
              'EWAH_bitmap_Bytes', 'EWAH_bitmap_BytesOnDisk', 'Bitmap_1s']
    indexData.columns = header

    indexData['Li_Bytes'] = indexData.Li_Len * 4
    indexData['Liwo1_Bytes'] = indexData.Liwo1_Len * 4
    indexData['C_Li_Bytes'] = indexData.C_Li_Len * 4
    indexData['C_Liwo1_Bytes'] = indexData.C_Liwo1_Len * 4
    indexData['Ones_Ratio'] = indexData['1s_Qnt'] / indexData.Li_Len

    # SS_Ratio (Space Saving): 1 - (CompressedSize / UncompressedSize) #
    indexData['SS_Ratio_Li'] = 1 - (indexData.C_Li_Bytes / indexData.Li_Bytes)
    indexData['SS_Ratio_Liwo1'] = 1 - (indexData.C_Liwo1_Bytes / indexData.Liwo1_Bytes)

    indexData['SS_Ratio_Hybrid_Un'] = 1 - (indexData.Liwo1_Bytes + indexData.Bitmap_Bytes) / indexData.Li_Bytes
    indexData['SS_Ratio_Hybrid_Co'] = 1 - (indexData.C_Liwo1_Bytes + indexData.EWAH_bitmap_Bytes) / indexData.Li_Bytes

    indexData['SS_Ratio_Hybrid_Un_woBitmap'] = 1 - (indexData.Liwo1_Bytes) / indexData.Li_Bytes
    indexData['SS_Ratio_Hybrid_Co_woBitmap'] = 1 - (indexData.C_Liwo1_Bytes) / indexData.Li_Bytes

    indexData['Target'] = indexData.SS_Ratio_Hybrid_Co > indexData.SS_Ratio_Li

    # Ver Correlacion de atributos calculados #
    features = ['Li_Len', '1s_Qnt', 'Ones_Ratio', '1rl_Avg', 'SS_Ratio_Li']
    indexData[features].corr().to_csv('correlacion.csv')
    indexData[features].describe().to_csv('describe.csv')
    # print np.arange(, step=0.1)
    # print np.linspace(.0, 1., num=11)

    # Normalize? #

    # Binning #

    indexData['Li_Len_Binned'] = binData(indexData['Li_Len'], 10000)
    indexData['1s_Qnt_Binned'] = binData(indexData['1s_Qnt'], 10000)
    indexData['Ones_Ratio_Binned'] = binData(indexData['Ones_Ratio'], 0.1, bounds=[.0, 1.1], binType='float')
    indexData['1rl_Avg_Binned'] = binData(indexData['1rl_Avg'], 10.0 )
    # indexData['SS_Ratio_Li'] = binData(indexData['1rl_Avg'], 0.1, bounds=[-.4, 1.1], binType='float')

    featuresBinned = ['Li_Len:Binned', '1s_Qnt_Binned']
    
    bound = int(ceil(indexData.shape[0] * 0.8))

    trainData = indexData.iloc[:bound][features]
    testData = indexData.iloc[bound:][features]

    trainTargets = indexData.iloc[:bound]['Target']
    testTargets = indexData.iloc[bound:]['Target']

    # False : Loose, True : Win #

    classifier = DecisionTreeClassifier(max_depth=5)
    # classifier = DecisionTreeClassifier(max_depth=10, min_samples_split=400,min_samples_leaf=2000, criterion="entropy")

    fit_classifier = classifier.fit(trainData, trainTargets)
    prediction = fit_classifier.predict(testData)

    total = float(indexData.shape[0])

    print ' ** Dataset Class Balance **'
    print 'Class - True: %.2f ' % (indexData.query('Target == True').shape[0] / total)
    print 'Class - False: %.2f ' % (indexData.query('Target == False').shape[0] / total)

    print ' ** Train Targets Balance **'
    print 'Class - True: %.2f ' % (trainTargets[trainTargets == True].shape[0] / total)
    print 'Class - False: %.2f ' % (trainTargets[trainTargets == False].shape[0] / total)

    print ' ** Test Targets Balance **'
    print 'Class - True: %.2f ' % (testTargets[testTargets == True].shape[0] / total)
    print 'Class - False: %.2f ' % (testTargets[testTargets == False].shape[0] / total)
    
    print "\n** Score ** : ", fit_classifier.score(testData, testTargets)
    roc_auc = roc_auc_score(testTargets, prediction)
    print "** Área Bajo la Curva ROC (AUC) -- (from prediction scores) ** -->", roc_auc
    precision = precision_score(testTargets, prediction)
    print "** Precision -- (from prediction scores) ** -->", roc_auc
    print precision

    cm = confusion_matrix(testTargets, prediction)
    print "** Matriz de Confusión **"
    print cm

    # Grafica del arbol de decision
    # from sklearn.externals.six import StringIO  
    # import pydot
    # from sklearn.feature_extraction import DictVectorizer
    # vec = DictVectorizer()
    # vectorized = vec.fit_transform(trainScaled.T.to_dict().values())
    # dot_data = StringIO() 
    export_graphviz(classifier, feature_names=['Li_Bytes', '1s_Qnt', 'Ones_Ratio', '1rl_Avg', 'SS_Ratio_Li'],
        filled=True, class_names=['False', 'True'], label='all', impurity=True) 
    # graph = pydot.graph_from_dot_data(dot_data.getvalue()) 
    # graph.write_pdf("tree.pdf")

    return 0

if __name__ == '__main__':
    main()