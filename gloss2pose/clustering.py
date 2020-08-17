import random
from copy import deepcopy

import numpy as np
import plotly.graph_objects as go
#from dtaidistance import dtw
from tslearn.metrics import dtw
from plotly.subplots import make_subplots
from scipy import interpolate
from sklearn.cluster import AgglomerativeClustering

def cluster_and_plot(trajectories_smile, test, labels, test_probs):

    THRESHOLD = 16.0

    trajectories = deepcopy(trajectories_smile)

    num_dims = trajectories_smile[('0',)][0].shape[1]

    distanceMatrixDictionary = {}

    iteration = 1
    while True:
        distanceMatrix = np.empty((len(trajectories), len(trajectories),))
        distanceMatrix[:] = np.nan

        for index1, (filter1, trajectory1) in enumerate(trajectories.items()):
            tempArray = []

            for index2, (filter2, trajectory2) in enumerate(trajectories.items()):

                if index1 > index2:
                    continue

                elif index1 == index2:
                    continue

                else:
                    unionFilter = filter1 + filter2
                    sorted(unionFilter)

                    if unionFilter in distanceMatrixDictionary.keys():
                        distanceMatrix[index1][index2] = distanceMatrixDictionary.get(unionFilter)

                        continue

                    metric = []
                    for subItem1 in trajectory1:

                        for subItem2 in trajectory2:
                            metricw = (dtw(subItem1[:, np.array([8, 9])], subItem2[:, np.array([8, 9])]) +
                                       dtw(subItem1[:, np.array([6, 7])], subItem2[:, np.array([6, 7])]) +
                                       dtw(subItem1[:, np.array([12, 13])], subItem2[:, np.array([12, 13])]) +
                                       dtw(subItem1[:, np.array([14, 15])], subItem2[:, np.array([14, 15])])
                                       ) / 4.0

                            metricf = (dtw(subItem1[:, np.array([124, 125])], subItem2[:, np.array([124, 125])]) +
                                       dtw(subItem1[:, np.array([130, 131])], subItem2[:, np.array([130, 131])]) +
                                       dtw(subItem1[:, np.array([136, 137])], subItem2[:, np.array([136, 137])]) +
                                       dtw(subItem1[:, np.array([142, 143])], subItem2[:, np.array([142, 143])]) +
                                       dtw(subItem1[:, np.array([164, 165])], subItem2[:, np.array([164, 165])]) +
                                       dtw(subItem1[:, np.array([166, 167])], subItem2[:, np.array([166, 167])])
                                       ) / 6.0

                            metrichl = (dtw(subItem1[:, np.array([176, 177])], subItem2[:, np.array([176, 177])]) +
                                        dtw(subItem1[:, np.array([184, 185])], subItem2[:, np.array([184, 185])]) +
                                        dtw(subItem1[:, np.array([192, 193])], subItem2[:, np.array([192, 193])]) +
                                        dtw(subItem1[:, np.array([200, 201])], subItem2[:, np.array([200, 201])]) +
                                        dtw(subItem1[:, np.array([208, 209])], subItem2[:, np.array([208, 209])])
                                        ) / 5.0

                            metrichr = (dtw(subItem1[:, np.array([218, 219])], subItem2[:, np.array([218, 219])]) +
                                        dtw(subItem1[:, np.array([226, 227])], subItem2[:, np.array([226, 227])]) +
                                        dtw(subItem1[:, np.array([234, 235])], subItem2[:, np.array([234, 235])]) +
                                        dtw(subItem1[:, np.array([242, 243])], subItem2[:, np.array([242, 243])]) +
                                        dtw(subItem1[:, np.array([250, 251])], subItem2[:, np.array([250, 251])])
                                        ) / 5.0


                            metricp = (metricw + metricf + 1.5 * metrichl + 1.5 * metrichr) / 4.0
                            metric.append(metricp)
                            #metric.append(dtw(subItem1, subItem2))

                    #metric = np.mean(metric)
                    metric = max(metric)

                    distanceMatrix[index1][index2] = metric
                    distanceMatrixDictionary[unionFilter] = metric

        minValue = np.min(list(distanceMatrixDictionary.values()))

        if minValue > THRESHOLD:
            break

        minIndices = np.where(distanceMatrix == minValue)
        minIndices = list(zip(minIndices[0], minIndices[1]))

        minIndex = minIndices[0]

        filter1 = list(trajectories.keys())[minIndex[0]]
        filter2 = list(trajectories.keys())[minIndex[1]]

        trajectory1 = trajectories.get(filter1)
        trajectory2 = trajectories.get(filter2)

        unionFilter = filter1 + filter2
        sorted(unionFilter)

        trajectoryGroup = trajectory1 + trajectory2

        trajectories = {key: value for key, value in trajectories.items()
                        if all(value not in unionFilter for value in key)}

        distanceMatrixDictionary = {key: value for key, value in distanceMatrixDictionary.items()
                                    if all(value not in unionFilter for value in key)}

        trajectories[unionFilter] = trajectoryGroup

        print(iteration, 'finished!')
        iteration += 1

        if len(list(trajectories.keys())) == 1:
            break

    trajectories.update(test)

    distanceMatrix = np.empty((len(trajectories), len(trajectories),))
    distanceMatrix[:] = np.nan

    for index1, (filter1, trajectory1) in enumerate(trajectories.items()):
        tempArray = []

        for index2, (filter2, trajectory2) in enumerate(trajectories.items()):

            if index1 > index2:
                continue

            elif index1 == index2:
                continue

            else:
                unionFilter = filter1 + filter2
                sorted(unionFilter)

                if unionFilter in distanceMatrixDictionary.keys():
                    distanceMatrix[index1][index2] = distanceMatrixDictionary.get(unionFilter)

                    continue

                metric = []
                for subItem1 in trajectory1:

                    for subItem2 in trajectory2:
                        metricw = (dtw(subItem1[:, np.array([8, 9])], subItem2[:, np.array([8, 9])]) +
                                   dtw(subItem1[:, np.array([6, 7])], subItem2[:, np.array([6, 7])]) +
                                   dtw(subItem1[:, np.array([12, 13])], subItem2[:, np.array([12, 13])]) +
                                   dtw(subItem1[:, np.array([14, 15])], subItem2[:, np.array([14, 15])])
                                   ) / 4.0

                        metricf = (dtw(subItem1[:, np.array([124, 125])], subItem2[:, np.array([124, 125])]) +
                                   dtw(subItem1[:, np.array([130, 131])], subItem2[:, np.array([130, 131])]) +
                                   dtw(subItem1[:, np.array([136, 137])], subItem2[:, np.array([136, 137])]) +
                                   dtw(subItem1[:, np.array([142, 143])], subItem2[:, np.array([142, 143])]) +
                                   dtw(subItem1[:, np.array([164, 165])], subItem2[:, np.array([164, 165])]) +
                                   dtw(subItem1[:, np.array([166, 167])], subItem2[:, np.array([166, 167])])
                                   ) / 6.0

                        metrichl = (dtw(subItem1[:, np.array([176, 177])], subItem2[:, np.array([176, 177])]) +
                                    dtw(subItem1[:, np.array([184, 185])], subItem2[:, np.array([184, 185])]) +
                                    dtw(subItem1[:, np.array([192, 193])], subItem2[:, np.array([192, 193])]) +
                                    dtw(subItem1[:, np.array([200, 201])], subItem2[:, np.array([200, 201])]) +
                                    dtw(subItem1[:, np.array([208, 209])], subItem2[:, np.array([208, 209])])
                                    ) / 5.0

                        metrichr = (dtw(subItem1[:, np.array([218, 219])], subItem2[:, np.array([218, 219])]) +
                                    dtw(subItem1[:, np.array([226, 227])], subItem2[:, np.array([226, 227])]) +
                                    dtw(subItem1[:, np.array([234, 235])], subItem2[:, np.array([234, 235])]) +
                                    dtw(subItem1[:, np.array([242, 243])], subItem2[:, np.array([242, 243])]) +
                                    dtw(subItem1[:, np.array([250, 251])], subItem2[:, np.array([250, 251])])
                                    ) / 5.0

                        metricp = (1.0 * metricw + 1.0 * metricf + 1.5 * metrichl + 1.5 * metrichr) / 4
                        #e = 0.0001
                        #metricp = (metricw * 1/(test_probs[:, 0] + e) + metricf * 1/(test_probs[:,1] + e) + metrichl * 1/(test_probs[:, 2] + e) + metrichr * (1/test_probs[:, 3] + e)) / 4
                        metric.append(metricp)
                        #metric.append(dtw(subItem1, subItem2))

                #metric = max(metric)
                metric = np.mean(metric)

                distanceMatrix[index1][index2] = metric
                distanceMatrixDictionary[unionFilter] = metric

    print(distanceMatrix)
    for key, value in distanceMatrixDictionary.items():
        print(str(key) + ': ' + str(value))

    for key, value in trajectories.items():
        print(key)

        label_inds = list(map(int, key))
        figure = make_subplots(rows=1, cols=1)
        colors = ["#" + ''.join([random.choice('0123456789ABCDEF') for j in range(6)]) for i in range(len(value))]

        for index, subValue in enumerate(value):
            figure.add_trace(go.Scatter(x=list(range(0, len(subValue))), y=subValue[:, 15],
                                        mode='lines', marker_color=colors[index], line=dict(width=4), line_shape='spline', name=labels[label_inds[index]]),
                             row=1, col=1,
                             )

            '''oldScale = np.arange(0, len(subValue))
            interpolateFunction = interpolate.interp1d(oldScale, subValue)

            newScale = np.linspace(0, len(subValue) - 1, MAX_LEN_OF_TRAJECTORY)
            subValue = interpolateFunction(newScale)

            figure.add_trace(go.Scatter(x=list(range(0, len(subValue))), y=subValue,
                                        mode='lines', marker_color=colors[index]), row=1, col=2)'''

        figure.update_layout(showlegend=True, height=600, width=900)
        figure.show()

    return
