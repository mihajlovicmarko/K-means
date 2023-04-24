class K_Means():
    def __init__(self, X):
        import random
        self.number_of_clusters_maximal = 4
        self.cost_exp_per_cluster = 5
        self.X = X
        
    def PointCost(self, mean_point, mean_point_confidence, point, point_confidance):
        cost = np.sqrt(np.square(mean_point[0] - point[0]) + np.square(mean_point[1] - point[1])) / (mean_point_confidence * 0.0)
        return cost
        
        
    def RandomStart(self, number_of_clusters):
        X = self.X
        self.cluster_means = {}
        self.dim = X.shape[1]
        dim = self.dim
        
        mean = self.SetNewMeans(1, X)
        
        for i in range(1, number_of_clusters+1):
            #self.cluster_means[str(i)] = (np.random.randint(-int(dim / 2), int(dim / 2)) + mean[0], mean[1] + np.random.randint(-np.int16(dim / 2), np.int16(dim / 2)))
            self.cluster_means[str(i)] = (np.random.randint(0, dim), np.random.randint(0, dim))
    def SetNewMeans(self, number_of_clusters, Y):
        for i in range(1, number_of_clusters+1):
            Y1 = np.equal(Y, i)
            Y1 = Y1 * Y
            
            w_weighted_sum = 0
            h_weighted_sum = 0
            
            
            for w in range(Y1.shape[0]):
                h_weighted_sum += np.sum(Y1[w]) * w
            #print(h_weighted_sum)
            self.h_avg = h_weighted_sum / (np.sum(Y1) + 1e-5)

            for h in range(Y1.shape[1]):
                w_weighted_sum += np.sum(Y1[:, h]) * h
            self.w_avg = w_weighted_sum / (np.sum(Y1) + 1e-5)        
           # print(np.sum(Y1)        )
            
            self.cluster_means[str(i)] = (self.w_avg, self.h_avg)
        if number_of_clusters == 1:
            return(self.w_avg, self.h_avg)
    
    def CalculateDeviation(self, mean):
        X = self.X
        Xw = np.sum(X, axis = 0)
        Xh = np.sum(X, axis = 1)
        ww1 = 0
        ww2 = 0
        
        hw1 = 0
        hw2 = 0
        
        
        for w in range(mean[0], X.shape[0]):
            ww1 += Xw[w] * (w - mean[0])
        
        for w in range(0, mean[0]):
            ww2 += Xw[w] * (mean[0] - w)
            
            
        for h in range(mean[1], X.shape[1]):
            hw1 += Xh[h] * (h - mean[1])
        
        for h in range(0, mean[1]):
            hw2 += Xh[h] * (mean[1] - h)
            
        return (ww1, ww2, hw1, hw2)
    
    def AssigningToClusters(self, n_c, n_steps, repeat_times):
        self.n_c = n_c 
        X = self.X    
        Y =  np.zeros(X.shape)
        bestY = Y
        LOSSMIN = 1e20
        for k in range(repeat_times):
            self.RandomStart(n_c)
            for j in range(n_steps):
                LOSS = 0
                if j % 10 == 0:
                    print(j / n_steps, end = "\r")
                for w in range(self.dim):
                    for h in range(self.dim):
                        if np.sum(X[w, h]) > 1e-3:
                            minLoss = 1e6
                            cMin = 0
                            for c in range(1, n_c+1):
                                w1, h1 = np.int16(self.cluster_means[str(c)])
                                loss = self.PointCost(self.cluster_means[str(c)], X[w1, h1], (w, h), X[w, h])
                                if np.sum(loss) < np.sum(minLoss):
                                    minLoss = loss
                                    cMin = c
                            Y[w, h] = cMin

                            LOSS += minLoss

                self.SetNewMeans(n_c, Y)
                self.Y = Y
            print(LOSS)
            if LOSS < LOSSMIN:
                #print(LOSS)
                LOSSMIN = LOSS
                bestY = Y
            
        return bestY, LOSS
    
    def Cords(self):
        yMax = 0
        for i in range(1, self.n_c+1):
            y = np.sum(self.Y * self.X)
            if y > yMax:
                iMax = i
                yMax = y
                
        return self.cluster_means[str(iMax)]
    
    def NormalDistribution(self, position,  mean, sdev):
        #print("aa")
        
        k = mean[0] - position[0] > 0
        k1 = mean[1] - position[1] > 0
        
        y = np.exp((1 - k) * -(mean[0] - position[0]) ** 2 / sdev[1] + k * -(mean[0] - position[0]) ** 2 / sdev[0] - k1 * (mean[1] - position[1]) ** 2 / sdev[2] + (1 - k1) * -(mean[1] - position[1]) ** 2 / sdev[3])
        return y 
    
    def ReducedPrediction(self, optClusterMeans, sdev):
        Y = np.zeros(self.X.shape)
        X = self.X
        for w in range(optClusterMeans[0] - 60, optClusterMeans[0] + 60):
            for h in range(optClusterMeans[1] - 60, optClusterMeans[1] + 60):
                try:
                    Y[w, h] = X[w, h] * self.NormalDistribution((w, h), optClusterMeans, sdev)
                except:
                    pass
        return(Y)
                    
                        
    
