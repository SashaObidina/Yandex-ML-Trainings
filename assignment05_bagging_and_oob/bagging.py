import numpy as np

class SimplifiedBaggingRegressor:
    def __init__(self, num_bags, oob=False):
        self.num_bags = num_bags
        self.oob = oob
        
    def _generate_splits(self, data: np.ndarray):
        '''
        Generate indices for every bag and store in self.indices_list list
        '''
        self.indices_list = []
        data_length = len(data)
        for bag in range(self.num_bags):
            # Your Code Here
          # выбираем случайные data_length индексов из индексов исходной выборки
          indices = np.random.choice([i for i in range(data_length)], size=data_length, replace=True)
          # добавляем полученный список индексов в список списков индексов
          self.indices_list.append(indices)
        
    def fit(self, model_constructor, data, target):
        '''
        Fit model on every bag.
        Model constructor with no parameters (and with no ()) is passed to this function.
        
        example:
        
        bagging_regressor = SimplifiedBaggingRegressor(num_bags=10, oob=True)
        bagging_regressor.fit(LinearRegression, X, y)
        '''
        self.data = None
        self.target = None
        self._generate_splits(data)
        assert len(set(list(map(len, self.indices_list)))) == 1, 'All bags should be of the same length!'
        assert list(map(len, self.indices_list))[0] == len(data), 'All bags should contain `len(data)` number of elements!'
        self.models_list = []
        for bag in range(self.num_bags):
            model = model_constructor()
            data_bag, target_bag = [data[i] for i in self.indices_list[bag]], [target[i] for i in self.indices_list[bag]] # Your Code Here
            self.models_list.append(model.fit(data_bag, target_bag)) # store fitted models here
        if self.oob:
            self.data = data
            self.target = target
        
    def predict(self, data):
        '''
        Get average prediction for every object from passed dataset
        '''
        # Your code here
        models_preds = [0]*len(data)
        for model in self.models_list:
            models_preds += model.predict(data)
        average_pred = models_preds/self.num_bags
        return average_pred
    
    def _get_oob_predictions_from_every_model(self):
        '''
        Generates list of lists, where list i contains predictions for self.data[i] object
        from all models, which have not seen this object during training phase
        '''
        list_of_predictions_lists = [[] for _ in range(len(self.data))]
        # Your Code Here
        data_length = len(self.data)
        for i in range(data_length):
            obj = self.data[i].reshape(1, -1)
            models_preds = []
            for bag in range(self.num_bags):
                if i not in self.indices_list[bag]:
                    models_preds.append(float(self.models_list[bag].predict(obj)))
            list_of_predictions_lists[i] = models_preds
        
        self.list_of_predictions_lists = np.array(list_of_predictions_lists, dtype=object)
    
    def _get_averaged_oob_predictions(self):
        '''
        Compute average prediction for every object from training set.
        If object has been used in all bags on training phase, return None instead of prediction
        '''
        self._get_oob_predictions_from_every_model()
        self.oob_predictions = [0]*len(self.data) # Your Code Here
        for i, models_preds in enumerate(self.list_of_predictions_lists):
            if not models_preds:
                self.oob_predictions[i] = None
            else:
                self.oob_predictions[i] = np.mean(models_preds)
        
        
    def OOB_score(self):
        '''
        Compute mean square error for all objects, which have at least one prediction
        '''
        self._get_averaged_oob_predictions()
        score = 0
        for i, oob_pred in enumerate(self.oob_predictions):
          if oob_pred != None:
            score += (oob_pred - self.target[i])**2
        return score/len(self.target) # Your Code Here
        #return np.nanmean((self.oob_predictions - self.target)**2)