import os
import torch
from tests import _PATH_DATA
import pytest

pytest.mark.skipif(not os.path.exists(os.path.join(_PATH_DATA,'processed','train.pt')), reason="Data files not found")
pytest.mark.skipif(not os.path.exists(os.path.join(_PATH_DATA,'processed','test.pt')), reason="Data files not found")


def test_data():
    #print(os.path.exists(os.path.join(_PATH_DATA,'processed','train2.pt')))
    #Loading data
    print("Loading data")
    train_data = torch.load(os.path.join(_PATH_DATA,'processed','train.pt'))
    test_data = torch.load(os.path.join(_PATH_DATA,'processed','test.pt'))
    
    #Testing length, shape and if all labels are represented in training set
    data = [train_data, test_data]
    label_list = []
    for d in data: 
        if str(d) == str(train_data): 
            assert len(d['images']) == 25000
        else:
            assert len(d['images']) == 5000
        for datapoint in d['images']:
            assert datapoint.shape == (1,28,28)
        for labelpoint in d['labels']:
            if labelpoint not in label_list:
                label_list.append(labelpoint.item())
        assert sorted(label_list) == [0,1,2,3,4,5,6,7,8,9]
        
#test_data()