import os

import pytest
import torch

from tests import _PATH_DATA


def test_data():

    print("Loading data")
    try:
        train_data = torch.load(os.path.join(_PATH_DATA,'processed','train.pt'))
        test_data = torch.load(os.path.join(_PATH_DATA,'processed','test.pt'))
    except Exception as e:
        pytest.skip("Data files not found")
        
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
        
