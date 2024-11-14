import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import copy
import pandas as pd
import numpy as np
from ipywidgets import IntProgress
from IPython.display import display
import random

'''
Functions utilized in the death prediction model

Note that this isn't optimized for CUDA
'''

class Model(nn.Module):
  def __init__(self, n_inputs=9, h1=70, h2=70, out_features=1):
    super().__init__()
    self.fc1 = nn.Linear(n_inputs, h1)
    self.fc2 = nn.Linear(h1, h2)
    self.out = nn.Linear(h2, out_features)

  def forward(self, x):
    x = F.relu(self.fc1(x))
    x = F.relu(self.fc2(x))
    x = self.out(x)

    return x
  

def set_seed(seed):
    '''
    Sets a specific random seed to make results consistent
    '''

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
        
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    return


def time_to_death_grouped(data, category):
    '''
    Groups predicted time until death value by selected category
      param data: inputted dataframe, includes predicted time until death and category columns, df
      param category: the category, i.e. column name, to group dataframe by, str
      return: dataframe grouped by category, df
    '''
    
    print(f'Average time to death estimate by {category}:\n')

    # group by category and calculate the mean predicted time until death
    grouped_data = data.groupby(category, as_index=False)['Predicted time until death'].mean(numeric_only=True)

    # sort the grouped data in descending order
    grouped_data = grouped_data.sort_values(by='Predicted time until death', ascending=False)
    
    # print and return the result
    print(grouped_data)
    print('\n')
    
    return grouped_data

def cross_validation(X, y, batch_size, n_iterations, scramble_trait=False, remove_trait=False):
  '''
  runs cross validation to determine loss of neural network model
    param X: all input values, df
    param y: all expected output values, df
    param batch_size: size of each batch to be run by each iteration of the NN, int
    param n_iterations: number of entries within cross validation, int
    param scramble_trait: whether to test the accuracy of the model with scrambled inputs by parameter during testing
    param remove_trait: whether to test the accuracy of the model with removed parameters during training
    return: 3 lists containing all of the models predictions, the actual values, and the loss values
  '''

  # set random seed
  set_seed(808) # arbitrary

  # get the data structures to return
  if (scramble_trait or remove_trait):
    trait_loss = {}
  else:
    all_approx = []
    all_actual = []
    all_losses = []

  # initialize the progress bar
  progress_bar = IntProgress(min=0, max=n_iterations, description=f'Training')
  display(progress_bar)

  # create a dictionary distinguishing the different subjects
  subj_indices = {}

  # get the indices distinguishing each individual subject and add them to a list
  working_subject = []
  n_subjects = 0

  # iterate over each row in the DataFrame
  for i, row in X.iterrows():
      # check if the time point is 0
      if row['time_point_in_study_weeks'] == 0:
          if i > 0:
              # append the index of the previous row (end of the previous subject)
              working_subject.append(i - 1)
              # add the start index of the current subject (the current index)
              subj_indices[f'subject_{n_subjects}'] = [working_subject[0], i - 1]
              # prepare for the next subject
              working_subject = [i]  # start new subject with the current index
              n_subjects += 1
          else:
              # this handles the case for the very first subject
              working_subject.append(i)

  # add the last subject manually, using only the first and last index
  if working_subject:
      subj_indices[f'subject_{n_subjects}'] = [working_subject[0], X.index[-1]]

  # split the dictionary into [n_iterations] subsets
  subjects = list(subj_indices.keys())
  random.shuffle(subjects)
  sample_pool = [subjects[i::n_iterations] for i in range(n_iterations)]
  
  n = 0
  while n < n_iterations:

    # set the train and test set indices depending on which iteration you are on
    train_subj = [subj_indices[item] for i, sublist in enumerate(sample_pool) if i != n for item in sublist]
    test_subj = [subj_indices[item] for item in sample_pool[n]]

    # create new dataframes in accordance with the train indices
    X_train = pd.concat([X.iloc[start:end+1] for start, end in train_subj], ignore_index=True)
    y_train = pd.concat([pd.Series(y[start:end+1]) for start, end in train_subj], ignore_index=True)

    #print(X_train.shape)

    # create new dataframes in accordance with the test indices
    X_test = pd.concat([X.iloc[start:end+1] for start, end in test_subj], ignore_index=True)
    y_test = pd.concat([pd.Series(y[start:end+1]) for start, end in test_subj], ignore_index=True)

    if scramble_trait: # scramble traits to determine their significance in the model
      model = train_nn(X_train, y_train, batch_size, bar=False)
      status = 'with_time'
      while True:
        for column in X_test.columns:
          if status == 'with_time': 
              working_data = X_test.copy()
          else: 
              working_data = X_test.copy()
              working_data['time_point_in_study_weeks'] = np.random.normal(loc=0, scale=1, size=X_test.shape[0])
              if column == 'time_point_in_study_weeks': continue
          # shuffle the target columns
          if column == 'C57BL6J or Sv129Ev': continue
          elif column == 'CD1 or C57BL6J':
              # apply noise to both strain columns
              working_data['CD1 or C57BL6J'] = np.random.normal(loc=0, scale=1, size=X_test.shape[0])
              working_data['C57BL6J or Sv129Ev'] = np.random.normal(loc=0, scale=1, size=X_test.shape[0])
              column = 'Strain'
          else:
              working_data[column] = np.random.normal(loc=0, scale=1, size=X_test.shape[0])
          # get the losses and add them to a trait/iteration specific dictionary
          losses, _, _ = test_nn(model, working_data, y_test, avg=False)
          if f'{column}_{status}' not in trait_loss:
            trait_loss[f'{column}_{status}'] = losses
          else:
            trait_loss[f'{column}_{status}'].extend(losses)
          # tell the loop to remove timepoint from now on, of if its already being removed to break out of the loop
        if status != 'no_time': status = 'no_time'
        else: break

    elif remove_trait:
      train_iter = X_train.copy()
      test_iter = X_test.copy()
      status = 'with_time'
      while True:
        for column in X:
          # indicate the current trait
          if status == 'with_time': # if time is removed or not
             pass #print(f'\nremoved trait: {column}') 
          else: 
             #print(f'\nremoved traits: {column} and timepoint') 
             if column == 'time_point_in_study_weeks': continue # if time is removed, and time is slated to be removed, skip this iteration
             else: pass
          # check the strain, we want to remove both at once
          if 'C57BL6J or Sv129Ev' in column:
             continue
          elif 'CD1 or C57BL6J' in column:
             X_train_rm = train_iter.drop(columns = [column, 'C57BL6J or Sv129Ev'])
             X_test_rm = test_iter.drop(columns = [column, 'C57BL6J or Sv129Ev'])
             column = 'Strain'
          else:
             X_train_rm = train_iter.drop(columns = [column])
             X_test_rm = test_iter.drop(columns = [column])
          # train and test the model
          model = train_nn(X_train_rm, y_train, batch_size, bar=False, print_epochs=False, print_every=0)
          losses, _, _= test_nn(model, X_test_rm, y_test, avg=False)
          print(f'{column}_{status} iteration {n} Loss: {np.mean(losses)}')
          # calculate average loss by trait
          if f'{column}_{status}' not in trait_loss:
            trait_loss[f'{column}_{status}'] = losses
          else:
            trait_loss[f'{column}_{status}'].extend(losses)
        if status == 'no_time':
          break
        else: # repeat with time removed
          status = 'no_time'
          train_iter = X_train.drop(columns = 'time_point_in_study_weeks')
          test_iter = X_test.drop(columns = 'time_point_in_study_weeks')

    else: # add the losses and values to their respective lists
      model = train_nn(X_train, y_train, batch_size, bar=False)
      losses, approx, actual = test_nn(model, X_test, y_test, avg=False)
      for a in approx:
          if a == None: break
          else: all_approx.append(a)
      for a in actual:
          if a == None: break
          else: all_actual.append(a)
      for l in losses:
          if l == None: break
          else: all_losses.append(l)
    n+=1
    progress_bar.value = n

  if (scramble_trait or remove_trait):
    all_losses_dict = copy.deepcopy(trait_loss)
    # average by trait
    for key in trait_loss:
       trait_loss[key] = np.mean(trait_loss[key])
    return trait_loss, all_losses_dict     
  
  else: # return calculated values
     return all_approx, all_actual, all_losses

def train_nn(X_train, y_train, batch_size, bar=False, print_epochs=True, print_every=50):
  '''
  trains neural network model

    param X_train: training set input, df
    param y_train: training set ouput, df
    param batch_size: batch size for neural network, i.e. number of input/output pairs computed at once, int
    param bar: whether to display and update the progress bar with each iteration
    param print_epochs: whether to print the number of epochs
    param print_every: how often to print the loss by number of epochs, if 0 doesn't print

    return: most optimal neural network model identified throughout training, pytorch object
  
  '''

  # set random seed
  set_seed(808) # arbitrary

  n_inputs = X_train.shape[1]

  X_train = X_train.values
  y_train = y_train.values

  X_train = torch.tensor(X_train, dtype=torch.float32)
  y_train = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)

  # determine the number of epochs from batch size and number of observations
  epochs = X_train.shape[0]//batch_size
  if X_train.shape[0]%batch_size > 0: epochs+=1
  if print_epochs: print(f'\nNumber of epochs: {epochs}\n')

  criterion = nn.L1Loss()
  dataset = TensorDataset(X_train, y_train)
  dataloader = DataLoader(dataset, batch_size=batch_size)
    
  while True: # certain sets of initialized weights don't train well, so when we encounter these we restart training for the iteration

    model = Model(n_inputs)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    model.train()

    # set boundaries on training inefficiency
    last_loss = 75
    best_loss = last_loss

    # display and update progress bar
    if bar:
      progress_bar = IntProgress(min=0, max=epochs, description='Training')
      display(progress_bar)

    for i in range(epochs):
        for inputs, targets in dataloader:

          outputs = model(inputs)
          loss = criterion(outputs, targets)

          optimizer.zero_grad()
          loss.backward()
          optimizer.step()

        # print every x epochs
        if print_every == 0:
           pass
        elif (i % print_every) == 0:
          print(f'Epoch: {i}, Loss: {loss}')
        
        # update best performing model
        if loss < best_loss:
           best_loss = loss
           best_model = copy.deepcopy(model)

        if bar: progress_bar.value = i+1

    return best_model

def test_nn(model, X_val, y_val, avg=True):
  '''
  Evaluates the performance of the neural network model.
  param model: trained neural network, PyTorch object
  param X_val: input values, DataFrame
  param y_val: expected output values, DataFrame
  param avg: whether to return averaged loss (True) or individual losses (False) and approx/actual, boolean
  return: 3 lists containing the model's predictions, the actual values, and the loss values, or just the loss
  '''

  model.eval()

  # convert DataFrame to numpy arrays and then to tensors
  X_val = torch.tensor(X_val.values, dtype=torch.float32)
  y_val = torch.tensor(y_val.values, dtype=torch.float32).view(-1, 1)  # ensure the correct shape
  
  criterion = nn.L1Loss()

  with torch.no_grad():

    outputs = model(X_val)
    if avg:

      # compute  and return loss
      loss = criterion(outputs, y_val)
      return loss.item()
    
    else: # get the loss for every prediction

      dataset = TensorDataset(X_val, y_val)
      dataloader = DataLoader(dataset)

      approx = []
      actual = []
      losses = []

      for inputs, targets in dataloader:

        outputs = model(inputs)
        loss = criterion(outputs, targets)

        losses.append(loss.item()) 
        approx.append(outputs)
        actual.append(targets)

      return losses, approx, actual
       
def generate_nn_pred(model, X):
  '''
  gets predictions for input values using trained NN

    param model: trained NN, torch object
    param X: input values, df

    returns: n predictions, numpy array
  '''
  model.eval()

  X = X.values
  X = torch.tensor(X, dtype=torch.float32)

  with torch.no_grad():
     outputs = model(X)
  outputs = outputs.detach().numpy() 

  return outputs   