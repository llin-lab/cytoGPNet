import os
import numpy as np
from scipy import io
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from models import MnistModel  # Assuming MnistModel is defined in 'models.py'
from utils import fix_gpu_memory  # Assuming fix_gpu_memory is defined in 'utils.py'

####################
# MODEL/DATA HYPERPARAMETERS
DATA_URL = '../exp/mnist/data/'
MODEL_URL = '../exp/mnist/model/'
RESULT_URL = '../exp/mnist/results/exp'

# EXPLANATION HYPERPARAMETERS
EXP_METHOD = 'BBMP'
RESULT_URL = os.path.join(RESULT_URL, EXP_METHOD)
if not os.path.isdir(RESULT_URL):
    os.mkdir(RESULT_URL)

NUM_NOISE = 64
FUSED_TYPE = 'None'
LAMBDA_1 = 1e-3
LAMBDA_2 = 1e-4
LR = 1e-2
MASK_SHAPE = (28, 28, 1)
EXP_BATCH_SIZE = 32
EXP_EPOCH = 250
EXP_DISPLAY_INTERVAL = 50
EXP_LAMBDA_PATIENCE = 20
EARLY_STOP_PATIENCE = 20
####################

# Set device to CUDA if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load data using torchvision
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = datasets.MNIST(root=DATA_URL, train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root=DATA_URL, train=False, download=True, transform=transform)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=EXP_BATCH_SIZE, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=EXP_BATCH_SIZE, shuffle=False)

# Initialize and load model
input_shape = (1, 28, 28)
num_classes = 10
test_model = MnistModel(input_shape=input_shape, num_class=num_classes)
test_model.load_state_dict(torch.load(MODEL_URL + 'local_model.pth'))  # Assuming model saved as PyTorch state_dict

test_model.to(device)
test_model.eval()

# Evaluate original testing accuracy
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = test_model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Original testing accuracy: %.2f%%' % (100 * correct / total))

# Explanation using BBMP method
if EXP_METHOD == 'BBMP':
    print('Conducting explanation using BBMP method.')
    
    # Define BBMPExp class (assuming it's defined in 'exp.py')
    from exp import BBMPExp
    
    # Initialize BBMPExp instance
    exp_test = BBMPExp(input_shape=input_shape[1:], mask_shape=MASK_SHAPE, model=test_model,
                       num_class=num_classes, optimizer=optim.Adam, lr=LR, regularizer='elasticnet')
    
    mask_true_all = []
    mask_error_all = []
    
    for i in range(len(test_dataset)):
        if (i + 1) % 20 == 0:
            print('**********************************')
            print(f'Finish explaining {i+1}/{len(test_dataset)} samples.')
            print('**********************************')
        
        # Get sample and labels
        x_exp, y_exp_true = test_dataset[i]
        x_exp = x_exp.unsqueeze(0).repeat(NUM_NOISE, 1, 1, 1).to(device)  # Repeat for noise
        y_exp_true = y_exp_true.unsqueeze(0).repeat(NUM_NOISE, 1).to(device)
        y_exp_error = torch.argmax(test_model(x_exp), dim=1).unsqueeze(1)
        
        # Fit the explanation
        mask_true = exp_test.fit(X=x_exp, y=y_exp_true, batch_size=EXP_BATCH_SIZE, epochs=EXP_EPOCH,
                                 lambda_1=LAMBDA_1, lambda_2=LAMBDA_2, display_interval=EXP_DISPLAY_INTERVAL,
                                 lambda_patience=EXP_LAMBDA_PATIENCE, early_stop_patience=EARLY_STOP_PATIENCE,
                                 fused_flag=FUSED_TYPE)
        
        mask_true_all.append(mask_true)
        
        mask_error = exp_test.fit(X=x_exp, y=y_exp_error, batch_size=EXP_BATCH_SIZE, epochs=EXP_EPOCH,
                                  lambda_1=LAMBDA_1, lambda_2=LAMBDA_2, display_interval=EXP_DISPLAY_INTERVAL,
                                  lambda_patience=EXP_LAMBDA_PATIENCE, early_stop_patience=EARLY_STOP_PATIENCE,
                                  fused_flag=FUSED_TYPE)
        
        mask_error_all.append(mask_error)
    
    mask_true_all = torch.stack(mask_true_all)
    mask_error_all = torch.stack(mask_error_all)

else:
    print('Explaining with another method.')

# Save results or further processing as needed
# io.savemat(RESULT_URL+'exp_masks_'+str(num_round), {'m_true': mask_true_all.numpy(), 'm_error': mask_error_all.numpy()})
#
# num_dis = int(np.sqrt(mask_true_all.shape[0]))
# merged_1 = utils.merge_images(x_error[0:num_dis,], mask_true_all[0:num_dis,],)
# plt.imsave(RESULT_URL + 'exp_mask_local_true_'+str(num_round), merged_1, cmap='gray')
#
# merged_2 = utils.merge_images(x_error[0:num_dis,], mask_error_all[0:num_dis,],)
# plt.imsave(RESULT_URL + 'exp_mask_local_error_'+str(num_round), merged_2, cmap='gray')