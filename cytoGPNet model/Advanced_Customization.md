## Advanced Customization

### 1. Customizing GP Kernel Functions

By default, **cytoGPNet** uses **additive Scaled RBF kernels** in the GP layer. The current implementation in `loadmodel.py` creates multiple RBF kernels for different dimensions.

#### Current Implementation:

The kernel is defined in the `GaussianProcessLayer` class in `loadmodel.py`:

```python
# First kernel for the initial dimensions
self.first_kernel = gpytorch.kernels.ScaleKernel(
    gpytorch.kernels.RBFKernel(ard_num_dims=input_dim, active_dims=tuple(range(input_dim)),
                               lengthscale_prior=gpytorch.priors.SmoothedBoxPrior(
                                   math.exp(-1), math.exp(1), sigma=0.1, transform=torch.exp)))
self.covar_module = self.first_kernel

# Additional kernels for multiple marker clusters (if applicable)
for i in range(1, int(inducing_points.shape[1] / input_dim)):
    self.additional_kernel = gpytorch.kernels.ScaleKernel(
        gpytorch.kernels.RBFKernel(ard_num_dims=input_dim,
                                   active_dims=tuple(range(input_dim*i, input_dim*(i+1))),
                               lengthscale_prior=gpytorch.priors.SmoothedBoxPrior(
                                   math.exp(-1), math.exp(1), sigma=0.1, transform=torch.exp)))
    self.covar_module += self.additional_kernel
```
#### Step-by-Step Modification:
To modify the kernel function:
1. Edit the `GaussianProcessLayer` class in `loadmodel.py` (around lines 155-170).
2. Replace the kernel initialization with your desired kernel:
**Single Matérn Kernel:**
```python
self.covar_module = gpytorch.kernels.ScaleKernel(
    gpytorch.kernels.MaternKernel(nu=1.5, ard_num_dims=input_dim))
```
**Polynomial Kernel:**
```python
self.covar_module = gpytorch.kernels.ScaleKernel(
    gpytorch.kernels.PolynomialKernel(power=2, ard_num_dims=input_dim))
```
**Spectral Mixture Kernel:**
```python
self.covar_module = gpytorch.kernels.SpectralMixtureKernel(
    num_mixtures=4, ard_num_dims=input_dim)
```
**Custom Additive Kernel (different from default):**
```python
rbf_kernel = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims=input_dim))
matern_kernel = gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel(nu=2.5, ard_num_dims=input_dim))
linear_kernel = gpytorch.kernels.ScaleKernel(gpytorch.kernels.LinearKernel(num_dimensions=input_dim))

self.covar_module = rbf_kernel + matern_kernel + linear_kernel
```
3. Adjust priors and constraints if needed:
```python
# Set lengthscale constraints
self.covar_module.base_kernel.lengthscale_constraint = gpytorch.constraints.Interval(0.01, 10.0)

# Set output scale prior
self.covar_module.outputscale_prior = gpytorch.priors.GammaPrior(2.0, 0.15)
```
#### Kernel Selection Guidelines:
- **RBF Kernel**: Default choice, assumes smooth functions
- **Matérn Kernels**: More flexible, better for non-smooth functions
  - `nu=0.5`: Exponential kernel, very rough
  - `nu=1.5`: Good balance of smoothness and flexibility  
  - `nu=2.5`: Smoother than 1.5
- **Linear Kernel**: For linear relationships
- **Spectral Mixture**: For capturing periodic patterns and complex structures

### 2. Adding Covariates to the Final Classifier
The current implementation uses a `Simple_Classifier` that takes the attention output and applies logistic regression. To add covariates, you need to modify both the model architecture and the training pipeline.
#### Current Implementation:
In `loadmodel.py`, the classifier is defined as:
```python
class Simple_Classifier(nn.Module):
    def __init__(self, nz, n_out=1):
        super(Simple_Classifier, self).__init__()
        self.nz = nz
        self.n_out = n_out
        self.net = nn.Linear(nz, n_out)

    def forward(self, x):
        if x.size(1) == 1:
            return torch.sigmoid(x)
        else:
            return torch.sigmoid(self.net(x))
```
#### Step-by-Step Modification:
1. Modify the `Simple_Classifier` class in loadmodel.py:
```python
class Simple_Classifier(nn.Module):
    def __init__(self, nz, n_out=1, num_covariates=0):
        super(Simple_Classifier, self).__init__()
        self.nz = nz
        self.n_out = n_out
        self.num_covariates = num_covariates
        
        # Input size now includes both attention features and covariates
        input_size = nz + num_covariates
        self.net = nn.Linear(input_size, n_out)

    def forward(self, x, covariates=None):
        # x: attention output of shape (batch_size, nz)
        # covariates: additional features of shape (batch_size, num_covariates)
        
        if covariates is not None and self.num_covariates > 0:
            # Concatenate attention features with covariates
            x = torch.cat([x, covariates], dim=1)
        
        return torch.sigmoid(self.net(x))
```
2. Prepare your covariate data by modifying your data loading pipeline.
For example, if you have demographic/clinical covariates in your metadata:
```python
# In your data loading script (e.g., csv_to_obj.py or loaddata.py)
# Add covariate extraction from metadata

def load_covariates(metadata_file, patient_ids):
    """
    Load covariates for each patient
    Returns: tensor of shape (n_patients, n_covariates)
    """
    metadata = pd.read_csv(metadata_file)
    
    # Example: dummy variables for experimental conditions
    conditions = ['condition_A', 'condition_B', 'condition_C', 'condition_D', 'condition_E', 'condition_F']
    covariates_list = []
    
    for pid in patient_ids:
        patient_row = metadata[metadata['patient_id'] == pid].iloc[0]
        
        # Create dummy variables (one-hot encoding)
        dummy_vars = [1 if patient_row['condition'] == cond else 0 for cond in conditions]
        
        # Add other covariates (age, gender, etc.)
        # dummy_vars.extend([patient_row['age'], patient_row['gender_encoded']])
        
        covariates_list.append(dummy_vars)
    
    return torch.tensor(covariates_list, dtype=torch.float32)
```
3. Update your training script (`train_simplified.py`):
```python
# Add covariate loading
dataset = CyTOF_Dataset(datadir=fold_path, name="train_Data.obj", mode='train')
cyto_tensor = torch.from_numpy(dataset.data[1]).float()
labels = torch.tensor(dataset.data[0]["label"].values).float()
patient_ids = dataset.data[0]["patient_id"].values

# Load covariates
covariates = load_covariates(
    metadata_file=os.path.join(args.data_dir, "metadata_whole.csv"),
    patient_ids=patient_ids
)
num_covariates = covariates.shape[1]

# Create dataset with covariates
data_loader = DataLoader(
    TensorDataset(cyto_concat, labels, covariates), 
    batch_size=args.batch_size, 
    shuffle=True
)

# Initialize classifier with covariates
classifier = Simple_Classifier(nz=1, num_covariates=num_covariates).to(device)

# Update training loop
for x_batch, y_batch, cov_batch in tqdm(data_loader, desc=f"Epoch {epoch+1}"):
    x_batch = x_batch.to(device)
    y_batch = y_batch.to(device)
    cov_batch = cov_batch.to(device)  # Covariates

    # ...existing encoding and GP processing...
    
    # Attention -> pooled sample
    pooled = attention(f_samples)  # (B,)
    
    # Pass both attention output and covariates to classifier
    preds = classifier(pooled.unsqueeze(-1), covariates=cov_batch).squeeze(-1)
    
    # Rest of training loop remains the same
    loss = F.binary_cross_entropy(preds, y_batch)
```
4. Update the testing script (`test.py`) similarly:
```python
# Load test covariates
test_covariates = load_covariates(
    metadata_file=os.path.join(args.data_dir, "metadata_whole.csv"),
    patient_ids=patient_ids
)

test_loader = DataLoader(
    TensorDataset(cyto_concat, labels, test_covariates), 
    batch_size=args.batch_size, 
    shuffle=False
)

# In the inference loop
for i, (x_batch, y_batch, cov_batch) in enumerate(tqdm(test_loader, desc="Testing")):
    # ...existing processing...
    
    pooled = attention(f_samples)
    preds = classifier(pooled.unsqueeze(-1), covariates=cov_batch).squeeze(-1)
    
    # Rest remains the same
```
#### Example Covariate Types:

- **Experimental conditions**: Dummy variables for different stimulation conditions
- **Demographics**: Age (continuous), gender (binary), race (categorical → dummy variables)
- **Clinical**: Disease stage, treatment history, biomarker levels
- **Technical**: Batch identifiers, processing dates

#### Important Considerations:

- **Feature scaling**: Normalize continuous covariates to prevent dominance over attention features
- **Multicollinearity**: Ensure covariates aren't highly correlated with each other
- **Overfitting**: With small datasets, be cautious about adding too many covariates
- **Validation**: Use cross-validation to assess whether covariates improve prediction performance