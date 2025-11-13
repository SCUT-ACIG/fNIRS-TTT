import matplotlib.pyplot as plt
import numpy as np
import os

# Configuration
datasets = ['A', 'B', 'C']  # Multiple datasets
models_name = ["fNIRS-T", "fNIRS-PreT", 'CT-Net',"fNIRS_TTT_M", "fNIRS_TTT_L"]
train_method = "KFold"
metrics = ['loss', 'acc']  # Differentiate between loss and accuracy

# Plotting styles
font_style1 = {'fontname': 'Times New Roman', 'fontsize': 20, 'weight': 'bold'}
font_style = {'fontname': 'Times New Roman', 'fontsize': 18, 'weight': 'bold'}

line_styles = {'train': '-', 'test': '--'}
colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']

# Function to load data
def load_data(path, filename):
    file_path = os.path.join(path, filename)
    with open(file_path, 'r') as f:
        data = eval(f.read())  # Converts the string representation of list into a Python list
    return np.array(data)

# Create a single figure with 2 rows and 3 columns
fig, axes = plt.subplots(2, 3, figsize=(16, 8))  # Adjust figure size as needed

# Store handles and labels for the legend
handles, labels = [], []

# Create subplots for each dataset and metric
for idx, dataset in enumerate(datasets):
    for metric_idx, metric in enumerate(metrics):
        ax = axes[metric_idx, idx]

        for model_idx, model_name in enumerate(models_name):
            path = f'save/{dataset}/{train_method}/{model_name}'
            
            # Load train and test data
            train_file = f'final_train_{metric}_history.txt'
            test_file = f'final_test_{metric}_history.txt'
            train_data = load_data(path, train_file)
            test_data = load_data(path, test_file)

            if model_name == "fNIRS_TTT_M": model_name = "fNIRS-TTT-M"
            if model_name == "fNIRS_TTT_L": model_name = "fNIRS-TTT-L"
            if model_name == "CT-Net-old": model_name = "CT-Net"

            # Plotting train and test
            train_line, = ax.plot(train_data, line_styles['train'], color=colors[model_idx], label=f'{model_name} train {metric}', linewidth=2)
            test_line, = ax.plot(test_data, line_styles['test'], color=colors[model_idx], label=f'{model_name} test {metric}', linewidth=2)

            # Collect handles and labels only once for the combined legend
            if idx == 0 and metric_idx == 0:
                handles.append(train_line)
                handles.append(test_line)

        # Labels, title
        ax.set_title(f'Dataset {dataset} {metric.capitalize()} over Epochs', **font_style1)
        ax.set_xlabel('Epochs', **font_style)
        ax.set_ylabel(f'{metric.capitalize()}', **font_style)

        if metric.lower() in ['acc', 'accuracy']:
            ax.set_ylim(65, 105) 

        # Border around the plot
        ax.spines['top'].set_linewidth(1)
        ax.spines['bottom'].set_linewidth(1)
        ax.spines['left'].set_linewidth(1)
        ax.spines['right'].set_linewidth(1)
        
        # Customize ticks font
        ax.tick_params(axis='x', labelsize=12, labelrotation=45)
        ax.tick_params(axis='y', labelsize=12)


# Modify handles and labels to remove "loss" or "acc" from legend labels
modified_labels = [label.replace(" train acc", " train").replace(" test acc", " test")
                   .replace(" train loss", " train").replace(" test loss", " test")
                   for label in [handle.get_label() for handle in handles]]

# Add the combined legend below the plots without "acc" or "loss" in labels
fig.legend(handles, modified_labels, loc='lower center', 
           bbox_to_anchor=(0.5, 0.005), fontsize=16, prop={'family': 'Times New Roman', 'size': 14}, ncol=5)

# Add the combined legend below the plots
# fig.legend(handles, [handle.get_label() for handle in handles], loc='lower center', 
#            bbox_to_anchor=(0.5, 0.02), fontsize=16, prop={'family': 'Times New Roman','size':14}, ncol=4)


# Save the figure to a file
plt.tight_layout()
plt.subplots_adjust(bottom=0.2)  # Adjust space at the bottom to fit the legend
plt.savefig('acc_loss_plot-1027.png', dpi=300)  # Save as PNG with high resolution
plt.savefig('acc_loss_plot-1027.tiff') 
plt.show()
