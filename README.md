\# Deep Learning for Automated ECG Clinical Metrics Extraction

This repository contains a PyTorch Lightning-based 1D Convolutional Neural Network (CNN) designed to automatically extract key clinical metrics—Heart Rate, PR interval, QRS duration, and QT interval—directly from raw electrocardiogram (ECG) waveform data. 

By automating the extraction of these intervals from high-volume waveform data, this tool aims to bridge the gap between raw diagnostic signals and the kind of rigorous clinical outcomes research that actually changes practice for large patient populations.

\#\# Dataset  
This model is trained on the \*\*PTB-XL\*\* dataset, a large publicly available clinical ECG dataset containing over 21,000 records.   
\* \*\*Inputs:\*\* Raw 12-lead ECG waveform signals.  
\* \*\*Targets:\*\* Expert-annotated clinical metrics (\`heart\_rate\_bpm\`, \`pr\_interval\_ms\`, \`qrs\_duration\_ms\`, \`qt\_interval\_ms\`).  
\* \*\*Preprocessing:\*\* Signals are Z-score normalized with a safety epsilon to prevent gradient instability, and records with missing target labels are automatically sanitized out of the training pipeline.

\#\# Model Architecture  
The core model (\`ECG\_Metrics\_Net\`) is a custom 1D-CNN optimized for time-series waveform analysis.  
\* \*\*Feature Extraction:\*\* 4 sequential 1D Convolutional blocks, each utilizing Batch Normalization, ReLU activation, and Max Pooling to hierarchically extract morphological features from the 12-lead input.  
\* \*\*Global Pooling:\*\* Adaptive Average Pooling reduces the temporal dimension before classification.  
\* \*\*Regression Head:\*\* A fully connected feed-forward network with Dropout (0.3) outputs a 4-dimensional vector corresponding to the target clinical metrics.  
\* \*\*Framework:\*\* Built with PyTorch Lightning for scalable, resume-aware training, utilizing \`AdamW\` optimization and L1 Loss (Mean Absolute Error).

\#\# Training Details  
\* \*\*Hardware:\*\* Trained on a single NVIDIA GPU via Google Colab.  
\* \*\*Stability Mechanisms:\*\* Implemented gradient clipping (\`value=0.5\`) and a reduced learning rate (\`1e-4\`) to prevent exploding gradients (\`NaN\` loss) commonly encountered with high-variance clinical data.  
\* \*\*Performance:\*\* The model successfully converged over 30 epochs, reducing the Validation Mean Absolute Error (MAE) from \~182.5 to a final plateau of \*\*\~30.11\*\*.

\#\# Repository Structure  
\* \`example\_physionet.py\`: Boilerplate code for interacting with the dataset structure.  
\* \`\*.csv\`: Target labels and SCP statements linking the waveform data to clinical annotations.  
\* \*(Note: Raw \`.dat\`, \`.hea\`, and \`.xyz\` waveform files, as well as \`.ckpt\` model weights, are excluded from version control due to size constraints. Researchers wishing to replicate this must download the PTB-XL dataset independently).\*

\#\# Future Work  
\* \*\*Inference Pipeline:\*\* Developing a standalone inference script to process novel, out-of-distribution \`.wfdb\` files for rapid clinical triage.  
\* \*\*Hyperparameter Tuning:\*\* Exploring adaptive learning rate schedulers (\`ReduceLROnPlateau\`) and heavier dropout regularization to further minimize the validation MAE.  
