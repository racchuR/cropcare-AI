# CropCare AI - ML Model

This directory contains the machine learning components for crop disease detection using PyTorch.

## Structure

```
ml-model/
├── train.py              # Model training script
├── inference.py          # Standalone inference script
├── data_preparation.py   # Dataset preparation utilities
├── requirements.txt      # Python dependencies
├── data/                # Dataset directory (create this)
│   ├── train/           # Training images
│   ├── val/             # Validation images
│   └── test/            # Test images
├── saved_models/        # Trained models (created after training)
└── README.md           # This file
```

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Prepare your dataset:
```bash
# Organize raw images into train/val/test splits
python data_preparation.py --action organize --source_dir /path/to/raw/images --dataset_dir data

# Validate images
python data_preparation.py --action validate --dataset_dir data

# Plot dataset distribution
python data_preparation.py --action plot --dataset_dir data
```

## Training

Train the model:
```bash
python train.py --data_dir data --epochs 50 --batch_size 32 --lr 0.001
```

### Training Parameters

- `--data_dir`: Path to dataset directory
- `--epochs`: Number of training epochs (default: 50)
- `--batch_size`: Batch size (default: 32)
- `--lr`: Learning rate (default: 0.001)
- `--save_dir`: Directory to save models (default: saved_models)

### Training Outputs

After training, you'll find:
- `best_model.pth`: Best model based on validation accuracy
- `final_model.pth`: Final model after all epochs
- `class_names.json`: Class names mapping
- `training_history.png`: Training curves
- `confusion_matrix.png`: Validation confusion matrix

## Inference

Use the trained model for predictions:
```bash
python inference.py --image path/to/image.jpg --model saved_models/best_model.pth
```

### Inference Parameters

- `--image`: Path to input image
- `--model`: Path to model file (default: saved_models/best_model.pth)
- `--classes`: Path to class names file (default: saved_models/class_names.json)
- `--top_k`: Number of top predictions to show (default: 3)

## Model Architecture

The model uses a ResNet18 backbone with the following modifications:
- Pretrained on ImageNet
- Modified final layer for crop disease classification
- Dropout layers for regularization
- 8 output classes for different diseases

## Dataset Format

Expected dataset structure:
```
data/
├── train/
│   ├── Healthy/
│   ├── Bacterial_Blight/
│   ├── Brown_Spot/
│   └── ...
├── val/
│   ├── Healthy/
│   ├── Bacterial_Blight/
│   └── ...
└── test/
    ├── Healthy/
    ├── Bacterial_Blight/
    └── ...
```

## Disease Classes

The model is trained to detect the following crop diseases:
1. Healthy
2. Bacterial Blight
3. Brown Spot
4. Leaf Blight
5. Leaf Scald
6. Leaf Spot
7. Rust
8. Smut

## Performance Monitoring

Training includes:
- Real-time loss and accuracy tracking
- Validation metrics
- Learning rate scheduling
- Model checkpointing
- Confusion matrix generation
- Classification report

## Integration with Backend

The trained model is integrated with the FastAPI backend through:
- `backend/ml_model/inference.py`: Backend inference module
- Model loading and prediction functions
- Treatment advice generation

## Tips for Better Results

1. **Data Quality**: Ensure high-quality, well-labeled images
2. **Data Augmentation**: The training script includes various augmentations
3. **Balanced Dataset**: Try to have balanced classes
4. **Hyperparameter Tuning**: Experiment with learning rates and batch sizes
5. **Transfer Learning**: The model uses pretrained weights for better performance

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**: Reduce batch size
2. **Poor Performance**: Check dataset quality and balance
3. **Model Not Loading**: Verify model path and architecture compatibility

### Performance Optimization

- Use GPU for training if available
- Adjust batch size based on available memory
- Use data loading with multiple workers
- Enable mixed precision training for faster training
