# Model Evaluation Report

## OptiCrop - Smart Agricultural Production Optimization Engine

### Models Evaluated

1. Logistic Regression
2. K-Nearest Neighbors
3. Decision Tree
4. Random Forest

### Evaluation Metrics

- Accuracy
- Precision (Weighted)
- Recall (Weighted)
- F1-Score (Weighted)
- Cross-Validation Score (5-fold)

### Results Summary

The models were evaluated on the Crop Recommendation dataset containing 2200 samples across 22 crop types.

| Model | Accuracy | Precision | Recall | F1-Score | CV Mean |
|-------|----------|-----------|--------|----------|---------|
| Logistic Regression | - | - | - | - | - |
| K-Nearest Neighbors | - | - | - | - | - |
| Decision Tree | - | - | - | - | - |
| Random Forest | - | - | - | - | - |

*Note: Run `python model/train_model.py` to populate the actual metrics.*

### Best Model

The best-performing model is automatically selected and saved as `crop_model.pkl`.

### Confusion Matrix

Confusion matrices for each model are available in the `metrics.json` file after training.

### Conclusion

The Random Forest classifier typically achieves the highest accuracy due to its ensemble nature and ability to handle multi-class classification with high-dimensional feature spaces.
