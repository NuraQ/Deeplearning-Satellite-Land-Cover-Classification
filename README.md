This project explores land cover classification using Sentinel-2 satellite imagery through a structured deep learning pipeline. The dataset consists of 24,000 RGB images (64×64) across 9 land cover classes. The goal was to compare different learning strategies and evaluate how model complexity impacts performance and generalization.

The first stage focused on feature extraction using a pretrained VGG16 network. The convolutional base was used to generate high-level feature representations, which were then fed into classical machine learning models including Logistic Regression, Random Forest, Support Vector Machine, and XGBoost. While these models performed well, they were limited by their inability to adapt spatial representations to the new domain.

The second stage involved full transfer learning and selective fine-tuning of pretrained CNN architectures, including VGG16, ResNet50, and MobileNetV2. By unfreezing deeper convolutional blocks and incorporating data augmentation, regularization, and learning rate scheduling, the models were able to adapt domain-specific high-level features. Fine-tuned VGG16 achieved the strongest validation performance.

An ensemble approach was then implemented by combining the fine-tuned models using majority voting. While the ensemble improved robustness and reduced variance across classes, it did not surpass the strongest individual model.

The final section provides a theoretical analysis of adversarial machine learning, examining CNN vulnerabilities to gradient-based attacks, texture bias, transferability, and potential defense strategies. This analysis highlights the importance of robustness in real-world vision systems.


Technologies

Python, TensorFlow/Keras, Scikit-learn, XGBoost, NumPy, HD
