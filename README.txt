DLRL LAB MANUAL — CLEANED PROGRAMS

Eight programs are provided as separate Python files. Comments have been removed and code formatting/syntax problems visible in the manual have been corrected. Several small robustness fixes are included; the algorithm and experiment topic are retained.

PROGRAMS
1. Word embeddings with a Skip-gram-style neural network — program1_word_embeddings.py
2. Deep neural network for Iris classification — program2_deep_neural_network.py
3. CNN for MNIST image classification — program3_cnn_mnist.py
4. Autoencoder for MNIST compression/reconstruction — program4_autoencoder_mnist.py
5. LSTM-based text classification — program5_text_classification.py
6. Seq2Seq LSTM time-series forecasting — program6_time_series_forecasting.py
7. ResNet50 transfer learning for image classification — program7_transfer_learning_resnet50.py
8. Interactive 2D Grid World — program8_grid_world.py

CHECK RESULTS IN THIS ENVIRONMENT
- All 8 files passed Python syntax compilation.
- Program 1 executed successfully.
- Program 2 executed successfully and achieved 96.67% accuracy in this run. Neural-network results vary between runs.
- Programs 3 and 4 could not complete because the MNIST dataset was not available locally and this environment could not access the internet to download it.
- Programs 5, 6, and 7 could not be executed here because TensorFlow is not installed in this environment.
- Program 8 passed syntax compilation; it is interactive and needs manual movement input to test a full game.

INSTALLATION
PyTorch experiments (1–4):
  python -m pip install torch torchvision matplotlib scikit-learn

TensorFlow experiments (5–7):
  python -m pip install tensorflow scikit-learn numpy

Run a program from this folder with:
  python program1_word_embeddings.py

NOTES
- Programs 3 and 4 download MNIST the first time they run, so an internet connection is required unless the dataset is already cached.
- Program 5 downloads the 20 Newsgroups dataset on first run, so it may also require internet access.
- Program 7 expects images arranged in data/train/<class_name>/ and data/val/<class_name>/ folders. Use the same class subfolders in both directories and provide sufficient images per class. The tiny example dataset in the manual is not a reliable evaluation dataset.
- Program 6 uses zero-valued decoder inputs as in the manual's basic demonstration. This runs as a basic sequence-to-sequence forecast, but a production forecasting model would usually use a carefully designed inference/decoder strategy and a strictly time-ordered validation split.
