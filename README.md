# Speech-Emotion-Recognition

**Abstract:**
In this project, we explore the realm of Speech Emotion Recognition (SER) using audio data from the Toronto Emotional Speech Set (TESS) dataset. We leverage machine learning techniques to extract features from audio recordings, and then employ a Support Vector Machine classifier to recognize emotions present in the speech. Our results highlight the effectiveness of this approach in accurately categorizing emotions in speech data.

**Introduction:**
Speech Emotion Recognition (SER) is an important field of research with numerous applications in human-computer interaction, affective computing, and more. The ability to accurately recognize emotions from speech can contribute to the improvement of various systems, such as virtual assistants, call centers, and mental health applications. In this study, we focus on using the TESS dataset, which contains a diverse range of emotional speech recordings, to build a SER model.

**TESS Dataset:**
TESS was developed by a team of researchers at the University of Toronto, Canada, to provide a controlled and diverse dataset for studying the recognition of emotional states in speech. The dataset is intended to serve as a benchmark for researchers and practitioners working on speech emotion recognition tasks.
The TESS dataset covers a range of emotional expressions, including:Angry,Disgust,Fear,Happy,Neutral,Sad.
Each emotion category includes recordings of actors expressing that specific emotion through vocalizations. The dataset aims to capture a broad spectrum of emotional content to facilitate the training and evaluation of emotion recognition models.

**Data Visualization:**
Data visualization refers to the use of charts, graphs, maps, and other visual elements to present data in a visual format. The main goal is to convey complex information in a clear and intuitive manner, making it easier to understand and interpret. Visualizations can provide insights into relationships between variables, trends over time, distributions, and comparisons between different subsets of data.

**Exploratory Data Analysis (EDA):**
Exploratory Data Analysis involves systematically examining and summarizing the main characteristics of a dataset to gain a better understanding of its content. EDA techniques include generating summary statistics, visualizing data distributions, identifying missing values, and exploring relationships between variables. EDA helps analysts make decisions about data preprocessing, feature selection, and modeling strategies.

**Data Augmentation:**
Data augmentation is a technique used to artificially increase the size of a dataset by applying various transformations to the existing data. In SER, this can involve altering speech recordings by changing pitch, speed, adding noise, or shifting time. Data augmentation aims to improve the robustness and generalization of the model by exposing it to a wider variety of variations that mimic real-world conditions.

**Feature Extraction:**
Feature extraction involves transforming raw audio data into a set of relevant features that capture essential information for emotion recognition. These features can include Mel-Frequency Cepstral Coefficients (MFCCs), chroma features, spectral contrast, and more. Feature extraction reduces the dimensionality of the data while retaining meaningful information that characterizes emotional content in speech.

**Data Preparation:**
Data preparation includes preprocessing and organizing the dataset for training and evaluation. This involves tasks like normalizing audio amplitudes, segmenting recordings into smaller frames, and splitting data into training and testing sets. Proper data preparation ensures that the input data is in a suitable format for the chosen machine learning model.

**Modeling:**
Modeling in SER involves selecting an appropriate machine learning or deep learning algorithm to learn the patterns and relationships between the extracted features and emotions. Common models include Convolutional Neural Networks (CNNs), and Recurrent Neural Networks (RNNs). The model is trained using the training dataset and optimized to predict emotion labels from the extracted features.

**Confusion Matrix:**
A confusion matrix is a tool used to evaluate the performance of a classification model, particularly in the context of emotion recognition. It displays the number of true positive, true negative, false positive, and false negative predictions made by the model. The confusion matrix helps assess how well the model is classifying emotions and identify any particular emotions that might be challenging to distinguish.

**Results:**
Our experiment demonstrates promising results in speech emotion recognition.The confusion matrix reveals that the model performs well across different emotions, with some emotions achieving higher accuracy than others.In this study, we present a successful approach to Speech Emotion Recognition using the TESS dataset.These findings underscore the potential of this approach for real-world applications in human-computer interaction and affective computing, enhancing user experiences across various platforms.

**Future Work:**
While our results are promising, there are several avenues for further improvement and exploration. Some potential areas of future work include:

Ensemble Methods: Exploring ensemble techniques to combine multiple classifiers for improved performance and robustness.

Multilingual Emotion Recognition: Extending the study to incorporate multilingual datasets and examining cross-lingual speech emotion recognition.

Real-time Applications: Adapting the model for real-time applications and studying its performance in dynamic and noisy environments.

Overall, this study lays the foundation for advancing speech emotion recognition techniques and their applications in diverse domains.
