



# Emotion Recognition From Facial Expressions
This project was apart of Machine Learning course at Birzeit University. 

Emotion recognition is one of the hot topics in computer vision and machine learning. Good emotion detectors depend heavily on the face landmarks detection, this repo will discuss a few techniques for face landmarks detection using ensemble learning, such as viola-jones and the later one-millisecond face alignment with an ensemble of regression trees, and we will examine how detecting a good landmark leads to good emotion recognition. After this, we will try different methods for feature extraction and representation, such as HOG (Histogram of Oriented Gradients) features for viola-jones algorithm and the position of the landmarks and their angle rotation with respect to the center landmark (usually the nose) for the Vahid-Sullivan algorithm.


## Approach

### Viola-Jones and HOG features
 My first approach was using the viola-jones algorithm for detecting face parts and then extracting HOG features for the localized face then training an SVM model. This simple approach resulted in surprising good results. 

The first part was face localization and detecting the face, then I cropped the face to minimize the error of unwanted background/ hair and other minor details. 

The next step is to take the hog features. 

• For each face, I resized the image to 64:128. 

• Compute the gradient vector of every pixel, as well as its magnitude and direction. 

• Divide the image into many 8x8 pixel cells. In each cell, the magnitude values of these 64 cells are binned and cumulatively added into 9 buckets of unsigned direction (no sign, so 0-180 degree rather than 0-360 degree; this is a practical choice based on empirical experiments). 

• Then we slide a 2x2 cells (thus 16x16 pixels) block across the image. In each block region, 4 histograms of 4 cells are concatenated into a one-dimensional vector of 36 values and then normalized to have unit weight. The final HOG feature vector is the concatenation of all the block vectors.

 Each 16×16 block is represented by a 36×1 vector. So when we concatenate them all into one large vector we obtain a 36×105 = 3780-dimensional vector.

 So, for each face, we will have 3780 features.

 Then I will feed the features into SVM classifier, but since it’s a binary classifier I used OneVsRest approach to train it for multiple classes. The evaluation criteria are done using 10-Fold cross-validation, as seen in the experiment section.
 
 ### Using facial landmarks relative positions
By knowing the position of each landmark in the face the variations and unwanted features will be minimized and we can train a very strong classifier. One solution is to use the positions of the landmarks as they are and train an SVM classifier on them. However, this approach is not scale-invariant nor rotation-invariant. 

To solve the scale-invariant problem, I will store the relative position of each landmark with respect to the mean landmark point.  The rotation-invariant problem can be solved by storing the angle of the slope between the landmark and the mean landmark. When doing this any rotation of the
landmarks will be considered in my classifier and this would make
it relatively rotation-invariant.

For each landmark, we will have 5 features, the original landmark point (X(i),Y(i)), the relative landmark point (X(i)Relative,Y(i)Relative) and the angle of rotation θ. So in total of 68 landmarks, we will have 68*5 = 340 features for each face. And again I will feed those features to SVM classifier with OneVsRest Approach.


## Experiments and Results

### Viola-Jones and HOG features
***

|                | HAPPY | CONTEMPT | ANGER | DISGUST  | FEAR    | SADNESS   | SURPRISE | NEUTRAL |
| ---------------|-------|----------| ----- | -------  | ------- | --------- | -------- | ------- |
| HAPPY          |69     |0         | 0     |0         |  0      | 0         | 0        |0 
| CONTEMPT       |0      |9         | 0     |0         |  0      | 0         | 1        |8 
| ANGER          |0      |0         | 35    |3         |  0      | 2         | 0        |5
| DISGUST        |1      |0         | 2     |56        |  0      | 0         | 0        |0
| FEAR           |1      |0         | 1     |0         |  18     | 0         | 2        |3
| SADNESS        |0      |0         | 4     |0         |  0      | 17        | 1        |6
| SURPRISE       |0      |1         | 0     |0         |  1      | 0         | 81       |0
| NEUTRAL        |1      |7         | 4     |1         |  0      | 3         | 1        |95 

|                | Precision |  Recall  | F1-score | Support  |   
| ---------------|-----------|----------| -------- | -------- | 
| HAPPY          |0.96       | 1.00     | 0.98     | 69  
| CONTEMPT       |0.53       | 0.50     | 0.51     | 18  
| ANGER          |0.76       | 0.78     | 0.77     | 45
| DISGUST        |0.93       | 0.95     | 0.94     | 59
| FEAR           |0.95       | 0.72     | 0.82     | 25
| SADNESS        |0.77       | 0.61     | 0.68     | 28   
| SURPRISE       |0.94       | 0.98     | 0.96     | 83
| NEUTRAL        |0.81       | 0.85     | 0.83     | 112


|                | Precision |  Recall  | F1-score | Support  |
| ---------------|-----------|----------| -------- | -------- | 
| Micro Avg      |0.87       | 0.87     | 0.87     | 439
| Macro Avg      |0.83       | 0.80     | 0.81     | 439
| Weighted Avg   |0.86       | 0.87     | 0.86     | 439

 ### Using facial landmarks relative positions
 ***
 
|                | HAPPY | CONTEMPT | ANGER | DISGUST  | FEAR    | SADNESS   | SURPRISE | NEUTRAL |
| ---------------|-------|----------| ----- | -------  | ------- | --------- | -------- | ------- |
| HAPPY          |67     | 1        | 0     | 1        | 0       | 0         | 0        | 0
| CONTEMPT       |0      | 7        | 1     | 0        | 0       | 0         | 0        | 10
| ANGER          |0      | 2        | 28    | 6        | 0       | 5         | 1        | 3
| DISGUST        |0      | 1        | 4     | 46       | 0       | 2         | 2        | 4
| FEAR           |3      | 0        | 0     | 1        | 16      | 0         | 3        | 2
| SADNESS        |0      | 4        | 1     | 0        | 1       | 9         | 1        | 12
| SURPRISE       |1      | 0        | 0     | 0        | 0       | 1         | 78       | 3
| NEUTRAL        |0      | 4        | 5     | 2        | 2       | 4         | 5        | 90 

|                | Precision |  Recall  | F1-score | Support  |   
| ---------------|-----------|----------| -------- | -------- | 
| HAPPY          |0.94       | 0.97     | 0.96     | 69
| CONTEMPT       |0.37       | 0.39     | 0.38     | 18
| ANGER          |0.72       | 0.62     | 0.67     | 45
| DISGUST        |0.82       | 0.78     | 0.80     | 59
| FEAR           |0.84       | 0.64     | 0.73     | 25
| SADNESS        |0.43       | 0.32     | 0.37     | 28
| SURPRISE       |0.87       | 0.94     | 0.90     | 83
| NEUTRAL        |0.73       | 0.80     | 0.76     | 112


|                | Precision |  Recall  | F1-score | Support  |
| ---------------|-----------|----------| -------- | -------- | 
| Micro Avg      |0.78       | 0.78     | 0.78     | 439
| Macro Avg      |0.71       | 0.68     | 0.70     | 439
| Weighted Avg   |0.77       | 0.78     | 0.77     | 439
 
 
### Testing on 10 external images

![Alt text](/externalTest.png?raw=true "Results of external test")

The model predicted 8 out of 10, this may be due to the overfitting in the neutral class because it has 112 images whereas only 18 images in contempt class. However, we can merge some classes to have a more balanced dataset and then more accurate model.

